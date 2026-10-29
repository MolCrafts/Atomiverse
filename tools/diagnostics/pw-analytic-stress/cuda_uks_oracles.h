#pragma once
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "cuda_uks_detail.h"
#include "cuda_uks_fixture.h"

#include "atomiverse/cpu/planewave/internal/ewald.h"
#include "atomiverse/cpu/planewave/internal/frozen_operator_virial.h"
#include "atomiverse/cpu/planewave/internal/pw_final_state.h"
#include "atomiverse/cpu/planewave/internal/pw_forces.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include "cuda_total_values.h"
#include <numeric>
namespace independent {
using namespace ATV;
inline Mat3 identity() {
    Mat3 f;
    for (int i = 0; i < 3; ++i)
        f(i, i) = 1;
    return f;
}
inline hp::GVectorList transform(hp::GVectorList g, const Mat3& f) {
    const auto m = f.inverse().transpose();
    for (auto& x : g.g)
        x = m * x;
    g.kvec = m * g.kvec;
    for (int i = 0; i < g.count(); ++i)
        g.ekin[i] = .5 * (g.g[i] + g.kvec).norm_squared();
    return g;
}
inline cpu::Frame deform(cpu::Frame frame, const Mat3& f) {
    frame.box.set_cell(f * frame.box.cell());
    for (int i = 0; i < frame.nlocal(); ++i) {
        auto x = f
                 * Vec3{frame.span(kAtom_x)[i], frame.span(kAtom_y)[i],
                        frame.span(kAtom_z)[i]};
        frame.span(kAtom_x)[i] = x[0];
        frame.span(kAtom_y)[i] = x[1];
        frame.span(kAtom_z)[i] = x[2];
    }
    return frame;
}
struct XcPrimal {
    hp::PWWorkspace ws;
    hp::XCBuilder builder;
    explicit XcPrimal(const hp::GVectorList& g) {
        ws.nspin = 2;
        ws.gvec = &g;
        ws.n_g = g.count();
        ws.fft_nx = g.nx;
        ws.fft_ny = g.ny;
        ws.fft_nz = g.nz;
        ws.fft = std::make_unique<hp::FFTPlan3D>(g.nx, g.ny, g.nz);
        ws.nr = ws.fft->nr();
        ws.nc = ws.fft->nc();
        ws.rho_G.resize({std::size_t(g.count())});
        ws.rho_spin_G.resize(2);
        for (auto& r : ws.rho_spin_G)
            r.resize({std::size_t(g.count())});
        builder.prepare(ws);
    }
    double energy(const hp::GVectorList& g, std::span<const Complex> rho,
                  const std::array<std::vector<Complex>, 2>& spin,
                  double volume, cpu::xc::XCType type) {
        ws.gvec = &g;
        ws.omega = volume;
        std::copy(rho.begin(), rho.end(), ws.rho_G.begin());
        for (int s = 0; s < 2; ++s)
            std::copy(spin[s].begin(), spin[s].end(), ws.rho_spin_G[s].begin());
        builder.step_spin(ws, type);
        return ws.E_xc;
    }
};
inline std::array<double, 6> primal(const Mat3& f, cpu::xc::XCType type,
                                    XcPrimal& xc) {
    const auto& d = diagnostic;
    const double determinant = f.det(), volume = d.volume * determinant;
    const auto wave = transform(d.wave, f), density = transform(d.density, f);
    auto positions = d.positions;
    for (auto& x : positions)
        x = f * x;
    auto rho = d.rho_orb;
    auto spin = d.spin_orb;
    for (auto& x : rho)
        x /= determinant;
    for (auto& r : spin)
        for (auto& x : r)
            x /= determinant;
    const pw_frozen::Orbitals o{d.coefficients, d.occupations, 1};
    auto frame = deform(pw_frozen::silicon(), f);
    std::vector<Vec3> ang;
    for (int i = 0; i < frame.nlocal(); ++i)
        ang.push_back({frame.span(kAtom_x)[i], frame.span(kAtom_y)[i],
                       frame.span(kAtom_z)[i]});
    hp::EwaldSummation ionic(
        ang, std::vector<AccReal>(frame.nlocal(), d.potential->z_val),
        frame.box);
    return {
        pw_frozen::value_t(wave, o),
        pw_frozen::value_h(density, rho, volume),
        pw_frozen::value_local(*d.potential, density, positions, rho, volume),
        pw_frozen::value_nl(*d.potential, wave, positions, o, volume),
        xc.energy(density, rho, spin, volume, type),
        ionic.total_energy()};
}
inline void frozen_strain(cpu::xc::XCType type) {
    auto& d = diagnostic;
    require(d.outcome.stress_requested, "FD requires actual owner W");
    validate_fourier_density();
    XcPrimal xc(d.density);
    auto e0 = primal(identity(), type, xc);
    for (int term = 0; term < 6; ++term)
        close(d.terms[term].energy_ha, e0[term],
              1e-10 + 1e-9 * std::abs(e0[term]),
              "independent production component energy");
    double worst = 0;
    for (double h : {1e-5, 3e-6})
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                auto p = identity(), m = identity();
                p(a, b) += h;
                m(a, b) -= h;
                const auto plus = primal(p, type, xc),
                           minus = primal(m, type, xc);
                double sum = 0;
                for (int term = 0; term < 6; ++term) {
                    const double expected =
                                     -(plus[term] - minus[term]) / (2 * h),
                                 actual = d.terms[term].virial_ha[3 * a + b],
                                 gate = 1e-8 + 1e-7 * std::abs(expected);
                    close(actual, expected, gate,
                          "frozenC/f component all9 affine FD");
                    sum += expected;
                    std::printf(
                        "STRAIN_FD h=%.17g a=%d b=%d term=%d actual=%.17g "
                        "FD=%.17g error=%.17g gate=%.17g\n",
                        h, a, b, term, actual, expected,
                        std::abs(actual - expected), gate);
                }
                const double actual = d.terms[6].virial_ha[3 * a + b],
                             gate = 1e-8 + 1e-7 * std::abs(sum);
                close(actual, sum, gate, "frozenC/f total all9 affine FD");
                worst = std::max(worst, std::abs(actual - sum));
            }
    const hp::FrozenBandView o{d.coefficients, d.occupations, 1};
    const std::array<hp::FrozenOperatorVirial, 4> cpu = {
        hp::kinetic_frozen_virial(d.wave, o),
        hp::hartree_frozen_virial(d.density, d.rho_orb, d.volume),
        hp::local_potential_frozen_virial(*d.potential, d.density, d.positions,
                                          d.rho_orb, d.volume),
        hp::nonlocal_frozen_virial(*d.potential, d.wave, d.positions, o,
                                   d.volume)};
    for (int term = 0; term < 4; ++term)
        for (int j = 0; j < 9; ++j)
            close(d.terms[term].virial_ha[j], cpu[term].virial_ha[j],
                  1e-10 + 1e-9 * std::abs(cpu[term].virial_ha[j]),
                  "CPU raw fullD supplemental W");
    // Restore exact unstrained point for the CPU spin-XC tensor supplement.
    primal(identity(), type, xc);
    constexpr int a[6] = {0, 1, 2, 0, 0, 1}, b[6] = {0, 1, 2, 1, 2, 2};
    for (int j = 0; j < 6; ++j)
        close(d.terms[4].virial_ha[3 * a[j] + b[j]], xc.builder.virial_ha()[j],
              1e-10 + 1e-9 * std::abs(xc.builder.virial_ha()[j]),
              "CPU same-spin XC tensor supplement");
    std::printf(
        "PASS CUDA_PW_UKS_FROZEN_STRAIN all9=1 steps=2 SCF_additional=0 "
        "final_band_additional=0 maxTotalError_Ha=%.17g\n",
        worst);
}
inline void frozen_force() {
    auto& d = diagnostic;
    require(d.outcome.forces_requested, "ionic FD requires actual owner F");
    const pw_frozen::Orbitals o{d.coefficients, d.occupations, 1};
    auto ionic_value = [&](const cpu::Frame& frame,
                           const std::vector<Vec3>& pos) {
        std::vector<Vec3> ang;
        for (int i = 0; i < frame.nlocal(); ++i)
            ang.push_back({frame.span(kAtom_x)[i], frame.span(kAtom_y)[i],
                           frame.span(kAtom_z)[i]});
        hp::EwaldSummation ionic(
            ang, std::vector<AccReal>(frame.nlocal(), d.potential->z_val),
            frame.box);
        return pw_frozen::value_local(*d.potential, d.density, pos, d.rho_orb,
                                      d.volume)
               + pw_frozen::value_nl(*d.potential, d.wave, pos, o, d.volume)
               + ionic.total_energy();
    };
    double worst = 0;
    for (double h : {1e-5, 3e-6})
        for (int i = 0; i < 3; ++i)
            for (int a = 0; a < 3; ++a) {
                auto plus = d.positions, minus = d.positions;
                plus[i][a] += h;
                minus[i][a] -= h;
                auto fp = pw_frozen::silicon(), fm = fp;
                const auto column = a == 0   ? kAtom_x
                                    : a == 1 ? kAtom_y
                                             : kAtom_z;
                fp.span(column)[i] += h / cpu::pw::qe_units::angstrom_to_bohr;
                fm.span(column)[i] -= h / cpu::pw::qe_units::angstrom_to_bohr;
                double fd = -(ionic_value(fp, plus) - ionic_value(fm, minus))
                            / (2 * h),
                       gate = 1e-8 + 1e-7 * std::abs(fd);
                close(d.forces[i][a], fd, gate,
                      "same-state ionic full9 force FD");
                worst = std::max(worst, std::abs(d.forces[i][a] - fd));
                std::printf("FORCE_FD h=%.17g atom=%d axis=%d actual=%.17g "
                            "FD=%.17g error=%.17g gate=%.17g\n",
                            h, i, a, d.forces[i][a], fd,
                            std::abs(d.forces[i][a] - fd), gate);
            }
    std::printf(
        "PASS CUDA_PW_UKS_FROZEN_FORCE full9=1 steps=2 SCF_additional=0 "
        "final_band_additional=0 maxError_HaBohr=%.17g\n",
        worst);
}
} // namespace independent

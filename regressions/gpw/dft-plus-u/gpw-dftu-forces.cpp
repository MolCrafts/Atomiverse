/// @file regressions/gpw/gpw-dftu-forces/reg_gpw-dftu-forces.cpp
/// @brief END-TO-END: Γ-UKS GPW analytic ionic forces with Dudarev DFT+U.
///
/// The Mulliken occupation n = ½Herm[(PS)+(SP)] restricted to the correlated
/// (atom, ℓ) block depends on geometry through S, so DFT+U adds an
/// occupation-matrix force term folded as W ← W − M into the Pulay
/// contraction (`internal/dft_plus_u.h`, `subtract_dftu_pulay`).
///
/// Gates:
///   A1 Γ-UKS+U SCF payload sanity (H₂ HS toy)
///   A2 decisive dual gate: analytic vs full-6N multi-SCF FD forces with U
///   A3 translational invariance |ΣF| + Newton-3 on the bond axis
///   A4 U activation: forces move when U switches on
///   B  CP2K ENERGY_FORCE pin: periodic Γ NiO stretched molecule, PLUS_U
///      Mulliken (offline pin, tools/cp2k_oracle — see REFERENCE.md)
///
/// ctest never spawns CP2K. Public GpwDft members + forces(frame) / gpw_forces_fd only.
/// Exit non-zero on failure. Target ≤ 60 s Release (8 threads).

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"
#include "harness/registry.h"

namespace atv_reg_gpw_gpw_dftu_forces {

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

using Forces = std::vector<std::array<double, 3>>;

/// H₂ toy in a cubic box of edge `L` [Bohr]; the Frame is in Å.
ATV::cpu::Frame h2_hs(double L) {
    using namespace ATV;
    constexpr double b2a = units::kBohrInAngstrom;
    // Compressed bond along x (same as gpw-dftu-kpoint-forces A).
    const double x[2] = {0.5 * L - 0.5, 0.5 * L + 0.5};
    cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_x)[i] = x[i] * b2a;
        frame.span(kAtom_y)[i] = 0.5 * L * b2a;
        frame.span(kAtom_z)[i] = 0.5 * L * b2a;
    }
    frame.box = cpu::Box::cubic(L * b2a);
    return frame;
}

/// Frame forces [eV/Å] → Ha/Bohr (the pins' unit).
Forces frame_forces_ha_bohr(const ATV::cpu::Frame& frame) {
    using namespace ATV;
    Forces F(static_cast<std::size_t>(frame.nlocal()));
    for (int i = 0; i < frame.nlocal(); ++i)
        F[static_cast<std::size_t>(i)] = {
            frame.span(kAtom_fx)[i] / units::kHartreePerBohrInEvPerAngstrom,
            frame.span(kAtom_fy)[i] / units::kHartreePerBohrInEvPerAngstrom,
            frame.span(kAtom_fz)[i] / units::kHartreePerBohrInEvPerAngstrom};
    return F;
}

/// eV/Å → Ha/Bohr, in place.
Forces to_ha_bohr(Forces F) {
    for (auto& f : F)
        for (double& c : f)
            c /= ATV::units::kHartreePerBohrInEvPerAngstrom;
    return F;
}

double max_abs_diff(const std::vector<std::array<double, 3>>& a,
                    const std::vector<std::array<double, 3>>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        for (int k = 0; k < 3; ++k)
            m = std::max(m, std::abs(a[i][k] - b[i][k]));
    return m;
}

} // namespace

int run() {
    using namespace ATV;
    using namespace ATV::cpu::gpw;
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    const double L = 12.0;
    cpu::Frame frame = h2_hs(L);
    // U−J = 2 eV: single-branch SCF on this DZV s-shell toy (4 eV is
    // multi-branch / unphysical Mulliken — see projector-divergence note).
    constexpr double kU_ev = 2.0;

    auto make_g = [&](double u_ev) {
        GpwDft m{{Kind{molrs::Element::H,
                       GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
                       GthPotential(molrs::Element::H, "GTH-PADE-q1")}},
                 100.0,
                 XcKind::PADE};
        m.n_alpha = 2;
        m.n_beta = 0;
        m.max_iter = 120;
        m.tol = 1e-8;
        m.mix = 0.35;
        m.local_pp_on_grid = true;
        m.nonperiodic = false;
        if (u_ev > 0.0)
            m.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, u_ev});
        return m;
    };

    // A1: Γ-UKS + U SCF payload
    GpwDft energy = make_g(kU_ev);
    energy.build(frame);
    const auto& er = energy.run(frame);
    check(er.converged, "A1 SCF converged");
    check(std::isfinite(er.e_total) && std::isfinite(er.e_dftu),
          "A1 finite e_total / e_dftu");
    check(std::fabs(er.e_dftu) > 1e-8, "A1 e_dftu non-trivial");
    std::printf("E(Γ-UKS+U, 100 Ry) = %.12f Ha  e_dftu=%.8f  iters=%d\n",
                er.e_total, er.e_dftu, er.iters);

    // A2: decisive dual gate — analytic vs full-6N multi-SCF central FD
    GpwDft m_an = make_g(kU_ev);
    m_an.build(frame);
    m_an.forces(frame);
    const Forces Fan = frame_forces_ha_bohr(frame);
    GpwDft m_fd = make_g(kU_ev);
    m_fd.build(frame);
    const Forces Ffd = to_ha_bohr(
        gpw_forces_fd(frame, m_fd, /*h=*/1e-3 * units::kBohrInAngstrom));
    for (std::size_t i = 0; i < Fan.size(); ++i)
        std::printf("  F%zu an=(% .9e % .9e % .9e)\n     fd=(% .9e % .9e "
                    "% .9e)\n",
                    i, Fan[i][0], Fan[i][1], Fan[i][2], Ffd[i][0], Ffd[i][1],
                    Ffd[i][2]);
    const double dual = max_abs_diff(Fan, Ffd);
    constexpr double kDualTol = 5.0e-5;
    std::printf("A2: max|F_an - F_fd| = %.3e Ha/Bohr (tol %.1e)\n", dual,
                kDualTol);
    check(dual <= kDualTol, "A2 analytic vs FD dual gate (U on, Γ)");

    // A3: translational invariance + Newton-3 on the bond axis
    {
        double sum[3] = {0.0, 0.0, 0.0};
        for (const auto& f : Fan)
            for (int k = 0; k < 3; ++k)
                sum[k] += f[k];
        const double s =
            std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
        std::printf("A3: |sum F| = %.3e Ha/Bohr\n", s);
        check(s <= 1e-6, "A3 |sum F| ~ 0");
        check(std::abs(Fan[0][0] + Fan[1][0]) <= 1e-6, "A3 Newton-3 on F_x");
        check(Fan[0][0] < 0.0 && Fan[1][0] > 0.0,
              "A3 compressed bond is repulsive");
    }

    // A4: the U term is live in the force path
    {
        GpwDft m0 = make_g(0.0);
        m0.build(frame);
        m0.forces(frame);
        const Forces F0 = frame_forces_ha_bohr(frame);
        const double dU = max_abs_diff(Fan, F0);
        std::printf("A4: max|F(U) - F(U=0)| = %.3e Ha/Bohr\n", dU);
        check(dU >= 1e-4, "A4 U activation floor on forces");
    }

    // B: CP2K ENERGY_FORCE pin — periodic Γ NiO stretched molecule (10 Å box,
    // SZV / PADE / U=6 eV on Ni d, Nα=13 Nβ=11). Offline CP2K 2026.1 pin
    // (OT, multigrid 250/40×4): see REFERENCE.md. ATV single grid 250 Ry.
    double wall_nio_s = 0.0;
    {
        // Geometry [Å], 10 Å cubic box.
        cpu::Frame nio(2);
        const int zz[2] = {28, 8};
        const double zpos[2] = {4.1, 5.9};
        for (int i = 0; i < 2; ++i) {
            nio.span(kAtom_type)[i] = zz[i];
            nio.span(kAtom_x)[i] = 5.0;
            nio.span(kAtom_y)[i] = 5.0;
            nio.span(kAtom_z)[i] = zpos[i];
        }
        nio.box = cpu::Box::cubic(10.0);
        GpwDft scf{{Kind{molrs::Element::Ni,
                         GaussianBasis(molrs::Element::Ni, "SZV-GTH-PADE-q18"),
                         GthPotential(molrs::Element::Ni, "GTH-PADE-q18")},
                    Kind{molrs::Element::O,
                         GaussianBasis(molrs::Element::O, "SZV-GTH-PADE"),
                         GthPotential(molrs::Element::O, "GTH-PADE-q6")}},
                   250.0,
                   XcKind::PADE};
        scf.max_iter = 300;
        scf.tol = 1e-6;
        scf.mix = 0.35;
        scf.nonperiodic = false;
        scf.local_pp_on_grid = true;
        scf.n_alpha = 13;
        scf.n_beta = 11;
        scf.plus_u.push_back(PlusU{molrs::Element::Ni, /*l=*/2, /*u_ev=*/6.0});

        const auto t_nio0 = clock::now();
        scf.build(nio);
        const auto& rr = scf.forces(nio);
        check(rr.converged, "B NiO periodic Γ-UKS+U SCF converged");
        const Forces Fnio = frame_forces_ha_bohr(nio);
        const auto t_nio1 = clock::now();
        wall_nio_s = std::chrono::duration<double>(t_nio1 - t_nio0).count();

        // CP2K 2026.1 pin (tools/cp2k_oracle/inputs/gpw-dftu-forces/
        // nio_pbc_gamma_ot_force.inp, 2026-08-01, git:5e54ba2, OMP=8):
        //   E = −184.929424563910 Ha, e_dftu = 0.04952309241681 Ha
        //   F(Ni) = (+3.18e-7, −1.92e-6, −8.35837089e-3) Ha/Bohr
        //   F(O)  = (−3.55e-8, +2.19e-6, −7.27092413e-3) Ha/Bohr
        // ENERGY asserted. Force twin is DIAGNOSTIC: CP2K PLUS_U_METHOD
        // MULLIKEN is on-site *net* P∘S (no occupation force); ATV uses
        // symmetrized *gross* ½(PS+SP) (geometry-dependent occupation force).
        // Decisive force gate is A2. See REFERENCE.md + projector notes.
        constexpr double kEcp2k = -184.929424563910;
        constexpr double kDFzCp2k = -8.35837089e-3 - (-7.27092413e-3);
        const double dFz = Fnio[0][2] - Fnio[1][2];
        std::printf("B: E=%.9f Ha (CP2K %.9f, |dE|=%.3e)  e_dftu=%.8f  "
                    "iters=%d  wall_nio=%.2f s\n",
                    rr.e_total, kEcp2k, std::fabs(rr.e_total - kEcp2k),
                    rr.e_dftu, rr.iters, wall_nio_s);
        std::printf("B: F_z(Ni)=% .6e F_z(O)=% .6e  dFz=% .6e "
                    "(CP2K net-Mulliken dFz=% .6e — model diff, diagnostic)\n",
                    Fnio[0][2], Fnio[1][2], dFz, kDFzCp2k);
        check(std::fabs(rr.e_total - kEcp2k) <= 5e-2,
              "B |E - CP2K| (single grid vs multigrid+OT)");
        check(std::fabs(Fnio[0][0]) < 5e-4 && std::fabs(Fnio[0][1]) < 5e-4,
              "B transverse force ~ 0 (axial molecule)");
        check(std::isfinite(Fnio[0][2]) && std::isfinite(Fnio[1][2]),
              "B finite axial forces");
    }

    const auto t1 = clock::now();
    const double wall_total = std::chrono::duration<double>(t1 - t0).count();
    // Fair wall vs CP2K twin (same NiO ENERGY_FORCE job, OMP=8): 13.24 s.
    constexpr double kT_cp2k = 13.24;
    const double ratio = wall_nio_s / kT_cp2k;
    std::printf("wall_total=%.1f s  wall_nio_scf+F=%.2f s  T_cp2k=%.2f s  "
                "ratio=%.3f\n",
                wall_total, wall_nio_s, kT_cp2k, ratio);
    check(ratio <= 10.0, "perf T_atv/T_cp2k <= 10 on NiO twin");

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    std::printf("PASS: gpw-dftu-forces\n");
    return 0;
}

} // namespace atv_reg_gpw_gpw_dftu_forces

ATV_REGRESSION("gpw/dft-plus-u", "gpw-dftu-forces",
               atv_reg_gpw_gpw_dftu_forces::run);

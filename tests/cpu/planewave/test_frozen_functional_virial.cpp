// Prepared-operator functional assembly, one unstrained band sweep, no SCF.
#include "atomiverse/cpu/planewave/internal/ewald.h"
#include "atomiverse/cpu/planewave/internal/frozen_functional_virial.h"
#include "atomiverse/cpu/planewave/internal/frozen_operator_virial.h"
#include "atomiverse/cpu/planewave/internal/poisson.h"
#include "atomiverse/cpu/planewave/internal/pw_final_state.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include <gtest/gtest.h>
namespace primitive_values {
using namespace ATV;
using namespace ATV::cpu::pw::internal;
using C = std::complex<AccReal>;
inline double value_local(const NcPotentialData& d, const GVectorList& g,
                          std::span<const Vec3> r, std::span<const C> rho,
                          double volume) {
    const auto vg = nc_vloc_to_gspace(d, g, volume);
    double e = 0;
    for (int i = 0; i < g.count(); ++i) {
        C structure{};
        for (const auto& a : r)
            structure += std::polar(1., -g.g[i].dot(a));
        e += volume * std::real(std::conj(rho[i]) * structure) * vg[i];
    }
    return e;
}
inline double value_nl(const NcPotentialData& d, const GVectorList& g,
                       std::span<const Vec3> r, FrozenBandView o,
                       double volume) {
    std::vector<Vec3> atoms(r.begin(), r.end());
    std::vector<std::vector<KBProjector>> proj;
    setup_kb_projectors(proj, g, d, atoms, volume);
    double e = 0;
    for (std::size_t b = 0; b < o.occupations.size(); ++b)
        for (const auto& site : proj)
            for (const auto& p : site)
                for (int m = 0; m < 2 * p.l + 1; ++m) {
                    C overlap{};
                    for (int i = 0; i < g.count(); ++i)
                        overlap += std::conj(p.beta_G(m, i))
                                   * o.coefficients[b * g.count() + i];
                    e += o.k_weight * o.occupations[b] * p.d
                         * std::norm(overlap);
                }
    return e;
}
inline double value_h(const GVectorList& g, std::span<const C> rho,
                      double volume) {
    PWWorkspace ws;
    ws.gvec = &g;
    ws.n_g = g.count();
    ws.rho_G.resize({rho.size()});
    ws.V_H_G.resize({rho.size()});
    std::copy(rho.begin(), rho.end(), ws.rho_G.begin());
    PoissonSolver poisson;
    poisson.step(ws);
    double e = 0;
    for (int i = 0; i < g.count(); ++i)
        e += .5 * volume * std::real(std::conj(rho[i]) * ws.V_H_G[i]);
    return e;
}
inline double value_t(const GVectorList& g, FrozenBandView o) {
    double e = 0;
    for (std::size_t b = 0; b < o.occupations.size(); ++b)
        for (int i = 0; i < g.count(); ++i)
            e += o.k_weight * o.occupations[b]
                 * std::norm(o.coefficients[b * g.count() + i]) * .5
                 * (g.g[i] + g.kvec).norm_squared();
    return e;
}
} // namespace primitive_values
namespace pw_frozen {
inline ATV::cpu::Frame silicon() {
    using namespace ATV;
    cpu::Frame frame(2);
    Mat3 h;
    h(0, 0) = .10;
    h(1, 0) = 2.70;
    h(2, 0) = 2.65;
    h(0, 1) = 2.75;
    h(1, 1) = .05;
    h(2, 1) = 2.70;
    h(0, 2) = 2.65;
    h(1, 2) = 2.75;
    h(2, 2) = .15;
    frame.box.set_cell(h);
    constexpr double p[2][3] = {{.2, .3, .4}, {1.57, 1.66, 1.75}};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_atomic_number)[i] = 14;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_mass)[i] = 28.0855;
        frame.span(kAtom_x)[i] = p[i][0];
        frame.span(kAtom_y)[i] = p[i][1];
        frame.span(kAtom_z)[i] = p[i][2];
    }
    return frame;
}
} // namespace pw_frozen
#include <cstdio>
using namespace ATV;
using namespace ATV::cpu::pw::internal;
using namespace primitive_values;
namespace {
void require(bool x, const char* m) {
    if (!x)
        throw std::runtime_error(m);
}
void close(double x, double y, double gate, const char* m) {
    if (!std::isfinite(x) || std::abs(x - y) > gate) {
        std::fprintf(stderr, "GATE %s got=%.17g expected=%.17g bound=%.17g\n",
                     m, x, y, gate);
        throw std::runtime_error(m);
    }
}
std::vector<C> snapshot(const PWWorkspace& w) {
    std::vector<C> v;
    auto add = [&](const auto& a) {
        for (auto x : a)
            v.emplace_back(x);
    };
    add(w.rho_G);
    add(w.rho_G_old);
    add(w.V_H_G);
    add(w.V_xc_G);
    add(w.V_eff_G);
    add(w.V_loc_G);
    add(w.wfc);
    add(w.eigenvalues);
    for (const auto& c : w.rho_spin_G)
        add(c);
    return v;
}
Mat3 identity() {
    Mat3 F;
    for (int a = 0; a < 3; ++a)
        F(a, a) = 1;
    return F;
}
cpu::Frame deform(cpu::Frame f, const Mat3& F) {
    f.box.set_cell(F * f.box.cell());
    for (int i = 0; i < f.nlocal(); ++i) {
        auto r =
            F
            * Vec3{f.span(kAtom_x)[i], f.span(kAtom_y)[i], f.span(kAtom_z)[i]};
        f.span(kAtom_x)[i] = r[0];
        f.span(kAtom_y)[i] = r[1];
        f.span(kAtom_z)[i] = r[2];
    }
    return f;
}
double primal(const cpu::Frame& f, const PWSetup& ref, const FinalBandState& s,
              bool spin, double scale, XCBuilder& xc) {
    PWSetup p;
    PWWorkspace ws;
    setup_pw_problem_fixed_basis(f, ref.strain_reference(), 12, 1, p, ws);
    auto& w = ws;
    require(p.gv0.h == ref.gv0.h && p.gv0.k == ref.gv0.k && p.gv0.l == ref.gv0.l
                && w.fft_nx == ref.gv0.nx && w.fft_ny == ref.gv0.ny
                && w.fft_nz == ref.gv0.nz,
            "fixed ordered density modes and FFT");
    for (std::size_t g = 0; g < s.orbital_density_G.size(); ++g)
        w.rho_G[g] = s.orbital_density_G[g] * scale;
    if (spin) {
        w.nspin = 2;
        w.rho_spin_G.resize(2);
        for (int c = 0; c < 2; ++c) {
            w.rho_spin_G[c].resize({s.orbital_density_G.size()});
            for (std::size_t g = 0; g < s.orbital_density_G.size(); ++g)
                w.rho_spin_G[c][g] = s.orbital_spin_density_G[c][g] * scale;
        }
    }
    xc.prepare(w);
    if (spin)
        xc.step_spin(w, cpu::xc::XCType::PBEsol);
    else
        xc.step(w, cpu::xc::XCType::PBEsol);
    double e = w.E_xc + p.ions.ew_total;
    for (const auto& b : s.local_blocks) {
        FrozenBandView v{{b.coefficients.data(), b.coefficients.size()},
                         b.occupations,
                         b.k_weight};
        e += value_t(p.mesh.gv_k[0], v)
             + value_nl(*p.ions.species.front(), p.mesh.gv_k[0],
                        w.atom_pos_bohr, v, w.omega);
    }
    const auto rho = std::span<const C>{w.rho_G.data(), w.rho_G.size()};
    return e + value_h(p.gv0, rho, w.omega)
           + value_local(*p.ions.species.front(), p.gv0, w.atom_pos_bohr, rho,
                         w.omega);
}
void run(bool spin) {
    auto f = pw_frozen::silicon();
    auto potential = std::make_shared<cpu::pw::GthPseudopotential>(
        molrs::Element::Si, "GTH-PBE-q4");
    cpu::pw::PW m{{potential}, 12, 1, cpu::xc::XCType::PBEsol};
    m.tol = 1e-10;
    m.spin_polarized = spin;
    m.multiplicity = 1;
    m.smearing =
        spin ? electronic::Smearing::fermi_dirac : electronic::Smearing::none;
    m.degauss = .08;
    m.extra_bands = 2;
    PWSetup p;
    PWWorkspace ws;
    setup_pw_problem(f, m.potentials, 12, 1, p, ws);
    if (spin) {
        ws.nspin = 2;
        ws.rho_spin_G = {.65 * ws.rho_G, .35 * ws.rho_G};
    }
    ws.V_H_G.fill(7);
    ws.V_xc_G.fill(-5);
    // Prepared state has no accepted SCF caches; define all audited stale
    // buffers so unchanged-input comparisons never read indeterminate values.
    ws.rho_G_old.fill(2);
    ws.V_eff_G.fill(-3);
    ws.eigenvalues = xt::ones<AccReal>({std::size_t(1)});
    ws.wfc = xt::ones<C>({std::size_t(1), std::size_t(1)});
    cpu::pw::PWResult reference;
    reference.energy = 17;
    reference.energy_ha = 8.5;
    reference.converged = false;
    reference.iterations = 7;
    reference.e_fermi = .2;
    reference.e_entropy = -.4;
    const auto before = snapshot(ws);
    const auto* fft = ws.fft.get();
    const auto* binding = ws.gvec;
    const auto bands = extract_final_bands(
        p.grid(), p.ions, p.mesh, ws.density(), ws.wfc, m, reference, 6);
    require(bands.density_mismatch_l2 > 1e-6,
            "must distinguish rhoorb from rhoref");
    std::vector<Vec3> positions;
    for (int i = 0; i < f.nlocal(); ++i)
        positions.push_back(
            {f.span(kAtom_x)[i], f.span(kAtom_y)[i], f.span(kAtom_z)[i]});
    EwaldSummation ew(positions, std::vector<AccReal>(positions.size(), 4),
                      f.box);
    XCBuilder scratch;
    const auto result = assemble_frozen_functional_virial(
        p.ions, p.gv0, p.mesh.gv_k, p.mesh.kg.weights, ws.density(), bands,
        m.xc, ew, scratch);
    require(result.max_antisymmetric_ha < 1e-9, "unprojected W9");
    require(std::abs(result.nonlocal.energy_ha) > 1e-3,
            "nontrivial Si nonlocal");
    close(primal(f, p, bands, spin, 1, scratch), result.total.energy_ha, 1e-11,
          "internal functional energy primal");
    auto metadata = bands;
    metadata.auxiliary_entropy += 2.75;
    metadata.auxiliary_mu_ha = 72;
    metadata.reference_result.energy = 1000;
    metadata.reference_result.energy_ha = 500;
    metadata.reference_result.e_entropy = -3;
    metadata.reference_result.iterations = 1984;
    const auto invariant = assemble_frozen_functional_virial(
        p.ions, p.gv0, p.mesh.gv_k, p.mesh.kg.weights, ws.density(), metadata,
        m.xc, ew, scratch);
    require(invariant.total.energy_ha == result.total.energy_ha
                && invariant.total.virial_ha == result.total.virial_ha,
            "functional independent of entropy/mu/reference bookkeeping");
    int rejects = 0;
    auto bad = bands;
    bad.local_blocks[0].epsilon_ha.pop_back();
    try {
        assemble_frozen_functional_virial(p.ions, p.gv0, p.mesh.gv_k,
                                          p.mesh.kg.weights, ws.density(), bad,
                                          m.xc, ew, scratch);
    } catch (const std::invalid_argument&) {
        ++rejects;
    }
    if (spin) {
        bad = bands;
        bad.orbital_spin_density_G.pop_back();
        try {
            assemble_frozen_functional_virial(p.ions, p.gv0, p.mesh.gv_k,
                                              p.mesh.kg.weights, ws.density(),
                                              bad, m.xc, ew, scratch);
        } catch (const std::invalid_argument&) {
            ++rejects;
        }
        bad = bands;
        bad.local_blocks[1].spin = 0;
        try {
            assemble_frozen_functional_virial(p.ions, p.gv0, p.mesh.gv_k,
                                              p.mesh.kg.weights, ws.density(),
                                              bad, m.xc, ew, scratch);
        } catch (const std::invalid_argument&) {
            ++rejects;
        }
    }
    require(rejects == (spin ? 3 : 1), "shape/spin admission rejects");
    constexpr int a[6] = {0, 1, 2, 0, 0, 1}, b[6] = {0, 1, 2, 1, 2, 2};
    double maxerr = 0;
    int energies = 0;
    for (double h : {3e-5, 1e-5})
        for (int k = 0; k < 6; ++k) {
            double e[2];
            for (int sign = 0; sign < 2; ++sign) {
                auto F = identity();
                F(a[k], b[k]) += sign == 0 ? h : -h;
                e[sign] =
                    primal(deform(f, F), p, bands, spin, 1 / F.det(), scratch);
                ++energies;
            }
            const double fd = -(e[0] - e[1]) / (2 * h),
                         want = .5
                                * (result.total.virial_ha[3 * a[k] + b[k]]
                                   + result.total.virial_ha[3 * b[k] + a[k]]);
            maxerr = std::max(maxerr, std::abs(fd - want));
            close(fd, want, 1e-8 + 1e-7 * std::abs(want),
                  "total frozen-functional affine derivative");
        }
    require(
        before == snapshot(ws) && fft == ws.fft.get() && binding == ws.gvec,
        "reference arrays/cache/bindings unchanged on success/rejection/FD");
    require(reference.energy == 17 && reference.energy_ha == 8.5
                && !reference.converged && reference.iterations == 7,
            "reference result untouched");
    std::printf("PASS ASSEMBLY_PRIMITIVE %s rho_mismatch=%.17g "
                "E_internal_Ha=%.17g max_dW_Ha=%.17g metadata_invariance=1 "
                "rejects=%d primal_energy_calls=%d SCF_calls=0 band_sweeps=1 "
                "spin_band_solves=%d\n",
                spin ? "UKS_FD_PBEsol" : "RKS_fixed_PBEsol",
                bands.density_mismatch_l2, result.total.energy_ha, maxerr,
                rejects, energies + 1, bands.local_band_solves);
}
} // namespace
TEST(PwFrozenFunctionalVirial, RksFixedPbeSolMatchesFrozenPrimal) {
    EXPECT_NO_THROW(run(false));
}
TEST(PwFrozenFunctionalVirial, UksFermiDiracPbeSolMatchesFrozenPrimal) {
    EXPECT_NO_THROW(run(true));
}

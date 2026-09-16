/// @file regressions/gpw/gpw-kpoints-si/reg_gpw-kpoints-si.cpp
/// @brief Monkhorst-Pack k-point GPW RKS on diamond Si (bct 4-atom cell,
///        DZVP-MOLOPT-GTH + GTH-PBE-q4, PBE) — internal exactness gates plus
///        the frozen absolute pin from REFERENCE.md.
///
/// Out-of-library consumer: public GPW headers only, flat main(), hard-coded
/// numbers, no external process. Gates (spec gpw-kpoints-si, acceptance.md):
///   A1 Γ-limit identity (k machinery vs legacy real Γ path, H₂ toy)
///   A2 supercell folding identity (mesh 1×1×2 vs z-doubled cell at κ_z=½)
///   A3 time-reversal identity (MP 2×2×2, 8 k vs TR-folded 4 k)
///   A4 k-sampling activation (|E_MP222 − E_Γ| > 1e-3 Ha)
///   A5 insulating gap across sampled k (> 0.1 eV) + k payload shapes
///   A6 convergence + finiteness
///   A7 absolute e_tot vs the frozen first-green pin (REFERENCE.md)

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

namespace {

constexpr double kA2B = 1.0 / 0.52917720859; // Å → Bohr
constexpr double kHa2Ev = 27.211386245988;

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

// Diamond Si in the body-centred-tetragonal 4-atom cell (a_t = a/√2, c = a),
// fractional (0,0,0), (1/2,1/2,1/2), (1/2,0,1/4), (0,1/2,3/4) — the
// orthorhombic-compatible setting of the fcc diamond lattice. nz = number of
// z-repeats (1 = 4 atoms, 2 = the folding supercell with 8 atoms).
std::vector<ATV::cpu::gpw::GpwAtom> si_bct(double a_bohr, int nz) {
    using namespace ATV;
    const double at = a_bohr / std::sqrt(2.0);
    const ATV::cpu::gpw::GaussianBasis bSi(molrs::Element::Si,
                                           "DZVP-MOLOPT-GTH");
    const ATV::cpu::gpw::GthPotential pSi(molrs::Element::Si, "GTH-PBE-q4");
    const double frac[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, 0.0, 0.25}, {0.0, 0.5, 0.75}};
    std::vector<ATV::cpu::gpw::GpwAtom> atoms;
    for (int r = 0; r < nz; ++r)
        for (int i = 0; i < 4; ++i) {
            const double pos[3] = {frac[i][0] * at, frac[i][1] * at,
                                   (frac[i][2] + r) * a_bohr};
            atoms.emplace_back(pos, bSi, pSi, 14);
        }
    return atoms;
}

ATV::cpu::gpw::GpwDft si_method(double a_bohr, int nz, double cutoff_ry) {
    ATV::cpu::gpw::GpwDft m;
    m.cell[0] = m.cell[1] = a_bohr / std::sqrt(2.0);
    m.cell[2] = a_bohr * nz;
    m.cutoff_ry = cutoff_ry;
    m.n_occ = 8 * nz; // 4 atoms × 4 valence e⁻ per repeat, closed shell
    m.max_iter = 60;
    m.tol = 1e-8;
    m.xc = ATV::cpu::gpw::XcKind::PBE;
    m.eps_default = 1e-7;
    return m;
}

} // namespace

int main() {
    using namespace ATV;
    using cpu::gpw::GaussianBasis;
    using cpu::gpw::GpwAtom;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GthPotential;
    using cpu::gpw::XcKind;

    const double a = 5.43 * kA2B; // Si lattice constant 5.43 Å → 10.26120 Bohr

    // ── A1: Γ-limit identity on a compact-basis H₂ toy ─────────────────────
    // (legacy real Γ path vs the complex k machinery at an explicit Γ point;
    // the diffuse-MOLOPT Si cell is impractical on the legacy Γ path, which
    // is part of what the pre-summed k path fixes).
    {
        const double L = 8.0; // Bohr
        const GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
        const GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
        const double r0[3] = {0.5 * L - 0.7, 0.5 * L, 0.5 * L};
        const double r1[3] = {0.5 * L + 0.7, 0.5 * L, 0.5 * L};
        std::vector<GpwAtom> h2;
        h2.emplace_back(r0, bH, pH, 1);
        h2.emplace_back(r1, bH, pH, 1);

        GpwDft gamma;
        gamma.cell[0] = gamma.cell[1] = gamma.cell[2] = L;
        gamma.cutoff_ry = 100.0;
        gamma.n_occ = 1;
        gamma.max_iter = 60;
        gamma.tol = 1e-9;
        gamma.xc = XcKind::PADE;
        gamma.build(h2);
        const double e_legacy = gamma.run(h2).e_total;

        GpwDft kg;
        kg.cell[0] = kg.cell[1] = kg.cell[2] = L;
        kg.cutoff_ry = 100.0;
        kg.n_occ = 1;
        kg.max_iter = 60;
        kg.tol = 1e-9;
        kg.xc = XcKind::PADE;
        kg.k_points.push_back({0.0, 0.0, 0.0});
        kg.k_point_weights.push_back(1.0);
        kg.build(h2);
        const auto& rkg = kg.run(h2);
        std::printf("A1: H2 legacy Γ = %.9f Ha, k-path Γ = %.9f Ha, "
                    "|Δ| = %.3e Ha\n",
                    e_legacy, rkg.e_total, std::abs(e_legacy - rkg.e_total));
        check(rkg.converged, "A1 k-path Γ SCF converged");
        check(std::abs(e_legacy - rkg.e_total) <= 1e-6,
              "A1 Γ-limit identity |ΔE| <= 1e-6 Ha");
    }

    // ── A2: supercell folding identity (95 Ry: commensurate grids) ─────────
    double e_mesh12 = 0.0;
    {
        const auto cell1 = si_bct(a, 1);
        GpwDft m1 = si_method(a, 1, 95.0);
        m1.k_mesh[0] = 1;
        m1.k_mesh[1] = 1;
        m1.k_mesh[2] = 2; // MP → κ_z = ±1/4, TR-folds to (0,0,+1/4), w = 1
        m1.build(cell1);
        const auto& r1 = m1.run(cell1);
        e_mesh12 = r1.e_total;
        check(r1.converged, "A2 mesh 1x1x2 SCF converged");

        const auto cell2 = si_bct(a, 2);
        GpwDft m2 = si_method(a, 2, 95.0);
        m2.k_points.push_back({0.0, 0.0, 0.5}); // the folded zone-boundary k
        m2.k_point_weights.push_back(1.0);
        m2.build(cell2);
        const auto& r2 = m2.run(cell2);
        check(r2.converged, "A2 supercell SCF converged");
        const double diff = std::abs(0.5 * r2.e_total - e_mesh12);
        std::printf("A2: E(1x1x2 mesh) = %.9f Ha, E(supercell)/2 = %.9f Ha, "
                    "|Δ| = %.3e Ha\n",
                    e_mesh12, 0.5 * r2.e_total, diff);
        check(diff <= 5e-5, "A2 supercell folding identity |Δ| <= 5e-5 Ha");
    }

    // ── A3: time-reversal identity (95 Ry) ─────────────────────────────────
    {
        const auto cell1 = si_bct(a, 1);
        GpwDft full = si_method(a, 1, 95.0);
        full.k_mesh[0] = full.k_mesh[1] = full.k_mesh[2] = 2;
        full.k_time_reversal = false; // all 8 k
        full.build(cell1);
        const auto& rf = full.run(cell1);
        check(rf.converged, "A3 full-mesh SCF converged");
        check(rf.k_frac.size() == 8, "A3 full mesh samples 8 k-points");

        GpwDft tr = si_method(a, 1, 95.0);
        tr.k_mesh[0] = tr.k_mesh[1] = tr.k_mesh[2] = 2;
        tr.k_time_reversal = true; // 4 k, doubled weights
        tr.build(cell1);
        const auto& rt = tr.run(cell1);
        check(rt.converged, "A3 TR-mesh SCF converged");
        check(rt.k_frac.size() == 4, "A3 TR mesh samples 4 k-points");
        const double diff = std::abs(rf.e_total - rt.e_total);
        std::printf("A3: E(8k) = %.9f Ha, E(4k TR) = %.9f Ha, |Δ| = %.3e Ha\n",
                    rf.e_total, rt.e_total, diff);
        check(diff <= 5e-6, "A3 time-reversal identity |Δ| <= 5e-6 Ha");
    }

    // ── Main run: MP 2×2×2 TR at 150 Ry (A4-A7) ────────────────────────────
    {
        const auto cell1 = si_bct(a, 1);
        GpwDft mp = si_method(a, 1, 150.0);
        mp.k_mesh[0] = mp.k_mesh[1] = mp.k_mesh[2] = 2;
        mp.build(cell1);
        const auto& rm = mp.run(cell1);
        check(rm.converged, "A6 MP 2x2x2 SCF converged");
        check(std::isfinite(rm.e_total), "A6 finite e_total");
        std::printf("MAIN: E(MP 2x2x2, 150 Ry) = %.10f Ha (iters=%d)\n",
                    rm.e_total, rm.iters);

        // A5: payload shapes + insulating gap across the sampled k.
        check(rm.k_frac.size() == 4 && rm.k_weights.size() == 4
                  && rm.eps_k.size() == 4,
              "A5 k payload shapes (4 TR k-points)");
        double vbm = -1e30, cbm = 1e30;
        for (const auto& eps : rm.eps_k) {
            if (eps.size() < 9) {
                check(false, "A5 eps_k has occupied+virtual bands");
                break;
            }
            vbm = std::max(vbm, eps[7]); // n_occ = 8 → highest occupied
            cbm = std::min(cbm, eps[8]); // lowest virtual
        }
        const double gap_ev = (cbm - vbm) * kHa2Ev;
        std::printf("A5: VBM = %.6f Ha, CBM = %.6f Ha, gap = %.3f eV\n", vbm,
                    cbm, gap_ev);
        check(gap_ev > 0.1, "A5 insulating gap > 0.1 eV across sampled k");

        // A4: k sampling matters — explicit Γ through the same machinery.
        GpwDft gm = si_method(a, 1, 150.0);
        gm.k_points.push_back({0.0, 0.0, 0.0});
        gm.k_point_weights.push_back(1.0);
        gm.build(cell1);
        const auto& rg = gm.run(cell1);
        check(rg.converged, "A4 Γ-point SCF converged");
        const double dk = std::abs(rm.e_total - rg.e_total);
        std::printf("A4: E(Γ) = %.10f Ha, |E_MP − E_Γ| = %.4e Ha\n", rg.e_total,
                    dk);
        check(dk > 1e-3, "A4 k-sampling activation |Δ| > 1e-3 Ha");

        // A7: offline CP2K 2026.1 pin (REFERENCE.md) — the exact twin input
        // tools/cp2k_oracle/inputs/gpw-kpoints-si.inp run 2026-07-31
        // (replaces the interim ATV-frozen first-green band).
        constexpr double kEPin = -15.802416412655166; // Ha (CP2K 2026.1)
        constexpr double kTolAbs = 1e-3;              // Ha
        const double dpin = std::abs(rm.e_total - kEPin);
        std::printf("A7: |E − pin| = %.3e Ha (pin %.10f)\n", dpin, kEPin);
        check(dpin <= kTolAbs, "A7 absolute e_tot within 1e-3 Ha of pin");
    }

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    std::printf("PASS: gpw-kpoints-si — Monkhorst-Pack k-point GPW RKS on "
                "diamond Si (folding, TR, Γ-limit, gap, pin)\n");
    return 0;
}

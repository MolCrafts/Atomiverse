/// @file regressions/gpw/gpw-bands-uks/reg_gpw-bands-uks.cpp
/// @brief END-TO-END: multi-k GPW collinear UKS band structure (BCT Si).
///
/// Dual gate vs offline CP2K 2026.1 Quickstep UKS + BAND_STRUCTURE twin
/// (same GTH-GPW class as reg_gpw-bands-si / reg_gpw-dos-uks). See REFERENCE.md.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/bands_dos.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

namespace {

constexpr double kA2B = 1.0 / 0.52917720859;
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

bool near_k(const std::array<double, 3>& a, const std::array<double, 3>& b,
            double tol = 1e-8) {
    return std::abs(a[0] - b[0]) < tol && std::abs(a[1] - b[1]) < tol
           && std::abs(a[2] - b[2]) < tol;
}

std::vector<ATV::cpu::gpw::GpwAtom> si_bct(double a_bohr) {
    using namespace ATV;
    const double at = a_bohr / std::sqrt(2.0);
    const cpu::gpw::GaussianBasis bSi(molrs::Element::Si, "DZVP-MOLOPT-GTH");
    const cpu::gpw::GthPotential pSi(molrs::Element::Si, "GTH-PBE-q4");
    const double frac[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, 0.0, 0.25}, {0.0, 0.5, 0.75}};
    std::vector<cpu::gpw::GpwAtom> atoms;
    for (int i = 0; i < 4; ++i) {
        const double pos[3] = {frac[i][0] * at, frac[i][1] * at,
                               frac[i][2] * a_bohr};
        atoms.emplace_back(pos, bSi, pSi, 14);
    }
    return atoms;
}

} // namespace

int main() {
    using namespace ATV::cpu::gpw;

    // Closed-shell UKS ≡ RKS energy twin (same class as gpw-dos-uks / bands-si).
    constexpr double kCp2kEtotHa = -15.802416412655198;
    constexpr double kEtotTolHa = 1e-3;
    // Path gap class of RKS twin (~0.89 eV); joint α/β at n_α=n_β=8.
    constexpr double kCp2kGapEv = 0.893516;
    constexpr double kGapTolEv = 0.08;

    const double a = 5.43 * kA2B;
    auto atoms = si_bct(a);

    GpwDft method;
    method.cell[0] = method.cell[1] = a / std::sqrt(2.0);
    method.cell[2] = a;
    method.cutoff_ry = 150.0;
    method.n_alpha = 8;
    method.n_beta = 8;
    method.max_iter = 60;
    method.tol = 1e-8;
    method.eps_default = 1e-7;
    method.xc = XcKind::PBE;
    method.k_mesh[0] = method.k_mesh[1] = method.k_mesh[2] = 2;
    method.k_time_reversal = true;
    method.local_pp_on_grid = true;
    method.build(atoms);

    Bands bands;
    bands.n_bands = 12;
    bands.set_diamond_path(4);

    const auto t0 = std::chrono::steady_clock::now();
    const BandsResult br = bands.compute(method, atoms);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();

    std::printf(
        "gpw-bands-uks: E=%.10f Ha  gap=%.6f eV  nspin=%d  wall=%.3f s  "
        "nk=%zu  conv=%d\n",
        br.energy_ha, br.gap_ha * kHa2Ev, br.nspin, wall_s, br.kfrac.size(),
        br.scf_converged ? 1 : 0);

    check(br.scf_converged, "SCF converged");
    check(br.nspin == 2, "nspin=2 (UKS)");
    check(std::isfinite(br.energy_ha), "E finite");
    check(std::abs(br.energy_ha - kCp2kEtotHa) <= kEtotTolHa,
          "E_tot vs CP2K (1e-3 Ha)");
    check(!br.eigenvalues_ha.empty() && !br.eigenvalues_beta_ha.empty(),
          "dual-spin path eigenvalues present");
    check(br.kfrac.size() == br.eigenvalues_ha.size()
              && br.kfrac.size() == br.eigenvalues_beta_ha.size(),
          "kfrac/evals size match");
    check(static_cast<int>(br.kfrac.size()) == 16,
          "diamond path n_seg=4 → 16 k");

    for (const auto& row : br.eigenvalues_ha) {
        check(static_cast<int>(row.size()) == bands.n_bands, "n_bands α");
        for (std::size_t b = 1; b < row.size(); ++b)
            check(row[b] + 1e-10 >= row[b - 1], "α evals sorted");
    }
    for (const auto& row : br.eigenvalues_beta_ha) {
        check(static_cast<int>(row.size()) == bands.n_bands, "n_bands β");
        for (std::size_t b = 1; b < row.size(); ++b)
            check(row[b] + 1e-10 >= row[b - 1], "β evals sorted");
    }

    // Closed-shell: α and β spectra should match to numerical residual.
    double max_ab = 0.0;
    for (std::size_t ik = 0; ik < br.eigenvalues_ha.size(); ++ik) {
        const auto& arow = br.eigenvalues_ha[ik];
        const auto& brow = br.eigenvalues_beta_ha[ik];
        for (std::size_t b = 0; b < arow.size(); ++b)
            max_ab = std::max(max_ab, std::abs(arow[b] - brow[b]));
    }
    std::printf("  max|ε_α−ε_β|=%.3e Ha\n", max_ab);
    check(max_ab < 1e-6, "closed-shell α≡β eigenvalues");

    const double gap_ev = br.gap_ha * kHa2Ev;
    check(br.gap_ha > 0.0, "path gap > 0");
    check(std::abs(gap_ev - kCp2kGapEv) <= kGapTolEv, "path gap vs CP2K class");

    const std::array<double, 3> G{0, 0, 0};
    int ik_g = -1;
    for (std::size_t ik = 0; ik < br.kfrac.size(); ++ik) {
        if (near_k(br.kfrac[ik], G) && ik_g < 0)
            ik_g = static_cast<int>(ik);
    }
    check(ik_g >= 0, "path includes Γ");
    if (ik_g >= 0) {
        check(br.vbm_k == ik_g, "VBM at Γ (twin character)");
        check(br.cbm_k == ik_g, "CBM at Γ (twin character)");
    }

    std::printf("ATV_WALL_S %.6f\n", wall_s);

    if (g_fail) {
        std::fprintf(stderr, "%d checks failed\n", g_fail);
        return 1;
    }
    std::printf("ALL CHECKS PASSED\n");
    return 0;
}

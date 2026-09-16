/// @file regressions/gpw/gpw-vdw-kpoint-forces/reg_gpw-vdw-kpoint-forces.cpp
/// @brief Multi-k GPW PBE + vdW-DF (DRSLL) analytic ionic forces.
///
/// Out-of-library consumer. SC nonlocal \(V_{\mathrm{nl}}\) rides the multi-k
/// force assembly (`assemble_gpw_forces_kpoints`). Gates (REFERENCE.md):
///   G1 E_disp / E_tot vs offline CP2K 2026.1 NON_LOCAL + KPOINTS ENERGY_FORCE;
///   G2 analytic forces vs CP2K FORCES| (twin-family absolute tol) + non-vacuous
///      multi-k payload; frozen ATV F_O0_x pin;
///   fair wall vs the same CP2K twin (1 MPI × 8 OMP).
/// Offline FD (documented in REFERENCE): max|F_ana − F_FD| ≈ 7e-7 Ha/Bohr on
/// single-grid multi-k O0 x — not re-run in ctest (3× multi-k SCFs ≫ 60 s).
/// No live oracle in ctest.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/molrs.hpp"

namespace {

constexpr double kA2B = 1.0 / 0.52917720859;

// Same water dimer as gpw-vdw-kpoints / gpw-vdw-df-ice [Å].
const double kAng[6][3] = {
    {0.000000, 0.000000, 0.000000},  // O
    {0.758602, 0.000000, 0.504284},  // H
    {0.758602, 0.000000, -0.504284}, // H
    {3.000000, 0.000000, 0.000000},  // O
    {3.758602, 0.000000, 0.504284},  // H
    {3.758602, 0.000000, -0.504284}, // H
};
const int kZ[6] = {8, 1, 1, 8, 1, 1};

// Offline CP2K 2026.1 ENERGY_FORCE pin — see REFERENCE.md.
// water_dimer_vdw_df_kp_forces.inp, cp2k.psmp git:5e54ba2, 1 MPI × OMP=8.
constexpr double kEDispCp2k = 0.15193559149613;   // Ha
constexpr double kETotCp2k = -34.186333514352626; // Ha
constexpr double kEDispTol = 1.0e-4;              // Ha
// Same Γ-dimer twin-family systematic as gpw-vdw-kpoints (~8e-3 Ha).
constexpr double kETotTol = 1.0e-2; // Ha

// FORCES| Atomic forces [hartree/bohr], atom order O H H O H H.
constexpr double kForceCp2k[6][3] = {
    {-8.35844769e-02, 1.49028331e-09, 2.51422839e-10},
    {4.16583157e-02, -5.35060271e-09, 1.31835454e-01},
    {4.16583337e-02, 4.10205702e-09, -1.31835315e-01},
    {-7.89346446e-02, 2.67651054e-11, 6.49135595e-09},
    {3.76866359e-02, -4.43451105e-09, 1.35635778e-01},
    {3.76865671e-02, 4.54720810e-09, -1.35635393e-01},
};
// Absolute force parity: same twin-family residual class as E_tot (grid /
// multigrid / density). Measured max|ΔF| ≈ 7.25e-3 Ha/Bohr on first green.
constexpr double kForceTol = 1.0e-2; // Ha/Bohr

// Frozen ATV multigrid multi-k force sample (Release, OMP=8, first green).
// Tight pin for product reproducibility; FD identity offline (REFERENCE).
constexpr double kF0xAtvPin = -8.976000e-02; // Ha/Bohr
constexpr double kF0xPinTol = 5.0e-5;        // Ha/Bohr

// Fair wall: CP2K internal timer TOTAL TIME on ENERGY_FORCE twin (OMP=8).
constexpr double kOracleWallS = 13.216;
constexpr double kWallBudget = 60.0; // s full regression
constexpr double kFairRatioMax = 10.0;

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

std::vector<ATV::cpu::gpw::GpwAtom> make_atoms() {
    using namespace ATV::cpu::gpw;
    const GaussianBasis bO(molrs::Element::O, "DZVP-MOLOPT-SR-GTH");
    const GaussianBasis bH(molrs::Element::H, "DZVP-MOLOPT-SR-GTH");
    const GthPotential pO(molrs::Element::O, "GTH-PBE-q6");
    const GthPotential pH(molrs::Element::H, "GTH-PBE-q1");
    std::vector<GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const double pos[3] = {kAng[i][0] * kA2B, kAng[i][1] * kA2B,
                               kAng[i][2] * kA2B};
        if (kZ[i] == 8)
            atoms.emplace_back(pos, bO, pO, 8);
        else
            atoms.emplace_back(pos, bH, pH, 1);
    }
    return atoms;
}

ATV::cpu::gpw::GpwDft make_twin() {
    ATV::cpu::gpw::GpwDft gpw;
    const double L = 12.0 * kA2B;
    gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
    gpw.cutoff_ry = 200.0;
    gpw.ngrids = 4;
    gpw.rel_cutoff = 40.0;
    gpw.n_occ = 8;
    gpw.max_iter = 60;
    gpw.tol = 1e-7;
    gpw.xc = ATV::cpu::gpw::XcKind::PBE;
    gpw.nonperiodic = false;
    gpw.vdw_df = true;
    gpw.k_mesh[0] = 2;
    gpw.k_mesh[1] = 1;
    gpw.k_mesh[2] = 1;
    // Match CP2K full MP mesh (no TR fold) — same as gpw-vdw-kpoints energy pin.
    gpw.k_time_reversal = false;
    return gpw;
}

} // namespace

int main() {
    using namespace ATV::cpu::gpw;
    using Clock = std::chrono::steady_clock;

    const auto t_all0 = Clock::now();
    const auto atoms = make_atoms();

    GpwDft mp = make_twin();
    mp.build(atoms);
    const auto t0 = Clock::now();
    const GpwResult& res = mp.run(atoms);
    check(res.converged, "G1 multi-k SCF converged");
    check(res.n_kpts >= 1 && !res.C_occ_k.empty(), "G1 multi-k C_occ_k payload");
    check(std::isfinite(res.e_disp)
              && std::abs(res.e_disp - kEDispCp2k) <= kEDispTol,
          "G1 |E_disp - CP2K|");
    check(std::abs(res.e_total - kETotCp2k) <= kETotTol,
          "G1 |E_tot - CP2K| (twin-family)");
    std::printf("G1: E_tot  ATV=%.14f  CP2K=%.14f  |Δ|=%.3e Ha\n", res.e_total,
                kETotCp2k, std::abs(res.e_total - kETotCp2k));
    std::printf("G1: E_disp ATV=%.14f  CP2K=%.14f  |Δ|=%.3e Ha  n_kpts=%d\n",
                res.e_disp, kEDispCp2k, std::abs(res.e_disp - kEDispCp2k),
                res.n_kpts);

    const auto F = gpw_forces_from_scf(atoms, mp, res);
    const double wall_twin_s =
        std::chrono::duration<double>(Clock::now() - t0).count();
    check(F.size() == 6, "G2 six force vectors");

    double max_df = 0.0;
    double max_f = 0.0;
    double sum[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double got = F[static_cast<std::size_t>(i)][k];
            const double want = kForceCp2k[i][k];
            max_df = std::max(max_df, std::abs(got - want));
            max_f = std::max(max_f, std::abs(want));
            sum[k] += got;
            check(std::isfinite(got), "G2 finite force component");
        }
        std::printf("G2: F[%d] ATV=(% .6e, % .6e, % .6e)\n", i,
                    F[static_cast<std::size_t>(i)][0],
                    F[static_cast<std::size_t>(i)][1],
                    F[static_cast<std::size_t>(i)][2]);
    }
    const double sum_mag =
        std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
    std::printf(
        "G2: max|F_atv-F_cp2k|=%.3e  signal=%.3e  |sum F|=%.3e Ha/Bohr\n",
        max_df, max_f, sum_mag);
    check(max_f > 1.0e-2, "G2 non-vacuous force signal");
    check(max_df <= kForceTol, "G2 max|F - CP2K FORCES|");
    // Translation residual (finite basis / grid) — soft bound.
    check(sum_mag <= 5.0e-2, "G2 |sum F| soft bound");
    check(std::abs(F[0][0] - kF0xAtvPin) <= kF0xPinTol,
          "G2 |F_O0_x - ATV pin|");

    std::printf("G2: twin wall ATV=%.3f s  CP2K=%.3f s  ratio=%.3f\n",
                wall_twin_s, kOracleWallS, wall_twin_s / kOracleWallS);
    check(wall_twin_s / kOracleWallS <= kFairRatioMax,
          "G2 fair wall T_atv/T_cp2k <= 10");

    const double wall_all =
        std::chrono::duration<double>(Clock::now() - t_all0).count();
    std::printf("PASS gpw-vdw-kpoint-forces (total wall %.1f s, twin %.1f s, "
                "ratio %.2f)\n",
                wall_all, wall_twin_s, wall_twin_s / kOracleWallS);
    check(wall_all <= kWallBudget, "full regression wall <= 60 s");

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    return 0;
}

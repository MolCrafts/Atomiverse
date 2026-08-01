/// @file tests/cpu/gpw/test_cpu_gpw_hirshfeld.cpp
/// @brief Hirshfeld volume ratios + density-scaled MBD/TS unit gates.

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/basis_data/hirshfeld_tables.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/hirshfeld_volumes.h"
#include "atomiverse/cpu/gpw/internal/mbd_dispersion.h"
#include "atomiverse/molrs.hpp"

using ATV::cpu::gpw::hirshfeld_rho_free;
using ATV::cpu::gpw::hirshfeld_volume_ratios;
using ATV::cpu::gpw::make_uniform_grid;
using ATV::cpu::gpw::mbd_rsscs_pbe;
using ATV::cpu::gpw::UniformGrid;

namespace {

constexpr double kAngToBohr = 1.0 / 0.52917720859;

std::vector<std::array<double, 3>> water_dimer_bohr() {
    const double ang[6][3] = {
        {0.000000, 0.000000, 0.000000},  // O
        {0.758602, 0.000000, 0.504284},  // H
        {0.758602, 0.000000, -0.504284}, // H
        {3.000000, 0.000000, 0.000000},  // O
        {3.758602, 0.000000, 0.504284},  // H
        {3.758602, 0.000000, -0.504284}, // H
    };
    std::vector<std::array<double, 3>> pos(6);
    for (int i = 0; i < 6; ++i)
        for (int d = 0; d < 3; ++d)
            pos[i][d] = ang[i][d] * kAngToBohr;
    return pos;
}

const std::vector<int> kZ = {8, 1, 1, 8, 1, 1};

/// @brief Collocate Σ_A ρ_A^free onto a uniform grid (min-image).
std::vector<double>
superpose_free_atoms(const UniformGrid& grid, const std::vector<int>& Z,
                     const std::vector<std::array<double, 3>>& pos,
                     const double cell[3], bool periodic) {
    std::vector<double> rho(static_cast<std::size_t>(grid.npts()), 0.0);
    double r[3];
    for (long g = 0; g < grid.npts(); ++g) {
        grid.point(g, r);
        double s = 0.0;
        for (std::size_t a = 0; a < Z.size(); ++a) {
            double dx = r[0] - pos[a][0];
            double dy = r[1] - pos[a][1];
            double dz = r[2] - pos[a][2];
            if (periodic) {
                dx -= cell[0] * std::nearbyint(dx / cell[0]);
                dy -= cell[1] * std::nearbyint(dy / cell[1]);
                dz -= cell[2] * std::nearbyint(dz / cell[2]);
            }
            const double rr = std::sqrt(dx * dx + dy * dy + dz * dz);
            s += hirshfeld_rho_free(Z[a], rr);
        }
        rho[static_cast<std::size_t>(g)] = s;
    }
    return rho;
}

} // namespace

// Isolated free atom: ρ = ρ_free on a large box ⇒ v ≈ 1 (grid truncation small).
TEST(GpwHirshfeld, IsolatedFreeAtomGivesUnitVolumeRatio) {
    const std::vector<int> Z = {8};
    std::vector<std::array<double, 3>> pos = {{{0.0, 0.0, 0.0}}};
    // Large cell so free-atom tails fit; moderate cutoff for the unit gate.
    const double L = 20.0; // Bohr
    const double cell[3] = {L, L, L};
    pos[0] = {0.5 * L, 0.5 * L, 0.5 * L};
    const UniformGrid grid = make_uniform_grid(cell, /*cutoff_ry=*/80.0);
    const auto rho = superpose_free_atoms(grid, Z, pos, cell, /*periodic=*/false);
    // Check integrated free-atom charge ≈ N_val.
    double q = 0.0;
    for (double v : rho)
        q += v;
    q *= grid.weight();
    std::printf("[Hirshfeld] isolated O free-atom int rho = %.6f (want 6)\n", q);
    EXPECT_NEAR(q, 6.0, 0.05);

    const auto v =
        hirshfeld_volume_ratios(grid, rho, Z, pos, cell, /*periodic=*/false);
    ASSERT_EQ(v.size(), 1u);
    std::printf("[Hirshfeld] isolated O v=%.6f\n", v[0]);
    EXPECT_NEAR(v[0], 1.0, 3e-2);
}

// Volume scaling of MBD@rsSCS matches a dual call with the same ratios.
TEST(GpwHirshfeld, MbdVolumeScalingMatchesDual) {
    const auto pos = water_dimer_bohr();
    const double cell[3] = {0.0, 0.0, 0.0};
    const std::vector<double> vols = {0.92, 0.80, 0.80, 0.91, 0.81, 0.81};
    const double e_free =
        mbd_rsscs_pbe(kZ, pos, cell, /*periodic=*/false).energy;
    const double e_scaled =
        mbd_rsscs_pbe(kZ, pos, cell, /*periodic=*/false, 0.83, 6.0, vols)
            .energy;
    std::printf("[Hirshfeld] MBD free=%.16e  scaled=%.16e  delta=%.3e Ha\n",
                e_free, e_scaled, e_scaled - e_free);
    EXPECT_LT(e_free, 0.0);
    EXPECT_LT(e_scaled, 0.0);
    EXPECT_GT(std::abs(e_scaled - e_free), 1e-6);
    // Dual recomputation (same call) is bit-identical.
    const double e2 =
        mbd_rsscs_pbe(kZ, pos, cell, /*periodic=*/false, 0.83, 6.0, vols)
            .energy;
    EXPECT_NEAR(e_scaled, e2, 1e-15);
}

// Full GpwDft path: mbd + hirshfeld writes e_disp and volume_ratios.
TEST(GpwHirshfeld, GpwDftMbdHirshfeldReportsVolumesAndEDisp) {
    using ATV::cpu::gpw::GaussianBasis;
    using ATV::cpu::gpw::GpwAtom;
    using ATV::cpu::gpw::GpwDft;
    using ATV::cpu::gpw::GthPotential;
    using ATV::cpu::gpw::XcKind;

    const double ang[6][3] = {
        {0.000000, 0.000000, 0.000000}, {0.758602, 0.000000, 0.504284},
        {0.758602, 0.000000, -0.504284}, {3.000000, 0.000000, 0.000000},
        {3.758602, 0.000000, 0.504284}, {3.758602, 0.000000, -0.504284},
    };
    const GaussianBasis bO(molrs::Element::O, "DZVP-MOLOPT-SR-GTH");
    const GaussianBasis bH(molrs::Element::H, "DZVP-MOLOPT-SR-GTH");
    const GthPotential pO(molrs::Element::O, "GTH-PBE-q6");
    const GthPotential pH(molrs::Element::H, "GTH-PBE-q1");
    std::vector<GpwAtom> atoms;
    for (int i = 0; i < 6; ++i) {
        const double pos[3] = {ang[i][0] * kAngToBohr, ang[i][1] * kAngToBohr,
                               ang[i][2] * kAngToBohr};
        if (i == 0 || i == 3)
            atoms.emplace_back(pos, bO, pO, 8);
        else
            atoms.emplace_back(pos, bH, pH, 1);
    }

    GpwDft gpw;
    const double L = 12.0 * kAngToBohr;
    gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
    gpw.cutoff_ry = 200.0;
    gpw.n_occ = 8;
    gpw.max_iter = 12;
    gpw.tol = 1e-6;
    gpw.mix = 0.4;
    gpw.xc = XcKind::PBE;
    gpw.mbd = true;
    gpw.hirshfeld = true;

    gpw.build(atoms);
    const auto& res = gpw.run(atoms);
    std::printf("[Hirshfeld] E_disp=%.16e  n_vol=%zu  e_tot=%.8f iters=%d\n",
                res.e_disp, res.volume_ratios.size(), res.e_total, res.iters);
    for (std::size_t i = 0; i < res.volume_ratios.size(); ++i)
        std::printf("  v[%zu]=%.6f\n", i, res.volume_ratios[i]);

    ASSERT_EQ(res.volume_ratios.size(), 6u);
    EXPECT_TRUE(std::isfinite(res.e_disp));
    EXPECT_LT(res.e_disp, 0.0);
    EXPECT_TRUE(std::isfinite(res.e_total));
    // Free-atom MBD on this geometry is ~−1.49e-3 Ha; Hirshfeld-scaled should
    // remain the same order of magnitude and stay attractive.
    EXPECT_LT(std::abs(res.e_disp), 1e-2);
    for (double v : res.volume_ratios) {
        // Valence GTH density vs compact free-atom exponential: ratios are
        // O(0.5–1.2) for water; keep a wide physical gate.
        EXPECT_GT(v, 0.2);
        EXPECT_LT(v, 2.0);
    }
    // Distinct from free-atom pin (volume scaling moved the energy).
    const double free_atom = -1.4928804171407961e-03;
    EXPECT_GT(std::abs(res.e_disp - free_atom), 1e-6);
}

// Multi-k RKS + mbd + hirshfeld: k-summed density → volume ratios + E_disp.
TEST(GpwHirshfeld, MultiKMbdHirshfeldReportsVolumesAndEDisp) {
    using ATV::cpu::gpw::GaussianBasis;
    using ATV::cpu::gpw::GpwAtom;
    using ATV::cpu::gpw::GpwDft;
    using ATV::cpu::gpw::GthPotential;
    using ATV::cpu::gpw::XcKind;

    const double ang[6][3] = {
        {0.000000, 0.000000, 0.000000}, {0.758602, 0.000000, 0.504284},
        {0.758602, 0.000000, -0.504284}, {3.000000, 0.000000, 0.000000},
        {3.758602, 0.000000, 0.504284}, {3.758602, 0.000000, -0.504284},
    };
    const GaussianBasis bO(molrs::Element::O, "DZVP-MOLOPT-SR-GTH");
    const GaussianBasis bH(molrs::Element::H, "DZVP-MOLOPT-SR-GTH");
    const GthPotential pO(molrs::Element::O, "GTH-PBE-q6");
    const GthPotential pH(molrs::Element::H, "GTH-PBE-q1");
    std::vector<GpwAtom> atoms;
    for (int i = 0; i < 6; ++i) {
        const double pos[3] = {ang[i][0] * kAngToBohr, ang[i][1] * kAngToBohr,
                               ang[i][2] * kAngToBohr};
        if (i == 0 || i == 3)
            atoms.emplace_back(pos, bO, pO, 8);
        else
            atoms.emplace_back(pos, bH, pH, 1);
    }

    GpwDft gpw;
    const double L = 12.0 * kAngToBohr;
    gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
    gpw.cutoff_ry = 160.0; // unit gate: lighter than regression 200 Ry
    gpw.n_occ = 8;
    gpw.max_iter = 10;
    gpw.tol = 1e-5;
    gpw.mix = 0.4;
    gpw.xc = XcKind::PBE;
    gpw.mbd = true;
    gpw.hirshfeld = true;
    gpw.k_mesh[0] = gpw.k_mesh[1] = gpw.k_mesh[2] = 2;
    gpw.k_time_reversal = true;

    gpw.build(atoms);
    const auto& res = gpw.run(atoms);
    std::printf("[Hirshfeld k] E_disp=%.16e  n_vol=%zu  n_kpts=%d  iters=%d\n",
                res.e_disp, res.volume_ratios.size(), res.n_kpts, res.iters);
    for (std::size_t i = 0; i < res.volume_ratios.size(); ++i)
        std::printf("  v[%zu]=%.6f\n", i, res.volume_ratios[i]);

    ASSERT_EQ(res.volume_ratios.size(), 6u);
    EXPECT_GT(res.n_kpts, 1);
    EXPECT_TRUE(std::isfinite(res.e_disp));
    EXPECT_LT(res.e_disp, 0.0);
    EXPECT_LT(std::abs(res.e_disp), 1e-2);
    for (double v : res.volume_ratios) {
        EXPECT_GT(v, 0.2);
        EXPECT_LT(v, 2.0);
    }
    const double free_atom = -1.4928804171407961e-03;
    EXPECT_GT(std::abs(res.e_disp - free_atom), 1e-6);
}

// Γ UKS + mbd + hirshfeld: total density ρ=ρα+ρβ → volume ratios + E_disp.
TEST(GpwHirshfeld, UksMbdHirshfeldReportsVolumesAndEDisp) {
    using ATV::cpu::gpw::GaussianBasis;
    using ATV::cpu::gpw::GpwAtom;
    using ATV::cpu::gpw::GpwDft;
    using ATV::cpu::gpw::GthPotential;
    using ATV::cpu::gpw::XcKind;

    // O₂-like high-spin dimer fragment (single O, open shell) — light unit gate.
    const GaussianBasis bO(molrs::Element::O, "DZVP-MOLOPT-SR-GTH");
    const GthPotential pO(molrs::Element::O, "GTH-PBE-q6");
    const double L = 10.0 * kAngToBohr;
    const double pos[3] = {0.5 * L, 0.5 * L, 0.5 * L};
    std::vector<GpwAtom> atoms;
    atoms.emplace_back(pos, bO, pO, 8);

    GpwDft gpw;
    gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
    gpw.cutoff_ry = 120.0;
    gpw.n_alpha = 4;
    gpw.n_beta = 2;
    gpw.max_iter = 10;
    gpw.tol = 1e-5;
    gpw.mix = 0.4;
    gpw.xc = XcKind::PBE;
    gpw.mbd = true;
    gpw.hirshfeld = true;

    gpw.build(atoms);
    const auto& res = gpw.run(atoms);
    std::printf("[Hirshfeld UKS] E_disp=%.16e  n_vol=%zu  s2=%.4f  iters=%d\n",
                res.e_disp, res.volume_ratios.size(), res.s2, res.iters);
    for (std::size_t i = 0; i < res.volume_ratios.size(); ++i)
        std::printf("  v[%zu]=%.6f\n", i, res.volume_ratios[i]);

    ASSERT_EQ(res.volume_ratios.size(), 1u);
    EXPECT_TRUE(std::isfinite(res.e_disp));
    // Single atom → MBD pair sum empty → E_disp ≈ 0; volumes still defined.
    EXPECT_NEAR(res.e_disp, 0.0, 1e-12);
    EXPECT_GT(res.volume_ratios[0], 0.2);
    EXPECT_LT(res.volume_ratios[0], 2.0);
}

// Multi-k UKS + mbd + hirshfeld: k-summed ρ=ρα+ρβ → volumes + E_disp
// (fold_dispersion_k_uks product path; pin-first slug gpw-hirshfeld-uks-kpoints).
TEST(GpwHirshfeld, MultiKUksMbdHirshfeldReportsVolumesAndEDisp) {
    using ATV::cpu::gpw::GaussianBasis;
    using ATV::cpu::gpw::GpwAtom;
    using ATV::cpu::gpw::GpwDft;
    using ATV::cpu::gpw::GthPotential;
    using ATV::cpu::gpw::XcKind;

    // High-spin O₂ fragment (2 O, open shell) on a small periodic cell.
    const GaussianBasis bO(molrs::Element::O, "DZVP-MOLOPT-SR-GTH");
    const GthPotential pO(molrs::Element::O, "GTH-PBE-q6");
    const double L = 8.0 * kAngToBohr;
    constexpr double R_ang = 1.2075;
    const double z0 = 0.5 * L - 0.5 * R_ang * kAngToBohr;
    const double z1 = 0.5 * L + 0.5 * R_ang * kAngToBohr;
    std::vector<GpwAtom> atoms;
    const double p0[3] = {0.5 * L, 0.5 * L, z0};
    const double p1[3] = {0.5 * L, 0.5 * L, z1};
    atoms.emplace_back(p0, bO, pO, 8);
    atoms.emplace_back(p1, bO, pO, 8);

    GpwDft gpw;
    gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
    gpw.cutoff_ry = 120.0;
    gpw.n_alpha = 7;
    gpw.n_beta = 5;
    gpw.max_iter = 12;
    gpw.tol = 1e-5;
    gpw.mix = 0.4;
    gpw.xc = XcKind::PBE;
    gpw.mbd = true;
    gpw.hirshfeld = true;
    gpw.k_mesh[0] = gpw.k_mesh[1] = gpw.k_mesh[2] = 2;
    gpw.k_time_reversal = true;

    gpw.build(atoms);
    const auto& res = gpw.run(atoms);
    std::printf(
        "[Hirshfeld multi-k UKS] E_disp=%.16e  n_vol=%zu  n_kpts=%d  s2=%.4f  "
        "iters=%d\n",
        res.e_disp, res.volume_ratios.size(), res.n_kpts, res.s2, res.iters);
    for (std::size_t i = 0; i < res.volume_ratios.size(); ++i)
        std::printf("  v[%zu]=%.6f\n", i, res.volume_ratios[i]);

    ASSERT_EQ(res.volume_ratios.size(), 2u);
    EXPECT_GT(res.n_kpts, 1);
    EXPECT_TRUE(std::isfinite(res.e_disp));
    EXPECT_LT(res.e_disp, 0.0);
    EXPECT_LT(std::abs(res.e_disp), 1e-2);
    for (double v : res.volume_ratios) {
        EXPECT_GT(v, 0.2);
        EXPECT_LT(v, 2.0);
    }
}

// Closed-shell water UKS (Nα=Nβ) + hirshfeld-MBD ≈ RKS hirshfeld-MBD (same ρ).
TEST(GpwHirshfeld, ClosedShellUksMatchesRksHirshfeldEDisp) {
    using ATV::cpu::gpw::GaussianBasis;
    using ATV::cpu::gpw::GpwAtom;
    using ATV::cpu::gpw::GpwDft;
    using ATV::cpu::gpw::GthPotential;
    using ATV::cpu::gpw::XcKind;

    const double ang[6][3] = {
        {0.000000, 0.000000, 0.000000}, {0.758602, 0.000000, 0.504284},
        {0.758602, 0.000000, -0.504284}, {3.000000, 0.000000, 0.000000},
        {3.758602, 0.000000, 0.504284}, {3.758602, 0.000000, -0.504284},
    };
    const GaussianBasis bO(molrs::Element::O, "DZVP-MOLOPT-SR-GTH");
    const GaussianBasis bH(molrs::Element::H, "DZVP-MOLOPT-SR-GTH");
    const GthPotential pO(molrs::Element::O, "GTH-PBE-q6");
    const GthPotential pH(molrs::Element::H, "GTH-PBE-q1");
    std::vector<GpwAtom> atoms;
    for (int i = 0; i < 6; ++i) {
        const double pos[3] = {ang[i][0] * kAngToBohr, ang[i][1] * kAngToBohr,
                               ang[i][2] * kAngToBohr};
        if (i == 0 || i == 3)
            atoms.emplace_back(pos, bO, pO, 8);
        else
            atoms.emplace_back(pos, bH, pH, 1);
    }

    const double L = 12.0 * kAngToBohr;
    auto run_disp = [&](bool uks) {
        GpwDft gpw;
        gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
        gpw.cutoff_ry = 160.0;
        gpw.max_iter = 12;
        gpw.tol = 1e-5;
        gpw.mix = 0.4;
        gpw.xc = XcKind::PBE;
        gpw.mbd = true;
        gpw.hirshfeld = true;
        if (uks) {
            gpw.n_alpha = 8;
            gpw.n_beta = 8;
        } else {
            gpw.n_occ = 8;
        }
        gpw.build(atoms);
        return gpw.run(atoms);
    };

    const auto rks = run_disp(false);
    const auto uks = run_disp(true);
    std::printf("[Hirshfeld RKS/UKS] E_disp RKS=%.16e  UKS=%.16e  d=%.3e\n",
                rks.e_disp, uks.e_disp, uks.e_disp - rks.e_disp);
    ASSERT_EQ(rks.volume_ratios.size(), 6u);
    ASSERT_EQ(uks.volume_ratios.size(), 6u);
    EXPECT_TRUE(std::isfinite(rks.e_disp));
    EXPECT_TRUE(std::isfinite(uks.e_disp));
    EXPECT_LT(rks.e_disp, 0.0);
    EXPECT_LT(uks.e_disp, 0.0);
    // Same total density at spin-unpolarized fixed point ⇒ E_disp parity.
    // Loose gate: SCF residual + DIIS path differences at light cutoff.
    EXPECT_NEAR(uks.e_disp, rks.e_disp, 5e-6);
}


// Force dual-gates + product fold live in test_cpu_gpw_vol_forces.cpp
// (unit-test class: geometry FD + light H₂ wiring; ≤10 s hard cap).
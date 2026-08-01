/// @file test_cpu_gpw_dos.cpp
/// @brief Unit tests for GPW multi-k DOS free math and parameter guards.
///        No full SCF e2e (that lives in regressions/gpw/gpw-dos-si).

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/bands_dos.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"

using namespace ATV::cpu::gpw;

TEST(GpwGapFromSpectrum, SyntheticIndirectGap) {
    // 2 k-points, 3 bands; n_occ=2.
    // k0: -1, -0.1, 1.0  → VBM candidate -0.1
    // k1: -1, -0.5, 0.2  → CBM 0.2
    std::vector<std::vector<double>> evals = {
        {-1.0, -0.1, 1.0},
        {-1.0, -0.5, 0.2},
    };
    double e_vbm = 0, e_cbm = 0;
    int vk = -1, ck = -1, vb = -1, cb = -1;
    const double gap =
        gap_from_spectrum(evals, /*n_occ=*/2, e_vbm, e_cbm, vk, ck, vb, cb);
    EXPECT_NEAR(e_vbm, -0.1, 1e-12);
    EXPECT_NEAR(e_cbm, 0.2, 1e-12);
    EXPECT_NEAR(gap, 0.3, 1e-12);
    EXPECT_EQ(vk, 0);
    EXPECT_EQ(ck, 1);
    EXPECT_EQ(vb, 1);
    EXPECT_EQ(cb, 2);
}

TEST(GpwGaussianDos, SinglePeakIntegratesToWeightTimesTwo) {
    // One eigenvalue at 0 with weight 1 → ∫ D ≈ 2 (closed-shell factor).
    std::vector<double> evals = {0.0};
    std::vector<double> weights = {1.0};
    std::vector<double> energy, dos;
    const double sigma = 0.05;
    gaussian_dos(evals, weights, /*e_min=*/-1.0, /*e_max=*/1.0, /*n_bins=*/2001,
                 sigma, energy, dos);

    EXPECT_NEAR(integrate_dos_to(energy, dos, /*e_cut=*/1.0), 2.0, 1e-3);
}

TEST(GpwGaussianDos, SpinOrbitalDegeneracyOne) {
    std::vector<double> evals = {0.0};
    std::vector<double> weights = {1.0};
    std::vector<double> energy, dos;
    const double sigma = 0.05;
    gaussian_dos(evals, weights, /*e_min=*/-1.0, /*e_max=*/1.0, /*n_bins=*/2001,
                 sigma, energy, dos, /*spin_degeneracy=*/1.0);
    EXPECT_NEAR(integrate_dos_to(energy, dos, /*e_cut=*/1.0), 1.0, 1e-3);
}

TEST(GpwIntegrateDosTo, PartialCutoffHalfPeak) {
    std::vector<double> evals = {0.0};
    std::vector<double> weights = {1.0};
    std::vector<double> energy, dos;
    const double sigma = 0.05;
    gaussian_dos(evals, weights, /*e_min=*/-1.0, /*e_max=*/1.0, /*n_bins=*/4001,
                 sigma, energy, dos);
    EXPECT_NEAR(integrate_dos_to(energy, dos, /*e_cut=*/0.0), 1.0, 2e-2);
    EXPECT_NEAR(integrate_dos_to(energy, dos, /*e_cut=*/-1.0), 0.0, 1e-6);
}

TEST(GpwDosGuards, RejectsGammaOnlyMesh) {
    GpwDft method;
    method.cell[0] = method.cell[1] = method.cell[2] = 10.0;
    method.n_occ = 1;
    method.cutoff_ry = 50.0;
    // No atoms needed for the multi-k gate — compute throws before run when
    // mesh is Γ-only. Still need a non-empty atom list for API contract if it
    // got past the gate; the gate fires first.
    Dos dos;
    dos.k_mesh = 1; // forces Γ after clear of explicit list
    dos.sigma_ha = 0.025;
    std::vector<GpwAtom> atoms; // empty ok — gate is before run
    EXPECT_THROW(dos.compute(method, atoms), std::invalid_argument);
}

TEST(GpwDosGuards, RejectsNonPositiveSigma) {
    Dos dos;
    dos.k_mesh = 2;
    dos.sigma_ha = 0.0;
    GpwDft method;
    std::vector<GpwAtom> atoms;
    EXPECT_THROW(dos.compute(method, atoms), std::invalid_argument);
}

TEST(GpwDosResult, DefaultIsRestricted) {
    DosResult dr;
    EXPECT_EQ(dr.nspin, 1);
    EXPECT_TRUE(dr.dos_alpha.empty());
    EXPECT_TRUE(dr.dos_beta.empty());
    EXPECT_NEAR(dr.e_fermi_ha, 0.0, 1e-15);
    EXPECT_EQ(dr.n_kpts, 0);
}

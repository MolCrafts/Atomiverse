/// @file test_pw_bands_dos.cpp
/// @brief Unit tests for band-path geometry, gap finder, Gaussian DOS.
///        No full SCF e2e (that lives in regressions/pw/bands-dos-si).

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/bands_dos.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV::cpu::pw;

TEST(BandsPath, DiamondPathIncludesGammaAndX) {
    Bands bands;
    bands.set_diamond_path(4);
    ASSERT_GE(bands.kfrac.size(), 5u);

    auto near = [](const std::array<double, 3>& a,
                   const std::array<double, 3>& b, double tol = 1e-12) {
        return std::abs(a[0] - b[0]) < tol && std::abs(a[1] - b[1]) < tol
               && std::abs(a[2] - b[2]) < tol;
    };

    const std::array<double, 3> G{0, 0, 0};
    const std::array<double, 3> X{0.5, 0.0, 0.5};
    bool has_G = false, has_X = false;
    for (const auto& k : bands.kfrac) {
        if (near(k, G))
            has_G = true;
        if (near(k, X))
            has_X = true;
    }
    EXPECT_TRUE(has_G);
    EXPECT_TRUE(has_X);

    // n_seg=4 → 4 samples per segment; 5 segments with shared joins:
    // start + 5*(4-1) = 1 + 15 = 16.
    EXPECT_EQ(bands.kfrac.size(), 16u);
}

TEST(GapFromSpectrum, SyntheticIndirectGap) {
    // 2 k-points, 3 bands; n_occ=2.
    // k0: -1.0, -0.5, 0.8   (VBM candidate -0.5)
    // k1: -0.9, -0.2, 0.3   (VBM -0.2, CBM 0.3) → VBM at k1, CBM at k1
    // Make CBM at different k:
    // k0: -1, -0.1, 1.0
    // k1: -1, -0.5, 0.2
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

TEST(GaussianDos, SinglePeakIntegratesToWeightTimesTwo) {
    // One eigenvalue at 0 with weight 1 → ∫ D ≈ 2 (closed-shell factor).
    std::vector<double> evals = {0.0};
    std::vector<double> weights = {1.0};
    std::vector<double> energy, dos;
    const double sigma = 0.05;
    gaussian_dos(evals, weights, /*e_min=*/-1.0, /*e_max=*/1.0, /*n_bins=*/2001,
                 sigma, energy, dos);

    double integ = 0;
    const double de = energy[1] - energy[0];
    for (std::size_t i = 0; i + 1 < energy.size(); ++i)
        integ += 0.5 * (dos[i] + dos[i + 1]) * de;
    EXPECT_NEAR(integ, 2.0, 1e-3);

    // Same integral via shared helper (metallic path uses cut = E_F).
    EXPECT_NEAR(integrate_dos_to(energy, dos, /*e_cut=*/1.0), 2.0, 1e-3);
}

TEST(GaussianDos, SpinOrbitalDegeneracyOne) {
    // Collinear spin-orbital: g=1 → ∫ D ≈ weight (not 2×).
    std::vector<double> evals = {0.0};
    std::vector<double> weights = {1.0};
    std::vector<double> energy, dos;
    const double sigma = 0.05;
    gaussian_dos(evals, weights, /*e_min=*/-1.0, /*e_max=*/1.0, /*n_bins=*/2001,
                 sigma, energy, dos, /*spin_degeneracy=*/1.0);
    EXPECT_NEAR(integrate_dos_to(energy, dos, /*e_cut=*/1.0), 1.0, 1e-3);
}

TEST(IntegrateDosTo, PartialCutoffHalfPeak) {
    // Single peak at 0, closed-shell weight 1 → full ∫ = 2.
    // Integrating only to 0 (peak centre, symmetric) ≈ 1.
    std::vector<double> evals = {0.0};
    std::vector<double> weights = {1.0};
    std::vector<double> energy, dos;
    const double sigma = 0.05;
    gaussian_dos(evals, weights, /*e_min=*/-1.0, /*e_max=*/1.0, /*n_bins=*/4001,
                 sigma, energy, dos);
    const double half = integrate_dos_to(energy, dos, /*e_cut=*/0.0);
    EXPECT_NEAR(half, 1.0, 2e-2);
    EXPECT_NEAR(integrate_dos_to(energy, dos, /*e_cut=*/-1.0), 0.0, 1e-6);
}

TEST(BandsResult, DefaultIsRestricted) {
    BandsResult br;
    EXPECT_EQ(br.nspin, 1);
    EXPECT_TRUE(br.eigenvalues_beta_ry.empty());
    EXPECT_NEAR(br.magnetization, 0.0, 1e-15);
    EXPECT_EQ(br.vbm_spin, 0);
    EXPECT_EQ(br.cbm_spin, 0);
    EXPECT_NEAR(br.e_fermi_ry, 0.0, 1e-15);
}

TEST(BandsCompute, SpinPlusMetallicSmearRequiresPositiveDegauss) {
    // Guard only: spin+smear is product; degauss must be > 0.
    ATV::cpu::Frame frame(1);
    frame.span(ATV::kAtom_type)[0] = 14;
    frame.span(ATV::kAtom_x)[0] = 0;
    frame.span(ATV::kAtom_y)[0] = 0;
    frame.span(ATV::kAtom_z)[0] = 0;
    frame.box = ATV::cpu::Box::cubic(5.0);

    PW method;
    method.gth_name = "GTH-PBE-q4"; // valid potential: degauss is what fails
    method.spin_polarized = true;
    method.multiplicity = 1;
    method.smearing = Smearing::fermi_dirac;
    method.degauss = 0.0; // invalid
    method.extra_bands = 2;
    method.ecut_ry = 5.0;
    method.k_mesh = 1;

    Bands bands;
    bands.n_bands = 4;
    bands.set_diamond_path(2);
    EXPECT_THROW(bands.compute(method, frame), std::invalid_argument);
}

TEST(BandsCompute, SpinPlusMetallicSmearNotRejectedByCombo) {
    // Prior residual: combo threw before any SCF. Now only bad degauss /
    // multiplicity / n_bands gates fire. Empty UPF fails inside setup (I/O),
    // not at a spin+smear reject — same pattern as DFPT spin+smear unit.
    ATV::cpu::Frame frame(1);
    frame.span(ATV::kAtom_type)[0] = 14;
    frame.span(ATV::kAtom_x)[0] = 0;
    frame.span(ATV::kAtom_y)[0] = 0;
    frame.span(ATV::kAtom_z)[0] = 0;
    frame.box = ATV::cpu::Box::cubic(5.0);

    PW method;
    method.spin_polarized = true;
    method.multiplicity = 1;
    method.smearing = Smearing::fermi_dirac;
    method.degauss = 0.02;
    method.extra_bands = 2;
    method.ecut_ry = 5.0;
    method.k_mesh = 1;
    method.gth_name = "GTH-PBE-q4"; // analytic Si, no file dependency
    method.max_iter = 1;

    Bands bands;
    bands.n_bands = 4;
    bands.set_diamond_path(2);
    try {
        (void)bands.compute(method, frame);
        // Toy empty-UPF path may complete a cheap SCF; either is fine.
        SUCCEED();
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_EQ(msg.find("collinear spin + metallic smearing not"),
                  std::string::npos)
            << "unexpected residual reject: " << msg;
        // Other invalid_argument (parity / n_bands) would be a real bug here.
        EXPECT_TRUE(msg.find("n_bands") != std::string::npos
                    || msg.find("multiplicity") != std::string::npos
                    || msg.find("degauss") != std::string::npos
                    || msg.find("parity") != std::string::npos
                    || msg.find("UPF") != std::string::npos
                    || msg.find("upf") != std::string::npos
                    || msg.find("pseudopotential") != std::string::npos)
            << "unexpected invalid_argument: " << msg;
    } catch (const std::exception&) {
        // I/O or runtime failures inside SCF setup are acceptable for this
        // unit (no full e2e); e2e is reg_bands-spin-metallic-si.
        SUCCEED();
    }
}

TEST(DosResult, DefaultIsRestricted) {
    DosResult dr;
    EXPECT_EQ(dr.nspin, 1);
    EXPECT_TRUE(dr.dos_alpha.empty());
    EXPECT_TRUE(dr.dos_beta.empty());
    EXPECT_NEAR(dr.magnetization, 0.0, 1e-15);
    EXPECT_NEAR(dr.e_fermi_ry, 0.0, 1e-15);
}

TEST(DosCompute, SpinPlusMetallicNoLongerGuardReject) {
    // Guard that once threw "spin-polarized + metallic smearing not
    // supported" is gone. Without a UPF the collinear-smear path fails
    // later in setup (I/O / toy path), not at the product gate — same
    // pattern as DFPT spin+smear unit tests.
    ATV::cpu::Frame frame(1);
    frame.span(ATV::kAtom_type)[0] = 14;
    frame.span(ATV::kAtom_x)[0] = 0;
    frame.span(ATV::kAtom_y)[0] = 0;
    frame.span(ATV::kAtom_z)[0] = 0;
    frame.box = ATV::cpu::Box::cubic(5.0);

    PW method;
    method.spin_polarized = true;
    method.multiplicity = 1;
    method.smearing = Smearing::fermi_dirac;
    method.degauss = 0.02;
    method.extra_bands = 2;
    method.ecut_ry = 5.0;
    method.k_mesh = 1;
    method.max_iter = 1;
    method.upf_path = "/nonexistent/Si.upf";

    Dos dos;
    dos.k_mesh = 1;
    dos.n_bands = 4;
    dos.sigma_ry = 0.05;
    try {
        (void)dos.compute(method, frame);
        // Toy/empty UPF may still complete a minimal path; either outcome
        // is fine as long as the old spin+smear guard message is gone.
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_EQ(msg.find("spin-polarized + metallic smearing not"),
                  std::string::npos);
        EXPECT_EQ(msg.find("spin + metallic"), std::string::npos);
    } catch (const std::exception&) {
        // I/O / setup failures are expected without a real UPF.
    }
}

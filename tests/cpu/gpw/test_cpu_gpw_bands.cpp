/// @file test_cpu_gpw_bands.cpp
/// @brief Unit tests for GPW band-path geometry and gap finder.
///        No full SCF e2e (that lives in regressions/gpw/gpw-bands-si).

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/bands_dos.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV::cpu::gpw;

TEST(GpwBandsPath, DiamondPathIncludesGammaAndX) {
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

TEST(GpwGapFromSpectrum, SyntheticIndirectGap) {
    // k0: VBM at band 1 = -0.1; k1: CBM at band 2 = 0.2 → gap 0.3
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

TEST(GpwBandsGuards, RejectsUksAndSmear) {
    const double L = 8.0;
    const GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
    const GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
    const double r0[3] = {0.5 * L - 0.7, 0.5 * L, 0.5 * L};
    const double r1[3] = {0.5 * L + 0.7, 0.5 * L, 0.5 * L};
    std::vector<GpwAtom> h2;
    h2.emplace_back(r0, bH, pH, 1);
    h2.emplace_back(r1, bH, pH, 1);

    GpwDft uks;
    uks.cell[0] = uks.cell[1] = uks.cell[2] = L;
    uks.cutoff_ry = 80.0;
    uks.n_alpha = 1;
    uks.n_beta = 1;
    uks.max_iter = 1;
    uks.k_mesh[0] = uks.k_mesh[1] = uks.k_mesh[2] = 1;
    uks.xc = XcKind::PADE;
    uks.build(h2);

    Bands bands;
    bands.n_bands = 2;
    bands.set_diamond_path(2);
    EXPECT_THROW(bands.compute(uks, h2), std::invalid_argument);

    GpwDft smear;
    smear.cell[0] = smear.cell[1] = smear.cell[2] = L;
    smear.cutoff_ry = 80.0;
    smear.n_occ = 1;
    smear.max_iter = 1;
    smear.k_mesh[0] = smear.k_mesh[1] = smear.k_mesh[2] = 2;
    smear.smearing = Smearing::fermi_dirac;
    smear.degauss = 0.02;
    smear.xc = XcKind::PADE;
    smear.build(h2);
    EXPECT_THROW(bands.compute(smear, h2), std::invalid_argument);
}

TEST(GpwBandsResult, DefaultIsRestricted) {
    BandsResult br;
    EXPECT_EQ(br.nspin, 1);
    EXPECT_TRUE(br.eigenvalues_ha.empty());
    EXPECT_NEAR(br.gap_ha, 0.0, 1e-15);
    EXPECT_FALSE(br.scf_converged);
}

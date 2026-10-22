/// @file test_cpu_gpw_bandpath.cpp
/// @brief Unit tests for GPW band-path geometry and gap finder.
///        No full SCF e2e (that lives in regressions/gpw/bands-dos).

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

TEST(GpwBandsGuards, RejectsDispersionOnPath) {
    const double L = 8.0;
    // H2 in an L-Bohr cube; Frame speaks Å.
    const double b2a = ATV::units::kBohrInAngstrom;
    ATV::cpu::Frame h2(2);
    const double xs[2] = {0.5 * L - 0.7, 0.5 * L + 0.7};
    for (int i = 0; i < 2; ++i) {
        h2.span(ATV::kAtom_type)[i] = 1;
        h2.span(ATV::kAtom_x)[i] = xs[i] * b2a;
        h2.span(ATV::kAtom_y)[i] = 0.5 * L * b2a;
        h2.span(ATV::kAtom_z)[i] = 0.5 * L * b2a;
    }
    h2.box = ATV::cpu::Box::cubic(L * b2a);

    // Multigrid / DFT+U / dispersion throw in Bands before SCF; d3 is the
    // lightest product-illegal flag that still builds on the k path.
    GpwDft disp{{Kind{molrs::Element::H,
                      GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
                      GthPotential(molrs::Element::H, "GTH-PADE-q1")}},
                /*cutoff_ry=*/80.0,
                XcKind::PADE};
    disp.n_occ = 1;
    disp.max_iter = 1;
    disp.k_mesh[0] = disp.k_mesh[1] = disp.k_mesh[2] = 2;
    disp.d3 = true;
    disp.local_pp_on_grid = true;
    disp.build(h2);

    Bands bands;
    bands.n_bands = 2;
    bands.set_diamond_path(2);
    EXPECT_THROW(bands.compute(disp, h2), std::invalid_argument);
}

TEST(GpwBandsResult, DefaultIsRestricted) {
    BandsResult br;
    EXPECT_EQ(br.nspin, 1);
    EXPECT_TRUE(br.eigenvalues_ha.empty());
    EXPECT_TRUE(br.eigenvalues_beta_ha.empty());
    EXPECT_NEAR(br.gap_ha, 0.0, 1e-15);
    EXPECT_NEAR(br.e_fermi_ha, 0.0, 1e-15);
    EXPECT_FALSE(br.scf_converged);
}

/// @file tests/cpu/observe/test_species_density_profile.cpp
/// @brief Unit: SpeciesDensityProfile per-species ρ_s(z) binning + run mean.

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/species_density_profile.h"

using namespace ATV;
using cpu::SpeciesDensityProfile;

namespace {

/// Two cations (type 1) at z = 1.5, 5.5; one anion (type 2) at z = 5.5.
/// Box 10×10×10 → bin volume with 10 bins = 10·10·1 = 100 Å³.
cpu::Frame three_ion_slab() {
    cpu::Frame f(3);
    const double zz[3] = {1.5, 5.5, 5.5};
    const int ty[3] = {1, 1, 2};
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_x)[i] = 5.0;
        f.span(kAtom_y)[i] = 5.0;
        f.span(kAtom_z)[i] = zz[i];
        f.span(kAtom_type)[i] = ty[i];
        f.span(kAtom_charge)[i] = (ty[i] == 1) ? 1.0 : -1.0;
        f.span(kAtom_mass)[i] = 100.0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(10.0);
    return f;
}

} // namespace

TEST(SpeciesDensityProfile, InvalidCtorThrows) {
    EXPECT_THROW(SpeciesDensityProfile({}, 10, 0.0, 10.0),
                 std::invalid_argument);
    EXPECT_THROW(SpeciesDensityProfile({1}, 0, 0.0, 10.0),
                 std::invalid_argument);
    EXPECT_THROW(SpeciesDensityProfile({1}, -3, 0.0, 10.0),
                 std::invalid_argument);
    EXPECT_THROW(SpeciesDensityProfile({1}, 10, 5.0, 5.0),
                 std::invalid_argument);
    EXPECT_THROW(SpeciesDensityProfile({1}, 10, 6.0, 5.0),
                 std::invalid_argument);
}

TEST(SpeciesDensityProfile, InstantaneousBinning) {
    auto f = three_ion_slab();
    SpeciesDensityProfile obs({1, 2}, /*n_bins=*/10, /*z_min=*/0.0,
                              /*z_max=*/10.0);
    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, f, 0, 0};
    obs.launch(lc);

    ASSERT_EQ(obs.n_species(), 2u);
    EXPECT_EQ(obs.species_type(0), 1);
    EXPECT_EQ(obs.species_type(1), 2);
    EXPECT_EQ(obs.n_bins(), 10);
    EXPECT_NEAR(obs.dz(), 1.0, 1e-15);
    EXPECT_NEAR(obs.z_center(1), 1.5, 1e-15);

    // Cations: bin 1 (z=1.5) and bin 5 (z=5.5), 1/(100 Å³) each.
    EXPECT_NEAR(obs.density(0)[1], 0.01, 1e-15);
    EXPECT_NEAR(obs.density(0)[5], 0.01, 1e-15);
    EXPECT_NEAR(obs.density(0)[0], 0.0, 1e-15);
    // Anion: bin 5 only.
    EXPECT_NEAR(obs.density(1)[5], 0.01, 1e-15);
    EXPECT_NEAR(obs.density(1)[1], 0.0, 1e-15);

    EXPECT_EQ(obs.binned_count(0), 2);
    EXPECT_EQ(obs.binned_count(1), 1);
}

TEST(SpeciesDensityProfile, WindowExcludesOutsideAtoms) {
    auto f = three_ion_slab();
    // Window [3, 8): the z=1.5 cation falls outside.
    SpeciesDensityProfile obs({1, 2}, /*n_bins=*/5, /*z_min=*/3.0,
                              /*z_max=*/8.0);
    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, f, 0, 0};
    obs.launch(lc);

    EXPECT_EQ(obs.binned_count(0), 1);
    EXPECT_EQ(obs.binned_count(1), 1);
    // z=5.5 → bin 2 of [3,8) with dz=1.
    EXPECT_NEAR(obs.density(0)[2], 0.01, 1e-15);
    EXPECT_NEAR(obs.density(1)[2], 0.01, 1e-15);
}

TEST(SpeciesDensityProfile, MeanAccumulatesOverLaunches) {
    auto f = three_ion_slab();
    SpeciesDensityProfile obs({1}, /*n_bins=*/10, /*z_min=*/0.0,
                              /*z_max=*/10.0);
    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, f, 0, 0};

    EXPECT_EQ(obs.n_samples(), 0);
    EXPECT_NEAR(obs.mean_density(0, 1), 0.0, 1e-15); // pre-launch: zero

    obs.launch(lc);
    // Move the z=1.5 cation to bin 2 and launch again.
    f.span(kAtom_z)[0] = 2.5;
    obs.launch(lc);

    EXPECT_EQ(obs.n_samples(), 2);
    // Bin 1 held one cation in launch 1 only → mean 0.005 Å⁻³.
    EXPECT_NEAR(obs.mean_density(0, 1), 0.005, 1e-15);
    EXPECT_NEAR(obs.mean_density(0, 2), 0.005, 1e-15);
    // Bin 5 held one cation both launches → mean 0.01 Å⁻³.
    EXPECT_NEAR(obs.mean_density(0, 5), 0.01, 1e-15);

    obs.reset();
    EXPECT_EQ(obs.n_samples(), 0);
    EXPECT_NEAR(obs.mean_density(0, 5), 0.0, 1e-15);
    // Instantaneous survives reset.
    EXPECT_NEAR(obs.density(0)[5], 0.01, 1e-15);
}

TEST(SpeciesDensityProfile, NormalizationSumsToCountOverVolume) {
    auto f = three_ion_slab();
    SpeciesDensityProfile obs({1, 2}, /*n_bins=*/20, /*z_min=*/0.0,
                              /*z_max=*/10.0);
    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, f, 0, 0};
    obs.launch(lc);

    const double area_dz = 10.0 * 10.0 * obs.dz();
    for (std::size_t s = 0; s < obs.n_species(); ++s) {
        double n_from_profile = 0.0;
        for (int i = 0; i < obs.n_bins(); ++i)
            n_from_profile += obs.density(s)[i] * area_dz;
        EXPECT_NEAR(n_from_profile, static_cast<double>(obs.binned_count(s)),
                    1e-12);
    }
}

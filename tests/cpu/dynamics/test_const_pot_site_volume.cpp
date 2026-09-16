/// @file tests/cpu/dynamics/test_const_pot_site_volume.cpp
/// @brief Unit: ConstPotBO per-site TF volume path (Voronoi companion).

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

constexpr double kD = 0.0593; ///< [1/Å³] fixed-density twin.

/// Two-plate 4-site toy with an electrolyte pair (types 1/2 electrodes).
cpu::Frame plate_toy() {
    cpu::Frame f(6);
    const double z[6] = {6.0, 8.0, 12.0, 14.0, 9.0, 11.0};
    const int typ[6] = {1, 1, 2, 2, 3, 3};
    const double q0[6] = {0.0, 0.0, 0.0, 0.0, 0.5, -0.5};
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_x)[i] = 0.0;
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_charge)[i] = q0[i];
        f.span(kAtom_type)[i] = typ[i];
        f.span(kAtom_mass)[i] = (typ[i] < 3) ? 196.97 : 1.0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::orthorhombic(4.0, 4.0, 20.0);
    return f;
}

cpu::md::ConstPotBO tf_conp() {
    cpu::md::ConstPotBO conp(2.0);
    conp.add_electrode(1, 0.0).add_electrode(2, 1.0);
    conp.xy_image_range = 2;
    conp.l_tf = 2.0;
    return conp;
}

void run_once(cpu::md::ConstPotBO& conp, cpu::Frame& f) {
    cpu::Context ctx;
    conp.build({ctx, f});
    conp.launch({ctx, f, /*stage=*/0, /*step=*/0});
}

} // namespace

TEST(ConstPotSiteVolume, UniformVolumeMatchesFixedDensity) {
    auto fd = plate_toy();
    auto dens = tf_conp();
    dens.site_density = kD;
    run_once(dens, fd);

    auto fv = plate_toy();
    auto vol = tf_conp();
    vol.site_volume.assign(6, 1.0 / kD); // v_i = 1/d on every row
    run_once(vol, fv);

    ASSERT_EQ(dens.charges().size(), vol.charges().size());
    for (std::size_t i = 0; i < dens.charges().size(); ++i)
        EXPECT_NEAR(vol.charges()[i], dens.charges()[i], 1e-12);
}

TEST(ConstPotSiteVolume, SmallerVolumeReducesCapacitance) {
    auto f1 = plate_toy();
    auto bulk = tf_conp();
    bulk.site_volume.assign(6, 1.0 / kD);
    run_once(bulk, f1);

    auto f2 = plate_toy();
    auto dense = tf_conp();
    dense.site_volume.assign(6, 0.5 / kD); // halved volume -> harder diagonal
    run_once(dense, f2);

    EXPECT_LT(dense.capacitance(), bulk.capacitance());
    EXPECT_GT(dense.capacitance(), 0.0);
}

TEST(ConstPotSiteVolume, ThrowsWhenBothDensityAndVolumeSet) {
    auto f = plate_toy();
    auto conp = tf_conp();
    conp.site_density = kD;
    conp.site_volume.assign(6, 1.0 / kD);
    cpu::Context ctx;
    EXPECT_THROW(conp.build({ctx, f}), std::invalid_argument);
}

TEST(ConstPotSiteVolume, ThrowsWhenNeitherDensityNorVolumeSet) {
    auto f = plate_toy();
    auto conp = tf_conp();
    cpu::Context ctx;
    EXPECT_THROW(conp.build({ctx, f}), std::invalid_argument);
}

TEST(ConstPotSiteVolume, ThrowsOnWrongSizeVolumeVector) {
    auto f = plate_toy();
    auto conp = tf_conp();
    conp.site_volume.assign(4, 1.0 / kD); // frame has 6 rows
    cpu::Context ctx;
    EXPECT_THROW(conp.build({ctx, f}), std::invalid_argument);
}

TEST(ConstPotSiteVolume, ThrowsOnNonPositiveElectrodeVolume) {
    auto f = plate_toy();
    auto conp = tf_conp();
    conp.site_volume.assign(6, 1.0 / kD);
    conp.site_volume[2] = 0.0; // an electrode row (type 2)
    cpu::Context ctx;
    EXPECT_THROW(conp.build({ctx, f}), std::invalid_argument);
}

TEST(ConstPotSiteVolume, NonElectrodeRowsIgnored) {
    // Zero volume on electrolyte rows (types 3) must not throw or change Q.
    auto fa = plate_toy();
    auto a = tf_conp();
    a.site_volume.assign(6, 1.0 / kD);
    run_once(a, fa);

    auto fb = plate_toy();
    auto b = tf_conp();
    b.site_volume.assign(6, 1.0 / kD);
    b.site_volume[4] = 0.0;
    b.site_volume[5] = 0.0;
    run_once(b, fb);

    for (std::size_t i = 0; i < a.charges().size(); ++i)
        EXPECT_NEAR(b.charges()[i], a.charges()[i], 1e-14);
}

/// @file tests/cpu/semiempirical/test_xtb_gfn1.cpp
/// Unit tests for GFN1-xTB (tables, throws, finite energy). Pin lives in
/// regressions/semiempirical/gfn1-xtb-organic.
#include <cmath>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/basis_data/gfn1_tables.h"
#include "atomiverse/cpu/semiempirical/gfn1.h"

using namespace ATV;
using namespace ATV::cpu::semiempirical;

namespace {

cpu::Frame make_h2(double r = 0.74) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    return f;
}

cpu::Frame make_ch4() {
    cpu::Frame f(5);
    f.span(kAtom_type)[0] = 6;
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    const double s = 1.09 / std::sqrt(3.0);
    const double hpos[4][3] = {
        {s, s, s}, {s, -s, -s}, {-s, s, -s}, {-s, -s, s}};
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_type)[i + 1] = 1;
        f.span(kAtom_x)[i + 1] = hpos[i][0];
        f.span(kAtom_y)[i + 1] = hpos[i][1];
        f.span(kAtom_z)[i + 1] = hpos[i][2];
    }
    for (int i = 0; i < 5; ++i) {
        f.span(kAtom_mass)[i] = (i == 0) ? 12.011 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

} // namespace

TEST(GFN1, TablesCoverHCNO) {
    using namespace basis_data;
    EXPECT_NEAR(kGfn1Elements[0].zeff, 1.116244, 1e-8);
    EXPECT_NEAR(kGfn1Elements[1].zeff, 4.428763, 1e-8);
    EXPECT_NEAR(kGfn1Elements[1].alpha, 1.281954, 1e-8);
    EXPECT_EQ(kGfn1NElements, 4u);
    int mask = 0;
    for (std::size_t i = 0; i < kGfn1NElements; ++i)
        mask |= (1 << kGfn1Elements[i].z);
    EXPECT_TRUE(mask & (1 << 1));
    EXPECT_TRUE(mask & (1 << 6));
    EXPECT_TRUE(mask & (1 << 7));
    EXPECT_TRUE(mask & (1 << 8));
    EXPECT_EQ(kGfn1Elements[0].n_ao, 2); // H: 1s+2s
    EXPECT_EQ(kGfn1Elements[1].n_ao, 4); // C: 2s+2p
    EXPECT_FALSE(kGfn1Elements[0].valence[1]);
}

TEST(GFN1, ProvidesForceFalse) {
    GFN1 g;
    EXPECT_FALSE(g.provides_force());
    EXPECT_TRUE(g.provides_energy());
}

TEST(GFN1, UnsupportedZThrows) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 9; // F — not in organic tables
    f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0;
    f.span(kAtom_mass)[0] = 19.0;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    cpu::Context ctx;
    GFN1 g;
    g.build({ctx, f});
    try {
        g.launch({ctx, f, 0, 0});
        FAIL() << "expected unsupported-Z throw";
    } catch (const std::runtime_error& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("Z=9"), std::string::npos);
        EXPECT_NE(msg.find("H/C/N/O"), std::string::npos);
    }
}

TEST(GFN1, WithForcesThrows) {
    auto f = make_h2();
    cpu::Context ctx;
    GFN1 g;
    g.with_forces = true;
    g.build({ctx, f});
    EXPECT_THROW(g.launch({ctx, f, 0, 0}), std::runtime_error);
}

TEST(GFN1, OpenShellThrows) {
    auto f = make_h2();
    cpu::Context ctx;
    GFN1 g;
    g.multiplicity = 3;
    g.build({ctx, f});
    EXPECT_THROW(g.launch({ctx, f, 0, 0}), std::runtime_error);
}

TEST(GFN1, H2EnergyFinite) {
    auto f = make_h2();
    cpu::Context ctx;
    GFN1 g;
    g.build({ctx, f});
    g.launch({ctx, f, 0, 0});
    EXPECT_TRUE(std::isfinite(g.last_energy()));
    EXPECT_LT(g.last_energy(), 0.0);
    EXPECT_NEAR(f.meta.get(kEnergy), g.last_energy(), 1e-12);
}

TEST(GFN1, CH4EnergyFinite) {
    auto f = make_ch4();
    cpu::Context ctx;
    GFN1 g;
    g.build({ctx, f});
    g.launch({ctx, f, 0, 0});
    EXPECT_TRUE(std::isfinite(g.last_energy()));
    EXPECT_LT(g.last_energy(), 0.0);
}

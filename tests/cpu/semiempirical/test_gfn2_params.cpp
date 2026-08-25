/// @file tests/cpu/semiempirical/test_gfn2_params.cpp
/// Unit tests for GFN2-xTB static tables and H2O energy path.
#include <cmath>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/basis_data/gfn2_xtb_tables.h"
#include "atomiverse/cpu/semiempirical/gfn2.h"

using namespace ATV;
using namespace ATV::cpu::semiempirical;

namespace {

cpu::Frame make_water() {
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = 0.96;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_x)[2] = -0.24;
    f.span(kAtom_y)[2] = 0.93;
    f.span(kAtom_z)[2] = 0.0;
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_mass)[i] = (i == 0) ? 15.999 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

} // namespace

TEST(Gfn2Params, TablesCoverHCNO) {
    using namespace basis_data;
    EXPECT_EQ(kGfn2NElements, 4u);
    int mask = 0;
    for (std::size_t i = 0; i < kGfn2NElements; ++i)
        mask |= (1 << kGfn2Elements[i].z);
    EXPECT_TRUE(mask & (1 << 1));
    EXPECT_TRUE(mask & (1 << 6));
    EXPECT_TRUE(mask & (1 << 7));
    EXPECT_TRUE(mask & (1 << 8));
}

TEST(Gfn2Params, OxygenQkernelIsTabulated) {
    using namespace basis_data;
    bool found = false;
    for (std::size_t i = 0; i < kGfn2NElements; ++i) {
        if (kGfn2Elements[i].z != 8)
            continue;
        found = true;
        EXPECT_NE(kGfn2Elements[i].qkernel, 0.0);
        EXPECT_NE(kGfn2Elements[i].dkernel, 0.0);
    }
    EXPECT_TRUE(found);
}

TEST(Gfn2Params, UnsupportedElementThrows) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 9; // F — not in the H/C/N/O foothold
    f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0;
    f.span(kAtom_mass)[0] = 19.0;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    cpu::Context ctx;
    GFN2 gfn2;
    gfn2.with_forces = false;
    gfn2.build({ctx, f});
    try {
        gfn2.launch({ctx, f, 0, 0});
        FAIL() << "expected throw for unsupported Z";
    } catch (const std::runtime_error& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("unsupported element"), std::string::npos);
        EXPECT_NE(msg.find("H/C/N/O"), std::string::npos);
    }
}

TEST(Gfn2Params, WaterEnergyFiniteAndSccConverged) {
    auto f = make_water();
    cpu::Context ctx;
    GFN2 gfn2;
    gfn2.with_forces = false;
    gfn2.scc_tol = 1e-8;
    gfn2.build({ctx, f});
    gfn2.launch({ctx, f, 0, 0});
    const double e = f.meta.get(kEnergy);
    EXPECT_TRUE(std::isfinite(e));
    EXPECT_GT(std::fabs(e), 1.0);
    EXPECT_LT(gfn2.last_scc_residual(), gfn2.scc_tol);
    EXPECT_GT(gfn2.last_scc_iterations(), 0);
    EXPECT_EQ(gfn2.last_mulliken_dq().size(), 3u);
    const double pin = -5.070383095219 * units::kHartreeInEv;
    EXPECT_LT(std::fabs(e - pin), 1.0e-3);
}

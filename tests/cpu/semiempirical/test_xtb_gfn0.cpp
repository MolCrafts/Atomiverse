/// @file tests/cpu/semiempirical/test_xtb_gfn0.cpp
/// Unit tests for production GFN0-xTB (H/C/N/O tables + energy skeleton).
#include <cmath>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/basis_data/gfn0_tables.h"
#include "atomiverse/cpu/semiempirical/gfn0.h"

using namespace ATV;
using namespace ATV::cpu::semiempirical;

namespace {

cpu::Frame make_h2o() {
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

TEST(XtbGfn0, TablesCoverHCNO) {
    using namespace basis_data;
    EXPECT_EQ(kGfn0NElements, 4u);
    int mask = 0;
    for (std::size_t i = 0; i < kGfn0NElements; ++i)
        mask |= (1 << kGfn0Elements[i].z);
    EXPECT_TRUE(mask & (1 << 1));
    EXPECT_TRUE(mask & (1 << 6));
    EXPECT_TRUE(mask & (1 << 7));
    EXPECT_TRUE(mask & (1 << 8));
}

TEST(XtbGfn0, UnsupportedZThrows) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 9; // F
    f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0;
    f.span(kAtom_mass)[0] = 19.0;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    cpu::Context ctx;
    GFN0 gfn0;
    gfn0.build({ctx, f});
    EXPECT_THROW(
        {
            try {
                gfn0.launch({ctx, f, 0, 0});
            } catch (const std::runtime_error& e) {
                EXPECT_NE(std::string(e.what()).find("unsupported"),
                          std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}

TEST(XtbGfn0, WaterEnergyFinite) {
    auto f = make_h2o();
    cpu::Context ctx;
    GFN0 gfn0;
    gfn0.build({ctx, f});
    gfn0.launch({ctx, f, 0, 0});
    EXPECT_TRUE(std::isfinite(gfn0.last_energy()));
    EXPECT_LT(gfn0.last_energy(), 0.0);
}

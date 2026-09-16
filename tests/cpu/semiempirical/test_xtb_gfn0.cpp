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

TEST(XtbGfn0, AnalyticForceMatchesCentralFD) {
    auto f = make_h2o();
    cpu::Context ctx;
    GFN0 gfn0;
    gfn0.with_forces = true;
    gfn0.build({ctx, f});
    gfn0.launch({ctx, f, 0, 0});
    const double fx0 = static_cast<double>(f.span(kAtom_fx)[0]);
    const double fy1 = static_cast<double>(f.span(kAtom_fy)[1]);
    EXPECT_GT(std::fabs(fx0) + std::fabs(fy1), 1e-4);

    auto energy_only = [](cpu::Frame g) {
        g.meta.get(kEnergy) = 0;
        for (int i = 0; i < 3; ++i)
            g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
        cpu::Context c;
        GFN0 m;
        m.with_forces = false;
        m.build({c, g});
        m.launch({c, g, 0, 0});
        return g.meta.get(kEnergy);
    };
    auto fd = [&](int atom, int k, double h) {
        auto shifted = [&](double s) {
            cpu::Frame g = make_h2o();
            StorageReal* col = (k == 0)   ? g.span(kAtom_x)
                               : (k == 1) ? g.span(kAtom_y)
                                          : g.span(kAtom_z);
            col[atom] += static_cast<StorageReal>(s);
            return energy_only(g);
        };
        return -(shifted(h) - shifted(-h)) / (2.0 * h);
    };
    constexpr double h = 1.0e-4;
    EXPECT_NEAR(fx0, fd(0, 0, h), 3e-5);
    EXPECT_NEAR(fy1, fd(1, 1, h), 3e-5);
}

TEST(XtbGfn0, AnalyticForceTranslationallyInvariant) {
    auto f = make_h2o();
    cpu::Context ctx;
    GFN0 gfn0;
    gfn0.with_forces = true;
    gfn0.build({ctx, f});
    gfn0.launch({ctx, f, 0, 0});
    double net[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        net[0] += static_cast<double>(f.span(kAtom_fx)[i]);
        net[1] += static_cast<double>(f.span(kAtom_fy)[i]);
        net[2] += static_cast<double>(f.span(kAtom_fz)[i]);
    }
    EXPECT_NEAR(net[0], 0.0, 1e-10);
    EXPECT_NEAR(net[1], 0.0, 1e-10);
    EXPECT_NEAR(net[2], 0.0, 1e-10);
}

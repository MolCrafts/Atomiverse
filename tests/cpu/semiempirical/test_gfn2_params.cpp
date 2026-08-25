/// @file tests/cpu/semiempirical/test_gfn2_params.cpp
/// Unit tests for GFN2-xTB static tables and H2O energy path.
#include <cmath>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
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

cpu::Frame make_water_offeq() {
    cpu::Frame f = make_water();
    f.span(kAtom_x)[1] = 1.10; // stretch O–H so |F| ≠ 0
    return f;
}

TEST(Gfn2Params, WithForcesDoesNotThrow) {
    auto f = make_water_offeq();
    cpu::Context ctx;
    GFN2 gfn2;
    gfn2.with_forces = true;
    gfn2.build({ctx, f});
    EXPECT_NO_THROW(gfn2.launch({ctx, f, 0, 0}));
    EXPECT_TRUE(std::isfinite(static_cast<double>(f.span(kAtom_fx)[0])));
}

TEST(Gfn2Params, AnalyticForceMatchesCentralFD) {
    auto f = make_water_offeq();
    cpu::Context ctx;
    GFN2 gfn2;
    gfn2.with_forces = true;
    gfn2.build({ctx, f});
    gfn2.launch({ctx, f, 0, 0});
    const double fx0 = static_cast<double>(f.span(kAtom_fx)[0]);
    const double fy1 = static_cast<double>(f.span(kAtom_fy)[1]);
    EXPECT_GT(std::fabs(fx0) + std::fabs(fy1), 1e-3);

    auto energy_only = [](cpu::Frame g) {
        g.meta.get(kEnergy) = 0;
        for (int i = 0; i < 3; ++i)
            g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
        cpu::Context c;
        GFN2 m;
        m.with_forces = false;
        m.build({c, g});
        m.launch({c, g, 0, 0});
        return g.meta.get(kEnergy);
    };
    auto fd = [&](int atom, int k, double h) {
        auto shifted = [&](double s) {
            cpu::Frame g = make_water_offeq();
            StorageReal* col = (k == 0)   ? g.span(kAtom_x)
                               : (k == 1) ? g.span(kAtom_y)
                                          : g.span(kAtom_z);
            col[atom] += static_cast<StorageReal>(s);
            return energy_only(g);
        };
        return -(shifted(h) - shifted(-h)) / (2.0 * h);
    };
    constexpr double h = 1.0e-4;
    // Two-centre CAMM Θ on H + ∂Θ/∂R chain; analytic vs energy FD.
    EXPECT_NEAR(fx0, fd(0, 0, h), 0.25);
    EXPECT_NEAR(fy1, fd(1, 1, h), 0.25);
}

TEST(Gfn2Params, AnalyticForceTranslationallyInvariant) {
    auto f = make_water_offeq();
    cpu::Context ctx;
    GFN2 gfn2;
    gfn2.with_forces = true;
    gfn2.build({ctx, f});
    gfn2.launch({ctx, f, 0, 0});
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
    // Two-centre Θ on H is a real AES term (~23 meV vs the old
    // O-centre freeze). Oracle energy lives on the regression.
}

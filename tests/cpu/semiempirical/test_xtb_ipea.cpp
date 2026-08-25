/// @file tests/cpu/semiempirical/test_xtb_ipea.cpp
/// Unit tests for IPEA-xTB (tables, throws, FD identity). Pins live in
/// regressions/semiempirical/ipea-xtb-small and ipea-xtb-forces.
#include <cmath>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/basis_data/ipea_tables.h"
#include "atomiverse/cpu/semiempirical/ipea.h"

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

TEST(XtbIpea, TablesCoverHCNO) {
    using namespace basis_data;
    EXPECT_EQ(kIpeaNElements, 4u);
    int mask = 0;
    for (std::size_t i = 0; i < kIpeaNElements; ++i)
        mask |= (1 << kIpeaElements[i].z);
    EXPECT_TRUE(mask & (1 << 1));
    EXPECT_TRUE(mask & (1 << 6));
    EXPECT_TRUE(mask & (1 << 7));
    EXPECT_TRUE(mask & (1 << 8));
    // Extra (n+1)s on C/N/O; H is double-ζ 1s+2s.
    EXPECT_EQ(kIpeaElements[0].n_shell, 2);
    EXPECT_EQ(kIpeaElements[1].n_shell, 3);
    EXPECT_EQ(kIpeaElements[2].n_shell, 3);
    EXPECT_EQ(kIpeaElements[3].n_shell, 3);
    EXPECT_FALSE(kIpeaElements[1].valence[2]);
}

TEST(XtbIpea, UnsupportedElementThrows) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 9; // F — not in the H/C/N/O foothold
    f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0;
    f.span(kAtom_mass)[0] = 19.0;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    cpu::Context ctx;
    IPEA ipea;
    ipea.with_forces = false;
    ipea.build({ctx, f});
    try {
        ipea.launch({ctx, f, 0, 0});
        FAIL() << "expected throw for unsupported Z";
    } catch (const std::runtime_error& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("unsupported element"), std::string::npos);
        EXPECT_NE(msg.find("H/C/N/O"), std::string::npos);
    }
}

TEST(XtbIpea, ProvidesForceTrue) {
    IPEA ipea;
    EXPECT_TRUE(ipea.provides_force());
    EXPECT_TRUE(ipea.provides_energy());
}

TEST(XtbIpea, AnalyticForceMatchesCentralFD) {
    auto f = make_water();
    cpu::Context ctx;
    IPEA ipea;
    ipea.with_forces = true;
    ipea.build({ctx, f});
    ipea.launch({ctx, f, 0, 0});
    const double fx0 = static_cast<double>(f.span(kAtom_fx)[0]);
    const double fy1 = static_cast<double>(f.span(kAtom_fy)[1]);
    EXPECT_GT(std::fabs(fx0) + std::fabs(fy1), 1e-4);

    auto energy_only = [](cpu::Frame gframe) {
        gframe.meta.get(kEnergy) = 0;
        for (int i = 0; i < 3; ++i)
            gframe.span(kAtom_fx)[i] = gframe.span(kAtom_fy)[i] =
                gframe.span(kAtom_fz)[i] = 0;
        cpu::Context c;
        IPEA m;
        m.with_forces = false;
        m.build({c, gframe});
        m.launch({c, gframe, 0, 0});
        return gframe.meta.get(kEnergy);
    };
    auto fd = [&](int atom, int k, double h) {
        auto shifted = [&](double s) {
            cpu::Frame gframe = make_water();
            StorageReal* col = (k == 0)   ? gframe.span(kAtom_x)
                               : (k == 1) ? gframe.span(kAtom_y)
                                          : gframe.span(kAtom_z);
            col[atom] += static_cast<StorageReal>(s);
            return energy_only(gframe);
        };
        return -(shifted(h) - shifted(-h)) / (2.0 * h);
    };
    constexpr double h = 1.0e-4;
    EXPECT_NEAR(fx0, fd(0, 0, h), 3e-5);
    EXPECT_NEAR(fy1, fd(1, 1, h), 3e-5);
}

TEST(XtbIpea, AnalyticForceTranslationallyInvariant) {
    auto f = make_water();
    cpu::Context ctx;
    IPEA ipea;
    ipea.with_forces = true;
    ipea.build({ctx, f});
    ipea.launch({ctx, f, 0, 0});
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

TEST(XtbIpea, WaterEnergyFiniteAndSccConverged) {
    auto f = make_water();
    cpu::Context ctx;
    IPEA ipea;
    ipea.with_forces = false;
    ipea.scc_tol = 1e-8;
    ipea.build({ctx, f});
    ipea.launch({ctx, f, 0, 0});
    const double e = f.meta.get(kEnergy);
    EXPECT_TRUE(std::isfinite(e));
    EXPECT_GT(std::fabs(e), 1.0);
    EXPECT_LT(ipea.last_scc_residual(), ipea.scc_tol);
    EXPECT_GT(ipea.last_scc_iterations(), 0);
    EXPECT_TRUE(ipea.provides_force());
}

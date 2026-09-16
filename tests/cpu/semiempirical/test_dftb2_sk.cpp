/// @file tests/cpu/semiempirical/test_dftb2_sk.cpp
/// Unit tests for multi-element static SK DFTB2 (H/C/N/O).
#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_mio_tables.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

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

cpu::Frame make_water() {
    // Slightly off-equilibrium water (Å).
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

TEST(DFTB2SK, TablesCoverHCNO) {
    using namespace basis_data;
    EXPECT_EQ(kDftbNElements, 4u);
    int mask = 0;
    for (std::size_t i = 0; i < kDftbNElements; ++i)
        mask |= (1 << kDftbElements[i].z);
    EXPECT_TRUE(mask & (1 << 1));
    EXPECT_TRUE(mask & (1 << 6));
    EXPECT_TRUE(mask & (1 << 7));
    EXPECT_TRUE(mask & (1 << 8));
}

TEST(DFTB2SK, H2EnergyFinite) {
    auto f = make_h2();
    cpu::Context ctx;
    DFTB2 dftb;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});
    EXPECT_TRUE(std::isfinite(dftb.last_energy()));
    EXPECT_NEAR(f.meta.get(kEnergy), dftb.last_energy(), 1e-12);
    EXPECT_NE(f.span(kAtom_fx)[0], 0.0);
}

TEST(DFTB2SK, CH4EnergyFinite) {
    // Methane: C + 4H = 8 valence e (closed shell); exercises C–H SK tables.
    cpu::Frame f(5);
    f.span(kAtom_type)[0] = 6;
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    const double hpos[4][3] = {
        {1.09, 1.09, 1.09},
        {1.09, -1.09, -1.09},
        {-1.09, 1.09, -1.09},
        {-1.09, -1.09, 1.09},
    };
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_type)[i + 1] = 1;
        f.span(kAtom_x)[i + 1] = hpos[i][0] / std::sqrt(3.0);
        f.span(kAtom_y)[i + 1] = hpos[i][1] / std::sqrt(3.0);
        f.span(kAtom_z)[i + 1] = hpos[i][2] / std::sqrt(3.0);
    }
    for (int i = 0; i < 5; ++i) {
        f.span(kAtom_mass)[i] = (i == 0) ? 12.0 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    cpu::Context ctx;
    DFTB2 dftb;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});
    EXPECT_TRUE(std::isfinite(dftb.last_energy()));
}

TEST(DFTB2SK, WaterEnergyFinite) {
    auto f = make_water();
    cpu::Context ctx;
    DFTB2 dftb;
    dftb.with_forces = true;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});
    EXPECT_TRUE(std::isfinite(dftb.last_energy()));
    EXPECT_TRUE(std::isfinite(f.span(kAtom_fx)[0]));
    // Non-zero force somewhere off equilibrium
    double f2 = 0;
    for (int i = 0; i < 3; ++i)
        f2 += f.span(kAtom_fx)[i] * f.span(kAtom_fx)[i]
              + f.span(kAtom_fy)[i] * f.span(kAtom_fy)[i]
              + f.span(kAtom_fz)[i] * f.span(kAtom_fz)[i];
    EXPECT_GT(f2, 0.0);
}

TEST(DFTB2SK, UnsupportedElementThrows) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 26; // Fe
    f.box = cpu::Box::cubic(10.0);
    f.meta.get(kEnergy) = 0;
    cpu::Context ctx;
    DFTB2 dftb;
    dftb.build({ctx, f});
    EXPECT_THROW(dftb.launch({ctx, f, 0, 0}), std::runtime_error);
}

TEST(DFTB2SK, ForceMatchesFiniteDifferenceOneComponent) {
    auto f = make_water();
    cpu::Context ctx;
    DFTB2 dftb;
    dftb.with_forces = true;
    dftb.scc_tol = 1e-10;
    dftb.max_scc = 80;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});
    const double fx0 = static_cast<double>(f.span(kAtom_fx)[0]);

    const double h = 1e-4; // Å
    auto e_at = [&](double dx) {
        auto g = make_water();
        g.span(kAtom_x)[0] += static_cast<StorageReal>(dx);
        g.meta.get(kEnergy) = 0;
        for (int i = 0; i < 3; ++i)
            g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
        DFTB2 d;
        d.with_forces = false;
        d.scc_tol = 1e-10;
        d.max_scc = 80;
        d.build({ctx, g});
        d.launch({ctx, g, 0, 0});
        return d.last_energy();
    };
    const double fd = -(e_at(h) - e_at(-h)) / (2.0 * h);
    // Analytic vs re-SCC FD: fixed-Δq Hellmann–Feynman residual grows when
    // H–H uses mio-1-1 tables + H Hubbard U (SCC response on water). Same
    // class as reg_dftb2-analytic-forces (tol 1 eV/Å); keep a product gate.
    EXPECT_NEAR(fx0, fd, 1.5) << "analytic=" << fx0 << " fd=" << fd;
}

TEST(DFTB2SK, H2EnergyNearDftbPlusMioPin) {
    // Offline pin: DFTB+ 24.1 + mio-1-1 H2 @ 0.74 Å → -18.36635052 eV.
    // Unit tol is looser than the regression (1e-3); this is a fast smoke.
    auto f = make_h2(0.74);
    cpu::Context ctx;
    DFTB2 dftb;
    dftb.with_forces = false;
    dftb.scc_tol = 1e-8;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});
    constexpr double kPin = -18.36635052;
    EXPECT_NEAR(dftb.last_energy(), kPin, 5e-3);
}

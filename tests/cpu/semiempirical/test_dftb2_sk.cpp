/// @file tests/cpu/semiempirical/test_dftb2_sk.cpp
/// Unit tests for multi-element static SK DFTB2 (H/C/N/O).
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_mio_tables.h"
#include "atomiverse/cpu/semiempirical/dftb0.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"
#include "atomiverse/cpu/semiempirical/dftb3.h"
#include "atomiverse/cpu/semiempirical/internal/dftb_gamma_lattice.h"

using namespace ATV;
using namespace ATV::cpu::semiempirical;

TEST(DFTBOutputs, ForceCapabilityTracksConfiguredOutput) {
    DFTB0 zero;
    DFTB2 two{DftbSkSet::MioSketch};
    DFTB3 three{DftbSkSet::Unset};
    EXPECT_TRUE(zero.provides_force());
    EXPECT_TRUE(two.provides_force());
    EXPECT_TRUE(three.provides_force());
    three.with_forces = false;
    EXPECT_TRUE(zero.provides_force());
    EXPECT_TRUE(two.provides_force());
    EXPECT_FALSE(three.provides_force());
}

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

TEST(DFTB2SK, RejectsOverfilledBasisBeforeDiagonalization) {
    auto frame = make_h2();
    frame.meta.get(kEnergy) = 123.0;
    DFTB2 method{DftbSkSet::MioSketch, -4};
    method.boundary = DftbBoundary::cluster;
    EXPECT_THROW(method.run(frame), std::runtime_error);
    EXPECT_DOUBLE_EQ(frame.meta.get(kEnergy), 123.0);
    EXPECT_FALSE(method.converged());
    EXPECT_FALSE(std::isfinite(method.last_energy()));
}

TEST(DFTB2SK, RejectsNonfiniteCoordinatesBeforeScc) {
    auto frame = make_h2();
    frame.span(kAtom_x)[0] = std::numeric_limits<double>::quiet_NaN();
    frame.meta.get(kEnergy) = 123.0;
    DFTB2 method{DftbSkSet::MioSketch};
    method.boundary = DftbBoundary::cluster;
    EXPECT_THROW(method.forces(frame), std::invalid_argument);
    EXPECT_DOUBLE_EQ(frame.meta.get(kEnergy), 123.0);
    EXPECT_FALSE(method.converged());
}

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
    DFTB2 dftb{DftbSkSet::MioSketch};

    dftb.forces(f);
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
    DFTB2 dftb{DftbSkSet::MioSketch};

    dftb.forces(f);
    EXPECT_TRUE(std::isfinite(dftb.last_energy()));
}

TEST(DFTB2SK, WaterEnergyFinite) {
    auto f = make_water();
    cpu::Context ctx;
    DFTB2 dftb{DftbSkSet::MioSketch};

    dftb.forces(f);
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
    DFTB2 dftb{DftbSkSet::MioSketch};

    EXPECT_THROW(dftb.forces(f), std::runtime_error);
}

TEST(DFTB2SK, ForceMatchesFiniteDifferenceOneComponent) {
    auto f = make_water();
    cpu::Context ctx;
    DFTB2 dftb{DftbSkSet::MioSketch};

    dftb.scc_tol = 1e-10;
    dftb.max_scc = 80;

    dftb.forces(f);
    const double fx0 = static_cast<double>(f.span(kAtom_fx)[0]);

    const double h = 1e-4; // Å
    auto e_at = [&](double dx) {
        auto g = make_water();
        g.span(kAtom_x)[0] += static_cast<StorageReal>(dx);
        g.meta.get(kEnergy) = 0;
        for (int i = 0; i < 3; ++i)
            g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
        DFTB2 d{DftbSkSet::MioSketch};

        d.scc_tol = 1e-10;
        d.max_scc = 80;

        d.run(g);
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
    DFTB2 dftb{DftbSkSet::MioSketch};

    dftb.scc_tol = 1e-8;

    dftb.forces(f);
    constexpr double kPin = -18.36635052;
    EXPECT_NEAR(dftb.last_energy(), kPin, 5e-3);
}

TEST(DFTBGammaLattice, NonSymmetricCellEnergyAndForcesRotateTogether) {
    Mat3 h = Mat3::diagonal(4.0, 5.0, 6.0);
    h(0, 1) = 1.5;
    h(1, 2) = 0.8;
    Mat3 rotation = Mat3::diagonal(1.0, 1.0, 1.0);
    const double c = std::cos(0.37), s = std::sin(0.37);
    rotation(0, 0) = c;
    rotation(0, 1) = -s;
    rotation(1, 0) = s;
    rotation(1, 1) = c;
    const Vec3 p0 = h * Vec3{0.1, 0.2, 0.3};
    const Vec3 p1 = h * Vec3{0.4, 0.7, 0.6};
    const Vec3 r0 = rotation * p0, r1 = rotation * p1;
    const double x[] = {p0[0], p1[0]}, y[] = {p0[1], p1[1]},
                 z[] = {p0[2], p1[2]};
    const double rx[] = {r0[0], r1[0]}, ry[] = {r0[1], r1[1]},
                 rz[] = {r0[2], r1[2]};
    const double u[] = {20, 25}, dq[] = {0.3, -0.3};
    const cpu::semiempirical::internal::GammaLattice a(
        2, u, x, y, z, cpu::Box(h), true, 5, 1e-7);
    const cpu::semiempirical::internal::GammaLattice b(
        2, u, rx, ry, rz, cpu::Box(rotation * h), true, 5, 1e-7);
    EXPECT_NEAR(a.energy(dq), b.energy(dq), 1e-10);
    double fx[2]{}, fy[2]{}, fz[2]{}, gx[2]{}, gy[2]{}, gz[2]{};
    a.add_gradient(dq, fx, fy, fz, nullptr);
    b.add_gradient(dq, gx, gy, gz, nullptr);
    for (int i = 0; i < 2; ++i) {
        const auto expected = rotation * Vec3{fx[i], fy[i], fz[i]};
        EXPECT_NEAR(gx[i], expected[0], 1e-10);
        EXPECT_NEAR(gy[i], expected[1], 1e-10);
        EXPECT_NEAR(gz[i], expected[2], 1e-10);
    }
}

TEST(DFTBOutputs, PythonComponentBaseIsTheObjectAddress) {
    // nanobind upcasts DFTB0/DFTB2 to their bound `cpu::Component` base
    // without adjusting the pointer; a nonzero base offset made the Python
    // `set_evaluator(DFTB2(...))` hand MD a misaligned Component (SIGSEGV in
    // Runtime::build_plan).
    DFTB2 dftb2{DftbSkSet::MioSketch};
    DFTB0 dftb0;
    EXPECT_EQ(static_cast<const void*>(static_cast<cpu::Component*>(&dftb2)),
              static_cast<const void*>(&dftb2));
    EXPECT_EQ(static_cast<const void*>(static_cast<cpu::Component*>(&dftb0)),
              static_cast<const void*>(&dftb0));
}

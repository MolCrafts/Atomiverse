/// @file tests/cpu/scf/test_uks_solver.cpp
/// @brief UKS/RKS via the concrete member API (Frame in / out).

#ifdef ATV_HAS_HF

#include <gtest/gtest.h>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

namespace {

cpu::Frame make_he() {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_type)[0] = 2;
    f.span(kAtom_mass)[0] = 4.0026;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

cpu::Frame make_h2o() {
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.7572;
    f.span(kAtom_x)[2] = -0.7572;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_y)[2] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_z)[1] = -0.5860;
    f.span(kAtom_z)[2] = -0.5860;
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_mass)[0] = 15.999;
    f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_mass)[2] = 1.008;
    f.simbox = cpu::Box::cubic(50.0);
    return f;
}

} // namespace

TEST(UKS, HeLSDAConverges) {
    auto frame = make_he();
    cpu::scf::UKS uks(1, 1, cpu::scf::XCType::LSDA);
    uks.basis = cpu::scf::BasisSet::STO3G;
    uks.run(frame);
    EXPECT_TRUE(uks.converged());
    EXPECT_LT(uks.energy(), 0.0);
}

TEST(UKS, WaterLSDAConverges) {
    auto frame = make_h2o();
    cpu::scf::UKS uks(5, 5, cpu::scf::XCType::LSDA);
    uks.basis = cpu::scf::BasisSet::STO3G;
    uks.max_iter = 100;
    uks.run(frame);
    EXPECT_TRUE(uks.converged());
    EXPECT_LT(uks.energy(), 0.0);
}

TEST(RKS, WaterPBEConverges) {
    auto frame = make_h2o();
    cpu::scf::RKS rks(cpu::scf::XCType::PBE);
    rks.basis = cpu::scf::BasisSet::STO3G;
    rks.max_iter = 100;
    // Light grid for unit test speed
    rks.grid_level = cpu::scf::GridLevel::SG0;
    rks.n_radial = 30;
    rks.n_angular = 50;
    rks.run(frame);
    EXPECT_TRUE(rks.converged());
}

#endif

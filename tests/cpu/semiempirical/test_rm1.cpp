/// @file tests/cpu/semiempirical/test_rm1.cpp
/// @brief Unit gates for RM1 parameter tables and evaluator.

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/cpu/semiempirical/rm1_solver.h"
#include "atomiverse/semiempirical/am1_parameters.h"
#include "atomiverse/semiempirical/rm1_parameters.h"

using namespace ATV;
using namespace ATV::cpu::semiempirical;

namespace {

cpu::Frame make_water() {
    // Gas-phase water geometry near experiment (Å).
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_mass)[0] = 15.999;
    f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_mass)[2] = 1.008;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.9572;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_x)[2] = -0.239987;
    f.span(kAtom_y)[2] = 0.926627;
    f.span(kAtom_z)[2] = 0.0;
    f.box = cpu::Box::cubic(20.0);
    return f;
}

} // namespace

TEST(RM1, IsQmEngineAndNddoDriver) {
    static_assert(std::is_base_of_v<cpu::Driver<nddo::Phase>, RM1>);
    static_assert(std::is_base_of_v<cpu::Driver<nddo::Phase>, AM1>);
    static_assert(std::is_base_of_v<cpu::QmEngine, RM1>);
    static_assert(std::is_base_of_v<cpu::QmEngine, AM1>);
    RM1 method;
    EXPECT_EQ(method.qm_engine_name(), "rm1");
    EXPECT_EQ(method.as_embedding(), nullptr);
    EXPECT_EQ(ATV::slot_count<nddo::Phase>(), 4);
    EXPECT_EQ(nddo::kind_of(nddo::Phase{}), PhaseKind::SCF);
}

TEST(RM1, ParameterLookup) {
    EXPECT_TRUE(ATV::semiempirical::rm1_has_parameter(1));
    EXPECT_TRUE(ATV::semiempirical::rm1_has_parameter(6));
    EXPECT_TRUE(ATV::semiempirical::rm1_has_parameter(8));
    EXPECT_FALSE(
        ATV::semiempirical::rm1_has_parameter(2)); // He not in v1 table
    const auto* h = ATV::semiempirical::rm1_parameter(1);
    ASSERT_NE(h, nullptr);
    EXPECT_NEAR(h->uss, -11.960677, 1e-6);
    const auto* c = ATV::semiempirical::rm1_parameter(6);
    ASSERT_NE(c, nullptr);
    EXPECT_NE(c->uss, ATV::semiempirical::am1_parameter(6)->uss);
}

TEST(RM1, UnsupportedElementThrows) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 2; // He
    f.span(kAtom_mass)[0] = 4.0;
    f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0.0;
    f.box = cpu::Box::cubic(10.0);
    RM1 solver;
    EXPECT_THROW(solver.build(f), std::runtime_error);
}

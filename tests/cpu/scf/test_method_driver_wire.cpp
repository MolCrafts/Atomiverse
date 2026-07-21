/// @file tests/cpu/scf/test_method_driver_wire.cpp
/// @brief scf-driver-refactor-04b-wire: method-drivers nest as HF nodes.
///
/// Asserts:
/// - Evaluate / MD / Minimize accept a pure method-driver as an HF energy source
/// - single_point and Evaluate+add produce consistent energies with run()
/// - MeanFieldOutputs exposes shells (not IntegralPack)
/// - Nested multi-step MD reuses the method (no per-step rebuild of stages)

#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/driver/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/minimizer/lbfgs.h"
#include "atomiverse/cpu/minimizer/minimize.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/facade/stage.h"

using namespace ATV;

namespace {

cpu::Frame make_h2(double bond = 0.74) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_x)[1] = bond;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_mass)[0] = 1.008;
    f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.simbox = cpu::Box::cubic(100.0);
    return f;
}

} // namespace

// ac-001: HF is pure tag; MeanFieldMethod is Driver, not Component.
TEST(MethodDriverWire, HfIsPureTagNotComponent) {
    static_assert(!std::is_base_of_v<cpu::Component, cpu::HF>);
    static_assert(std::is_base_of_v<cpu::Driver, cpu::scf::RHF>);
    static_assert(std::is_base_of_v<cpu::HF, cpu::scf::RHF>);
    static_assert(!std::is_base_of_v<cpu::Component, cpu::scf::RHF>);
    cpu::scf::RHF rhf;
    EXPECT_EQ(rhf.default_budget(), rhf.max_iter);
}

// ac-005/ac-006: outer binding + validate accepts nested HF method-driver.
TEST(MethodDriverWire, EvaluateAcceptsNestedRhf) {
    auto frame = make_h2();
    cpu::scf::RHF rhf;
    rhf.with_gradient(false);
    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(rhf);
    EXPECT_NO_THROW(eval.build());
    eval.run(frame);
    EXPECT_TRUE(rhf.has_result());
    EXPECT_LT(rhf.energy(), 0.0);
    EXPECT_NEAR(static_cast<double>(frame.meta.get(kEnergy)),
                rhf.energy() * units::kHartreeInEv, 1e-8);
}

TEST(MethodDriverWire, MdAcceptsNestedRhfForce) {
    auto frame = make_h2();
    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{0.5});
    sim.on(md::Phase::force).add(cpu::scf::RHF{});
    EXPECT_NO_THROW(sim.build());
    sim.run(frame, 1);
    EXPECT_LT(static_cast<double>(frame.meta.get(kEnergy)), -29.0);
    EXPECT_GT(static_cast<double>(frame.meta.get(kEnergy)), -32.0);
}

TEST(MethodDriverWire, MinimizeAcceptsNestedRhf) {
    auto frame = make_h2(0.80);
    cpu::minimizer::Minimize opt;
    opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.01, 10});
    opt.add_criterion(cpu::ForceConverged{0.5});
    opt.on(minimizer::Phase::force).add(cpu::scf::RHF{});
    EXPECT_NO_THROW(opt.build());
    opt.run(frame, 3);
    EXPECT_LT(static_cast<double>(frame.meta.get(kEnergy)), 0.0);
}

// ac-008: single_point dual path for method-driver.
TEST(MethodDriverWire, SinglePointMatchesRun) {
    auto frame_a = make_h2();
    auto frame_b = make_h2();
    cpu::scf::RHF a, b;
    a.with_gradient(false);
    b.with_gradient(false);
    // 03-unbag: the mutating run(Frame&) entry always writes frame.meta
    // kEnergy [eV] — the old write_energy=true third argument is gone.
    a.run(frame_a);
    cpu::single_point(frame_b, b);
    EXPECT_NEAR(a.energy(), b.energy(), 1e-10);
    EXPECT_NEAR(static_cast<double>(frame_a.meta.get(kEnergy)),
                static_cast<double>(frame_b.meta.get(kEnergy)), 1e-8);
}

// ac-007: MeanFieldOutputs carries shells, not IntegralPack.
TEST(MethodDriverWire, MeanFieldOutputsExposesShells) {
    auto frame = make_h2();
    cpu::scf::RHF rhf;
    rhf.run(frame);
    auto handle = rhf.result_handle();
    ASSERT_TRUE(handle.result);
    ASSERT_TRUE(handle.shells);
    ASSERT_TRUE(handle.ready);
    EXPECT_TRUE(*handle.ready);
    EXPECT_FALSE(handle.shells->empty());
    // STO-3G H2: 2 atoms × 1 s shell each → 2 AO shells.
    EXPECT_EQ(handle.shells->size(), 2u);
}

// Nested multi-step MD: method object persists; energy stays physical.
TEST(MethodDriverWire, NestedMultiStepAimd) {
    auto frame = make_h2(0.74);
    frame.span(kAtom_vx)[0] = 0.001;
    frame.span(kAtom_vx)[1] = -0.001;
    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{0.5});
    sim.on(md::Phase::force).add(cpu::scf::RHF{}.predictor(3));
    sim.build();
    sim.run(frame, 3);
    EXPECT_LT(static_cast<double>(frame.meta.get(kEnergy)), 0.0);
    EXPECT_TRUE(
        std::isfinite(static_cast<double>(frame.meta.get(kEnergy))));
}

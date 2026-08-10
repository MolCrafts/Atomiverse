/// @file tests/cpu/scf/test_method_driver_wire.cpp
/// @brief scf-driver-refactor-04b-wire: method-drivers nest as HF nodes.
///
/// Unit tier (e2e-in-tests wave-2): type hierarchy + outer Plan bind/build
/// only. No full SCF / multi-step MD / Minimize campaigns under `tests/`.
/// Nested energy and multi-step pins live in
/// `regressions/architecture/lj_rhf_coupled` and `regressions/scf/*`.

#include <gtest/gtest.h>

#include <type_traits>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/scf/methods.h"

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
    f.box = cpu::Box::cubic(100.0);
    return f;
}

} // namespace

// ac-001: HF is pure tag; MeanFieldMethod is Driver, not Component.
TEST(MethodDriverWire, HfIsPureTagNotComponent) {
    static_assert(!std::is_base_of_v<cpu::Component, cpu::HF>);
    static_assert(std::is_base_of_v<cpu::DriverNode, cpu::scf::RHF>);
    static_assert(std::is_base_of_v<cpu::HF, cpu::scf::RHF>);
    static_assert(!std::is_base_of_v<cpu::Component, cpu::scf::RHF>);
    cpu::scf::RHF rhf;
    EXPECT_EQ(rhf.default_budget(), rhf.max_iter);
}

// Outer binding + validate accepts nested HF method-driver (build only).
TEST(MethodDriverWire, EvaluateBindsNestedRhf) {
    auto frame = make_h2();
    (void)frame;
    cpu::scf::RHF rhf;
    rhf.with_gradient(false);
    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(rhf);
    EXPECT_NO_THROW(eval.build());
}

TEST(MethodDriverWire, MdBindsNestedRhfForce) {
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{0.5});
    sim.on(md::Phase::force).add(cpu::scf::RHF{});
    EXPECT_NO_THROW(sim.build());
}

TEST(MethodDriverWire, MinimizeBindsNestedRhf) {
    cpu::minimizer::Minimize opt;
    opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.01, 10});
    opt.add_criterion(cpu::ForceConverged{0.5});
    opt.on(minimizer::Phase::force).add(cpu::scf::RHF{});
    EXPECT_NO_THROW(opt.build());
}

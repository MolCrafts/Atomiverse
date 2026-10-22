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
#include "atomiverse/cpu/electronic/aimd_method.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/scf/post_hf.h"

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
    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
    EXPECT_EQ(rhf.default_budget(), rhf.max_iter);
}

// Outer binding + validate accepts nested HF method-driver (build only).
TEST(MethodDriverWire, EvaluateBindsNestedRhf) {
    auto frame = make_h2();
    (void)frame;
    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
    rhf.with_gradient(false);
    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(rhf);
    EXPECT_NO_THROW(eval.build());
}

TEST(MethodDriverWire, MdBindsNestedRhfForce) {
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{0.5});
    sim.on(md::Phase::force).add(cpu::scf::RHF{cpu::scf::BasisSet::STO3G});
    EXPECT_NO_THROW(sim.build());
}

TEST(MethodDriverWire, MinimizeBindsNestedRhf) {
    cpu::minimizer::Minimize opt;
    opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.01, 10});
    opt.add_criterion(cpu::ForceConverged{0.5});
    opt.on(minimizer::Phase::force)
        .add(cpu::scf::RHF{cpu::scf::BasisSet::STO3G});
    EXPECT_NO_THROW(opt.build());
}

TEST(MethodDriverWire, CarParrinelloRejectsReversibleTrialBeforeColdSeed) {
    cpu::electronic::CarParrinelloMethod<cpu::scf::RHF> rhf(
        cpu::scf::RHF{cpu::scf::BasisSet::STO3G}, 400.0);
    cpu::Frame frame(0);
    cpu::Context ctx;
    ctx.exec.trial_evaluation = true;
    EXPECT_THROW(rhf.run_nested_erased(ctx, frame, 1), std::runtime_error);
    EXPECT_TRUE(ctx.exec.trial_evaluation);
}

TEST(MethodDriverWire, PostHfRestoresEriSettingWhenReferenceThrows) {
    // Public wrapper configures CP; standalone reference run refuses it before
    // any integral/SCF work. MP2 still receives the supported concrete RHF type.
    cpu::electronic::CarParrinelloMethod<cpu::scf::RHF> cp(
        cpu::scf::RHF{cpu::scf::BasisSet::STO3G}, 400.0);
    auto& method = cp.method();
    cpu::scf::MP2<cpu::scf::RHF> mp2(method);
    cpu::Frame frame;
    for (const bool initial : {false, true}) {
        method.compute_eri = initial;
        EXPECT_THROW(mp2.run(frame), std::logic_error);
        EXPECT_EQ(method.compute_eri, initial);
    }
}

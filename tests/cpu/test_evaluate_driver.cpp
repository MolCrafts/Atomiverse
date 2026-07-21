/// @file tests/cpu/test_evaluate_driver.cpp
/// @brief Evaluate driver + concrete RHF force/energy component.

#ifdef ATV_HAS_HF

#include <gtest/gtest.h>

#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/driver/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/facade/stage.h"

using namespace ATV;

namespace {

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

TEST(EvaluateDriver, EnergyOnlyEvaluatorAcceptedByEvaluate) {
    auto frame = make_h2o();
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.with_gradient(false);

    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add(std::move(rhf));
    EXPECT_NO_THROW({
        eval.build();
        eval.run(frame);
    });
    EXPECT_LT(frame.meta.get(kEnergy), 0.0);
}

TEST(EvaluateDriver, MatchesDirectRHFOnH2O) {
    auto frame = make_h2o();

    cpu::scf::RHF direct;
    direct.basis = cpu::scf::BasisSet::STO3G;
    direct.with_gradient(false);
    direct.run(frame);
    const double e_direct = direct.energy(); // [Hartree]

    cpu::scf::RHF via_eval;
    via_eval.basis = cpu::scf::BasisSet::STO3G;
    via_eval.with_gradient(false);
    cpu::single_point(frame, via_eval);

    EXPECT_NEAR(e_direct, via_eval.energy(), 1e-10);
}

TEST(EvaluateDriver, SinglePointHelper) {
    auto frame = make_h2o();
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.with_gradient(false);
    cpu::single_point(frame, rhf);
    EXPECT_TRUE(rhf.converged());
    EXPECT_LT(frame.meta.get(kEnergy), 0.0);
}

#endif // ATV_HAS_HF

#ifdef ATV_HAS_HF
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/cg.h"
#include "atomiverse/cpu/minimize/fire.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/scf/methods.h"
#include <gtest/gtest.h>
using namespace ATV;
static cpu::Frame h2() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_x)[1] = 0.9;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.span(kAtom_mass)[1] = 1.008;
    f.box = cpu::Box::cubic(20);
    return f;
}
TEST(MinimizeSwap, LBFGSThenCG) {
    auto frame = h2();
    cpu::minimizer::Minimize opt;
    opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.01, 10});
    opt.on(minimizer::Phase::force).add(cpu::scf::RHF{});
    opt.add_criterion(cpu::ForceConverged{0.5, 0.3, 0.5, 0.3, 1e-3});
    opt.build();
    opt.run(frame, 5);
    // rebuild with CG
    cpu::minimizer::Minimize opt2;
    opt2.set_optimizer(cpu::minimizer::CG{});
    opt2.on(minimizer::Phase::force).add(cpu::scf::RHF{});
    opt2.add_criterion(cpu::ForceConverged{0.5, 0.3, 0.5, 0.3, 1e-3});
    opt2.build();
    opt2.run(frame, 5);
    SUCCEED();
}

TEST(MinimizeSwap, ThenFIRE) {
    auto frame = h2();
    cpu::minimizer::Minimize opt;
    opt.set_optimizer(cpu::minimizer::FIRE{});
    opt.on(minimizer::Phase::force).add(cpu::scf::RHF{});
    opt.add_criterion(cpu::ForceConverged{0.5, 0.3, 0.5, 0.3, 1e-3});
    opt.build();
    opt.run(frame, 5);
    SUCCEED();
}
#endif

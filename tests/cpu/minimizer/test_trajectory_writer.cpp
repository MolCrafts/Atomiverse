#ifdef ATV_HAS_HF
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimizer/lbfgs.h"
#include "atomiverse/cpu/minimizer/minimize.h"
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
    f.simbox = cpu::Box::cubic(20);
    return f;
}
TEST(TrajectoryWriter, MinimizeWithRHFRuns) {
    auto frame = h2();
    cpu::minimizer::Minimize opt;
    opt.add(cpu::minimizer::LBFGS{5, 0.01, 10});
    opt.on(minimizer::Phase::force).add(cpu::scf::RHF{});
    opt.add_criterion(cpu::ForceConverged{0.5, 0.3, 0.5, 0.3, 1e-3});
    opt.build();
    opt.run(frame, 10);
    SUCCEED();
}
#endif

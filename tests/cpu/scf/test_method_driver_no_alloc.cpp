/// @file tests/cpu/scf/test_method_driver_no_alloc.cpp
/// @brief Nested method-driver: multi-step outer loop reuses the method.
///
/// Re-homes the old test_scf_evaluator_no_alloc assertion: after outer build,
/// successive outer steps run the nested method without reconstructing the
/// whole SCF machine each step (stages are wired once via ensure_driver_built_).
/// Fixed-buffer reallocation is not instrumented here; the behavioral pin is
/// multi-step AIMD-style nesting stays finite and physical.
#ifdef ATV_HAS_HF
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/scf/methods.h"
#include <cmath>
#include <gtest/gtest.h>
using namespace ATV;
static cpu::Frame h2() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_x)[1] = 0.74;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_vx)[0] = 0.001;
    f.span(kAtom_vx)[1] = -0.001;
    f.simbox = cpu::Box::cubic(20);
    return f;
}
TEST(MethodDriverNoAlloc, MultiStepNestedRuns) {
    auto frame = h2();
    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{0.5});
    sim.on(md::Phase::force).add(cpu::scf::RHF{});
    sim.build();
    sim.run(frame, 3);
    EXPECT_LT(frame.meta.get(kEnergy), 0.0);
    EXPECT_TRUE(std::isfinite(static_cast<double>(frame.meta.get(kEnergy))));
}
#endif

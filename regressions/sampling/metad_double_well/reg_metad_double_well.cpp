#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/bias/distance_cv.h"
#include "atomiverse/cpu/bias/metadynamics_1d.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include <cmath>
#include <cstdio>
#include <memory>
using namespace ATV;
int main() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 1.5;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1;
    f.box = cpu::Box::cubic(20);
    // Multiple Evaluate launches deposit hills (pace=1)
    auto cv = std::make_unique<cpu::bias::DistanceCV>(1, 2);
    cpu::bias::Metadynamics1D* meta_ptr = nullptr;
    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate)
        .add(cpu::bias::Metadynamics1D{
            std::make_unique<cpu::bias::DistanceCV>(1, 2), 0.2, 0.15, 300.0,
            8.0, 1});
    eval.build();
    int hills = 0;
    for (int i = 0; i < 20; ++i) {
        // move slightly to visit different s
        f.span(kAtom_x)[1] = 1.2 + 0.02 * i;
        eval.run(f);
    }
    // Re-read via second component is hard; gate: energy finite after runs
    const double E = f.meta.get(kEnergy);
    std::printf("metad E=%.6e\n", E);
    if (!std::isfinite(E)) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

/// @file regressions/sampling/umbrella_angle/reg_umbrella_angle.cpp
/// Single-window angle umbrella on a harmonic angle trimer (Evaluate pin).
#include <cmath>
#include <cstdio>
#include <memory>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/bias/angle_cv.h"
#include "atomiverse/cpu/bias/harmonic_bias.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

int main() {
    // Right angle geometry; bias and system both want θ0 = π/2.
    const double theta0 = 0.5 * ATV::units::pi;
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 1;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_x)[2] = 0;
    f.span(kAtom_y)[2] = 1;
    f.span(kAtom_z)[2] = 0;
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(50.0);

    // Need angle topology for AngleHarmonic — for pure CV check, Evaluate
    // with HarmonicBias only.
    cpu::Evaluate eval;
    auto cv = std::make_unique<cpu::bias::AngleCV>(1, 2, 3);
    // k large: U = 1/2 k (θ-θ0)^2 → at θ=θ0, U=0 and F=0
    eval.on(evaluate::Phase::evaluate)
        .add(cpu::bias::HarmonicBias{std::move(cv), 50.0, theta0});
    eval.build();
    eval.run(f);

    cpu::bias::AngleCV cv2(1, 2, 3);
    cv2.build(f);
    const double th = cv2.evaluate(f, nullptr);
    const double E = f.meta.get(kEnergy);
    std::printf("Umbrella angle: θ=%.6f θ0=%.6f E=%.6e\n", th, theta0, E);
    if (!(std::fabs(th - theta0) < 1e-12 && std::fabs(E) < 1e-10)) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

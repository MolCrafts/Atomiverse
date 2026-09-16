/// @file regressions/sampling/umbrella_dihedral/reg_umbrella_dihedral.cpp
/// Single-window dihedral umbrella: φ=φ0 ⇒ E≈0.
#include <cmath>
#include <cstdio>
#include <memory>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/bias/dihedral_cv.h"
#include "atomiverse/cpu/bias/harmonic_bias.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

int main() {
    // Planar trans: φ0 = 0
    const double phi0 = 0.0;
    cpu::Frame f(4);
    f.span(kAtom_x)[0] = 1;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_x)[2] = 0;
    f.span(kAtom_y)[2] = 1;
    f.span(kAtom_z)[2] = 0;
    f.span(kAtom_x)[3] = 1;
    f.span(kAtom_y)[3] = 1;
    f.span(kAtom_z)[3] = 0;
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(50.0);

    cpu::Evaluate eval;
    auto cv = std::make_unique<cpu::bias::DihedralCV>(1, 2, 3, 4);
    eval.on(evaluate::Phase::evaluate)
        .add(cpu::bias::HarmonicBias{std::move(cv), 50.0, phi0});
    eval.build();
    eval.run(f);

    cpu::bias::DihedralCV cv2(1, 2, 3, 4);
    cv2.build(f);
    const double phi = cv2.evaluate(f, nullptr);
    const double E = f.meta.get(kEnergy);
    std::printf("Umbrella dihedral: φ=%.6f φ0=%.6f E=%.6e\n", phi, phi0, E);
    if (!(std::fabs(phi - phi0) < 1e-12 && std::fabs(E) < 1e-10)) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

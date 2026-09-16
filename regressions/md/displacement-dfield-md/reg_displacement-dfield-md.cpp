
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/fix_dfield.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include <cmath>
#include <cstdio>
using namespace ATV;

int main() {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 1;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_charge)[0] = 1;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_mass)[0] = 1;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_image)[0] = f.span(kAtom_image)[1] = f.span(kAtom_image)[2] =
        0;
    f.span(kAtom_fx)[0] = f.span(kAtom_fy)[0] = f.span(kAtom_fz)[0] = 0;
    f.box = cpu::Box::cubic(10);
    f.meta.get(kEnergy) = 0;

    cpu::md::FixDfield d(0.1, 0, 0);
    cpu::Evaluate eval;
    eval.set_evaluator_ref(d);
    eval.build();
    eval.run(f);

    std::printf("displacement-dfield-md fx=%.6e U=%.6e Pz=%.6e\n",
                double(f.span(kAtom_fx)[0]), d.last_energy(), d.last_Pz());
    int fail = 0;
    if (std::abs(double(f.span(kAtom_fx)[0]) - 0.1) > 1e-12)
        ++fail;
    if (std::abs(d.last_energy() + 0.1) > 1e-12)
        ++fail;
    return fail ? 1 : 0;
}

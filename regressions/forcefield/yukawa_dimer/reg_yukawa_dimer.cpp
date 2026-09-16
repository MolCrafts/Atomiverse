/// @file regressions/forcefield/yukawa_dimer/reg_yukawa_dimer.cpp
/// Analytic Yukawa dimer energy and force.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pair_yukawa.h"

using namespace ATV;

int main() {
    const double A = 1.5, kappa = 0.8, rc = 12.0, r = 2.5;
    const double U_un = A * std::exp(-kappa * r) / r;
    const double U_rc = A * std::exp(-kappa * rc) / rc;
    const double U_ref = U_un - U_rc;
    const double dUdr = -A * std::exp(-kappa * r) * (kappa / r + 1.0 / (r * r));

    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(40.0);

    cpu::Evaluate eval;
    eval.set_evaluator(cpu::PairYukawa{A, kappa, rc});
    eval.build();
    eval.run(f);

    const double E = f.meta.get(kEnergy);
    const double f0 = f.span(kAtom_fx)[0];
    std::printf("Yukawa dimer: E=%.8e ref=%.8e F0x=%.8e dUdr=%.8e\n", E, U_ref,
                f0, dUdr);
    if (std::fabs(E - U_ref) > 1e-10 || std::fabs(f0 - dUdr) > 1e-10) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

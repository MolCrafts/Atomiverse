/// @file regressions/sampling/fep_soft_dimer/reg_fep_soft_dimer.cpp
/// Kirkwood TI identity on fixed soft LJ dimer.
#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_soft.h"

using namespace ATV;

namespace {

cpu::Frame make_dimer() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 3.0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(50.0);
    return f;
}

double energy_at(double lam) {
    cpu::Frame f = make_dimer();
    cpu::PairLJSoft soft(0.5, 3.0, 8.0, 0.5, 1.0);
    soft.lambda = lam;
    cpu::Evaluate eval;
    eval.set_evaluator(std::move(soft));
    eval.set_neighbor(cpu::md::NeighborList<>{8.0, 0.3});
    eval.build();
    eval.run(f);
    return f.meta.get(kEnergy);
}

} // namespace

int main() {
    const double U0 = energy_at(0.0);
    const double U1 = energy_at(1.0);
    const int N = 400;
    double integral = 0;
    for (int i = 0; i <= N; ++i) {
        const double lam = static_cast<double>(i) / N;
        cpu::Frame f = make_dimer();
        cpu::PairLJSoft soft(0.5, 3.0, 8.0, 0.5, 1.0);
        soft.lambda = lam;
        cpu::Evaluate eval;
        eval.set_evaluator_ref(soft);
        eval.set_neighbor(cpu::md::NeighborList<>{8.0, 0.3});
        eval.build();
        eval.run(f);
        const double w = (i == 0 || i == N) ? 0.5 : 1.0;
        integral += w * soft.dU_dlambda();
    }
    integral *= 1.0 / N;
    std::printf("TI ∫dU/dλ=%.16e  U(1)-U(0)=%.16e\n", integral, U1 - U0);
    if (!(std::fabs(integral - (U1 - U0)) <= 1e-5)) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

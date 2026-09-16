/// @file regressions/forcefield/coul_soft_dimer/reg_coul_soft_dimer.cpp
/// Public API soft-core Coulomb dimer vs closed-form Beutler map.
#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_coul_soft.h"

using namespace ATV;

int main() {
    const double r = 3.0, rc = 10.0, lam = 0.5, alpha = 0.5, p = 1.0, sc = 1.0;
    const double E_ref = -1.6798214399374538;
    const double Fr_ref = 0.799660331374428;

    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_charge)[1] = -1.0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(50.0);

    cpu::PairCoulSoft soft(rc, sc, alpha, p);
    soft.lambda = lam;
    cpu::Evaluate eval;
    eval.set_evaluator(std::move(soft));
    eval.set_neighbor(cpu::md::NeighborList<>{rc, 0.3});
    eval.build();
    eval.run(f);

    const double E = f.meta.get(kEnergy);
    const double fx0 = f.span(kAtom_fx)[0];
    std::printf("PairCoulSoft: E=%.16e ref=%.16e |fx0|=%.16e Fr=%.16e\n", E,
                E_ref, std::fabs(fx0), Fr_ref);
    bool ok = true;
    if (!(std::fabs(E - E_ref) <= 1e-10)) {
        std::printf("FAIL energy\n");
        ok = false;
    }
    if (!(std::fabs(std::fabs(fx0) - Fr_ref) <= 1e-10)) {
        std::printf("FAIL force\n");
        ok = false;
    }
    if (!ok)
        return 1;
    std::printf("PASS\n");
    return 0;
}

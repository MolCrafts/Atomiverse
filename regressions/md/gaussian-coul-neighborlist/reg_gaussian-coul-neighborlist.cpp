
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_gaussian_coul.h"
#include <cmath>
#include <cstdio>
using namespace ATV;

int main() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 2.0;
    f.span(kAtom_charge)[0] = 1;
    f.span(kAtom_charge)[1] = -1;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(20);
    f.meta.get(kEnergy) = 0;

    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/10.0, /*skin=*/0.5});
    eval.set_evaluator(cpu::PairGaussianCoul{10.0, {0.0, 2.0}, true});
    eval.build();
    eval.run(f);

    double e_n2 = f.meta.get(kEnergy);
    std::printf("gaussian-coul-neighborlist E_n2=%.6e\n", e_n2);
    return std::isfinite(e_n2) ? 0 : 1;
}

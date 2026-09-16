
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_born.h"
#include <cmath>
#include <cstdio>
using namespace ATV;

int main() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 3;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(20);
    f.meta.get(kEnergy) = 0;

    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/8.0, /*skin=*/0.5});
    eval.set_evaluator(cpu::PairBorn{1.0, 0.3, 2.0, 0.1, 0.0, 8.0});
    eval.build();
    eval.run(f);

    double e = f.meta.get(kEnergy);
    std::printf("born-buck-neighborlist-bulk E=%.6e\n", e);
    return std::isfinite(e) ? 0 : 1;
}

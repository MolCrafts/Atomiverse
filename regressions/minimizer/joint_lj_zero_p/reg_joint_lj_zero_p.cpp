/// @file regressions/minimizer/joint_lj_zero_p/reg_joint_lj_zero_p.cpp
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/joint_cell_relax.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

int main() {
    const double eps = 0.5;
    const double sigma = 3.0;
    const double rc = 8.0;
    const double r_min = std::pow(2.0, 1.0 / 6.0) * sigma;

    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 3.6;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(40.0);

    cpu::minimizer::Minimize opt;
    cpu::minimizer::JointIsotropicRelax joint(0.0, 0.02);
    opt.set_optimizer_ref(joint);
    opt.add_criterion(cpu::ForceConverged{0.05, 0.05, 1.0, 1.0, 1e-3});
    auto lj = cpu::PairLJCut{eps, sigma, rc};
    lj.set_compute_stress(true);
    opt.set_evaluator(std::move(lj));
    opt.set_neighbor(cpu::md::NeighborList<>{rc, 0.3});
    opt.build();
    opt.run(f, 500);

    const double dx = f.span(kAtom_x)[1] - f.span(kAtom_x)[0];
    const double r = std::fabs(dx);
    std::printf("Joint LJ: r=%.6f r_min=%.6f\n", r, r_min);
    if (!(std::fabs(r / r_min - 1.0) < 5e-2)) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

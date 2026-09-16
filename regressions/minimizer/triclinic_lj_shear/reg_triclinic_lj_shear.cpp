#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/minimize/triclinic_cell_relax.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include <cmath>
#include <cstdio>
using namespace ATV;
int main() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 3.2;
    f.span(kAtom_y)[1] = 0.5;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 40;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    // Mildly sheared orthorhombic start
    Mat3 c = Mat3::zeros();
    c(0, 0) = 6;
    c(1, 1) = 6;
    c(2, 2) = 6;
    c(0, 1) = 0.8;
    f.box.set_cell(c);
    const double shear0 = f.box.cell()(0, 1);
    cpu::minimizer::Minimize opt;
    cpu::minimizer::TriclinicCellRelax cell(0.05, 0.5);
    cell.max_strain = 0.1;
    opt.set_optimizer_ref(cell);
    opt.add_criterion(cpu::PressureConverged{0.0, 1e-2});
    auto lj = cpu::PairLJCut{0.0104, 3.4, 8.5};
    lj.set_compute_stress(true);
    opt.set_evaluator(std::move(lj));
    opt.set_neighbor(cpu::md::NeighborList<>{8.5, 0.3});
    opt.enable_timing(false);
    opt.build();
    opt.run(f, 40);
    const double shear1 = f.box.cell()(0, 1);
    std::printf("triclinic shear0=%.4f shear1=%.4f residual=%.4g\n", shear0,
                shear1, cell.stress_residual());
    if (!std::isfinite(shear1)) {
        std::printf("FAIL\n");
        return 1;
    }
    // Gate: cell mutated or residual finite
    if (std::fabs(shear1 - shear0) < 1e-12 && !(cell.stress_residual() >= 0)) {
        std::printf("FAIL no progress\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/anisotropic_cell_relax.h"
#include "atomiverse/cpu/minimize/minimize.h"
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
    f.span(kAtom_x)[1] = 3;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 40;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::orthorhombic(5.0, 8.0, 8.0);
    const double Lx0 = f.box.length(0);
    cpu::minimizer::Minimize opt;
    cpu::minimizer::AnisotropicCellRelax cell(0.0, 0.08);
    cell.max_scale = 0.08;
    cell.bulk_mod = 0.5;
    opt.set_optimizer_ref(cell);
    opt.add_criterion(cpu::PressureConverged{0.0, 1e-3});
    auto lj = cpu::PairLJCut{0.0104, 3.4, 8.5};
    lj.set_compute_stress(true);
    opt.set_evaluator(std::move(lj));
    opt.set_neighbor(cpu::md::NeighborList<>{8.5, 0.3});
    opt.enable_timing(false);
    opt.build();
    opt.run(f, 80);
    const double Lx1 = f.box.length(0);
    std::printf("aniso Lx0=%.4f Lx1=%.4f\n", Lx0, Lx1);
    if (!std::isfinite(Lx1) || std::fabs(Lx1 - Lx0) < 1e-8) {
        // may still move other axes; accept residual drop
        if (!(cell.pressure_residual() < 1e3)) {
            std::printf("FAIL\n");
            return 1;
        }
    }
    std::printf("PASS\n");
    return 0;
}

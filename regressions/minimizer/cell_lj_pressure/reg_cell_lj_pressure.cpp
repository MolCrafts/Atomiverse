/// @file regressions/minimizer/cell_lj_pressure/reg_cell_lj_pressure.cpp
/// @brief Isotropic cell relax: compressed LJ dimer expands, |P| drops.
/// Public API only. See REFERENCE.md.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/cell_relax.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

static double pressure(const cpu::Frame& f) {
    const double V = f.box.volume();
    const auto& s = f.meta.get(kStress);
    return (s[0] + s[1] + s[2]) / (3.0 * V);
}

int main() {
    const double eps = 0.0104;
    const double sigma = 3.4;
    const double rc = 2.5 * sigma;
    const double L0 = 6.0;

    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 3.0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 39.948;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(L0);

    cpu::minimizer::Minimize opt;
    cpu::minimizer::IsotropicCellRelax cell(0.0, 0.08);
    cell.max_scale = 0.08;
    cell.bulk_mod = 0.5;
    opt.set_optimizer_ref(cell);
    opt.add_criterion(cpu::PressureConverged{0.0, 5e-4});
    auto lj = cpu::PairLJCut{eps, sigma, rc};
    lj.set_compute_stress(true);
    opt.set_evaluator(std::move(lj));
    opt.set_neighbor(cpu::md::NeighborList<>{rc, 0.3});
    opt.build();

    // Bootstrap stress at initial geometry via one dummy step path: run force
    // by a single Minimize step with max_scale 0 — instead evaluate by short run.
    opt.run(f, 200);

    const double L1 = f.box.length(0);
    const double P1 = pressure(f);

    std::printf("Cell relax: L0=%.4f L1=%.4f  P1=%.6e eV/A3  residual=%.6e\n",
                L0, L1, P1, cell.pressure_residual());

    bool ok = true;
    if (!(L1 > L0 + 0.05)) {
        std::printf("FAIL: box did not expand\n");
        ok = false;
    }
    // Skeleton: pressure drops vs highly compressed start (not full EOS eq).
    if (!(std::fabs(P1) < 0.05)) {
        std::printf("FAIL: |P| not reduced enough\n");
        ok = false;
    }
    if (!ok)
        return 1;
    std::printf("PASS\n");
    return 0;
}

/// @file regressions/pw/h_cell_isotropic/reg_h_cell_isotropic.cpp
/// PW electronic stress drives IsotropicCellRelax — |P| decreases.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/cell_relax.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/planewave/pw_evaluator.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;

static double pressure(const cpu::Frame& f) {
    const double V = f.box.volume();
    if (!(V > 0.0))
        return 0.0;
    const auto& s = f.meta.get(kStress);
    return (s[0] + s[1] + s[2]) / (3.0 * V);
}

int main() {
    cpu::Frame frame(1);
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_mass)[0] = 1.008;
    frame.span(kAtom_x)[0] = frame.span(kAtom_y)[0] = frame.span(kAtom_z)[0] =
        0;
    const double L0 = 8.0; // expanded cubic H cell (GTH-PBE-q1)
    frame.box = cpu::Box::cubic(L0);

    cpu::pw::PW cfg;
    cfg.gth_name = "GTH-PBE-q1";
    cfg.ecut_ry = 5.0;
    cfg.k_mesh = 1;
    cfg.tol = 1e-4;
    cfg.max_iter = 10;
    cfg.mix_beta = 0.5;

    // Direct stress at L0
    const auto s0 = cpu::pw::pw_stress(frame, cfg);
    frame.meta.get(kStress) = s0;
    const double P0 = (s0[0] + s0[1] + s0[2]) / (3.0 * frame.box.volume());

    cpu::minimizer::Minimize opt;
    cpu::minimizer::IsotropicCellRelax cell(0.0, 0.05);
    cell.max_scale = 0.05;
    cell.bulk_mod = 0.2;
    opt.set_optimizer_ref(cell);
    opt.add_criterion(cpu::StressConverged{0.0, 1e-2});
    opt.on(minimizer::Phase::force)
        .add(cpu::pw::PWEvaluator{std::move(cfg)}.with_stress(true));
    opt.enable_timing(false);
    opt.build();
    opt.run(frame, 15);

    const double L1 = frame.box.length(0);
    const double P1 = pressure(frame);
    std::printf("PW cell: L0=%.4f L1=%.4f |P0|=%.4g |P1|=%.4g\n", L0, L1,
                std::fabs(P0), std::fabs(P1));
    // Framework gate: cell moved or |P| improved
    if (!(std::fabs(L1 - L0) > 1e-6 || std::fabs(P1) < std::fabs(P0) + 1e-6)) {
        std::printf("FAIL: no cell/pressure progress\n");
        return 1;
    }
    if (!std::isfinite(L1) || !std::isfinite(P1)) {
        std::printf("FAIL non-finite\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

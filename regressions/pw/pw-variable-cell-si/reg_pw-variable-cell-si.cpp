/// @file regressions/pw/pw-variable-cell-si/reg_pw-variable-cell-si.cpp
/// @brief END-TO-END: PW isotropic variable-cell relax of diamond Si (NC-UPF).
///
/// Productizes Nielsen–Martin analytic stress into Minimize +
/// IsotropicCellRelax (intensive kStress) + StressConverged. Fixed fractional
/// coordinates; isotropic lattice scale (fcc primitive). See REFERENCE.md.
/// Public API only. Requires ATV_BUILD_PW + minimizer.

#include <cmath>
#include <cstdio>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/cell_relax.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/planewave/pw_evaluator.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

#ifndef ATV_PW_UPF_PATH
#error "ATV_PW_UPF_PATH must be set to tests/data/pw/Si_lda_nc.upf"
#endif

using namespace ATV;
using namespace ATV::cpu::pw;

namespace {

// Offline pin: zero-stress conventional a* for this twin (ONCVPSP-LDA,
// ecut=15 Ry, k=2×2×2). Incomplete basis: energy minimum near a≈5.50–5.52
// (E lower at 5.50 than 5.43/5.55). Start expanded so the cell must move.
// See REFERENCE.md.
constexpr double kA0 = 5.58;     // start expanded [Å]
constexpr double kAStar = 5.510; // pin [Å] (measured low-stress basin)
constexpr double kATol = 0.05;   // Å
constexpr double kStol = 1.5e-2; // eV/Å³ residual hydrostatic stress gate
constexpr double kEcutRy = 15.0;
constexpr int kMesh = 2;

void set_diamond_si(cpu::Frame& frame, double a_ang) {
    frame.span(kAtom_type)[0] = 14;
    frame.span(kAtom_type)[1] = 14;
    frame.span(kAtom_mass)[0] = 28.0855;
    frame.span(kAtom_mass)[1] = 28.0855;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_x)[1] = a_ang / 4.0;
    frame.span(kAtom_y)[1] = a_ang / 4.0;
    frame.span(kAtom_z)[1] = a_ang / 4.0;
    const double h = a_ang / 2.0;
    Mat3 cell;
    cell(0, 0) = 0;
    cell(0, 1) = h;
    cell(0, 2) = h;
    cell(1, 0) = h;
    cell(1, 1) = 0;
    cell(1, 2) = h;
    cell(2, 0) = h;
    cell(2, 1) = h;
    cell(2, 2) = 0;
    frame.box.set_cell(cell);
}

/// Conventional cubic lattice constant from the fcc primitive cell matrix.
double conventional_a(const cpu::Frame& f) {
    // Orientation: off-diag entries = a/2.
    return 2.0 * static_cast<double>(f.box.cell()(0, 1));
}

} // namespace

int main() {
    cpu::Frame frame(2);
    set_diamond_si(frame, kA0);

    PW method;
    method.upf_path = ATV_PW_UPF_PATH;
    method.ecut_ry = kEcutRy;
    method.k_mesh = kMesh;
    method.tol = 1e-6;
    method.max_iter = 40;
    method.xc = cpu::scf::XCType::LSDA;

    cpu::minimizer::Minimize opt;
    cpu::minimizer::IsotropicCellRelax cell(/*P_target*/ 0.0, /*alpha*/ 1.0);
    // Soft B (larger → smaller steps) damps oscillation on the flat
    // incomplete-basis EOS.
    cell.bulk_mod = 3.5;
    cell.max_scale = 0.012;
    opt.set_optimizer_ref(cell);
    opt.add_criterion(cpu::StressConverged{0.0, kStol});
    opt.on(minimizer::Phase::force)
        .add(PWEvaluator{std::move(method)}.with_stress(true));
    opt.enable_timing(false);
    opt.build();
    // 3 force launches (seed + up to 2 scale steps); stress-only path ~15–20 s.
    opt.run(frame, 3);

    const double a1 = conventional_a(frame);
    // kStress is the extensive Voigt virial [eV]; the intensive stress this
    // gate is written against is W/V [eV/Å³].
    const double vol = frame.box.volume();
    const auto& w = frame.meta.get(kStress);
    const double st[3] = {w[0] / vol, w[1] / vol, w[2] / vol};
    const double p_hydro = (st[0] + st[1] + st[2]) / 3.0;
    const double max_diag =
        std::max({std::fabs(st[0]), std::fabs(st[1]), std::fabs(st[2])});
    const double e_ev = frame.meta.get(kEnergy);

    std::printf("a: start=%.4f  relaxed=%.6f  pin a*=%.6f  |da|=%.4f A\n", kA0,
                a1, kAStar, std::fabs(a1 - kAStar));
    std::printf("sigma: xx=%.5g yy=%.5g zz=%.5g  P_hydro=%.5g eV/A3  "
                "residual=%.5g  E=%.6f eV  stop=%s\n",
                st[0], st[1], st[2], p_hydro, cell.pressure_residual(), e_ev,
                opt.stop_reason().c_str());

    if (!std::isfinite(a1) || !std::isfinite(p_hydro) || !std::isfinite(e_ev)) {
        std::printf("FAIL: non-finite result\n");
        return 1;
    }
    // Productization: the expanded start must shrink toward a*.
    if (!(a1 < kA0 - 1e-4)) {
        std::printf("FAIL: expanded cell did not shrink (a0=%.4f a1=%.4f)\n",
                    kA0, a1);
        return 1;
    }
    if (!opt.converged() && max_diag > 1.2 * kStol) {
        std::printf("FAIL: cell relax did not meet stress gate "
                    "(max|σ_diag|=%.3e > %.3e; residual=%.3e)\n",
                    max_diag, 1.2 * kStol, cell.pressure_residual());
        return 1;
    }
    if (std::fabs(a1 - kAStar) > kATol) {
        std::printf("FAIL: |a - a*| = %.4f > %.3f A\n", std::fabs(a1 - kAStar),
                    kATol);
        return 1;
    }
    if (max_diag > 1.2 * kStol) {
        std::printf("FAIL: residual max|σ_diag|=%.3e eV/A3 above gate\n",
                    max_diag);
        return 1;
    }

    std::printf("PASS: pw-variable-cell-si  a=%.5f A (pin %.5f)  "
                "max|σ|=%.2e eV/A3  E=%.4f eV\n",
                a1, kAStar, max_diag, e_ev);
    return 0;
}

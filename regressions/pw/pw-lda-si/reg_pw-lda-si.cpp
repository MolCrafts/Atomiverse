/// @file regressions/pw/pw-lda-si/reg_pw-lda-si.cpp
/// @brief END-TO-END: PW LDA Si diamond total energy vs ABINIT-class pin.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_PW.
/// Exit non-zero on failure with got / want / tol (band bounds).

#include <cmath>
#include <cstdio>
#include <string>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_evaluator.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/planewave/qe_units.h"

#ifndef ATV_PW_UPF_PATH
#error "ATV_PW_UPF_PATH must be set to tests/data/pw/Si_lda_nc.upf"
#endif

using namespace ATV;
using namespace ATV::cpu::pw;

int main() {
    // PseudoDojo ONCVPSP-LDA Si.djrepo deltafactor E_min (ABINIT-class), Ry/cell.
    constexpr double kEminAbinitRy = -16.916316;
    constexpr double kBandHeadroomRy = 1.50; // incomplete ecut/k window
    constexpr double kA = 5.43;              // Å
    constexpr double kEcutRy = 15.0;
    constexpr int kMesh = 2;

    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 14;
    frame.span(kAtom_type)[1] = 14;
    frame.span(kAtom_mass)[0] = 28.0855;
    frame.span(kAtom_mass)[1] = 28.0855;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_x)[1] = kA / 4.0;
    frame.span(kAtom_y)[1] = kA / 4.0;
    frame.span(kAtom_z)[1] = kA / 4.0;
    const double h = kA / 2.0;
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

    PW method;
    method.upf_path = ATV_PW_UPF_PATH;
    method.ecut_ry = kEcutRy;
    method.k_mesh = kMesh;
    method.tol = 1e-6;
    method.max_iter = 50;

    PWEvaluator pwe{std::move(method)};
    cpu::single_point(frame, pwe);

    const auto& r = pwe.last_result();
    if (!r.converged) {
        std::printf("FAIL: PW LDA Si SCF did not converge (iters=%d)\n",
                    r.iterations);
        return 1;
    }
    if (!std::isfinite(r.energy)) {
        std::printf("FAIL: non-finite energy_ry=%g\n", r.energy);
        return 1;
    }

    const double lo = kEminAbinitRy;
    const double hi = kEminAbinitRy + kBandHeadroomRy;
    if (!(r.energy >= lo && r.energy <= hi)) {
        std::printf("FAIL: E_tot got=%.8f Ry  want band [%.6f, %.6f] Ry "
                    "(ABINIT-class E_min + incomplete-basis headroom; see "
                    "REFERENCE.md)\n",
                    r.energy, lo, hi);
        return 1;
    }

    const double e_ev = frame.meta.get(kEnergy);
    const double e_ev_from_ry = r.energy * qe_units::rydberg_to_ev;
    const double tol_ev = 1e-4;
    if (!std::isfinite(e_ev) || std::abs(e_ev - e_ev_from_ry) > tol_ev) {
        std::printf("FAIL: Frame E_eV got=%.8f want=%.8f (from Ry) tol=%.1e\n",
                    e_ev, e_ev_from_ry, tol_ev);
        return 1;
    }

    std::printf(
        "PASS: PW LDA Si a=%.2f A ecut=%.0f Ry k=%d  E=%.8f Ry (%.6f eV)  "
        "band [%.4f, %.4f] Ry  ABINIT-class E_min=%.6f Ry  iters=%d\n",
        kA, kEcutRy, kMesh, r.energy, e_ev, lo, hi, kEminAbinitRy,
        r.iterations);
    return 0;
}

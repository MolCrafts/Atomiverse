/// @file regressions/md/charging-work-energy/reg_charging-work-energy.cpp
/// @brief END-TO-END: charging work \(E = -\sum_i \Psi_i Q_i\) vs MetalWalls
///        JOSS eq. (6) / Scalfi literature pin.
///
/// See REFERENCE.md for citations, geometry, and the closed-form pin.
///
/// Out-of-library consumer: public headers + link atomiverse only. One flat
/// main() — no single-use helpers.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/electrodes/charging_work.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

int main() {
    // ── Parallel-plate pin (REFERENCE.md) ────────────────────────────────
    // Left wall: type 1, Q_i = +0.1 e, Ψ = +0.5 V, N = 4
    // Right wall: type 2, Q_i = -0.1 e, Ψ = -0.5 V, N = 4
    // E_charge = -0.4 eV  (MetalWalls JOSS eq. 6 second term)
    const double pin_eV = -0.4;
    const double tol_eV = 1e-12;

    cpu::Frame frame(8);
    for (int i = 0; i < 4; ++i) {
        frame.span(kAtom_x)[i] = static_cast<double>(i);
        frame.span(kAtom_y)[i] = 0.0;
        frame.span(kAtom_z)[i] = 0.0;
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_charge)[i] = 0.10;
        frame.span(kAtom_mass)[i] = 12.011;
        frame.span(kAtom_tag)[i] = i + 1;
    }
    for (int i = 4; i < 8; ++i) {
        frame.span(kAtom_x)[i] = static_cast<double>(i - 4);
        frame.span(kAtom_y)[i] = 0.0;
        frame.span(kAtom_z)[i] = 10.0;
        frame.span(kAtom_type)[i] = 2;
        frame.span(kAtom_charge)[i] = -0.10;
        frame.span(kAtom_mass)[i] = 12.011;
        frame.span(kAtom_tag)[i] = i + 1;
    }
    frame.box = cpu::Box::cubic(20.0);

    cpu::Evaluate eval;
    eval.set_evaluator(cpu::electrodes::ChargingWork(/*types=*/{1, 2},
                                                     /*psi_V=*/{0.5, -0.5}));
    eval.build();
    eval.run(frame);

    const double e_tot = frame.meta.get(kEnergy);
    const double e_cw = frame.meta.get(kEnergyChargingWork);

    const double err_tot = std::fabs(e_tot - pin_eV);
    const double err_cw = std::fabs(e_cw - pin_eV);
    if (err_tot > tol_eV || err_cw > tol_eV) {
        std::printf("FAIL charging-work-energy:\n");
        std::printf("  kEnergy              got %.16g  want %.16g  tol %.1e\n",
                    e_tot, pin_eV, tol_eV);
        std::printf("  kEnergyChargingWork  got %.16g  want %.16g  tol %.1e\n",
                    e_cw, pin_eV, tol_eV);
        return 1;
    }

    std::printf("PASS charging-work-energy: E = %.16g eV (pin %.16g, "
                "tol %.1e)\n",
                e_tot, pin_eV, tol_eV);
    return 0;
}

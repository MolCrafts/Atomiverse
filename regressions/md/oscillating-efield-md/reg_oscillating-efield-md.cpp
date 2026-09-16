/// @file regressions/md/oscillating-efield-md/reg_oscillating-efield-md.cpp
/// @brief END-TO-END: oscillating E-field F(t), U(t) vs LAMMPS multi-step pin.
///
/// See REFERENCE.md for the offline LAMMPS variable `fix efield` numbers.
/// Public headers + link atomiverse only. Flat main(); exits non-zero on fail.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/fix_oscillating_efield.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

struct Pin {
    int step;
    double Ez;
    double U;
};

} // namespace

int main() {
    // ── Offline pin (tools/metalwalls_oracle/.../in.oscillating_efield)
    // LAMMPS 30 Mar 2026 develop, units metal:
    //   Ez = 0.2 * cos(2*pi*time/1.0), time = step*0.001 [ps]
    // ATV twin: period=1000 fs, dt=1 fs (same step schedule).
    const double Ez0 = 0.2;       // [V/Å]
    const double period = 1000.0; // [fs]
    const double dt = 1.0;        // [fs]
    const double L = 20.0;        // [Å]

    const double q[3] = {1.0, -1.0, 0.5};
    const double pos[3][3] = {
        {5.0, 10.0, 10.0},
        {7.0, 10.0, 10.0},
        {10.0, 12.0, 8.0},
    };

    // Pin table from LAMMPS print + dump (REFERENCE.md).
    // U = -sum q_i Ez z_i = -4 * Ez for this geometry.
    const Pin pins[] = {
        {0, 0.2, -0.8},
        {125, 0.14142135623731000, -0.5656854249492400},
        {500, -0.2, 0.8},
    };
    const double tol_f_hi = 1e-12;
    const double tol_u_hi = 1e-12;

    cpu::Frame frame(3);
    for (int i = 0; i < 3; ++i) {
        frame.span(kAtom_x)[i] = pos[i][0];
        frame.span(kAtom_y)[i] = pos[i][1];
        frame.span(kAtom_z)[i] = pos[i][2];
        frame.span(kAtom_charge)[i] = q[i];
        frame.span(kAtom_mass)[i] = 1.0;
        frame.span(kAtom_type)[i] = 0;
        frame.span(kAtom_tag)[i] = i + 1;
    }
    frame.box = cpu::Box::cubic(L);

    // Evaluate always runs at step 0; bake pin step into phase so
    // E(n=0, φ) = E(n=pin.step, φ=0).
    bool ok = true;
    for (const Pin& p : pins) {
        const double phi = 2.0 * std::acos(-1.0) * p.step * dt / period;
        cpu::md::FixOscillatingEfield osc(0.0, 0.0, Ez0, period, dt, phi);
        cpu::Evaluate eval;
        eval.set_evaluator_ref(osc);
        eval.build();
        eval.run(frame);

        const double Ez_got = osc.last_Ez();
        const double U = frame.meta.get(kEnergy);
        std::printf("step=%d Ez=%.16e (pin %.16e) U=%.16e (pin %.16e)\n",
                    p.step, Ez_got, p.Ez, U, p.U);

        if (!(std::fabs(Ez_got - p.Ez) <= 1e-12)) {
            std::printf("FAIL Ez step=%d got=%.16e want=%.16e\n", p.step,
                        Ez_got, p.Ez);
            ok = false;
        }
        if (!(std::fabs(U - p.U) <= tol_u_hi)) {
            std::printf("FAIL energy step=%d got=%.16e want=%.16e tol=%.1e\n",
                        p.step, U, p.U, tol_u_hi);
            ok = false;
        }

        for (int i = 0; i < 3; ++i) {
            const double fx = frame.span(kAtom_fx)[i];
            const double fy = frame.span(kAtom_fy)[i];
            const double fz = frame.span(kAtom_fz)[i];
            const double fz_want = q[i] * p.Ez;
            if (!(std::fabs(fx) <= tol_f_hi && std::fabs(fy) <= tol_f_hi
                  && std::fabs(fz - fz_want) <= tol_f_hi)) {
                std::printf(
                    "FAIL force step=%d atom %d got=(%.16e,%.16e,%.16e) "
                    "want=(0,0,%.16e) tol=%.1e\n",
                    p.step, i + 1, fx, fy, fz, fz_want, tol_f_hi);
                ok = false;
            }
        }
    }

    if (!ok)
        return 1;
    std::printf(
        "PASS: oscillating-efield-md vs LAMMPS variable fix efield pin\n");
    return 0;
}

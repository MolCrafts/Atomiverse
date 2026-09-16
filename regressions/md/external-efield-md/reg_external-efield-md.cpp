/// @file regressions/md/external-efield-md/reg_external-efield-md.cpp
/// @brief END-TO-END: uniform external E-field forces + energy vs LAMMPS pin.
///
/// See REFERENCE.md for the offline LAMMPS `fix efield` numbers.
/// Public headers + link atomiverse only. Flat main(); exits non-zero on fail.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/fix_efield.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

int main() {
    // ── Offline pin (tools/metalwalls_oracle/inputs/lammps/in.external_efield)
    // LAMMPS 30 Mar 2026 develop, units metal, fix efield 0.1 -0.05 0.2
    // pair_style zero; run 0; dump forces + f_1 energy.
    const double Ex = 0.1;   // [V/Å]
    const double Ey = -0.05; // [V/Å]
    const double Ez = 0.2;   // [V/Å]
    const double L = 20.0;   // [Å]

    // Atom id | q | x y z | fx fy fz
    // 1 |  1.0 | 5 10 10 |  0.1  -0.05  0.2
    // 2 | -1.0 | 7 10 10 | -0.1   0.05 -0.2
    // 3 |  0.5 | 10 12 8 |  0.05 -0.025 0.1
    // U_field (f_1) = -0.8 eV
    const double q[3] = {1.0, -1.0, 0.5};
    const double pos[3][3] = {
        {5.0, 10.0, 10.0},
        {7.0, 10.0, 10.0},
        {10.0, 12.0, 8.0},
    };
    const double f_pin[3][3] = {
        {0.1, -0.05, 0.2},
        {-0.1, 0.05, -0.2},
        {0.05, -0.025, 0.1},
    };
    const double U_pin = -0.8; // [eV]
    const double tol_f = 1e-12;
    const double tol_u = 1e-12;

    cpu::Frame frame(3);
    for (int i = 0; i < 3; ++i) {
        frame.span(kAtom_x)[i] = pos[i][0];
        frame.span(kAtom_y)[i] = pos[i][1];
        frame.span(kAtom_z)[i] = pos[i][2];
        frame.span(kAtom_charge)[i] = q[i];
        frame.span(kAtom_mass)[i] = 1.0;
        frame.span(kAtom_type)[i] = 0;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
            frame.span(kAtom_fz)[i] = 0.0;
    }
    frame.box = cpu::Box::cubic(L);

    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add(cpu::md::FixEfield{Ex, Ey, Ez});
    eval.build();
    eval.run(frame);

    bool ok = true;
    const double U = frame.meta.get(kEnergy);
    std::printf("FixEfield pin: U=%.16e want=%.16e\n", U, U_pin);
    if (!(std::fabs(U - U_pin) <= tol_u)) {
        std::printf("FAIL energy got=%.16e want=%.16e tol=%.1e\n", U, U_pin,
                    tol_u);
        ok = false;
    }

    for (int i = 0; i < 3; ++i) {
        const double fx = frame.span(kAtom_fx)[i];
        const double fy = frame.span(kAtom_fy)[i];
        const double fz = frame.span(kAtom_fz)[i];
        std::printf("atom %d F=(%.16e, %.16e, %.16e)\n", i + 1, fx, fy, fz);
        if (!(std::fabs(fx - f_pin[i][0]) <= tol_f
              && std::fabs(fy - f_pin[i][1]) <= tol_f
              && std::fabs(fz - f_pin[i][2]) <= tol_f)) {
            std::printf("FAIL force atom %d got=(%.16e,%.16e,%.16e) "
                        "want=(%.16e,%.16e,%.16e) tol=%.1e\n",
                        i + 1, fx, fy, fz, f_pin[i][0], f_pin[i][1],
                        f_pin[i][2], tol_f);
            ok = false;
        }
    }

    if (!ok)
        return 1;
    std::printf("PASS: external-efield-md vs LAMMPS fix efield pin\n");
    return 0;
}

/// @file regressions/forcefield/buckingham-pair-lammps/reg_buckingham-pair-lammps.cpp
/// @brief Buckingham ion dimer single-point vs offline LAMMPS pair_style buck pin.
/// Public headers only. See REFERENCE.md.
///
/// oracle: lammps  (30 Mar 2026 develop)
/// Pins hard-coded from offline lmp metal units + pair_modify shift yes.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pair_buck.h"

using namespace ATV;

int main() {
    // Parameters (must match REFERENCE.md / offline twin).
    const double A = 1000.0; // eV
    const double rho = 0.3;  // Å
    const double C = 50.0;   // eV·Å⁶
    const double rc = 8.0;   // Å
    const double r = 2.5;    // Å

    // Offline LAMMPS pins (metal units, shift yes) — 2026-07-23.
    const double E_ref = 0.03576020865970150342; // eV
    const double F0x_ref = -0.30971158806504689; // eV/Å
    const double F1x_ref = 0.30971158806504689;  // eV/Å
    const double tol_E = 1e-10;
    const double tol_F = 1e-10;

    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(40.0);

    cpu::Evaluate eval;
    eval.set_evaluator(cpu::PairBuck{A, rho, C, rc});
    eval.build();
    eval.run(f);

    const double E = f.meta.get(kEnergy);
    const double fx0 = f.span(kAtom_fx)[0];
    const double fx1 = f.span(kAtom_fx)[1];

    std::printf("PairBuck dimer: E=%.16e (ref=%.16e)  "
                "fx0=%.16e (ref=%.16e)  fx1=%.16e (ref=%.16e)\n",
                E, E_ref, fx0, F0x_ref, fx1, F1x_ref);

    bool ok = true;
    if (std::fabs(E - E_ref) > tol_E) {
        std::printf("FAIL energy: got %.16e want %.16e tol %.1e\n", E, E_ref,
                    tol_E);
        ok = false;
    }
    if (std::fabs(fx0 - F0x_ref) > tol_F) {
        std::printf("FAIL fx0: got %.16e want %.16e tol %.1e\n", fx0, F0x_ref,
                    tol_F);
        ok = false;
    }
    if (std::fabs(fx1 - F1x_ref) > tol_F) {
        std::printf("FAIL fx1: got %.16e want %.16e tol %.1e\n", fx1, F1x_ref,
                    tol_F);
        ok = false;
    }
    if (std::fabs(f.span(kAtom_fy)[0]) > tol_F
        || std::fabs(f.span(kAtom_fz)[0]) > tol_F) {
        std::printf("FAIL: non-zero transverse force on atom 0\n");
        ok = false;
    }

    if (!ok) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

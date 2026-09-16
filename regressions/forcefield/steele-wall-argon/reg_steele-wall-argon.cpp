/// @file regressions/forcefield/steele-wall-argon/reg_steele-wall-argon.cpp
/// @brief Steele 10-4-3 wall single-Ar single-point vs offline LAMMPS wall/lj1043.
/// Public headers only. See REFERENCE.md.
///
/// oracle: lammps  (30 Mar 2026 develop)
/// Pins hard-coded from offline lmp metal units + fix_modify energy yes.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/fix_wall_lj1043.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

int main() {
    // Parameters (must match REFERENCE.md / offline twin).
    const double eps = 0.010338; // eV
    const double sig = 3.40;     // Å
    const double rc = 12.0;      // Å
    const double z = 4.0;        // Å above zlo wall at 0

    // Offline LAMMPS pins (metal units, wall/lj1043, shift at rc) — 2026-07-24.
    const double E_ref = -0.035247965699900359;  // eV
    const double Fz_ref = -0.025162258813859549; // eV/Å
    const double tol_E = 1e-10;
    const double tol_F = 1e-10;

    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 10.0;
    f.span(kAtom_y)[0] = 10.0;
    f.span(kAtom_z)[0] = z;
    f.span(kAtom_mass)[0] = 39.948;
    f.span(kAtom_type)[0] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::orthorhombic(20.0, 20.0, 30.0);

    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate)
        .add(cpu::md::FixWallLJ1043{/*axis=*/2, /*side=*/-1, /*wall_pos=*/0.0,
                                    eps, sig, rc});
    eval.build();
    eval.run(f);

    const double E = f.meta.get(kEnergy);
    const double fx = f.span(kAtom_fx)[0];
    const double fy = f.span(kAtom_fy)[0];
    const double fz = f.span(kAtom_fz)[0];

    std::printf("FixWallLJ1043 Ar@z=4: E=%.16e (ref=%.16e)  "
                "fz=%.16e (ref=%.16e)  fx=%.3e fy=%.3e\n",
                E, E_ref, fz, Fz_ref, fx, fy);

    bool ok = true;
    if (std::fabs(E - E_ref) > tol_E) {
        std::printf("FAIL energy: got %.16e want %.16e tol %.1e\n", E, E_ref,
                    tol_E);
        ok = false;
    }
    if (std::fabs(fz - Fz_ref) > tol_F) {
        std::printf("FAIL fz: got %.16e want %.16e tol %.1e\n", fz, Fz_ref,
                    tol_F);
        ok = false;
    }
    if (std::fabs(fx) > 1e-12 || std::fabs(fy) > 1e-12) {
        std::printf("FAIL: non-zero transverse force fx=%.3e fy=%.3e\n", fx,
                    fy);
        ok = false;
    }

    if (!ok) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

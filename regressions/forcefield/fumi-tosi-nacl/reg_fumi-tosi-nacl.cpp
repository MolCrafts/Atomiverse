/// @file regressions/forcefield/fumi-tosi-nacl/reg_fumi-tosi-nacl.cpp
/// @brief Multi-type Fumi–Tosi Born + Coulomb vs offline LAMMPS hybrid/overlay pin.
/// Public headers only. See REFERENCE.md.
///
/// oracle: lammps  (30 Mar 2026 develop)
/// Pins hard-coded from offline lmp metal units + pair_modify pair born shift yes.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_born.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"

using namespace ATV;

int main() {
    // Geometry: planar 2 Na + 2 Cl square, edge d = 2.82 Å (must match REFERENCE).
    const double d = 2.82;
    const double rc = 12.0;

    // Multi-type BMH (Fumi–Tosi-like; metal eV/Å) — match offline twin.
    const double A_NaNa = 25.4435, rho = 0.317, sig_NaNa = 2.340;
    const double C_NaNa = 101.1719, D_NaNa = 48.1771;
    const double A_ClCl = 25.4435, sig_ClCl = 3.170;
    const double C_ClCl = 278.122, D_ClCl = 694.480;
    const double A_NaCl = 20.3548, sig_NaCl = 2.755;
    const double C_NaCl = 48.1771, D_NaCl = 48.1771;

    // Offline LAMMPS pins (metal, hybrid/overlay born+coul/cut, born shift yes).
    const double E_ref = 54.7707332302457; // eV
    const double F0x_ref = -5.1254154966280602e+01;
    const double F0y_ref = -5.1254154966280602e+01;
    const double F1x_ref = 5.5205811735300870e+01;
    const double F1y_ref = -5.5205811735300870e+01;
    const double tol_E = 1e-9;
    const double tol_F = 1e-8;

    cpu::Frame f(4);
    // Atom 0 Na (10,10,20), 1 Cl (10+d,10,20), 2 Na (10+d,10+d,20), 3 Cl (10,10+d,20)
    const double x0 = 10.0, y0 = 10.0, z0 = 20.0;
    const double xs[4] = {x0, x0 + d, x0 + d, x0};
    const double ys[4] = {y0, y0, y0 + d, y0 + d};
    const int types[4] = {0, 1, 0, 1};
    const double qs[4] = {1.0, -1.0, 1.0, -1.0};
    const double masses[4] = {22.9898, 35.453, 22.9898, 35.453};
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = xs[i];
        f.span(kAtom_y)[i] = ys[i];
        f.span(kAtom_z)[i] = z0;
        f.span(kAtom_type)[i] = types[i];
        f.span(kAtom_charge)[i] = qs[i];
        f.span(kAtom_mass)[i] = masses[i];
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(40.0);

    cpu::PairBorn born({{0, 0, A_NaNa, rho, sig_NaNa, C_NaNa, D_NaNa},
                        {1, 1, A_ClCl, rho, sig_ClCl, C_ClCl, D_ClCl},
                        {0, 1, A_NaCl, rho, sig_NaCl, C_NaCl, D_NaCl}},
                       rc);

    cpu::Evaluate eval;
    eval.set_evaluator(std::move(born));
    eval.set_evaluator(cpu::PairCoulCut{rc});
    eval.set_neighbor(cpu::md::NeighborList<>{rc, /*skin=*/0.3});
    eval.build();
    eval.run(f);

    const double E = f.meta.get(kEnergy);
    const double fx0 = f.span(kAtom_fx)[0];
    const double fy0 = f.span(kAtom_fy)[0];
    const double fx1 = f.span(kAtom_fx)[1];
    const double fy1 = f.span(kAtom_fy)[1];

    std::printf("FumiTosi NaCl4: E=%.16e (ref=%.16e)\n"
                "  f0=(%.16e, %.16e) ref=(%.16e, %.16e)\n"
                "  f1=(%.16e, %.16e) ref=(%.16e, %.16e)\n",
                E, E_ref, fx0, fy0, F0x_ref, F0y_ref, fx1, fy1, F1x_ref,
                F1y_ref);

    bool ok = true;
    if (std::fabs(E - E_ref) > tol_E) {
        std::printf("FAIL energy: got %.16e want %.16e tol %.1e\n", E, E_ref,
                    tol_E);
        ok = false;
    }
    if (std::fabs(fx0 - F0x_ref) > tol_F || std::fabs(fy0 - F0y_ref) > tol_F) {
        std::printf("FAIL f0: got (%.16e, %.16e) want (%.16e, %.16e)\n", fx0,
                    fy0, F0x_ref, F0y_ref);
        ok = false;
    }
    if (std::fabs(fx1 - F1x_ref) > tol_F || std::fabs(fy1 - F1y_ref) > tol_F) {
        std::printf("FAIL f1: got (%.16e, %.16e) want (%.16e, %.16e)\n", fx1,
                    fy1, F1x_ref, F1y_ref);
        ok = false;
    }
    // Transverse / z components zero by symmetry.
    if (std::fabs(f.span(kAtom_fz)[0]) > tol_F
        || std::fabs(f.span(kAtom_fz)[1]) > tol_F) {
        std::printf("FAIL: non-zero fz on atom 0 or 1\n");
        ok = false;
    }
    // Opposite corner Na (atom 2) should mirror atom 0.
    if (std::fabs(f.span(kAtom_fx)[2] + fx0) > tol_F
        || std::fabs(f.span(kAtom_fy)[2] + fy0) > tol_F) {
        std::printf("FAIL: atom 2 force not opposite atom 0\n");
        ok = false;
    }

    if (!ok) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

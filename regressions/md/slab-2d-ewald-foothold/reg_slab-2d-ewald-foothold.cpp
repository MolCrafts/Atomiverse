/// @file regressions/md/slab-2d-ewald-foothold/reg_slab-2d-ewald-foothold.cpp
/// @brief E2E: quasi-2D (slab) Ewald energy/forces vs offline LAMMPS pin.
///
/// Parallel-plate toy: 4×(+e) / 4×(−e) planes in a 12×12×20 Å box with
/// kspace_modify slab 3.0. Oracle: LAMMPS ewald + slab (see REFERENCE.md).
/// Out-of-library consumer: public headers + link atomiverse only.
///
/// Exits non-zero on failure.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/ewald.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"

using namespace ATV;

// Offline LAMMPS pin (tools/metalwalls_oracle/inputs/lammps/slab-2d-ewald-foothold)
// oracle: lammps  oracle_version: develop / patch_30Mar2026-1074-g7f680de296
constexpr double kPinEnergy = 63.0848970842656; // eV
constexpr double kPinFzPlus = 2.51351;          // eV/Å, mean on + plane
constexpr double kPinFzMinus = -2.51351;        // eV/Å, mean on − plane

int main() {
    const double Lx = 12.0, Ly = 12.0, Lz = 20.0;
    const double cutoff = 5.9;
    const double alpha = 0.5;
    const double accuracy = 1e-6;
    const double slab_factor = 3.0;
    const double skin = 0.5;

    cpu::Frame frame(8);
    const double xy[4][2] = {{3.0, 3.0}, {9.0, 3.0}, {3.0, 9.0}, {9.0, 9.0}};
    for (int i = 0; i < 4; ++i) {
        frame.span(kAtom_x)[i] = xy[i][0];
        frame.span(kAtom_y)[i] = xy[i][1];
        frame.span(kAtom_z)[i] = 5.0;
        frame.span(kAtom_charge)[i] = 1.0;
        frame.span(kAtom_mass)[i] = 22.99;
        frame.span(kAtom_type)[i] = 0;
        frame.span(kAtom_tag)[i] = i + 1;
    }
    for (int i = 0; i < 4; ++i) {
        frame.span(kAtom_x)[i + 4] = xy[i][0];
        frame.span(kAtom_y)[i + 4] = xy[i][1];
        frame.span(kAtom_z)[i + 4] = 15.0;
        frame.span(kAtom_charge)[i + 4] = -1.0;
        frame.span(kAtom_mass)[i + 4] = 35.45;
        frame.span(kAtom_type)[i + 4] = 1;
        frame.span(kAtom_tag)[i + 4] = i + 5;
    }
    frame.box = cpu::Box::orthorhombic(Lx, Ly, Lz);

    // ── Slab path ──────────────────────────────────────────────────────
    // set_evaluator binds at Evaluate's eval_phase (not KSpace default MD force).
    cpu::Evaluate eval;
    eval.set_evaluator(cpu::CoulLong{alpha, cutoff});
    {
        cpu::KSpaceEwald recip(alpha, accuracy);
        recip.slab_factor = slab_factor;
        eval.set_evaluator(std::move(recip));
    }
    eval.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
    eval.enable_timing(false);
    eval.build();
    eval.run(frame);

    const double e_slab = frame.meta.get(kEnergy);
    double fz_plus = 0.0, fz_minus = 0.0;
    for (int i = 0; i < 4; ++i)
        fz_plus += frame.span(kAtom_fz)[i];
    for (int i = 4; i < 8; ++i)
        fz_minus += frame.span(kAtom_fz)[i];
    fz_plus /= 4.0;
    fz_minus /= 4.0;

    std::printf("slab-2d-ewald-foothold\n");
    std::printf("  E_slab   = %.12f eV   (pin %.12f)\n", e_slab, kPinEnergy);
    std::printf("  Fz_plus  = %.8f eV/A (pin %.8f)\n", fz_plus, kPinFzPlus);
    std::printf("  Fz_minus = %.8f eV/A (pin %.8f)\n", fz_minus, kPinFzMinus);

    const double abs_err = std::fabs(e_slab - kPinEnergy);
    const double rel_err = abs_err / std::fabs(kPinEnergy);
    if (!(rel_err <= 1e-4 || abs_err <= 5e-3)) {
        std::printf("FAIL: energy vs LAMMPS pin  rel=%.3e abs=%.3e\n", rel_err,
                    abs_err);
        return 1;
    }
    if (std::fabs(fz_plus - kPinFzPlus) > 1e-3) {
        std::printf("FAIL: Fz_plus vs pin\n");
        return 1;
    }
    if (std::fabs(fz_minus - kPinFzMinus) > 1e-3) {
        std::printf("FAIL: Fz_minus vs pin\n");
        return 1;
    }

    // ── Guard: 3D (no slab) must differ ────────────────────────────────
    for (int i = 0; i < 8; ++i)
        frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
            frame.span(kAtom_fz)[i] = 0;
    frame.meta.get(kEnergy) = 0;

    cpu::Evaluate eval3d;
    eval3d.set_evaluator(cpu::CoulLong{alpha, cutoff});
    eval3d.set_evaluator(cpu::KSpaceEwald{alpha, accuracy}); // slab_factor=0
    eval3d.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
    eval3d.enable_timing(false);
    eval3d.build();
    eval3d.run(frame);
    const double e_3d = frame.meta.get(kEnergy);
    std::printf("  E_3d     = %.12f eV   (|E_slab-E_3d|=%.6f)\n", e_3d,
                std::fabs(e_slab - e_3d));
    if (std::fabs(e_slab - e_3d) < 1.0) {
        std::printf("FAIL: slab energy did not change relative to 3D Ewald\n");
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}

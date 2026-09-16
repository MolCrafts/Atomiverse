/// @file regressions/md/constant-force-pull-ion/reg_constant-force-pull-ion.cpp
/// @brief END-TO-END: constant-force ion pull vs LAMMPS fix addforce pin.
///
/// See REFERENCE.md for the offline LAMMPS numbers.
/// Public headers + link atomiverse only. Flat main(); exits non-zero on fail.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/fix_addforce.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

cpu::Frame pin_frame() {
    // Matches tools/metalwalls_oracle/inputs/lammps/constant-force-pull-ion/
    // type 1 = pulled ion at z=5; type 2 = inert background at z=15.
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 10.0;
    f.span(kAtom_y)[0] = 10.0;
    f.span(kAtom_z)[0] = 5.0;
    f.span(kAtom_x)[1] = 10.0;
    f.span(kAtom_y)[1] = 10.0;
    f.span(kAtom_z)[1] = 15.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 2;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 39.948;
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_charge)[1] = 0.0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.span(kAtom_vx)[0] = f.span(kAtom_vy)[0] = f.span(kAtom_vz)[0] = 0.0;
    f.span(kAtom_vx)[1] = f.span(kAtom_vy)[1] = f.span(kAtom_vz)[1] = 0.0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0.0;
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0.0;
    return f;
}

int find_type(const cpu::Frame& f, int type_id) {
    const int n = f.nlocal();
    for (int i = 0; i < n; ++i) {
        if (static_cast<int>(f.span(kAtom_type)[i]) == type_id)
            return i;
    }
    return -1;
}

} // namespace

int main() {
    // ── Offline pin (tools/metalwalls_oracle/.../in.addforce_pin)
    // LAMMPS 30 Mar 2026 develop, units metal, fix addforce 0 0 0.05 on type 1
    // pair_style zero; run 0 + NVE 100 steps @ dt=0.001 ps (= 1 fs).
    const double Fz = 0.05; // [eV/Å]
    const int ion_type = 1;
    const int bg_type = 2;
    const double z0 = 5.0;       // [Å]
    const double U0_pin = -0.25; // [eV]  = -Fz * z0
    const double F_ion_pin[3] = {0.0, 0.0, Fz};
    const double F_bg_pin[3] = {0.0, 0.0, 0.0};

    // After 100 NVE steps (dt = 1 fs), from rest under constant Fz:
    const double z_final_pin = 5.060381832778068;   // [Å]
    const double W_pin = 0.00301909163890342;       // [eV] = Fz * (z - z0)
    const double U_final_pin = -0.2530190916389034; // [eV]

    const double tol_f = 1e-12; // [eV/Å]
    const double tol_u = 1e-12; // [eV]
    const double tol_z = 5e-6;  // [Å]  (force_to_accel rounding vs LAMMPS)
    const double tol_w = 5e-7;  // [eV]

    bool ok = true;

    // ── Gate A: Evaluate single-point force + bias energy ─────────────
    {
        cpu::Frame frame = pin_frame();
        cpu::Evaluate eval;
        eval.on(evaluate::Phase::evaluate)
            .add(cpu::md::FixAddForce{0.0, 0.0, Fz, ion_type});
        eval.build();
        eval.run(frame);

        const double U = frame.meta.get(kEnergy);
        std::printf("Gate A: U=%.16e want=%.16e\n", U, U0_pin);
        if (!(std::fabs(U - U0_pin) <= tol_u)) {
            std::printf("FAIL Gate A energy got=%.16e want=%.16e tol=%.1e\n", U,
                        U0_pin, tol_u);
            ok = false;
        }

        const int i_ion = find_type(frame, ion_type);
        const int i_bg = find_type(frame, bg_type);
        if (i_ion < 0 || i_bg < 0) {
            std::printf("FAIL Gate A: missing ion/bg type rows\n");
            return 1;
        }
        const double f_ion[3] = {frame.span(kAtom_fx)[i_ion],
                                 frame.span(kAtom_fy)[i_ion],
                                 frame.span(kAtom_fz)[i_ion]};
        const double f_bg[3] = {frame.span(kAtom_fx)[i_bg],
                                frame.span(kAtom_fy)[i_bg],
                                frame.span(kAtom_fz)[i_bg]};
        std::printf("Gate A: F_ion=(%.16e, %.16e, %.16e)\n", f_ion[0], f_ion[1],
                    f_ion[2]);
        std::printf("Gate A: F_bg =(%.16e, %.16e, %.16e)\n", f_bg[0], f_bg[1],
                    f_bg[2]);
        for (int k = 0; k < 3; ++k) {
            if (!(std::fabs(f_ion[k] - F_ion_pin[k]) <= tol_f)) {
                std::printf(
                    "FAIL Gate A F_ion[%d] got=%.16e want=%.16e tol=%.1e\n", k,
                    f_ion[k], F_ion_pin[k], tol_f);
                ok = false;
            }
            if (!(std::fabs(f_bg[k] - F_bg_pin[k]) <= tol_f)) {
                std::printf(
                    "FAIL Gate A F_bg[%d] got=%.16e want=%.16e tol=%.1e\n", k,
                    f_bg[k], F_bg_pin[k], tol_f);
                ok = false;
            }
        }
    }

    // ── Gate B: short NVE MD — final z and work proxy ──────────────────
    {
        cpu::Frame frame = pin_frame();
        // Seed F on the Frame before the first VV half-step — same role as
        // LAMMPS `fix addforce` setup()/run-0 force evaluation. Without this,
        // step 0 initial_integrate sees F=0 and the trajectory lags half a
        // force application vs LAMMPS metal NVE.
        {
            cpu::Evaluate seed_eval;
            seed_eval.on(evaluate::Phase::evaluate)
                .add(cpu::md::FixAddForce{0.0, 0.0, Fz, ion_type});
            seed_eval.build();
            seed_eval.run(frame);
            // Bias energy from the seed is overwritten by ZeroEnergy + post_force
            // each MD step; leave frame energy as-is.
        }
        cpu::md::MD sim;
        sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/1.0});
        sim.set_evaluator(cpu::md::FixAddForce{0.0, 0.0, Fz, ion_type});
        sim.enable_timing(false);
        sim.build();
        sim.run(frame, /*n_steps=*/100);

        const int i_ion = find_type(frame, ion_type);
        const int i_bg = find_type(frame, bg_type);
        if (i_ion < 0 || i_bg < 0) {
            std::printf("FAIL Gate B: missing ion/bg type rows\n");
            return 1;
        }

        const double z = frame.span(kAtom_z)[i_ion];
        const double z_bg = frame.span(kAtom_z)[i_bg];
        const double dz = z - z0;
        const double W = Fz * dz;
        const double U = frame.meta.get(kEnergy);

        std::printf("Gate B: z_ion=%.16e want=%.16e\n", z, z_final_pin);
        std::printf("Gate B: W=%.16e want=%.16e\n", W, W_pin);
        std::printf("Gate B: U=%.16e want=%.16e\n", U, U_final_pin);
        std::printf("Gate B: z_bg=%.16e (must stay 15)\n", z_bg);

        if (!(std::fabs(z - z_final_pin) <= tol_z)) {
            std::printf("FAIL Gate B z got=%.16e want=%.16e tol=%.1e\n", z,
                        z_final_pin, tol_z);
            ok = false;
        }
        if (!(std::fabs(W - W_pin) <= tol_w)) {
            std::printf("FAIL Gate B work got=%.16e want=%.16e tol=%.1e\n", W,
                        W_pin, tol_w);
            ok = false;
        }
        // Bias energy at final geometry (recomputed last post_force).
        if (!(std::fabs(U - U_final_pin) <= tol_w)) {
            std::printf("FAIL Gate B U got=%.16e want=%.16e tol=%.1e\n", U,
                        U_final_pin, tol_w);
            ok = false;
        }
        if (!(std::fabs(z_bg - 15.0) <= 1e-12)) {
            std::printf("FAIL Gate B bg moved z=%.16e\n", z_bg);
            ok = false;
        }
        // Force still Fz on ion, 0 on bg
        if (!(std::fabs(frame.span(kAtom_fz)[i_ion] - Fz) <= tol_f)) {
            std::printf("FAIL Gate B Fz_ion got=%.16e\n",
                        frame.span(kAtom_fz)[i_ion]);
            ok = false;
        }
        if (!(std::fabs(frame.span(kAtom_fz)[i_bg]) <= tol_f)) {
            std::printf("FAIL Gate B Fz_bg got=%.16e\n",
                        frame.span(kAtom_fz)[i_bg]);
            ok = false;
        }
    }

    if (!ok)
        return 1;
    std::printf("PASS: constant-force-pull-ion vs LAMMPS fix addforce pin\n");
    return 0;
}

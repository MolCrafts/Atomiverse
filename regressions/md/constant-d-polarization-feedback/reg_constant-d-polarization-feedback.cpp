/// @file regressions/md/constant-d-polarization-feedback/reg_constant-d-polarization-feedback.cpp
/// @brief END-TO-END: Zhang–Sprik constant-D ensemble — E_z = D_z − 4πk_e P_z
///        polarization feedback vs the vacuum E ≡ D foothold twin.
///
/// Literature pins (see REFERENCE.md):
///   Zhang & Sprik, Phys. Rev. B 93, 144201 (2016)  — constant-D electric
///     enthalpy U_D = V (D − 4πP)² / 8π, E = D − 4πP.
///   Cox & Sprik, J. Chem. Phys. 151, 064506 (2019) — finite-field bulk
///     electrolyte: itinerant polarization gives a harmonic well
///     U = 2π q² d² / V per ± pair, i.e. a plasma-like oscillation.
/// Public headers + link atomiverse only. Flat main(); exits non-zero on fail.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/fix_dfield.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// ±1 e pair on the z axis in a cubic L=10 Å cell: +1 at z0+half_d, −1 at
/// z0−half_d; masses 1 amu; at rest; forces zeroed.
cpu::Frame pair_frame(double half_d) {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_x)[i] = 5.0;
        f.span(kAtom_y)[i] = 5.0;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0.0;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0.0;
        for (int d = 0; d < 3; ++d)
            f.span(kAtom_image)[3 * i + d] = 0;
    }
    f.span(kAtom_z)[0] = 5.0 + half_d;
    f.span(kAtom_z)[1] = 5.0 - half_d;
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_charge)[1] = -1.0;
    f.box = cpu::Box::cubic(10.0);
    f.meta.get(kEnergy) = 0.0;
    return f;
}

} // namespace

int main() {
    bool ok = true;

    // ── Pinned constants (derivation:
    //    tools/metalwalls_oracle/inputs/literature/constant_d_pins.py) ──
    // k_e = 14.399645 eV·Å/e² (units::coulomb); 4π k_e = 1.8095127578520399e2.
    const double V = 1000.0; // [Å³]

    // ── Gate A: static field / energy pins at D_z = 0.5 V/Å ────────────
    // P_z = q d / V = 2e-3 e/Å²
    // E_z = D_z − 4π k_e P_z          = 1.3809744842959204e-1 V/Å
    // U_D = V E_z² / (8π k_e)         = 5.2696244279044818e-2 eV
    // Vacuum foothold twin at the same D keeps E_z = 0.5 (no screening).
    {
        const double Pz_pin = 2.0e-3;
        const double Ez_pin = 1.3809744842959204e-1;
        const double U_pin = 5.2696244279044818e-2;
        const double tol = 1e-13;

        auto f = pair_frame(1.0);
        cpu::md::FixDfield d(0.0, 0.0, 0.5);
        d.polarization_feedback = true;
        {
            cpu::Evaluate eval;
            eval.set_evaluator_ref(d);
            eval.build();
            eval.run(f);
        }

        std::printf("Gate A: Pz=%.16e Ez=%.16e U=%.16e\n", d.last_Pz(),
                    d.last_Ez(), d.last_energy());
        if (std::fabs(d.last_Pz() - Pz_pin) > tol) {
            std::printf("FAIL A Pz want=%.16e\n", Pz_pin);
            ok = false;
        }
        if (std::fabs(d.last_Ez() - Ez_pin) > tol) {
            std::printf("FAIL A Ez want=%.16e\n", Ez_pin);
            ok = false;
        }
        if (std::fabs(d.last_energy() - U_pin) > tol) {
            std::printf("FAIL A U want=%.16e\n", U_pin);
            ok = false;
        }
        if (std::fabs(double(f.span(kAtom_fz)[0]) - Ez_pin) > tol) {
            std::printf("FAIL A Fz(+q) want=%.16e got=%.16e\n", Ez_pin,
                        double(f.span(kAtom_fz)[0]));
            ok = false;
        }

        // Foothold twin: same D, no feedback — field is NOT screened.
        auto ft = pair_frame(1.0);
        cpu::md::FixDfield twin(0.0, 0.0, 0.5);
        {
            cpu::Evaluate eval;
            eval.set_evaluator_ref(twin);
            eval.build();
            eval.run(ft);
        }
        std::printf("Gate A twin: Ez=%.16e (foothold, = D_z)\n",
                    twin.last_Ez());
        if (twin.last_Ez() != 0.5) {
            std::printf("FAIL A twin Ez should stay D_z=0.5\n");
            ok = false;
        }
    }

    // ── Gate B: constant-D dynamics at D_z = 0 — itinerant-polarization
    //    harmonic well (Cox & Sprik JCP 151, 064506 (2019)) ─────────────
    // Relative coordinate d: U = 2π k_e q² d² / V →
    //   ω = sqrt(4π k_e q² / (μ V) · force_to_accel)
    //     = 5.9091697154893236e-2 fs⁻¹   (period 106.33 fs)
    //   d(t) = d0 cos(ωt), d0 = 2 Å; U0 = 3.6190255157040796e-1 eV.
    // Pins are the CONTINUOUS analytic values; dt = 0.1 fs keeps the VV
    // discretization error below 5e-6 Å (tolerance 5e-5).
    {
        const double dt = 0.1;    // [fs]
        const double d0 = 2.0;    // [Å]
        const double omega = 5.9091697154893236e-2; // [1/fs]
        const double U0 = 3.6190255157040796e-1;    // [eV]
        const double tol_z = 5e-5;

        auto f = pair_frame(d0 / 2.0);
        // Seed forces before the first VV half-kick (same role as LAMMPS
        // fix setup(): step-0 initial_integrate must see F(t=0)).
        {
            cpu::md::FixDfield seed(0.0, 0.0, 0.0);
            seed.polarization_feedback = true;
            cpu::Evaluate seed_eval;
            seed_eval.set_evaluator_ref(seed);
            seed_eval.build();
            seed_eval.run(f);
            f.meta.get(kEnergy) = 0.0;
        }

        cpu::md::MD sim;
        sim.set_threads(1); // 2 atoms: fork/join on a fat node dwarfs physics
        sim.set_integrator(cpu::md::VelocityVerlet{dt});
        cpu::md::FixDfield dfield(0.0, 0.0, 0.0);
        dfield.polarization_feedback = true;
        sim.set_observer_ref(dfield);
        sim.enable_timing(false);
        sim.build();

        // Quarter period (~266 steps): d ≈ 0; run in two legs so we can pin
        // the mid-trajectory point too.
        sim.run(f, 266);
        const double d_266 = double(f.span(kAtom_z)[0]) - double(f.span(kAtom_z)[1]);
        std::printf("Gate B: step 266  d=%.16e want=%.16e\n", d_266,
                    d0 * std::cos(omega * 26.6));
        if (std::fabs(d_266 - d0 * std::cos(omega * 26.6)) > tol_z) {
            std::printf("FAIL B quarter-period separation\n");
            ok = false;
        }

        sim.run(f, 266); // half period total (532 steps)
        const double d_532 = double(f.span(kAtom_z)[0]) - double(f.span(kAtom_z)[1]);
        const double d_want = d0 * std::cos(omega * 53.2); // ≈ −d0
        std::printf("Gate B: step 532  d=%.16e want=%.16e\n", d_532, d_want);
        if (std::fabs(d_532 - d_want) > tol_z) {
            std::printf("FAIL B half-period separation\n");
            ok = false;
        }

        // Energy conservation: KE + U_D = U0 (field does the only work).
        double ke = 0.0;
        for (int i = 0; i < 2; ++i) {
            const double vz = double(f.span(kAtom_vz)[i]);
            ke += 0.5 * units::mass_v2_to_energy * 1.0 * vz * vz;
        }
        const double e_tot = ke + f.meta.get(kEnergy);
        std::printf("Gate B: KE+U=%.16e U0=%.16e\n", e_tot, U0);
        if (std::fabs(e_tot - U0) > 1e-6) {
            std::printf("FAIL B energy conservation\n");
            ok = false;
        }
        // Field response: E_z(t) = −4π k_e P_z(t) tracked the motion.
        std::printf("Gate B: Ez(final)=%.16e Pz(final)=%.16e\n",
                    dfield.last_Ez(), dfield.last_Pz());
        if (std::fabs(dfield.last_Ez() - (-4.0 * units::pi * units::coulomb
                                          * dfield.last_Pz()))
            > 1e-13) {
            std::printf("FAIL B Ez = -4*pi*k_e*Pz self-consistency\n");
            ok = false;
        }
    }

    // ── Gate C: vacuum-foothold twin — no polarization feedback, D = 0 →
    //    zero force, atoms never move, E_z stays D_z ─────────────────────
    {
        const double dt = 0.1;
        auto f = pair_frame(1.0);
        cpu::md::MD sim;
        sim.set_threads(1); // 2 atoms: fork/join on a fat node dwarfs physics
        sim.set_integrator(cpu::md::VelocityVerlet{dt});
        cpu::md::FixDfield dfield(0.0, 0.0, 0.0); // foothold: E ≡ D = 0
        sim.set_observer_ref(dfield);
        sim.enable_timing(false);
        sim.build();
        sim.run(f, 532);
        const double z0 = double(f.span(kAtom_z)[0]);
        const double z1 = double(f.span(kAtom_z)[1]);
        std::printf("Gate C: z+=%.16e z-=%.16e Ez=%.16e\n", z0, z1,
                    dfield.last_Ez());
        if (z0 != 6.0 || z1 != 4.0) {
            std::printf("FAIL C vacuum twin must not move\n");
            ok = false;
        }
        if (dfield.last_Ez() != 0.0) {
            std::printf("FAIL C vacuum twin Ez must stay D_z=0\n");
            ok = false;
        }
    }

    if (!ok)
        return 1;
    std::printf("PASS: constant-d-polarization-feedback vs Zhang–Sprik "
                "constant-D literature pins\n");
    return 0;
}

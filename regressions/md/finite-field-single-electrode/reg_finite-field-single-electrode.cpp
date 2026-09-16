/// @file regressions/md/finite-field-single-electrode/reg_finite-field-single-electrode.cpp
/// @brief E2E: finite-field term on the constant-potential BO right-hand side,
///        probed on ONE **isolated** Gaussian electrode site.
///
/// Offline pin: analytic Reed Gaussian BO + external-field term
/// (see REFERENCE.md). Public headers + link atomiverse only.
///
/// Scope, verified against MetalWalls c8ad175d (2026-07-29): the RHS
/// convention (Φ_ext = −E·r), F = qE, and P = Σ q r / V are MetalWalls'
/// own formulas. The *number* below is not a MetalWalls-reproducible
/// observable — MW refuses `external_field` with `charge_neutrality false`
/// (configuration.F90:624-630) because a free-Q single-site solve in 3D PBC
/// depends on the Ewald split and on the choice of origin. `xy_image_range`
/// is 0 here: this is the ISOLATED Gaussian self-capacitance, not a periodic
/// elastance. See REFERENCE.md for the MW cross-check and the 2.5 % gap it
/// exposes on the nearest MW-legal (neutral, two-site) system.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/fix_efield.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/classical_polarization.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

constexpr double kSqrt2OverPi = 0.79788456080286535588;

double A_self(double eta) { return units::coulomb * eta * kSqrt2OverPi; }

} // namespace

int main() {
    // ── Pin geometry (REFERENCE.md) ──────────────────────────────────
    constexpr double kEta = 2.0; // [1/Å]
    constexpr double kEz = 0.1;  // [V/Å]
    constexpr double kZ = 10.0;  // [Å]
    constexpr double kL = 20.0;  // [Å]
    constexpr double kPsi = 0.0; // [V]
    const double Aii = A_self(kEta);
    const double Q_pin = (kPsi + kEz * kZ) / Aii;
    const double V = kL * kL * kL;
    const double Pz_pin = Q_pin * kZ / V;

    int nfail = 0;

    // ── Standalone ConstPotBO + ClassicalPolarization (Frame-only MD) ─
    {
        cpu::Frame frame(1);
        frame.span(kAtom_x)[0] = 0.0;
        frame.span(kAtom_y)[0] = 0.0;
        frame.span(kAtom_z)[0] = kZ;
        frame.span(kAtom_charge)[0] = 0.0;
        frame.span(kAtom_type)[0] = 1;
        frame.span(kAtom_mass)[0] = 196.966553;
        frame.span(kAtom_tag)[0] = 1;
        frame.span(kAtom_vx)[0] = frame.span(kAtom_vy)[0] =
            frame.span(kAtom_vz)[0] = 0.0;
        frame.box = cpu::Box::cubic(kL);

        cpu::md::ConstPotBO conp(kEta);
        conp.add_electrode(/*type=*/1, kPsi);
        conp.Ez = kEz;
        conp.electroneutrality = false;

        cpu::md::ClassicalPolarization pol;
        cpu::md::MD sim;
        sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/2.0, /*skin=*/0.3});
        sim.set_evaluator(cpu::PairLJCut{/*eps=*/0.0, /*sigma=*/1.0,
                                         /*cutoff=*/2.0});
        sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.0});
        sim.set_observer_ref(conp);
        sim.set_observer_ref(pol);
        sim.build();
        sim.run(frame, 1);

        const double Q = conp.total_charge(0);
        std::printf("finite-field-single-electrode\n");
        std::printf("  Aii     = %.12e V/e\n", Aii);
        std::printf("  Q       = %.12e e  (pin %.12e)\n", Q, Q_pin);
        if (std::abs(Q - Q_pin) > 1e-10) {
            std::fprintf(stderr, "FAIL Q |%g - %g| > 1e-10\n", Q, Q_pin);
            ++nfail;
        }

        std::printf("  Pz      = %.12e e/Å²  (pin %.12e)\n", pol.Pz(), Pz_pin);
        if (std::abs(pol.Pz() - Pz_pin) > 1e-12) {
            std::fprintf(stderr, "FAIL Pz |%g - %g| > 1e-12\n", pol.Pz(),
                         Pz_pin);
            ++nfail;
        }
        if (std::abs(pol.Px()) > 1e-14 || std::abs(pol.Py()) > 1e-14) {
            std::fprintf(stderr, "FAIL Px/Py non-zero\n");
            ++nfail;
        }
    }

    // ── Dual-gate FixEfield force on electrolyte ion ─────────────────
    {
        cpu::Frame frame(1);
        frame.span(kAtom_x)[0] = 1.0;
        frame.span(kAtom_y)[0] = 2.0;
        frame.span(kAtom_z)[0] = 3.0;
        frame.span(kAtom_charge)[0] = 1.0;
        frame.span(kAtom_type)[0] = 2;
        frame.span(kAtom_mass)[0] = 1.0;
        frame.span(kAtom_tag)[0] = 1;
        frame.span(kAtom_fx)[0] = frame.span(kAtom_fy)[0] =
            frame.span(kAtom_fz)[0] = 0.0;
        frame.box = cpu::Box::cubic(kL);

        cpu::Evaluate eval;
        eval.on(evaluate::Phase::evaluate)
            .add(cpu::md::FixEfield{0.0, 0.0, kEz});
        eval.build();
        eval.run(frame);

        const double fz = static_cast<double>(frame.span(kAtom_fz)[0]);
        std::printf("  Fz(ion) = %.12e eV/Å  (pin %.12e)\n", fz, kEz);
        if (std::abs(fz - kEz) > 1e-12) {
            std::fprintf(stderr, "FAIL FixEfield Fz |%g - %g| > 1e-12\n", fz,
                         kEz);
            ++nfail;
        }
    }

    // ── MD binding path (ConstPotBO + FixEfield + pol, 1 VV step) ────
    {
        // Electrode + frozen mass; inert LJ so MD pipeline is valid.
        cpu::Frame frame(1);
        frame.span(kAtom_x)[0] = 0.0;
        frame.span(kAtom_y)[0] = 0.0;
        frame.span(kAtom_z)[0] = kZ;
        frame.span(kAtom_charge)[0] = 0.0;
        frame.span(kAtom_type)[0] = 1;
        frame.span(kAtom_mass)[0] = 196.966553;
        frame.span(kAtom_tag)[0] = 1;
        frame.span(kAtom_vx)[0] = frame.span(kAtom_vy)[0] =
            frame.span(kAtom_vz)[0] = 0.0;
        frame.box = cpu::Box::cubic(kL);

        cpu::md::ConstPotBO conp(kEta);
        conp.add_electrode(1, kPsi);
        conp.Ez = kEz;
        conp.electroneutrality = false;
        conp.rebuild_matrix_every_step = true;

        cpu::md::ClassicalPolarization pol;
        cpu::md::FixEfield field(0.0, 0.0, kEz);

        cpu::md::MD sim;
        // Inert LJ: cutoff ≪ any self-distance; only electrode self.
        sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/2.0, /*skin=*/0.3});
        sim.set_evaluator(cpu::PairLJCut{/*eps=*/0.0, /*sigma=*/1.0,
                                         /*cutoff=*/2.0});
        sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
        sim.set_observer_ref(conp);
        sim.set_observer_ref(field);
        sim.set_observer_ref(pol);
        sim.build();
        sim.run(frame, 1);

        const double Q_md = conp.total_charge(0);
        std::printf("  MD Q    = %.12e e  (pin %.12e)\n", Q_md, Q_pin);
        std::printf("  MD Pz   = %.12e e/Å²\n", pol.Pz());
        if (std::abs(Q_md - Q_pin) > 1e-10) {
            std::fprintf(stderr, "FAIL MD Q |%g - %g| > 1e-10\n", Q_md, Q_pin);
            ++nfail;
        }
        if (std::abs(pol.Pz() - Pz_pin) > 1e-10) {
            // Slightly looser: one VV step may move the atom by O(dt²).
            // With zero velocity and large mass, motion is negligible.
            std::fprintf(stderr, "FAIL MD Pz |%g - %g|\n", pol.Pz(), Pz_pin);
            ++nfail;
        }
    }

    if (nfail) {
        std::fprintf(stderr, "%d check(s) failed\n", nfail);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

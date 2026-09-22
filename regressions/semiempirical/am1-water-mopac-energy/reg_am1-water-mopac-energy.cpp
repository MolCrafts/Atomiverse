/// @file regressions/semiempirical/am1-water-mopac-energy/reg_am1-water-mopac-energy.cpp
/// @brief AM1 water — absolute OpenMOPAC energy / ΔHf / charge pin.
///
/// Oracle: **OpenMOPAC** (NDDO family lock; DFTB+ is a forbidden pin source
/// for AM1). Transcript: archived MOPAC 7.00 output of `AM1 PRECISE EF T=48H`
/// on water, H. Zipse, LMU München computational-chemistry course archive
/// (`cup.uni-muenchen.de/ch/compchem/energy/MOPAC_output.html`), twin at
/// `tools/tb_oracle/inputs/mopac/am1-water/`. AM1 H and O parameters:
/// Dewar, Zoebisch, Healy, Stewart, *J. Am. Chem. Soc.* **107**, 3902 (1985) —
/// the two citations MOPAC itself prints for this run.
///
/// Nothing here executes an oracle binary; every reference number is a
/// transcribed constant. See REFERENCE.md for the tolerance rationale and the
/// physical-constant convention that sets the residual floor.
///
/// Geometry convention: a published AM1 ΔHf belongs to the **AM1-optimized**
/// structure, so this case pins at the optimized internal coordinates printed
/// by the *same* run (r = 0.961267 Å, ∠HOH = 103.53103°) — geometry and energy
/// from one self-consistent transcript. A second pin at that run's *input*
/// geometry (1.0 Å, 105°, MOPAC `CYCLE 1 HEAT`) shows the agreement is not an
/// artifact of sitting at a stationary point.
///
/// Exit non-zero on failure.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/am1_evaluator.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"

namespace {

// ── Pinned OpenMOPAC (MOPAC 7.00) AM1 water transcript ──────────────────────
// Optimized structure (printed internal coordinates of the converged run).
constexpr double kOptOhAngstrom = 0.961267;
constexpr double kOptHohDegrees = 103.53103;
// FINAL HEAT OF FORMATION / TOTAL ENERGY at that structure.
constexpr double kMopacTotalEnergyEv = -348.56254;
constexpr double kMopacHofKcalMol = -59.24079;
// NET ATOMIC CHARGES at that structure.
constexpr double kMopacChargeO = -0.3827;
constexpr double kMopacChargeH = 0.1914;
// Input structure of the same run, and its `CYCLE 1` heat of formation.
constexpr double kInputOhAngstrom = 1.0;
constexpr double kInputHohDegrees = 105.0;
constexpr double kMopacInputHofKcalMol = -57.64871;

// ── Tolerances (see REFERENCE.md § Tolerance rationale) ─────────────────────
constexpr double kEnergyTolEv = 2.0e-3;
constexpr double kHofTolKcalMol = 0.05;
constexpr double kChargeTolE = 2.0e-4;
// Stationarity of the published optimum on ATV's own AM1 surface.
constexpr double kStationaryTolEvA = 5.0e-3;
// The same quantity at the un-optimized input geometry must be far outside it.
constexpr double kOffOptimumFloorEvA = 0.5;

/// @brief Water in MOPAC's C2v frame: O at origin, H1 on +x, H2 in the xy plane.
/// @param oh_angstrom O–H bond length [Å] (both bonds; C2v).
/// @param hoh_degrees H–O–H bond angle [degrees].
/// @return Frame with Z in `kAtom_type`, positions [Å], 30 Å cubic box.
ATV::cpu::Frame water(double oh_angstrom, double hoh_degrees) {
    const double theta = hoh_degrees * 3.14159265358979323846 / 180.0;
    ATV::cpu::Frame f(3);
    f.span(ATV::kAtom_type)[0] = 8;
    f.span(ATV::kAtom_x)[0] = 0;
    f.span(ATV::kAtom_y)[0] = 0;
    f.span(ATV::kAtom_z)[0] = 0;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_x)[1] = static_cast<ATV::StorageReal>(oh_angstrom);
    f.span(ATV::kAtom_y)[1] = 0;
    f.span(ATV::kAtom_z)[1] = 0;
    f.span(ATV::kAtom_type)[2] = 1;
    f.span(ATV::kAtom_x)[2] =
        static_cast<ATV::StorageReal>(oh_angstrom * std::cos(theta));
    f.span(ATV::kAtom_y)[2] =
        static_cast<ATV::StorageReal>(oh_angstrom * std::sin(theta));
    f.span(ATV::kAtom_z)[2] = 0;
    f.box = ATV::cpu::Box::cubic(30.0); // isolation only
    return f;
}

bool fail(const char* what, double got, double want, double tol,
          const char* unit) {
    std::fprintf(stderr,
                 "FAIL: %s = %.8f, MOPAC = %.8f, |Δ| = %.3e > %.3e %s\n", what,
                 got, want, std::abs(got - want), tol, unit);
    return false;
}

} // namespace

int main() {
    using namespace ATV;

    bool ok = true;

    // ── 1. Public Evaluate + AM1Evaluator path: total energy [eV] ───────────
    // MOPAC TOTAL ENERGY = ELECTRONIC ENERGY + CORE-CORE REPULSION, which is
    // exactly the quantity AM1 writes to frame.meta[kEnergy].
    {
        auto frame = water(kOptOhAngstrom, kOptHohDegrees);
        cpu::Evaluate eval;
        eval.on(evaluate::Phase::evaluate)
            .add(cpu::semiempirical::AM1Evaluator{}.with_gradient(true));
        eval.build();
        eval.run(frame);

        const double e = static_cast<double>(frame.meta.get(kEnergy));
        if (!std::isfinite(e)
            || std::abs(e - kMopacTotalEnergyEv) > kEnergyTolEv) {
            ok = fail("E_total(AM1 optimum)", e, kMopacTotalEnergyEv,
                      kEnergyTolEv, "eV");
        } else {
            std::printf("E_total  = %.8f eV   (MOPAC %.5f, Δ = %+.2e eV)\n", e,
                        kMopacTotalEnergyEv, e - kMopacTotalEnergyEv);
        }
    }

    // ── 2. Concrete-method path: ΔHf [kcal/mol] and Mulliken charges [e] ────
    // ΔHf = (E_tot − Σ EISOL)·23.061 + Σ ΔHf_atom (openmopac calc_of_hof), so
    // this pin gates the SCF energy AND the AM1 thermochemistry constants.
    {
        auto f = water(kOptOhAngstrom, kOptHohDegrees);
        cpu::semiempirical::AM1 am1;
        am1.build(f);
        const auto& r = am1.run(f);

        if (!r.converged) {
            std::fprintf(stderr, "FAIL: AM1 SCF did not converge\n");
            return 1;
        }
        const double hof = r.heat_of_formation_kcal_mol;
        if (!std::isfinite(hof)
            || std::abs(hof - kMopacHofKcalMol) > kHofTolKcalMol) {
            ok = fail("ΔHf(AM1 optimum)", hof, kMopacHofKcalMol, kHofTolKcalMol,
                      "kcal/mol");
        } else {
            std::printf("ΔHf      = %.6f kcal/mol (MOPAC %.5f, Δ = %+.2e)\n",
                        hof, kMopacHofKcalMol, hof - kMopacHofKcalMol);
        }

        const double q_o = r.mulliken_charges[0];
        const double q_h1 = r.mulliken_charges[1];
        const double q_h2 = r.mulliken_charges[2];
        if (std::abs(q_o - kMopacChargeO) > kChargeTolE) {
            ok = fail("q(O)", q_o, kMopacChargeO, kChargeTolE, "e");
        } else if (std::abs(q_h1 - kMopacChargeH) > kChargeTolE
                   || std::abs(q_h2 - kMopacChargeH) > kChargeTolE) {
            ok = fail("q(H)", q_h1, kMopacChargeH, kChargeTolE, "e");
        } else {
            std::printf("q(O)/q(H) = %+.5f / %+.5f e  (MOPAC %+.4f / %+.4f)\n",
                        q_o, q_h1, kMopacChargeO, kMopacChargeH);
        }

        // ── 3. The published optimum is stationary on ATV's own surface ─────
        // MOPAC stopped here at RMS GRADIENT = 0.00141 kcal/mol/Å; if ATV's AM1
        // is the same functional, this geometry must be a minimum of ATV's
        // energy too. That is a gradient-level agreement check independent of
        // the energy pin — and it is what licenses pinning an optimized-
        // geometry ΔHf at fixed coordinates: E is quadratic here, so the
        // transcript's coordinate truncation cannot masquerade as an error.
        am1.compute_forces(f, 1.0e-4);
        double max_f = 0.0;
        for (double v : am1.last_result().forces_ev_a) {
            max_f = std::max(max_f, std::abs(v));
        }
        if (!(max_f <= kStationaryTolEvA)) {
            std::fprintf(stderr,
                         "FAIL: MOPAC's AM1 optimum is not stationary for ATV: "
                         "max|F| = %.3e > %.3e eV/Å\n",
                         max_f, kStationaryTolEvA);
            ok = false;
        } else {
            std::printf("max|F|   = %.3e eV/Å at the MOPAC optimum (< %.1e)\n",
                        max_f, kStationaryTolEvA);
        }
    }

    // ── 4. Same pin off the stationary point (MOPAC `CYCLE 1 HEAT`) ─────────
    // Agreement at a non-stationary geometry rules out the objection that the
    // optimum pin only works because the gradient happens to vanish there, and
    // it gives the stationarity gate of step 3 a measured range.
    {
        auto f = water(kInputOhAngstrom, kInputHohDegrees);
        cpu::semiempirical::AM1 am1;
        am1.build(f);
        const double hof = am1.run(f).heat_of_formation_kcal_mol;
        if (!std::isfinite(hof)
            || std::abs(hof - kMopacInputHofKcalMol) > kHofTolKcalMol) {
            ok = fail("ΔHf(MOPAC input geometry)", hof, kMopacInputHofKcalMol,
                      kHofTolKcalMol, "kcal/mol");
        } else {
            std::printf("ΔHf(1.0 Å, 105°) = %.6f kcal/mol (MOPAC CYCLE 1 %.5f, "
                        "Δ = %+.2e)\n",
                        hof, kMopacInputHofKcalMol,
                        hof - kMopacInputHofKcalMol);
        }

        am1.compute_forces(f, 1.0e-4);
        double max_f = 0.0;
        for (double v : am1.last_result().forces_ev_a) {
            max_f = std::max(max_f, std::abs(v));
        }
        if (!(max_f > kOffOptimumFloorEvA)) {
            std::fprintf(
                stderr,
                "FAIL: stationarity gate is vacuous — the un-optimized "
                "geometry gives max|F| = %.3e <= %.1e eV/Å\n",
                max_f, kOffOptimumFloorEvA);
            ok = false;
        } else {
            std::printf("max|F|   = %.3e eV/Å off the optimum (%.0fx the "
                        "step-3 stationarity gate)\n",
                        max_f, max_f / kStationaryTolEvA);
        }
    }

    if (!ok) {
        return 1;
    }
    std::printf("PASS: AM1 H2O vs OpenMOPAC (MOPAC 7.00 `AM1 PRECISE EF`; "
                "Dewar JACS 107, 3902 (1985) parameters)\n");
    return 0;
}

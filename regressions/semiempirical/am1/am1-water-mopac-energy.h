/// @file regressions/semiempirical/am1/am1-water-mopac-energy.h
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

#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/cpu/semiempirical/rm1_solver.h"
#include "harness/registry.h"

namespace atv_reg_semiempirical_am1_water_mopac_energy {

namespace {

// OpenMOPAC 23.2.5 executed fixtures (no OLDCON), extracted from raw AUX/DISP.
// Source, inputs, output hashes and independent gradient qualification:
// tools/tb_oracle/inputs/mopac/live-23.2.5/README.md.
struct ModernMopacPin {
    const char* name;
    bool rm1;
    int count;
    int species[4];
    double xyz[4][3];
    double energy_ev;
    double hof_kcal;
    double charges[4];
    double forces[4][3];
};
constexpr ModernMopacPin kModernPins[] = {
    {"am1-ammonia",
     false,
     4,
     {7, 1, 1, 1},
     {{0.1, -0.1, 0.05},
      {1.05, 0.05, 0.27},
      {-0.34, 0.78, 0.36},
      {-0.25, -0.81, 0.59}},
     -248.5036722497572,
     -5.26782900723447,
     {-0.3626133185678, 0.11905107529003, 0.1172048010525, 0.12635744222526},
     {{-0.19215263876601718, 2.743638026750998, -0.41696832526616456},
      {0.8154020939752601, -0.3400131436324174, -0.0485201169567256},
      {0.02385454528199443, -0.9387768024315132, -0.515136372746632},
      {-0.6471040000386029, -1.4648480793527843, 0.9806248144251638}}},
    {"am1-co",
     false,
     2,
     {6, 8},
     {{0.1, -0.2, 0.3}, {0.95, 0.4, 0.9}},
     -447.10041000458017,
     -4.42489046978335,
     {0.22032838516751, -0.22032838516752},
     {{2.56031743710396, 1.8072829295143868, 1.807282930651255},
      {-2.5603174356826974, -1.8072829295143868, -1.8072829296480135}}},
    {"am1-water",
     false,
     3,
     {8, 1, 1},
     {{0.0, 0.0, 0.0}, {0.96, 0.0, 0.0}, {-0.24, 0.93, 0.1}},
     -348.56200724457074,
     -59.224043903444,
     {-0.38533744728156, 0.19230579459317, 0.19303165268838},
     {{-0.1703189427360047, 0.08528675280672224, 0.009170629056145156},
      {0.027718017848574316, 0.09345834965479423, 0.010049284941969187},
      {0.1426009248769363, -0.1787451023674337, -0.019219913998665066}}},
    {"rm1-ammonia",
     true,
     4,
     {7, 1, 1, 1},
     {{0.1, -0.1, 0.05},
      {1.05, 0.05, 0.27},
      {-0.34, 0.78, 0.36},
      {-0.25, -0.81, 0.59}},
     -252.82350284232146,
     -4.04980276976039,
     {-0.43044152159328, 0.14132470281443, 0.13937282635203, 0.14974399242681},
     {{-0.2353918301235511, 2.667792374998589, -0.5674877411528912},
      {1.323867574401197, -0.25067656876319855, -0.07378553417360097},
      {-0.1990951856413792, -0.5220974008951202, -0.48675293189482677},
      {-0.8893805579013708, -1.8950184039984979, 1.128026206595137}}},
    {"rm1-co",
     true,
     2,
     {6, 8},
     {{0.1, -0.2, 0.3}, {0.95, 0.4, 0.9}},
     -440.2956922176226,
     -9.10383187284424,
     {0.24815357886139, -0.24815357886139},
     {{4.470989056088783, 3.1559922985932487, 3.1559922985932487},
      {-4.470989053606886, -3.1559922985932487, -3.155992296841321}}},
    {"rm1-water",
     true,
     3,
     {8, 1, 1},
     {{0.0, 0.0, 0.0}, {0.96, 0.0, 0.0}, {-0.24, 0.93, 0.1}},
     -345.569010265186,
     -57.7866559366557,
     {-0.37160586077324, 0.18547955047946, 0.18612631029378},
     {{-0.00840153347781067, 0.26683654084536207, 0.028692111584403705},
      {-0.19440902177406577, 0.10381085564503859, 0.011162457651535769},
      {0.2028105553600352, -0.3706473962953056, -0.039854569237088626}}},
};

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

inline int run_cpu() {
    using namespace ATV;

    bool ok = true;

    // ── 1. Public Evaluate + nested AM1 path: total energy [eV] ───────────
    // MOPAC TOTAL ENERGY = ELECTRONIC ENERGY + CORE-CORE REPULSION, which is
    // exactly the quantity AM1 writes to frame.meta[kEnergy].
    {
        auto frame = water(kOptOhAngstrom, kOptHohDegrees);
        cpu::Evaluate eval;
        eval.on(evaluate::Phase::evaluate).add(cpu::semiempirical::AM1{});
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
        // One SCF: `forces()` fills energies, charges and `forces_ev_a` (the
        // latter gated in step 3).
        const auto& r = am1.forces(f);

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
        double max_f = 0.0;
        for (double v : r.forces_ev_a) {
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
        // One SCF: `forces()` fills ΔHf and `forces_ev_a` (gated below).
        const double hof = am1.forces(f).heat_of_formation_kcal_mol;
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

    // Strict modern-convention gate supplements the historical transcript.
    // Total-energy print rounding <=2.17e-7 eV. Independent MOPAC-only
    // Richardson qualification bounds native FD gradients to 7.84e-7 eV/A;
    // 1e-5 allows that documented oracle approximation (not an ATV fit).
    for (const auto& pin : kModernPins) {
        cpu::Frame frame(pin.count);
        for (int i = 0; i < pin.count; ++i) {
            frame.span(kAtom_type)[i] = pin.species[i];
            frame.span(kAtom_x)[i] = pin.xyz[i][0];
            frame.span(kAtom_y)[i] = pin.xyz[i][1];
            frame.span(kAtom_z)[i] = pin.xyz[i][2];
        }
        frame.box = cpu::Box::cubic(30.0);
        cpu::semiempirical::AM1 am1{0, 1, 1000, 1.e-10, 1.e-12};
        cpu::semiempirical::RM1 rm1{0, 1, 1000, 1.e-10, 1.e-12};
        const auto& result = pin.rm1 ? rm1.forces(frame) : am1.forces(frame);
        if (!result.converged || !std::isfinite(result.total_energy_ev)
            || std::abs(result.total_energy_ev - pin.energy_ev) > 1.e-6)
            ok = fail(pin.name, result.total_energy_ev, pin.energy_ev, 1.e-6,
                      "eV");
        if (!std::isfinite(result.heat_of_formation_kcal_mol)
            || std::abs(result.heat_of_formation_kcal_mol - pin.hof_kcal)
                   > 1.e-6)
            ok = fail(pin.name, result.heat_of_formation_kcal_mol, pin.hof_kcal,
                      1.e-6, "kcal/mol");
        double max_q = 0.0, max_f = 0.0;
        for (int i = 0; i < pin.count; ++i) {
            const double q = result.mulliken_charges[i];
            max_q = std::max(max_q, std::abs(q - pin.charges[i]));
            if (!std::isfinite(q) || std::abs(q - pin.charges[i]) > 1.e-7)
                ok = fail(pin.name, q, pin.charges[i], 1.e-7, "e");
            for (int axis = 0; axis < 3; ++axis) {
                const double f = result.forces_ev_a[3 * i + axis];
                max_f = std::max(max_f, std::abs(f - pin.forces[i][axis]));
                if (!std::isfinite(f)
                    || std::abs(f - pin.forces[i][axis]) > 1.e-5)
                    ok = fail(pin.name, f, pin.forces[i][axis], 1.e-5, "eV/A");
            }
        }
        std::printf("OpenMOPAC 23.2.5 %s: |dE|=%.3e eV max|dq|=%.3e e "
                    "max|dF|=%.3e eV/A\n",
                    pin.name, std::abs(result.total_energy_ev - pin.energy_ev),
                    max_q, max_f);
    }

    if (!ok) {
        return 1;
    }
    std::printf("PASS: AM1 H2O vs OpenMOPAC (MOPAC 7.00 `AM1 PRECISE EF`; "
                "Dewar JACS 107, 3902 (1985) parameters)\n");
    return 0;
}

} // namespace atv_reg_semiempirical_am1_water_mopac_energy

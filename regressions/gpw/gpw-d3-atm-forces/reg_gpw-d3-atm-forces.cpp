/// @file regressions/gpw/gpw-d3-atm-forces/reg_gpw-d3-atm-forces.cpp
/// @brief END-TO-END: DFT-D3 three-body ATM (C9) analytic ionic forces on the
///        GPW path vs offline CP2K twins — isolated AND small periodic cell.
///
/// Extends the `gpw-d3-atm` energy foothold with the dedicated ATM *force*
/// pin. Per boundary condition, two CP2K 2026.1 `ENERGY_FORCE` runs on the
/// same geometry — `CALCULATE_C9_TERM .TRUE.` and `.FALSE.` — differ only by
/// the three-body term. D3 is geometry-only (it never enters the Kohn-Sham
/// potential), so the difference of the printed `FORCES|` tables cancels the
/// electronic force to solver noise (~1e-15 here) and isolates exactly
///
///     F_ATM = -grad E_ATM,
///
/// the quantity this capability adds. Gates:
///   1. E_disp (ATM on) vs CP2K, isolated 12 A + periodic 6 A cells.
///   2. Standalone ATM-only force vs the CP2K F(C9) - F(two-body) pin, both
///      cells — the periodic one exercises the lattice-image triple
///      enumeration (CP2K qs_dispersion_d3 doabc CASE 2/3, fac0 = 1/3).
///   3. Product wiring: full GPW SCF via public GPWEvaluator with
///      d3_atm = true; the force assembly with d3_atm on minus off (same
///      converged density) must equal the standalone ATM gradient.
///   4. Total product forces vs CP2K's ATM-run FORCES| table (grid-noise
///      bounded) and translation invariance |sum F| on the isolated system.
/// The analytic-vs-FD dual gate lives in tests/cpu/gpw/test_cpu_gpw_dftd3.cpp
/// (`GpwDftD3Atm.AnalyticForcesMatchFiniteDifference` isolated,
/// `GpwDftD3Atm.PeriodicAtmForcesMatchFiniteDifference` periodic images).
///
/// Public API only. No live CP2K. See REFERENCE.md.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/dftd3.h"
#include "atomiverse/cpu/gpw/internal/dftd3_params.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

constexpr double kAng2Bohr = 1.8897259885789;

// ── Offline CP2K 2026.1 pins (see REFERENCE.md) ────────────────────────────
// Isolated water dimer, 12 A PERIODIC NONE (MT), R_CUTOFF 20.
constexpr double kEDispAtmIso = -0.00081414118193; // Ha, CALCULATE_C9_TERM T
// Periodic water dimer, 6 A PERIODIC XYZ, R_CUTOFF 10.
constexpr double kEDispAtmPer = -0.00195809486938; // Ha, CALCULATE_C9_TERM T
constexpr double kEDispTwoPer = -0.00195887984277; // Ha, CALCULATE_C9_TERM F
constexpr double kEDispTol = 1.0e-9;               // Ha

// CP2K FORCES| tables [Ha/Bohr], atom order O H H O H H.
// Isolated, CALCULATE_C9_TERM .TRUE.:
constexpr double kFAtmIsoCp2k[6][3] = {
    {-8.13885057E-02, 3.39382605E-08, -7.48922277E-09},
    {4.05547994E-02, 1.08883799E-08, 1.31321374E-01},
    {4.05547997E-02, -3.70556959E-08, -1.31321365E-01},
    {-7.98423417E-02, -1.52539145E-08, 2.13908340E-08},
    {3.73506217E-02, 9.81842399E-09, 1.35158535E-01},
    {3.73506306E-02, -1.84099620E-09, -1.35158557E-01},
};
// Isolated, CALCULATE_C9_TERM .FALSE. (same run pair, same parallelization):
constexpr double kFTwoIsoCp2k[6][3] = {
    {-8.13880434E-02, 3.39382581E-08, -7.48923114E-09},
    {4.05546400E-02, 1.08883804E-08, 1.31321564E-01},
    {4.05546403E-02, -3.70556938E-08, -1.31321555E-01},
    {-7.98421128E-02, -1.52539153E-08, 2.13908083E-08},
    {3.73504355E-02, 9.81842344E-09, 1.35158735E-01},
    {3.73504444E-02, -1.84099650E-09, -1.35158757E-01},
};
// Periodic 6 A, CALCULATE_C9_TERM .TRUE.:
constexpr double kFAtmPerCp2k[6][3] = {
    {-7.70297833E-02, -1.96186400E-07, -1.29836754E-08},
    {3.81505346E-02, 1.23208981E-07, 1.29913786E-01},
    {3.81504343E-02, 1.24270789E-07, -1.29913785E-01},
    {-7.70296389E-02, -2.00667422E-07, 1.79666280E-07},
    {3.81504401E-02, 1.77442868E-07, 1.29913776E-01},
    {3.81507350E-02, -2.61961211E-08, -1.29913946E-01},
};
// Periodic 6 A, CALCULATE_C9_TERM .FALSE.:
constexpr double kFTwoPerCp2k[6][3] = {
    {-7.70302942E-02, -1.96186403E-07, -1.29836767E-08},
    {3.81508710E-02, 1.23208981E-07, 1.29913227E-01},
    {3.81507708E-02, 1.24270788E-07, -1.29913227E-01},
    {-7.70303293E-02, -2.00667421E-07, 1.79666261E-07},
    {3.81507042E-02, 1.77442867E-07, 1.29913227E-01},
    {3.81509991E-02, -2.61961207E-08, -1.29913397E-01},
};

// ATM-only force pin tolerance. The pin difference cancels the electronic
// force to ~1e-15 (the y components differ in the 15th digit), so the floor
// is the FORCES| print resolution (9 significant digits on ~1e-1 components,
// i.e. ~1e-9 Ha/Bohr) against ATM signals of 4.6e-7 (iso) / 6.9e-7 (per).
constexpr double kFAtmTol = 1.0e-8; // Ha/Bohr
// The ATM term must actually engage (foothold-style nonzero gate).
constexpr double kFAtmMin = 1.0e-7; // Ha/Bohr
// Product wiring: force assembly (d3_atm on - off) on the SAME converged
// density vs the standalone ATM gradient — same code path, zero SCF residual.
constexpr double kWiringTol = 1.0e-7; // Ha/Bohr
// Total product force vs CP2K, bounded below by the oracle's own grid error
// (CP2K FORCES| Sum = 5.42e-3 Ha/Bohr on this twin) — see gpw-d3-forces.
constexpr double kFTotalTol = 1.0e-2; // Ha/Bohr
// Translation invariance of the isolated-system force field (MT placement).
constexpr double kNetForceTol = 1.0e-3; // Ha/Bohr
constexpr double kWallBudget = 60.0;    // s

/// Max |(F_on - F_off) - (cp2k_on - cp2k_off)| over 6 atoms x 3 axes, and the
/// pin's own max |cp2k_on - cp2k_off| (the ATM signal), for one cell.
void atm_diff_errors(const std::vector<std::array<double, 3>>& f_on,
                     const std::vector<std::array<double, 3>>& f_off,
                     const double cp2k_on[6][3], const double cp2k_off[6][3],
                     double& max_err, double& max_signal) {
    max_err = 0.0;
    max_signal = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double want = cp2k_on[i][k] - cp2k_off[i][k];
            const double got =
                f_on[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)]
                - f_off[static_cast<std::size_t>(i)]
                       [static_cast<std::size_t>(k)];
            max_err = std::max(max_err, std::abs(got - want));
            max_signal = std::max(max_signal, std::abs(want));
        }
    }
}

} // namespace

int main() {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    const int z[6] = {8, 1, 1, 8, 1, 1};
    // CP2K &COORD (A), shared by both twins.
    const double raw[6][3] = {
        {0.000000, 0.000000, 0.000000},  {0.758602, 0.000000, 0.504284},
        {0.758602, 0.000000, -0.504284}, {3.000000, 0.000000, 0.000000},
        {3.758602, 0.000000, 0.504284},  {3.758602, 0.000000, -0.504284},
    };

    // ── (1) Standalone ATM, isolated 12 A cell ────────────────────────────
    // Centred (dispersion is translation invariant; the MT product path below
    // requires it — see gpw-d3-forces REFERENCE.md "Placement").
    constexpr double kLIso = 12.0; // A
    std::vector<int> Zv(6);
    std::vector<std::array<double, 3>> pos_iso(6);
    for (int i = 0; i < 6; ++i) {
        Zv[static_cast<std::size_t>(i)] = z[i];
        pos_iso[static_cast<std::size_t>(i)] = {(raw[i][0] + 4.5) * kAng2Bohr,
                                                (raw[i][1] + 6.0) * kAng2Bohr,
                                                (raw[i][2] + 6.0) * kAng2Bohr};
    }
    const double cell_iso[3] = {kLIso * kAng2Bohr, kLIso * kAng2Bohr,
                                kLIso * kAng2Bohr};
    const cpu::gpw::D3Result iso_on = cpu::gpw::dftd3_pbe(
        Zv, pos_iso, cell_iso, /*periodic=*/false, cpu::gpw::D3Params{},
        cpu::gpw::kD3Cp2kPairCutoffBohr, /*calculate_c9=*/true);
    const cpu::gpw::D3Result iso_off = cpu::gpw::dftd3_pbe(
        Zv, pos_iso, cell_iso, /*periodic=*/false, cpu::gpw::D3Params{},
        cpu::gpw::kD3Cp2kPairCutoffBohr, /*calculate_c9=*/false);

    if (!std::isfinite(iso_on.energy)
        || std::abs(iso_on.energy - kEDispAtmIso) > kEDispTol) {
        std::printf("FAIL: iso |E_disp(ATM)-cp2k|=%.3e Ha  got=%.14e  "
                    "want=%.14e (tol %.1e)\n",
                    std::abs(iso_on.energy - kEDispAtmIso), iso_on.energy,
                    kEDispAtmIso, kEDispTol);
        return 1;
    }
    double err_iso = 0.0;
    double sig_iso = 0.0;
    atm_diff_errors(iso_on.forces, iso_off.forces, kFAtmIsoCp2k, kFTwoIsoCp2k,
                    err_iso, sig_iso);
    if (err_iso > kFAtmTol || sig_iso < kFAtmMin) {
        std::printf("FAIL: iso ATM force pin  max|dF-cp2k|=%.3e (tol %.1e)  "
                    "signal=%.3e (min %.1e) Ha/Bohr\n",
                    err_iso, kFAtmTol, sig_iso, kFAtmMin);
        return 1;
    }
    std::printf("iso  ATM: E_disp=%.14e Ha (|d|=%.2e)  max|dF-cp2k|=%.3e  "
                "signal=%.3e Ha/Bohr\n",
                iso_on.energy, std::abs(iso_on.energy - kEDispAtmIso), err_iso,
                sig_iso);

    // ── (2) Standalone ATM, periodic 6 A cell (lattice-image triples) ─────
    constexpr double kLPer = 6.0; // A
    std::vector<std::array<double, 3>> pos_per(6);
    for (int i = 0; i < 6; ++i)
        pos_per[static_cast<std::size_t>(i)] = {raw[i][0] * kAng2Bohr,
                                                raw[i][1] * kAng2Bohr,
                                                raw[i][2] * kAng2Bohr};
    const double cell_per[3] = {kLPer * kAng2Bohr, kLPer * kAng2Bohr,
                                kLPer * kAng2Bohr};
    // CP2K R_CUTOFF 10.0 A is a half cutoff: pairs are summed to 2x.
    const double rcut_per = 2.0 * 10.0 * kAng2Bohr;
    const cpu::gpw::D3Result per_on = cpu::gpw::dftd3_pbe(
        Zv, pos_per, cell_per, /*periodic=*/true, cpu::gpw::D3Params{},
        rcut_per, /*calculate_c9=*/true);
    const cpu::gpw::D3Result per_off = cpu::gpw::dftd3_pbe(
        Zv, pos_per, cell_per, /*periodic=*/true, cpu::gpw::D3Params{},
        rcut_per, /*calculate_c9=*/false);

    if (!std::isfinite(per_on.energy)
        || std::abs(per_on.energy - kEDispAtmPer) > kEDispTol
        || std::abs(per_off.energy - kEDispTwoPer) > kEDispTol) {
        std::printf("FAIL: per E_disp  ATM got=%.14e want=%.14e  two-body "
                    "got=%.14e want=%.14e (tol %.1e Ha)\n",
                    per_on.energy, kEDispAtmPer, per_off.energy, kEDispTwoPer,
                    kEDispTol);
        return 1;
    }
    double err_per = 0.0;
    double sig_per = 0.0;
    atm_diff_errors(per_on.forces, per_off.forces, kFAtmPerCp2k, kFTwoPerCp2k,
                    err_per, sig_per);
    if (err_per > kFAtmTol || sig_per < kFAtmMin) {
        std::printf("FAIL: per ATM force pin  max|dF-cp2k|=%.3e (tol %.1e)  "
                    "signal=%.3e (min %.1e) Ha/Bohr\n",
                    err_per, kFAtmTol, sig_per, kFAtmMin);
        return 1;
    }
    std::printf("per  ATM: E_disp=%.14e Ha (|d|=%.2e)  max|dF-cp2k|=%.3e  "
                "signal=%.3e Ha/Bohr\n",
                per_on.energy, std::abs(per_on.energy - kEDispAtmPer), err_per,
                sig_per);

    // ── (3) Product path: GPW SCF + forces with d3_atm through the public
    //        evaluator (isolated cell; CP2K CUTOFF 300 / EPS_SCF 1e-7) ─────
    cpu::Frame frame(6);
    for (int i = 0; i < 6; ++i) {
        frame.span(kAtom_type)[i] = z[i];
        frame.span(kAtom_x)[i] = raw[i][0] + 4.5;
        frame.span(kAtom_y)[i] = raw[i][1] + 6.0;
        frame.span(kAtom_z)[i] = raw[i][2] + 6.0;
        frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
            frame.span(kAtom_fz)[i] = 0;
    }
    frame.box = cpu::Box::cubic(kLIso);

    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

    cpu::gpw::GPWEvaluator eval({O, H});
    eval.cutoff_ry = 300.0;
    eval.max_iter = 120;
    eval.tol = 1e-7;
    eval.mix = 0.4;
    eval.xc = cpu::gpw::XcKind::PBE;
    eval.nonperiodic = true;
    eval.d3 = true;
    eval.d3_atm = true;
    eval.compute_forces = true;
    eval.compute_stress = false;

    cpu::Evaluate driver;
    driver.on(evaluate::Phase::evaluate).add_ref(eval);
    driver.enable_timing(false);
    driver.build();
    driver.run(frame);

    if (!eval.last_result().converged) {
        std::printf("FAIL: GPW SCF did not converge (iters=%d)\n",
                    eval.last_result().iters);
        return 1;
    }
    if (std::abs(eval.last_result().e_disp - kEDispAtmIso) > kEDispTol) {
        std::printf("FAIL: product |E_disp(ATM)-cp2k|=%.3e Ha  got=%.14e\n",
                    std::abs(eval.last_result().e_disp - kEDispAtmIso),
                    eval.last_result().e_disp);
        return 1;
    }

    // Force assembly twice on the SAME converged density — d3_atm on vs off.
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const cpu::gpw::Kind& k = (z[i] == 8) ? O : H;
        const double p[3] = {pos_iso[static_cast<std::size_t>(i)][0],
                             pos_iso[static_cast<std::size_t>(i)][1],
                             pos_iso[static_cast<std::size_t>(i)][2]};
        atoms.push_back(cpu::gpw::GpwAtom(p, k.basis, k.pot, z[i]));
    }

    cpu::gpw::GpwDft m_atm;
    m_atm.cell[0] = m_atm.cell[1] = m_atm.cell[2] = kLIso * kAng2Bohr;
    m_atm.cutoff_ry = 300.0;
    m_atm.n_occ = 8;
    m_atm.max_iter = 120;
    m_atm.tol = 1e-7;
    m_atm.mix = 0.4;
    m_atm.xc = cpu::gpw::XcKind::PBE;
    m_atm.nonperiodic = true;
    m_atm.local_pp_on_grid = true;
    m_atm.d3 = true;
    m_atm.d3_atm = true;
    m_atm.build(atoms);
    const auto F_on =
        cpu::gpw::gpw_forces_from_scf(atoms, m_atm, eval.last_result());

    cpu::gpw::GpwDft m_two; // GpwDft is non-copyable — build the twin fresh
    m_two.cell[0] = m_two.cell[1] = m_two.cell[2] = kLIso * kAng2Bohr;
    m_two.cutoff_ry = 300.0;
    m_two.n_occ = 8;
    m_two.max_iter = 120;
    m_two.tol = 1e-7;
    m_two.mix = 0.4;
    m_two.xc = cpu::gpw::XcKind::PBE;
    m_two.nonperiodic = true;
    m_two.local_pp_on_grid = true;
    m_two.d3 = true;
    m_two.d3_atm = false;
    m_two.build(atoms);
    const auto F_off =
        cpu::gpw::gpw_forces_from_scf(atoms, m_two, eval.last_result());

    double max_wire = 0.0;
    double max_total_err = 0.0;
    double net[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double on =
                F_on[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
            const double off =
                F_off[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
            if (!std::isfinite(on) || !std::isfinite(off)) {
                std::printf("FAIL: non-finite force atom %d axis %d\n", i, k);
                return 1;
            }
            const double f_atm_standalone =
                iso_on.forces[static_cast<std::size_t>(i)]
                             [static_cast<std::size_t>(k)]
                - iso_off.forces[static_cast<std::size_t>(i)]
                                [static_cast<std::size_t>(k)];
            max_wire =
                std::max(max_wire, std::abs((on - off) - f_atm_standalone));
            max_total_err =
                std::max(max_total_err, std::abs(on - kFAtmIsoCp2k[i][k]));
            net[k] += on;
        }
    }
    const double net_norm =
        std::sqrt(net[0] * net[0] + net[1] * net[1] + net[2] * net[2]);
    if (max_wire > kWiringTol) {
        std::printf("FAIL: wiring max|(F_atm_on-F_atm_off)-F_ATM|=%.3e > "
                    "%.1e Ha/Bohr\n",
                    max_wire, kWiringTol);
        return 1;
    }
    if (max_total_err > kFTotalTol) {
        std::printf("FAIL: max|F_atv-F_cp2k(ATM)|=%.3e > %.1e Ha/Bohr\n",
                    max_total_err, kFTotalTol);
        return 1;
    }
    if (net_norm > kNetForceTol) {
        std::printf("FAIL: |sum F| = %.3e > %.1e Ha/Bohr (MT placement / "
                    "translation invariance)\n",
                    net_norm, kNetForceTol);
        return 1;
    }

    const double wall_s =
        std::chrono::duration<double>(clock::now() - t0).count();
    if (wall_s > kWallBudget) {
        std::printf("FAIL: wall=%.2f s exceeds %.0f s e2e budget\n", wall_s,
                    kWallBudget);
        return 1;
    }

    std::printf("PASS: gpw-d3-atm-forces  iso max|dF-cp2k|=%.3e (sig %.3e)  "
                "per max|dF-cp2k|=%.3e (sig %.3e)  wiring=%.3e  "
                "max|F_total-cp2k|=%.3e  |sum F|=%.3e Ha/Bohr  iters=%d  "
                "wall=%.2f s\n",
                err_iso, sig_iso, err_per, sig_per, max_wire, max_total_err,
                net_norm, eval.last_result().iters, wall_s);
    return 0;
}

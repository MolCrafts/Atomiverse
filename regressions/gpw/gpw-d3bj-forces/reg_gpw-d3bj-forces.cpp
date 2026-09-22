/// @file regressions/gpw/gpw-d3bj-forces/reg_gpw-d3bj-forces.cpp
/// @brief END-TO-END: periodic water-dimer GPW PBE-D3(BJ) analytic ionic forces
///        vs an offline CP2K Quickstep twin.
///
/// Sibling of `gpw-d3-forces` (isolated, zero damping); this one exercises the
/// **rational (Becke-Johnson) damping** path on a **periodic** cell, so the
/// dispersion lattice sum and its gradient are both in play. Hard pins from
/// offline CP2K 2026.1 `TYPE DFTD3(BJ)`:
///   1. `E_disp` — CP2K's printed "Dispersion energy".
///   2. `F_disp` — CP2K's F(D3BJ) − F(plain) per atom. D3 is geometry-only, so
///      the SCF density is unchanged and the difference is exactly −∇E_disp.
///   3. Total product forces vs CP2K's `FORCES|` table.
///   4. Translation invariance of the Atomiverse force field (Σ_A F_A ≈ 0).
/// Wiring: the product path's F(D3BJ) − F(plain) must equal the standalone
/// D3(BJ) gradient. The analytic-vs-multi-SCF-FD gradient identity lives in
/// tests/cpu/gpw/test_cpu_gpw_forces.cpp (`AnalyticMatchesFdWithD3Bj`).
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
constexpr double kEDispCp2k = -0.00137627769481; // Ha, "Dispersion energy"
constexpr double kEDispTol = 1.0e-8;             // Ha

// CP2K FORCES| table for the D3(BJ) run [Ha/Bohr], order O H H O H H.
constexpr double kForceD3BjCp2k[6][3] = {
    {-8.18231267E-02, 3.13886473E-08, -7.96369314E-09},
    {4.08872662E-02, 1.13178277E-08, 1.31270247E-01},
    {4.08872656E-02, -3.52097593E-08, -1.31270237E-01},
    {-8.05124430E-02, -1.50606809E-08, 2.18547987E-08},
    {3.75636473E-02, 1.02088685E-08, 1.35137187E-01},
    {3.75636570E-02, -2.12332662E-09, -1.35137209E-01},
};
// CP2K FORCES| table for the plain (no dispersion) periodic run [Ha/Bohr].
constexpr double kForcePlainCp2k[6][3] = {
    {-8.19227425E-02, 3.13886469E-08, -7.96370458E-09},
    {4.08225109E-02, 1.13178281E-08, 1.31264307E-01},
    {4.08225103E-02, -3.52097601E-08, -1.31264297E-01},
    {-8.04237663E-02, -1.50606797E-08, 2.18548003E-08},
    {3.76338722E-02, 1.02088694E-08, 1.35133732E-01},
    {3.76338819E-02, -2.12332661E-09, -1.35133754E-01},
};

// Difference of two independently converged CP2K SCFs (EPS_SCF 1e-7), so it
// carries that solver's force noise: measured floor 1.2e-6 Ha/Bohr against a
// max|F_disp| ~ 1.0e-4 signal.
constexpr double kFDispTol = 5.0e-6; // Ha/Bohr
// Bounded below by the oracle's own grid error: CP2K's `FORCES| Sum` on this
// twin is 5.43e-3 Ha/Bohr. Same justification as gpw-d3-forces.
constexpr double kFTotalTol = 1.0e-2; // Ha/Bohr
// Translation invariance. Exact for a lattice sum in the continuum, but grid
// collocation of the Gaussians breaks it (egg-box), so the residual is a
// discretisation measure. Measured here: 1.34e-3 Ha/Bohr at 300 Ry, against
// the same quantity printed by the oracle on the identical system —
// CP2K `FORCES| Sum` = 5.43e-3, i.e. Atomiverse is 4x tighter. The gate exists
// to catch the Martyna-Tuckerman placement bug class, whose signature is
// |sum F| ~ 1-2 Ha/Bohr (gpw-fd-phonon-si), so 3e-3 keeps ~500x separation
// while not fighting the grid. The isolated MT twin (gpw-d3-forces) holds the
// tighter 1e-3 and measures 2.2e-4.
constexpr double kNetForceTol = 3.0e-3; // Ha/Bohr
constexpr double kWiringTol = 1.0e-7;   // Ha/Bohr — same code both sides
constexpr double kWallBudget = 60.0;    // s

} // namespace

int main() {
    using clock = std::chrono::steady_clock;

    constexpr double L = 12.0; // Å, PERIODIC XYZ
    const int z[6] = {8, 1, 1, 8, 1, 1};
    // Periodic cell: minimum-image wrapping makes the absolute placement
    // immaterial, so this keeps CP2K's coordinates verbatim (unlike the
    // isolated MT twin in gpw-d3-forces, which must sit inside the cell).
    const double xs[6] = {0.000000, 0.758602, 0.758602,
                          3.000000, 3.758602, 3.758602};
    const double ys[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double zs[6] = {0.000000, 0.504284, -0.504284,
                          0.000000, 0.504284, -0.504284};

    // ── (1) Standalone D3(BJ): energy + gradient vs the CP2K pins ─────────
    std::vector<int> Zv(6);
    std::vector<std::array<double, 3>> pos_bohr(6);
    for (int i = 0; i < 6; ++i) {
        Zv[static_cast<std::size_t>(i)] = z[i];
        pos_bohr[static_cast<std::size_t>(i)] = {
            xs[i] * kAng2Bohr, ys[i] * kAng2Bohr, zs[i] * kAng2Bohr};
    }
    const double cell_b[3] = {L * kAng2Bohr, L * kAng2Bohr, L * kAng2Bohr};
    // PBE-D3(BJ): s6=1, s8=0.7875, a1=0.4289, a2=4.4407 Bohr; CP2K CN form and
    // CP2K's 2 x R_CUTOFF pair cutoff (kD3Cp2kPairCutoffBohr).
    const cpu::gpw::D3Result d3 = cpu::gpw::dftd3_pbe_bj(
        Zv, pos_bohr, cell_b, /*periodic=*/true, cpu::gpw::D3Params{},
        cpu::gpw::kD3Cp2kPairCutoffBohr, /*s6=*/1.0, /*s8=*/0.7875,
        /*a1=*/0.4289, /*a2=*/4.4407, /*cp2k_cn=*/true,
        /*calculate_c9=*/false);

    if (!std::isfinite(d3.energy)
        || std::abs(d3.energy - kEDispCp2k) > kEDispTol) {
        std::printf("FAIL: |E_disp-cp2k|=%.6e Ha  got=%.14e  want=%.14e "
                    "(tol %.1e)\n",
                    std::abs(d3.energy - kEDispCp2k), d3.energy, kEDispCp2k,
                    kEDispTol);
        return 1;
    }

    double max_fdisp_err = 0.0;
    double max_fdisp = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double want = kForceD3BjCp2k[i][k] - kForcePlainCp2k[i][k];
            const double got = d3.forces[static_cast<std::size_t>(i)]
                                        [static_cast<std::size_t>(k)];
            max_fdisp_err = std::max(max_fdisp_err, std::abs(got - want));
            max_fdisp = std::max(max_fdisp, std::abs(want));
        }
    }
    if (max_fdisp_err > kFDispTol) {
        std::printf("FAIL: max|F_disp - (F_d3bj-F_plain)_cp2k|=%.3e > %.1e "
                    "Ha/Bohr (signal max|F_disp|=%.3e)\n",
                    max_fdisp_err, kFDispTol, max_fdisp);
        return 1;
    }
    std::printf("D3(BJ) standalone: E_disp=%.14e Ha (|d|=%.2e)  "
                "max|F_disp-cp2k|=%.3e Ha/Bohr (signal %.3e)\n",
                d3.energy, std::abs(d3.energy - kEDispCp2k), max_fdisp_err,
                max_fdisp);

    // ── (2) Product GPW with D3(BJ) forces ────────────────────────────────
    cpu::Frame frame(6);
    for (int i = 0; i < 6; ++i) {
        frame.span(kAtom_type)[i] = z[i];
        frame.span(kAtom_x)[i] = xs[i];
        frame.span(kAtom_y)[i] = ys[i];
        frame.span(kAtom_z)[i] = zs[i];
        frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
            frame.span(kAtom_fz)[i] = 0;
    }
    frame.box = cpu::Box::cubic(L);

    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

    cpu::gpw::GPWEvaluator eval({O, H});
    eval.cutoff_ry = 300.0; // CP2K CUTOFF 300
    eval.max_iter = 120;
    eval.tol = 1e-7; // CP2K EPS_SCF 1e-7
    eval.mix = 0.4;
    eval.xc = cpu::gpw::XcKind::PBE;
    eval.nonperiodic = false; // CP2K PERIODIC XYZ
    eval.d3bj = true;
    eval.compute_forces = true;
    eval.compute_stress = false;

    const auto t0 = clock::now();
    cpu::Evaluate driver;
    driver.on(evaluate::Phase::evaluate).add_ref(eval);
    driver.enable_timing(false);
    driver.build();
    driver.run(frame);
    const auto t1 = clock::now();
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();

    if (!eval.last_result().converged) {
        std::printf("FAIL: GPW SCF did not converge (iters=%d)\n",
                    eval.last_result().iters);
        return 1;
    }
    const double e_disp_product = eval.last_result().e_disp;
    if (!std::isfinite(e_disp_product)
        || std::abs(e_disp_product - kEDispCp2k) > kEDispTol) {
        std::printf("FAIL: product |E_disp-cp2k|=%.6e Ha  got=%.14e\n",
                    std::abs(e_disp_product - kEDispCp2k), e_disp_product);
        return 1;
    }

    // ── (3) Wiring, total forces, translation invariance ──────────────────
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const cpu::gpw::Kind& k = (z[i] == 8) ? O : H;
        const double p[3] = {xs[i] * kAng2Bohr, ys[i] * kAng2Bohr,
                             zs[i] * kAng2Bohr};
        atoms.push_back(cpu::gpw::GpwAtom(p, k.basis, k.pot, z[i]));
    }

    cpu::gpw::GpwDft method_bj;
    method_bj.cell[0] = method_bj.cell[1] = method_bj.cell[2] = L * kAng2Bohr;
    method_bj.cutoff_ry = 300.0;
    method_bj.n_occ = 8; // 2 x (q6 + q1 + q1) = 16 valence e-
    method_bj.max_iter = 120;
    method_bj.tol = 1e-7;
    method_bj.mix = 0.4;
    method_bj.xc = cpu::gpw::XcKind::PBE;
    method_bj.nonperiodic = false;
    method_bj.local_pp_on_grid = true;
    method_bj.d3bj = true;
    method_bj.build(atoms);
    const auto F_on =
        cpu::gpw::gpw_forces_from_scf(atoms, method_bj, eval.last_result());

    cpu::gpw::GpwDft plain; // GpwDft is non-copyable — build the twin fresh
    plain.cell[0] = plain.cell[1] = plain.cell[2] = L * kAng2Bohr;
    plain.cutoff_ry = 300.0;
    plain.n_occ = 8;
    plain.max_iter = 120;
    plain.tol = 1e-7;
    plain.mix = 0.4;
    plain.xc = cpu::gpw::XcKind::PBE;
    plain.nonperiodic = false;
    plain.local_pp_on_grid = true;
    plain.d3bj = false;
    plain.build(atoms);
    const auto F_plain = cpu::gpw::gpw_forces_analytic(atoms, plain);

    double max_wire = 0.0;
    double max_total_err = 0.0;
    double net[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double on =
                F_on[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
            const double pl = F_plain[static_cast<std::size_t>(i)]
                                     [static_cast<std::size_t>(k)];
            if (!std::isfinite(on) || !std::isfinite(pl)) {
                std::printf("FAIL: non-finite force atom %d axis %d\n", i, k);
                return 1;
            }
            const double f_disp = d3.forces[static_cast<std::size_t>(i)]
                                           [static_cast<std::size_t>(k)];
            max_wire = std::max(max_wire, std::abs((on - pl) - f_disp));
            max_total_err =
                std::max(max_total_err, std::abs(on - kForceD3BjCp2k[i][k]));
            net[k] += on;
        }
    }
    const double net_norm =
        std::sqrt(net[0] * net[0] + net[1] * net[1] + net[2] * net[2]);
    if (net_norm > kNetForceTol) {
        std::printf("FAIL: |sum F| = %.3e > %.1e Ha/Bohr — periodic force "
                    "field is not translation invariant\n",
                    net_norm, kNetForceTol);
        return 1;
    }
    if (max_wire > kWiringTol) {
        std::printf("FAIL: wiring max|(F_d3bj-F_plain)-F_standalone|=%.3e > "
                    "%.1e Ha/Bohr\n",
                    max_wire, kWiringTol);
        return 1;
    }
    if (max_total_err > kFTotalTol) {
        std::printf("FAIL: max|F_atv-F_cp2k|=%.3e > %.1e Ha/Bohr\n",
                    max_total_err, kFTotalTol);
        return 1;
    }
    if (wall_s > kWallBudget) {
        std::printf("FAIL: wall=%.2f s exceeds %.0f s e2e budget\n", wall_s,
                    kWallBudget);
        return 1;
    }

    std::printf("PASS: gpw-d3bj-forces  |dE_disp|=%.2e Ha  "
                "max|F_disp-cp2k|=%.3e  max|F_total-cp2k|=%.3e  "
                "max|wiring|=%.3e  |sum F|=%.3e Ha/Bohr  iters=%d  "
                "wall=%.2f s\n",
                std::abs(e_disp_product - kEDispCp2k), max_fdisp_err,
                max_total_err, max_wire, net_norm, eval.last_result().iters,
                wall_s);
    return 0;
}

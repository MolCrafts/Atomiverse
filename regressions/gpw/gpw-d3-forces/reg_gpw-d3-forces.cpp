/// @file regressions/gpw/gpw-d3-forces/reg_gpw-d3-forces.cpp
/// @brief END-TO-END: isolated water-dimer GPW PBE-D3 (zero damping) analytic
///        ionic forces vs an offline CP2K Quickstep twin.
///
/// Hard pins (offline CP2K 2026.1, `TYPE DFTD3` / `REFERENCE_FUNCTIONAL PBE`):
///   1. `E_disp` — CP2K's printed "Dispersion energy".
///   2. `F_disp` — CP2K's F(D3) − F(plain) per atom, i.e. the dispersion
///      contribution to the ionic forces, isolated from the electronic part by
///      running the same geometry twice (D3 is geometry-only, so the SCF
///      density is unchanged and the difference is exactly −∇E_disp).
///   3. Total product forces vs CP2K's `FORCES| Atomic forces` table.
/// Wiring: the product path's F(D3) − F(plain) must equal the standalone D3
/// force. The analytic-vs-multi-SCF-FD gradient identity lives in
/// tests/cpu/gpw/test_cpu_gpw_forces.cpp (`AnalyticMatchesFdWithD3`) — 6N SCFs
/// on this cell exceed the <= 60 s regression budget.
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
constexpr double kEDispCp2k = -0.00081399213205; // Ha, "Dispersion energy"
constexpr double kEDispTol = 1.0e-8;             // Ha

// CP2K FORCES| table for the D3 run [Ha/Bohr], atom order O H H O H H.
constexpr double kForceD3Cp2k[6][3] = {
    {-8.13880434E-02, 3.39382525E-08, -7.48922451E-09},
    {4.05546400E-02, 1.08883797E-08, 1.31321564E-01},
    {4.05546403E-02, -3.70556942E-08, -1.31321555E-01},
    {-7.98421128E-02, -1.52539175E-08, 2.13908200E-08},
    {3.73504355E-02, 9.81842421E-09, 1.35158735E-01},
    {3.73504444E-02, -1.84099666E-09, -1.35158757E-01},
};
// CP2K FORCES| table for the plain (no dispersion) run [Ha/Bohr].
constexpr double kForcePlainCp2k[6][3] = {
    {-8.15048974E-02, 3.39382577E-08, -7.48922385E-09},
    {4.06415374E-02, 1.08883798E-08, 1.31287871E-01},
    {4.06415377E-02, -3.70556926E-08, -1.31287862E-01},
    {-8.00748724E-02, -1.52539157E-08, 2.13908078E-08},
    {3.74383448E-02, 9.81842284E-09, 1.35165239E-01},
    {3.74383537E-02, -1.84099712E-09, -1.35165261E-01},
};

// F_disp pin tolerance. The pin is a difference of two independently converged
// CP2K SCF runs (EPS_SCF 1e-7), so it carries that solver's force noise; the
// measured floor on this twin is ~1.3e-6 Ha/Bohr against a signal of
// max|F_disp| ~ 2.3e-4 Ha/Bohr (>= 40x margin at this tolerance).
constexpr double kFDispTol = 5.0e-6; // Ha/Bohr
// Total-force tolerance vs CP2K. This comparison is bounded from below by the
// *oracle's* own grid error: CP2K's printed `FORCES| Sum` on this twin is
// 5.42e-3 Ha/Bohr, i.e. its force field is not translation invariant at
// CUTOFF 300 / REL_CUTOFF 40. Atomiverse's own net force on the same system is
// 2.2e-4 (300 Ry) shrinking to 7.6e-5 (500 Ry), and the raw component
// disagreement falls 7.5e-3 -> 3.2e-3 over the same range — the signature of
// discretisation on both sides, not a defect. 1e-2 is therefore the honest
// component tolerance here; the tight, physically meaningful checks are the
// dispersion pins above and the net-force gate below.
constexpr double kFTotalTol = 1.0e-2; // Ha/Bohr
// Translation invariance of the Atomiverse force field: for an isolated system
// Sum_A F_A must vanish. This is the gate that catches Martyna-Tuckerman
// placement bugs — an off-centre molecule (clipped cluster density) gives
// |Sum F| of order 1 Ha/Bohr while the energy still looks plausible, which is
// how gpw-fd-phonon-si silently produced garbage phonons.
constexpr double kNetForceTol = 1.0e-3; // Ha/Bohr
// Product wiring: F(d3) − F(plain) inside Atomiverse against its own
// standalone D3 gradient (same code, so only SCF residual enters).
constexpr double kWiringTol = 1.0e-7; // Ha/Bohr
constexpr double kWallBudget = 60.0;  // s

} // namespace

int main() {
    using clock = std::chrono::steady_clock;

    constexpr double L = 12.0; // Å, PERIODIC NONE (MT) in CP2K
    const int z[6] = {8, 1, 1, 8, 1, 1};
    // CP2K's coordinates, translated to the cell centre. An isolated system's
    // absolute position is physically irrelevant (CP2K's own MT solver recentres
    // internally), but Atomiverse's cluster boundary condition is not
    // translation invariant, so the molecule must sit inside the cell — see
    // GpwDft::build's MT margin guard and REFERENCE.md "Placement".
    constexpr double kShiftX = 6.0 - 1.5; // centre the O–O axis on x = 6 Å
    constexpr double kShiftYZ = 6.0;
    const double xs[6] = {0.000000 + kShiftX, 0.758602 + kShiftX,
                          0.758602 + kShiftX, 3.000000 + kShiftX,
                          3.758602 + kShiftX, 3.758602 + kShiftX};
    const double ys[6] = {kShiftYZ, kShiftYZ, kShiftYZ,
                          kShiftYZ, kShiftYZ, kShiftYZ};
    const double zs[6] = {0.000000 + kShiftYZ, 0.504284 + kShiftYZ,
                          -0.504284 + kShiftYZ, 0.000000 + kShiftYZ,
                          0.504284 + kShiftYZ, -0.504284 + kShiftYZ};

    // ── (1) Standalone D3: energy + gradient vs the CP2K pins ─────────────
    std::vector<int> Zv(6);
    std::vector<std::array<double, 3>> pos_bohr(6);
    for (int i = 0; i < 6; ++i) {
        Zv[static_cast<std::size_t>(i)] = z[i];
        pos_bohr[static_cast<std::size_t>(i)] = {
            xs[i] * kAng2Bohr, ys[i] * kAng2Bohr, zs[i] * kAng2Bohr};
    }
    const double cell_b[3] = {L * kAng2Bohr, L * kAng2Bohr, L * kAng2Bohr};
    const cpu::gpw::D3Result d3 =
        cpu::gpw::dftd3_pbe(Zv, pos_bohr, cell_b, /*periodic=*/false,
                            cpu::gpw::D3Params{}, /*rcut_bohr=*/40.0,
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
            const double want = kForceD3Cp2k[i][k] - kForcePlainCp2k[i][k];
            const double got =
                d3.forces[static_cast<std::size_t>(i)]
                         [static_cast<std::size_t>(k)];
            max_fdisp_err = std::max(max_fdisp_err, std::abs(got - want));
            max_fdisp = std::max(max_fdisp, std::abs(want));
        }
    }
    if (max_fdisp_err > kFDispTol) {
        std::printf("FAIL: max|F_disp - (F_d3-F_plain)_cp2k|=%.3e > %.1e "
                    "Ha/Bohr (signal max|F_disp|=%.3e)\n",
                    max_fdisp_err, kFDispTol, max_fdisp);
        return 1;
    }
    std::printf("D3 standalone: E_disp=%.14e Ha (|d|=%.2e)  "
                "max|F_disp-cp2k|=%.3e Ha/Bohr (signal %.3e)\n",
                d3.energy, std::abs(d3.energy - kEDispCp2k), max_fdisp_err,
                max_fdisp);

    // ── (2) Product GPW with D3 forces ────────────────────────────────────
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
    eval.nonperiodic = true; // CP2K PERIODIC NONE + PSOLVER MT
    eval.d3 = true;
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

    // ── (3) Wiring + total forces ─────────────────────────────────────────
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const cpu::gpw::Kind& k = (z[i] == 8) ? O : H;
        const double p[3] = {xs[i] * kAng2Bohr, ys[i] * kAng2Bohr,
                             zs[i] * kAng2Bohr};
        atoms.push_back(cpu::gpw::GpwAtom(p, k.basis, k.pot, z[i]));
    }

    cpu::gpw::GpwDft method_d3;
    method_d3.cell[0] = method_d3.cell[1] = method_d3.cell[2] = L * kAng2Bohr;
    method_d3.cutoff_ry = 300.0;
    method_d3.n_occ = 8; // 2 x (q6 + q1 + q1) = 16 valence e-
    method_d3.max_iter = 120;
    method_d3.tol = 1e-7;
    method_d3.mix = 0.4;
    method_d3.xc = cpu::gpw::XcKind::PBE;
    method_d3.nonperiodic = true;
    method_d3.local_pp_on_grid = true;
    method_d3.d3 = true;
    method_d3.build(atoms);
    const auto F_on =
        cpu::gpw::gpw_forces_from_scf(atoms, method_d3, eval.last_result());

    cpu::gpw::GpwDft plain; // GpwDft is non-copyable — build the twin fresh
    plain.cell[0] = plain.cell[1] = plain.cell[2] = L * kAng2Bohr;
    plain.cutoff_ry = 300.0;
    plain.n_occ = 8;
    plain.max_iter = 120;
    plain.tol = 1e-7;
    plain.mix = 0.4;
    plain.xc = cpu::gpw::XcKind::PBE;
    plain.nonperiodic = true;
    plain.local_pp_on_grid = true;
    plain.d3 = false;
    plain.build(atoms);
    const auto F_plain = cpu::gpw::gpw_forces_analytic(atoms, plain);

    double max_wire = 0.0;
    double max_total_err = 0.0;
    double net[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double on = F_on[static_cast<std::size_t>(i)]
                                  [static_cast<std::size_t>(k)];
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
                std::max(max_total_err, std::abs(on - kForceD3Cp2k[i][k]));
            net[k] += on;
        }
    }
    const double net_norm =
        std::sqrt(net[0] * net[0] + net[1] * net[1] + net[2] * net[2]);
    if (net_norm > kNetForceTol) {
        std::printf("FAIL: |sum F| = %.3e > %.1e Ha/Bohr — the isolated-system "
                    "force field is not translation invariant (check MT "
                    "placement / cell margin)\n",
                    net_norm, kNetForceTol);
        return 1;
    }
    if (max_wire > kWiringTol) {
        std::printf("FAIL: wiring max|(F_d3-F_plain)-F_standalone|=%.3e > "
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

    std::printf("PASS: gpw-d3-forces  |dE_disp|=%.2e Ha  "
                "max|F_disp-cp2k|=%.3e  max|F_total-cp2k|=%.3e  "
                "max|wiring|=%.3e  |sum F|=%.3e Ha/Bohr  iters=%d  "
                "wall=%.2f s\n",
                std::abs(e_disp_product - kEDispCp2k), max_fdisp_err,
                max_total_err, max_wire, net_norm, eval.last_result().iters,
                wall_s);
    return 0;
}

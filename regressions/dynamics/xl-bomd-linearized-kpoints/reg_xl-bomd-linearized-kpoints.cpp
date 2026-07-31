/// @file regressions/dynamics/xl-bomd-linearized-kpoints/reg_xl-bomd-linearized-kpoints.cpp
/// @brief END-TO-END: zero-SCF linearized (shadow-functional) XL-BOMD on the
///        multi-k path — periodic H₂ NVE (1×1×2, 2 k), three legs:
///        A = SCF-XL multi-k (shipped product, anchor), B = linearized
///        multi-k RKS (product under test; warm iters == 0; per-step shadow
///        E_pot gated against the CP2K 2026.1 converged-BOMD `.ener` trace),
///        C = linearized multi-k UKS with n_alpha = n_beta = 1 (dual-spin
///        machinery via the spin-degeneracy identity: C must reproduce B).
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_GPW + MD.
/// Exit non-zero on failure.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/electronic/xl_bomd_force.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

using namespace ATV;

namespace {

constexpr double kBohr2Ang = 0.52917720859;
constexpr double kLBohr = 12.0;
constexpr double kR0Bohr = 1.0; // compressed bond (gpw-kpoint-forces twin)
constexpr double kDt = 0.5;     // fs
constexpr int kProd = 4;        // production steps (CP2K STEPS 4)

/// CP2K 2026.1 `.ener` Pot rows [Ha] — offline oracle
/// (tools/cp2k_oracle/inputs/xl-bomd-linearized-kpoints, re-run 2026-07-31;
/// converged-SCF k-point BOMD = the BO surface the shadow must track).
constexpr double kCp2kPot[kProd + 1] = {
    -1.074050184, -1.097682104, -1.125544577, -1.130531455, -1.121507812};

cpu::Frame make_frame() {
    cpu::Frame frame(2);
    const double L_ang = kLBohr * kBohr2Ang;
    const double x0[2] = {(kLBohr / 2.0 - 0.5 * kR0Bohr) * kBohr2Ang,
                          (kLBohr / 2.0 + 0.5 * kR0Bohr) * kBohr2Ang};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_mass)[i] = 1.008;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_x)[i] = x0[i];
        frame.span(kAtom_y)[i] = L_ang / 2.0;
        frame.span(kAtom_z)[i] = L_ang / 2.0;
        frame.span(kAtom_vx)[i] = 0.0;
        frame.span(kAtom_vy)[i] = 0.0;
        frame.span(kAtom_vz)[i] = 0.0;
    }
    // Antisymmetric stretch [Å/fs] — no COM translation (CP2K &VELOCITY).
    frame.span(kAtom_vx)[0] = -0.0010;
    frame.span(kAtom_vx)[1] = 0.0010;
    frame.box = cpu::Box::cubic(L_ang);
    return frame;
}

cpu::gpw::GPWEvaluator make_gpw(int n_alpha, int n_beta) {
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")};
    // CP2K twin settings: CUTOFF 300, EPS_SCF 1e-7, ALPHA 0.3, PADE.
    cpu::gpw::GPWEvaluator gpw(
        /*kinds*/ {H}, /*cutoff_ry*/ 300.0, /*max_iter*/ 60, /*tol*/ 1e-7,
        /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ false,
        /*compute_forces*/ true);
    gpw.k_mesh[0] = 1;
    gpw.k_mesh[1] = 1;
    gpw.k_mesh[2] = 2;
    gpw.k_time_reversal = false; // two sampled k (CP2K FULL_GRID)
    gpw.n_alpha = n_alpha;
    gpw.n_beta = n_beta;
    return gpw;
}

struct LegResult {
    std::vector<double> pot_ha; // per-step E_pot [Ha], index 0 = after seed
    std::vector<double> etot_ev; // per-step E_pot + E_kin [eV]
    double r_hh = 0.0;           // final H–H distance [Å]
    int seed_iters = 0;
    int max_warm_iters = 0;
    bool warm_all_converged = true;
    int n_kpts = 0;
    double wall_s = 0.0;
};

/// One 1-seed + kProd-step NVE leg. `linearized` picks the wrapper ctor.
bool run_leg(const char* name, int n_alpha, int n_beta, bool linearized,
             LegResult& out) {
    cpu::Frame frame = make_frame();
    cpu::gpw::GPWEvaluator base = make_gpw(n_alpha, n_beta);
    cpu::electronic::XlBomdForce eval =
        linearized
            ? cpu::electronic::XlBomdForce{std::move(base), 5,
                                           cpu::electronic::XlBomdLinearized{}}
            : cpu::electronic::XlBomdForce{std::move(base), 5};

    const auto t0 = std::chrono::steady_clock::now();
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    sim.set_evaluator_ref(eval);
    sim.enable_timing(false);
    sim.build();
    // Prime the MD component lifecycle with a 0-step run: component build()
    // fires on the FIRST run() (continuation contract), and
    // GPWEvaluator::build deliberately resets the XL histories — so it must
    // happen BEFORE the seeding single point, not after (or the seed is
    // silently wiped and warm step 1 degrades to a full SCF).
    if (sim.run(frame, 0) != 0) {
        std::printf("FAIL[%s]: 0-step priming run stepped\n", name);
        return false;
    }
    // Step 0: single point at t=0 — the full-SCF cold seed. It writes
    // energy + forces (the VV bootstrap: CP2K MD evaluates forces at step 0
    // before the first VV step, so ATV step n aligns with `.ener` row n)
    // and advances every per-k XL history once, so MD step 1 is already a
    // warm shadow step ("1 cold seed + 4 production").
    cpu::single_point(frame, eval);
    if (!(eval.last_result().iters > 0) || !eval.last_result().converged) {
        std::printf("FAIL[%s]: seed must be full SCF (iters=%d conv=%d)\n",
                    name, eval.last_result().iters,
                    static_cast<int>(eval.last_result().converged));
        return false;
    }
    out.seed_iters = eval.last_result().iters;
    out.n_kpts = eval.last_result().n_kpts;
    if (out.n_kpts < 2) {
        std::printf("FAIL[%s]: expected n_kpts >= 2, got %d\n", name,
                    out.n_kpts);
        return false;
    }
    if (static_cast<int>(eval.last_result().P_k.size()) != out.n_kpts) {
        std::printf("FAIL[%s]: P_k payload missing after multi-k seed\n",
                    name);
        return false;
    }
    const double pe0 = eval.last_result().e_total * units::kHartreeInEv;
    // KE(t=0) from the initial velocities (single_point computes no KE);
    // 1 eV = 1/force_to_accel amu Å²/fs².
    double ke0 = 0.0;
    for (int i = 0; i < 2; ++i) {
        const double vx = static_cast<double>(frame.span(kAtom_vx)[i]);
        ke0 += 0.5 * 1.008 * vx * vx / units::force_to_accel;
    }
    if (!std::isfinite(pe0)) {
        std::printf("FAIL[%s]: non-finite seed energy PE=%g\n", name, pe0);
        return false;
    }
    out.pot_ha.push_back(eval.last_result().e_total);
    out.etot_ev.push_back(pe0 + ke0);

    for (int s = 0; s < kProd; ++s) {
        if (sim.run(frame, 1) != 1) {
            std::printf("FAIL[%s]: production step %d incomplete\n", name, s);
            return false;
        }
        const double pe = frame.meta.get(kEnergy);
        const double ke = frame.meta.get(kKinetic);
        if (!std::isfinite(pe) || !std::isfinite(ke) || !(ke >= 0.0)) {
            std::printf("FAIL[%s]: non-finite energy at step %d PE=%g KE=%g\n",
                        name, s, pe, ke);
            return false;
        }
        out.max_warm_iters =
            std::max(out.max_warm_iters, eval.last_result().iters);
        out.warm_all_converged =
            out.warm_all_converged && eval.last_result().converged;
        out.pot_ha.push_back(pe / units::kHartreeInEv);
        out.etot_ev.push_back(pe + ke);
        for (int i = 0; i < 2; ++i)
            if (!std::isfinite(static_cast<double>(frame.span(kAtom_x)[i]))
                || !std::isfinite(static_cast<double>(frame.span(kAtom_y)[i]))
                || !std::isfinite(
                    static_cast<double>(frame.span(kAtom_z)[i]))) {
                std::printf("FAIL[%s]: non-finite coordinate step %d atom "
                            "%d\n",
                            name, s, i);
                return false;
            }
    }
    out.wall_s = std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0)
                     .count();

    const double dx = static_cast<double>(frame.span(kAtom_x)[1]
                                          - frame.span(kAtom_x)[0]);
    const double dy = static_cast<double>(frame.span(kAtom_y)[1]
                                          - frame.span(kAtom_y)[0]);
    const double dz = static_cast<double>(frame.span(kAtom_z)[1]
                                          - frame.span(kAtom_z)[0]);
    out.r_hh = std::sqrt(dx * dx + dy * dy + dz * dz);
    return true;
}

double worst_rel_drift(const LegResult& leg) {
    const double e0 = leg.etot_ev.front();
    const double scale = std::max(1.0, std::fabs(e0));
    double worst = 0.0;
    for (double e : leg.etot_ev)
        worst = std::max(worst, std::fabs(e - e0) / scale);
    return worst;
}

} // namespace

int main() {
    // Tolerances (all measured values recorded in REFERENCE.md):
    // CP2K per-step Pot band [Ha] — engine model band 4.07e-4 (documented in
    // gpw-bomd-kpoints) + XL/shadow trajectory margin.
    constexpr double cp2k_band_ha = 2e-3;
    // A/B: linearized shadow PE vs SCF-XL PE per step [Ha] — the shadow
    // functional tracks the SCF-XL surface to O((D−n)²) on this window.
    constexpr double ab_band_ha = 1e-3;
    // C/B spin-degeneracy identity per step [Ha] (dual-spin path arithmetic
    // vs closed-shell path; seeds converge independently to tol 1e-7).
    constexpr double uks_band_ha = 1e-4;
    // NVE conserved-quantity envelope (family policy; CP2K itself shows
    // 4.4e-3 relative on this aggressive compressed-bond window).
    constexpr double drift_tol = 0.05;
    // Soft H–H window [Å] after production.
    constexpr double rmin = 0.4;
    constexpr double rmax = 1.5;

    LegResult A, B, C;
    if (!run_leg("A scf-xl rks-k", 0, 0, /*linearized=*/false, A))
        return 1;
    if (!run_leg("B linearized rks-k", 0, 0, /*linearized=*/true, B))
        return 1;
    if (!run_leg("C linearized uks-k", 1, 1, /*linearized=*/true, C))
        return 1;

    // ── B (product under test) gates ────────────────────────────────────
    if (B.max_warm_iters != 0) {
        std::printf("FAIL: linearized warm steps did iterative SCF "
                    "(max iters=%d, want 0)\n",
                    B.max_warm_iters);
        return 1;
    }
    if (!B.warm_all_converged) {
        std::printf("FAIL: linearized warm step not marked converged\n");
        return 1;
    }
    if (C.max_warm_iters != 0 || !C.warm_all_converged) {
        std::printf("FAIL: UKS linearized warm steps (iters=%d conv=%d)\n",
                    C.max_warm_iters, static_cast<int>(C.warm_all_converged));
        return 1;
    }
    // A stays the SCF-converged product (anchor sanity).
    if (A.max_warm_iters <= 0) {
        std::printf("FAIL: SCF-XL anchor should iterate (iters=%d)\n",
                    A.max_warm_iters);
        return 1;
    }

    double worst_cp2k = 0.0, worst_ab = 0.0, worst_uks = 0.0;
    for (int s = 0; s <= kProd; ++s) {
        const std::size_t k = static_cast<std::size_t>(s);
        worst_cp2k = std::max(worst_cp2k,
                              std::fabs(B.pot_ha[k] - kCp2kPot[s]));
        worst_ab = std::max(worst_ab, std::fabs(B.pot_ha[k] - A.pot_ha[k]));
        worst_uks = std::max(worst_uks, std::fabs(C.pot_ha[k] - B.pot_ha[k]));
    }
    const double drift_a = worst_rel_drift(A);
    const double drift_b = worst_rel_drift(B);
    const double drift_c = worst_rel_drift(C);

    std::printf("XL-BOMD linearized multi-k H2 NVE (K=5, nk=%d, 1 seed + %d "
                "steps):\n",
                B.n_kpts, kProd);
    std::printf("  leg A (SCF-XL):     seed_iters=%d warm_iters<=%d drift=%.3e "
                "rHH=%.3f wall=%.2fs\n",
                A.seed_iters, A.max_warm_iters, drift_a, A.r_hh, A.wall_s);
    std::printf("  leg B (linearized): seed_iters=%d warm_iters=%d  drift=%.3e "
                "rHH=%.3f wall=%.2fs\n",
                B.seed_iters, B.max_warm_iters, drift_b, B.r_hh, B.wall_s);
    std::printf("  leg C (UKS lin.):   seed_iters=%d warm_iters=%d  drift=%.3e "
                "rHH=%.3f wall=%.2fs\n",
                C.seed_iters, C.max_warm_iters, drift_c, C.r_hh, C.wall_s);
    for (int s = 0; s <= kProd; ++s)
        std::printf("  step %d  Pot[Ha]: cp2k=%.9f  A=%.9f  B=%.9f  C=%.9f\n",
                    s, kCp2kPot[s], A.pot_ha[static_cast<std::size_t>(s)],
                    B.pot_ha[static_cast<std::size_t>(s)],
                    C.pot_ha[static_cast<std::size_t>(s)]);
    std::printf("  worst |B - cp2k| = %.3e Ha (band %.1e)\n", worst_cp2k,
                cp2k_band_ha);
    std::printf("  worst |B - A|    = %.3e Ha (band %.1e)\n", worst_ab,
                ab_band_ha);
    std::printf("  worst |C - B|    = %.3e Ha (band %.1e)\n", worst_uks,
                uks_band_ha);

    if (!(worst_cp2k <= cp2k_band_ha)) {
        std::printf("FAIL: linearized shadow E_pot left the CP2K BO band\n");
        return 1;
    }
    if (!(worst_ab <= ab_band_ha)) {
        std::printf("FAIL: linearized vs SCF-XL A/B trace out of band\n");
        return 1;
    }
    if (!(worst_uks <= uks_band_ha)) {
        std::printf("FAIL: UKS spin-degeneracy identity violated\n");
        return 1;
    }
    if (!(drift_b <= drift_tol) || !(drift_c <= drift_tol)) {
        std::printf("FAIL: NVE drift envelope exceeded (B=%.3e C=%.3e "
                    "tol=%.2f)\n",
                    drift_b, drift_c, drift_tol);
        return 1;
    }
    for (const LegResult* leg : {&A, &B, &C})
        if (!(leg->r_hh > rmin) || !(leg->r_hh < rmax)) {
            std::printf("FAIL: H–H bond out of window r=%.3f Å (%.2f–%.2f)\n",
                        leg->r_hh, rmin, rmax);
            return 1;
        }

    std::printf("PASS: xl-bomd-linearized-kpoints (zero-SCF multi-k shadow "
                "XL NVE; CP2K-anchored)\n");
    return 0;
}

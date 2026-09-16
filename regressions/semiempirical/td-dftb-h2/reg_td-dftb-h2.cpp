/// @file regressions/semiempirical/td-dftb-h2/reg_td-dftb-h2.cpp
/// @brief Linear-response TD-DFTB (Casida) on SCC-DFTB2 — H2 (+ H2O windows).
///
/// oracle: literature (Niehaus et al., PRB 63, 085108 (2001)).
/// No citable H2/mio-1-1 TD-DFTB value; snapshot is a drift guard.
/// Public API only (`cpu::semiempirical::DFTB2::td_casida` + Frame).
/// Exit non-zero on failure. <= 60 s.
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"
#include "atomiverse/cpu/semiempirical/td_dftb2.h"

using namespace ATV;

// ── Self-consistency snapshot, NOT an oracle number ────────────────────
// H2 at r = 0.74 A, MioSketch (H onsite/U and the whole H-H SK + repulsive
// grid ARE transcribed mio-1-1). Triplet kernel uses published PBE W_ss(H)
// via spin_w_ha (HOAO collapse). One sigma_g -> sigma_u channel. These are
// Atomiverse's own converged values, pinned to catch silent drift — no DFTB+
// identity is claimed (no published TD-DFTB/mio-1-1 H2 value was located).
// Gap/S1 unchanged by W (Δq≡0; singlet kernel is γ). T1 re-pinned after
// published W_ss(H) (dftb-data-00-honesty, 2026-09-04); supersedes sketch-W
// 15.01024822 eV.
static constexpr double kH2GapEv = 15.44081455;
static constexpr double kH2SingletEv = 16.39162450;
static constexpr double kH2TripletEv = 13.67637744;
static constexpr double kSnapshotTolEv = 1.0e-6;

// Structural tolerances (see REFERENCE.md).
static constexpr double kZeroCouplingTolEv = 1.0e-10;
static constexpr double kSymmetryTolEv2 = 1.0e-12;
static constexpr double kChargeSumTolE = 1.0e-10;

static cpu::Frame make_h2(double r_ang) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = static_cast<StorageReal>(r_ang);
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    return f;
}

static cpu::Frame make_water() {
    cpu::Frame f(3);
    const double xs[3] = {0.0, 1.05, -0.26};
    const double ys[3] = {0.0, 0.0, 1.0};
    const int zs[3] = {8, 1, 1};
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_type)[i] = zs[i];
        f.span(kAtom_x)[i] = static_cast<StorageReal>(xs[i]);
        f.span(kAtom_y)[i] = static_cast<StorageReal>(ys[i]);
        f.span(kAtom_z)[i] = 0;
        f.span(kAtom_mass)[i] = (i == 0) ? 15.999 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

int main() {
    cpu::semiempirical::DFTB2 dftb;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    dftb.boundary = cpu::semiempirical::DftbBoundary::cluster;
    dftb.scc_tol = 1e-12;
    dftb.max_scc = 120;
    dftb.spin_w_ha = cpu::semiempirical::basis_data::kDftbSpinConstantsPbeHa;

    // ── H2: the single sigma_g -> sigma_u channel ──────────────────────
    cpu::Frame h2 = make_h2(0.74);
    const auto s = dftb.td_casida(h2, 1);
    const auto t =
        dftb.td_casida(h2, 1, cpu::semiempirical::TdDftbChannel::triplet);
    const auto bare = dftb.td_casida(
        h2, 1, cpu::semiempirical::TdDftbChannel::singlet, 0, 0, 0.0);

    std::printf("TD-DFTB H2 r=0.74 A: gap=%.8f  S1=%.8f  T1=%.8f eV\n",
                s.ks_gap_ev[0], s.energy_ev[0], t.energy_ev[0]);
    std::printf("  q_H1=%.10f  q_H2=%.10f e\n", s.transition_charge[0],
                s.transition_charge[1]);
    std::printf("  f_S1=%.6e  f_T1=%.6e (triplet dark)\n",
                s.oscillator_strength.empty() ? -1.0 : s.oscillator_strength[0],
                t.oscillator_strength.empty() ? -1.0
                                              : t.oscillator_strength[0]);
    if (s.oscillator_strength.empty() || s.oscillator_strength[0] < -1e-12) {
        std::printf("FAIL H2 singlet oscillator strength missing/negative\n");
        return 1;
    }
    if (t.oscillator_strength.empty()
        || std::fabs(t.oscillator_strength[0]) > 1e-12) {
        std::printf("FAIL H2 triplet must be dipole-dark (f≈0)\n");
        return 1;
    }

    if (s.n_pair != 1 || t.n_pair != 1) {
        std::printf("FAIL H2 must have exactly one occ->virt pair (got %d)\n",
                    s.n_pair);
        return 1;
    }
    if (!std::isfinite(s.energy_ev[0]) || !(s.energy_ev[0] > 0.0)) {
        std::printf("FAIL H2 singlet not finite/positive: %g\n",
                    s.energy_ev[0]);
        return 1;
    }
    if (!std::isfinite(t.energy_ev[0]) || !(t.energy_ev[0] > 0.0)) {
        std::printf("FAIL H2 triplet not finite/positive: %g\n",
                    t.energy_ev[0]);
        return 1;
    }
    // gamma_AB > 0 blue-shifts the singlet; W_A < 0 red-shifts the triplet.
    if (!(s.energy_ev[0] > s.ks_gap_ev[0])
        || !(t.energy_ev[0] < t.ks_gap_ev[0])) {
        std::printf("FAIL H2 ordering: T1=%.8f < gap=%.8f < S1=%.8f expected\n",
                    t.energy_ev[0], s.ks_gap_ev[0], s.energy_ev[0]);
        return 1;
    }

    // Decisive structural gate: coupling_scale = 0 must reproduce the bare
    // single-particle gap exactly.
    const double d_bare = std::fabs(bare.energy_ev[0] - bare.ks_gap_ev[0]);
    std::printf("  lambda=0: |omega - (eps_a - eps_i)| = %.3e eV\n", d_bare);
    if (d_bare > kZeroCouplingTolEv) {
        std::printf("FAIL H2 zero-coupling limit %.3e > %.1e eV\n", d_bare,
                    kZeroCouplingTolEv);
        return 1;
    }

    // sigma_g x sigma_u is ungerade: the transition charges are antisymmetric
    // and, for a symmetric two-level system, q = 1/(2 sqrt(1 - S12^2)) > 1/2.
    const double q1 = s.transition_charge[0];
    const double q2 = s.transition_charge[1];
    if (std::fabs(q1 + q2) > kChargeSumTolE || !(std::fabs(q1) > 0.5)) {
        std::printf("FAIL H2 transition charges q1=%.10f q2=%.10f\n", q1, q2);
        return 1;
    }

    // Stretching the bond closes the sigma_g/sigma_u splitting.
    cpu::Frame h2_long = make_h2(1.20);
    const auto s_long = dftb.td_casida(h2_long, 1);
    std::printf("  S1(1.20 A)=%.8f eV (must be below S1(0.74 A))\n",
                s_long.energy_ev[0]);
    if (!(s_long.energy_ev[0] < s.energy_ev[0])) {
        std::printf("FAIL H2 excitation did not red-shift on stretching\n");
        return 1;
    }

    // Snapshot pin (drift guard only — see REFERENCE.md).
    const double d_gap = std::fabs(s.ks_gap_ev[0] - kH2GapEv);
    const double d_s = std::fabs(s.energy_ev[0] - kH2SingletEv);
    const double d_t = std::fabs(t.energy_ev[0] - kH2TripletEv);
    std::printf("  snapshot |dgap|=%.3e |dS1|=%.3e |dT1|=%.3e eV\n", d_gap, d_s,
                d_t);
    if (d_gap > kSnapshotTolEv || d_s > kSnapshotTolEv
        || d_t > kSnapshotTolEv) {
        std::printf("FAIL H2 snapshot pin drifted beyond %.1e eV\n",
                    kSnapshotTolEv);
        return 1;
    }

    // ── H2O: a multi-pair Casida matrix (symmetry + sum rules) ─────────
    cpu::Frame h2o = make_water();
    const auto w = dftb.td_casida(h2o, 4);
    const auto w_bare = dftb.td_casida(
        h2o, 4, cpu::semiempirical::TdDftbChannel::singlet, 0, 0, 0.0);
    std::printf("TD-DFTB H2O: n_pair=%d  S1..S4 = %.6f %.6f %.6f %.6f eV\n",
                w.n_pair, w.energy_ev[0], w.energy_ev[1], w.energy_ev[2],
                w.energy_ev[3]);

    if (w.n_pair < 4) {
        std::printf("FAIL H2O pair space too small (%d)\n", w.n_pair);
        return 1;
    }
    double max_asym = 0.0;
    for (int p = 0; p < w.n_pair; ++p)
        for (int q = 0; q < w.n_pair; ++q) {
            const double d = std::fabs(
                w.casida_ev2[static_cast<std::size_t>(p) * w.n_pair + q]
                - w.casida_ev2[static_cast<std::size_t>(q) * w.n_pair + p]);
            if (d > max_asym)
                max_asym = d;
        }
    std::printf("  max |Omega_pq - Omega_qp| = %.3e eV^2\n", max_asym);
    if (max_asym > kSymmetryTolEv2) {
        std::printf("FAIL H2O Casida asymmetry %.3e > %.1e eV^2\n", max_asym,
                    kSymmetryTolEv2);
        return 1;
    }

    double max_qsum = 0.0;
    for (int p = 0; p < w.n_pair; ++p) {
        double sum = 0.0;
        for (int a = 0; a < w.n_atoms; ++a)
            sum += w.transition_charge[static_cast<std::size_t>(p) * w.n_atoms
                                       + a];
        if (std::fabs(sum) > max_qsum)
            max_qsum = std::fabs(sum);
    }
    std::printf("  max |sum_A q^ia_A| = %.3e e\n", max_qsum);
    if (max_qsum > kChargeSumTolE) {
        std::printf("FAIL H2O transition-charge sum rule %.3e > %.1e e\n",
                    max_qsum, kChargeSumTolE);
        return 1;
    }

    double max_bare = 0.0;
    for (std::size_t i = 0; i < w_bare.energy_ev.size(); ++i) {
        const double d = std::fabs(w_bare.energy_ev[i] - w_bare.ks_gap_ev[i]);
        if (d > max_bare)
            max_bare = d;
    }
    std::printf("  H2O lambda=0 max |omega - gap| = %.3e eV\n", max_bare);
    if (max_bare > kZeroCouplingTolEv) {
        std::printf("FAIL H2O zero-coupling limit %.3e > %.1e eV\n", max_bare,
                    kZeroCouplingTolEv);
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}

/// @file regressions/md/npt_pme_nacl/reg_npt_pme_nacl.cpp
/// @brief END-TO-END REGRESSION: rock-salt NaCl, Tosi–Fumi Born–Mayer–Huggins
///        + PME electrostatics, isotropic NPT at zero pressure.
///
/// Public composition only (headers + link atomiverse). Five checks:
///   1. static Madelung energy of the bare point-charge lattice (literature
///      constant),
///   2. pure-Coulomb 1/s scaling under uniform cell/coordinate scale,
///   3. **the virial identity** tr(σ) = −3V dE/dV for the FULL Hamiltonian,
///      taken by central difference against the engine's own `kStress` — the
///      one gate that does not need a literature value and the one that
///      catches a force field whose pressure is not the derivative of its
///      energy,
///   4. the model's mechanical equilibrium: P(a) = 0 within 1.5 % of the
///      experimental NaCl lattice constant, and the cohesive energy there
///      against the experimental lattice energy,
///   5. a 4 ps NPT run at P = 0: the cell relaxes to a lattice constant
///      within 3 % of experiment and the MTK extended Hamiltonian stays
///      inside a documented O(dt²) envelope.
///
/// See REFERENCE.md. Exits non-zero on failure. Requires ATV_BUILD_KSPACE.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/pme.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/pair_born.h"

using namespace ATV;

namespace {

// ── Geometry ───────────────────────────────────────────────────────────
constexpr double kA0Exp = 5.6402;      // NaCl lattice constant at 298 K [Å]
constexpr double kDExp = 0.5 * kA0Exp; // nearest-neighbour spacing [Å]
constexpr int kNSide = 4;              // 4³ = 64 ions (even ⇒ neutral)
constexpr int kN = kNSide * kNSide * kNSide;
constexpr double kMadelung = 1.747564594633182;

// Experimental static lattice energy of NaCl, per ion pair [eV].
constexpr double kLatticeEnergyExp = -7.94;

// 1 eV/Å³ = 160.21766208 GPa — printing only; every gate is in eV/Å³.
constexpr double kEvPerA3InGPa = 160.21766208;

// ── Interactions ───────────────────────────────────────────────────────
// Real-space cutoff must stay below L/2 for the minimum-image convention;
// L = 4·d ≈ 11.28 Å at the experimental spacing, so 5.0 Å leaves room for the
// cell to contract ~10 % before the convention would bite.
constexpr double kCut = 5.0;          // [Å]
constexpr double kAlpha = 3.5 / kCut; // [1/Å] — erfc(3.5) ≈ 7e-7
constexpr double kPmeSpacing = 0.5;   // [Å]
constexpr int kPmeOrder = 6;

// Type 0 = Na⁺, type 1 = Cl⁻.
constexpr int kNa = 0;
constexpr int kCl = 1;

/// @brief Tosi–Fumi NaCl in Atomiverse's `PairBorn` sign convention.
///
/// `PairBorn` evaluates A·exp((σ−r)/ρ) − C/r⁶ + D/r⁸ while Fumi–Tosi write
/// the r⁻⁸ term with a minus sign, so D enters negated. See REFERENCE.md for
/// the SI → eV/Å conversion of every row.
std::vector<cpu::BornTypeCoeff> tosi_fumi_nacl() {
    constexpr double rho = 0.317; // [Å]
    return {
        //      i     j        A [eV]      rho      sigma [Å]  C [eV Å⁶]  D [eV Å⁸]
        {kNa, kNa, 0.2637038, rho, 2.340, 0.6553585, -0.3114513},
        {kNa, kCl, 0.2109630, rho, 2.755, 4.3690564, -5.4176299},
        {kCl, kCl, 0.1582223, rho, 3.170, 45.2509408, -90.8139570},
    };
}

/// @brief Rock-salt NaCl at nearest-neighbour spacing @p d [Å].
cpu::Frame nacl(double d) {
    cpu::Frame f(kN);
    int idx = 0;
    for (int i = 0; i < kNSide; ++i)
        for (int j = 0; j < kNSide; ++j)
            for (int k = 0; k < kNSide; ++k) {
                const bool na = ((i + j + k) % 2 == 0);
                f.span(ATV::kAtom_x)[idx] = i * d;
                f.span(ATV::kAtom_y)[idx] = j * d;
                f.span(ATV::kAtom_z)[idx] = k * d;
                f.span(ATV::kAtom_charge)[idx] = na ? 1.0 : -1.0;
                f.span(ATV::kAtom_mass)[idx] = na ? 22.9898 : 35.453;
                f.span(ATV::kAtom_type)[idx] = na ? kNa : kCl;
                f.span(ATV::kAtom_tag)[idx] = idx + 1;
                ++idx;
            }
    f.box = cpu::Box::cubic(kNSide * d);
    return f;
}

/// @brief Bare point-charge electrostatics — the Madelung / scaling oracle.
double electrostatic_energy(cpu::Frame& frame) {
    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{kCut, /*skin=*/0.3});
    eval.set_evaluator(cpu::CoulLong{kAlpha, kCut});
    eval.set_evaluator(cpu::KSpacePME{kAlpha, kPmeSpacing, kPmeOrder});
    eval.build();
    eval.run(frame);
    return frame.meta.get(ATV::kEnergy);
}

struct Point {
    double energy;   ///< total potential energy [eV]
    double pressure; ///< tr(kStress)/(3V) [eV/Å³]
    double volume;   ///< [Å³]
};

/// @brief Full Hamiltonian single point: Born–Mayer–Huggins + Ewald/PME.
Point single_point(double d) {
    cpu::Frame f = nacl(d);
    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{kCut, /*skin=*/0.3});
    eval.set_evaluator(cpu::PairBorn{tosi_fumi_nacl(), kCut});
    eval.set_evaluator(cpu::CoulLong{kAlpha, kCut});
    eval.set_evaluator(cpu::KSpacePME{kAlpha, kPmeSpacing, kPmeOrder});
    eval.build();
    eval.run(f);
    const auto& s = f.meta.get(ATV::kStress);
    const double v = f.box.volume();
    return {f.meta.get(ATV::kEnergy), (s[0] + s[1] + s[2]) / (3.0 * v), v};
}

} // namespace

int main() {
    // ── (1) Madelung ───────────────────────────────────────────────────
    cpu::Frame frame = nacl(kDExp);
    const double e0 = electrostatic_energy(frame);
    const double e_ref = -0.5 * kN * kMadelung * units::coulomb / kDExp;
    const double madelung_rel = std::fabs(e0 / e_ref - 1.0);
    std::printf("NaCl PME Madelung: E = %.6f eV  ref = %.6f eV  |rel| = %.3e\n",
                e0, e_ref, madelung_rel);
    if (!(madelung_rel <= 5e-4)) {
        std::printf("FAIL: Madelung relative error %.3e > 5e-4\n",
                    madelung_rel);
        return 1;
    }

    // ── (2) Uniform scale s → E(s) = E(1)/s ────────────────────────────
    const double s = 0.99;
    cpu::Frame frame_s = nacl(kDExp * s);
    const double e_s = electrostatic_energy(frame_s);
    const double scale_rel = std::fabs(e_s * s / e0 - 1.0);
    std::printf("Uniform scale s=%.2f: E(s)*s/E(1) - 1 = %.3e\n", s, scale_rel);
    if (!(scale_rel <= 5e-4)) {
        std::printf("FAIL: scaling law broken (stale KSpace cell data?)\n");
        return 1;
    }

    // ── (3) Virial identity: tr(σ) = −3V dE/dV ─────────────────────────
    // Every coordinate and the cell scale together, so this is a pure
    // volume derivative at fixed fractional coordinates. Exact, not
    // asymptotic: no pair sits near the cutoff, so the pair set does not
    // change across the difference. A force field whose stress is not the
    // derivative of its energy fails here and nowhere else — and a barostat
    // driven by such a stress walks the cell to a volume the Hamiltonian
    // never asked for.
    {
        const double h = 1e-6;
        const Point p = single_point(kDExp * (1.0 + h));
        const Point m = single_point(kDExp * (1.0 - h));
        const Point c = single_point(kDExp);
        const double p_fd = -(p.energy - m.energy) / (p.volume - m.volume);
        const double rel = std::fabs(c.pressure - p_fd) / std::fabs(p_fd);
        std::printf("Virial identity at a=%.4f Å: P_vir = %+.9e  "
                    "P_fd = %+.9e eV/Å³  |rel| = %.3e\n",
                    kA0Exp, c.pressure, p_fd, rel);
        if (!(rel <= 1e-5)) {
            std::printf("FAIL: tr(kStress)/3V is not -dE/dV (|rel| = %.3e). "
                        "The pressure the barostat integrates is not the "
                        "derivative of the energy it is sampling.\n",
                        rel);
            return 1;
        }
    }

    // ── (4) Mechanical equilibrium + cohesive energy ────────────────────
    // Tosi–Fumi is fitted to NaCl, so its zero-pressure lattice constant and
    // its cohesive energy are both experiment, not our output.
    const Point at_exp = single_point(kDExp);
    const double e_per_pair = 2.0 * at_exp.energy / kN;
    std::printf("Cohesive energy at a=%.4f Å: %.4f eV/ion-pair  "
                "(experiment %.2f)   P = %+.4f GPa\n",
                kA0Exp, e_per_pair, kLatticeEnergyExp,
                at_exp.pressure * kEvPerA3InGPa);
    if (!(std::fabs(e_per_pair / kLatticeEnergyExp - 1.0) <= 0.02)) {
        std::printf("FAIL: cohesive energy %.4f eV/ion-pair is more than 2%% "
                    "from the experimental %.2f\n",
                    e_per_pair, kLatticeEnergyExp);
        return 1;
    }
    // Bracket the zero-pressure spacing: P > 0 (wants to expand) at the
    // experimental spacing, P < 0 (wants to contract) 1.5 % beyond it.
    const Point wide = single_point(kDExp * 1.015);
    std::printf("P(a=%.4f) = %+.4f GPa   P(a=%.4f) = %+.4f GPa\n", kA0Exp,
                at_exp.pressure * kEvPerA3InGPa, kA0Exp * 1.015,
                wide.pressure * kEvPerA3InGPa);
    if (!(at_exp.pressure > 0.0 && wide.pressure < 0.0)) {
        std::printf("FAIL: the model's zero-pressure lattice constant is not "
                    "within 1.5%% of the experimental %.4f Å\n",
                    kA0Exp);
        return 1;
    }

    // ── (5) NPT at P = 0 ───────────────────────────────────────────────
    // Started 3 % compressed, so the barostat has to push the cell OUT: a
    // stress that is too small (or missing a term) shows up as a cell that
    // fails to recover, which is exactly the failure mode a bare
    // "did the volume change?" check cannot see.
    cpu::Frame dyn = nacl(kDExp * 0.97);
    const double V0 = dyn.box.volume();

    constexpr double dt = 0.5;      // [fs]
    constexpr double T = 300.0;     // [K]
    constexpr double tau_T = 100.0; // [fs]
    constexpr double tau_P = 500.0; // [fs]
    constexpr int n_steps = 8000;   // 4 ps ≈ 8 barostat times
    constexpr int n_chunks = 40;

    cpu::md::MD sim;
    sim.set_observer(cpu::md::InitVelocity{T, /*seed=*/42u});
    sim.set_integrator(cpu::md::NoseHooverNPT{T, tau_T, 0.0, tau_P, dt});
    sim.set_neighbor(cpu::md::NeighborList<>{kCut, /*skin=*/0.5});
    sim.set_evaluator(cpu::PairBorn{tosi_fumi_nacl(), kCut});
    sim.set_evaluator(cpu::CoulLong{kAlpha, kCut});
    sim.set_evaluator(cpu::KSpacePME{kAlpha, kPmeSpacing, kPmeOrder});
    sim.enable_timing(false);
    sim.build();

    sim.run(dyn, 1);
    const double H0 = dyn.meta.get(ATV::kConserved);
    if (!std::isfinite(H0)) {
        std::printf(
            "FAIL: extended Hamiltonian is non-finite after first step\n");
        return 1;
    }

    double worst_rel = 0.0;
    double H_last = H0;
    double v_sum = 0.0;
    int v_samples = 0;
    double v_min = dyn.box.volume(), v_max = dyn.box.volume();
    for (int c = 0; c < n_chunks; ++c) {
        sim.run(dyn, n_steps / n_chunks);
        const double H = dyn.meta.get(ATV::kConserved);
        const double V = dyn.box.volume();
        if (!std::isfinite(H) || !std::isfinite(dyn.meta.get(ATV::kEnergy))
            || !std::isfinite(dyn.meta.get(ATV::kKinetic))) {
            std::printf("FAIL: non-finite thermo at chunk %d (E=%.3e K=%.3e "
                        "H=%.3e V=%.3f)\n",
                        c, dyn.meta.get(ATV::kEnergy),
                        dyn.meta.get(ATV::kKinetic), H, V);
            return 1;
        }
        v_min = std::min(v_min, V);
        v_max = std::max(v_max, V);
        worst_rel = std::max(worst_rel,
                             std::fabs(H - H0) / std::max(1.0, std::fabs(H0)));
        H_last = H;
        if (c >= n_chunks / 2) { // second half only: skip the relaxation
            v_sum += V;
            ++v_samples;
        }
    }
    const double V1 = dyn.box.volume();
    const double a_mean = std::cbrt(v_sum / v_samples) / kNSide * 2.0;
    std::printf("NPT %d steps (dt=%.2f fs): V0=%.3f V1=%.3f  H0=%.6e "
                "H_end=%.6e  max|ΔH|/max(1,|H0|)=%.3e\n",
                n_steps, dt, V0, V1, H0, H_last, worst_rel);
    std::printf("  <a> over the last 2 ps = %.4f Å  (experiment %.4f Å, "
                "%+.2f %%);  V/V0 in [%.3f, %.3f]\n",
                a_mean, kA0Exp, 100.0 * (a_mean / kA0Exp - 1.0), v_min / V0,
                v_max / V0);

    if (!(std::fabs(V1 - V0) / V0 > 1e-3)) {
        std::printf("FAIL: NPT left the cell frozen (|ΔV|/V0 = %.3e)\n",
                    std::fabs(V1 - V0) / V0);
        return 1;
    }
    if (!(std::fabs(a_mean / kA0Exp - 1.0) <= 0.03)) {
        std::printf("FAIL: zero-pressure lattice constant %.4f Å is more than "
                    "3%% from the experimental %.4f Å\n",
                    a_mean, kA0Exp);
        return 1;
    }
    if (!(dyn.meta.get(ATV::kKinetic) > 0.1)) {
        std::printf("FAIL: kinetic energy collapsed (K=%.3e)\n",
                    dyn.meta.get(ATV::kKinetic));
        return 1;
    }
    const double H_bound = 5e-3;
    if (!(worst_rel <= H_bound)) {
        std::printf("FAIL: conserved-H drift %.3e exceeds bound %.1e\n",
                    worst_rel, H_bound);
        return 1;
    }

    std::printf("PASS: NaCl Madelung + scaling + virial identity + Tosi–Fumi "
                "equilibrium + NPT conserved/cell relaxation\n");
    return 0;
}

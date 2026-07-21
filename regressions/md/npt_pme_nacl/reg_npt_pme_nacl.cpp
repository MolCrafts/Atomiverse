/// @file regressions/md/npt_pme_nacl/reg_npt_pme_nacl.cpp
/// @brief END-TO-END REGRESSION: charged NaCl, PME electrostatics, isotropic NPT.
///
/// Public composition only (headers + link atomiverse). Checks:
///   1. static Madelung energy (literature constant),
///   2. pure-Coulomb 1/s scaling under uniform cell/coordinate scale,
///   3. short NoseHooverNPT run mutates the orthorhombic cell while the MTK
///      extended Hamiltonian stays inside a documented O(dt²) envelope.
///
/// See REFERENCE.md. Exits non-zero on failure. Requires ATV_BUILD_KSPACE.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/pme.h"
#include "atomiverse/cpu/md/init.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/nose_hoover_npt.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

int main() {
    // ── Rock-salt NaCl (even n ⇒ charge-neutral) ───────────────────────
    constexpr double d = 2.82; // nearest-neighbour spacing [Å]
    constexpr int n_side = 4;  // 4³ = 64 ions
    constexpr int N = n_side * n_side * n_side;
    constexpr double kMadelung = 1.747564594633182;
    const double L = n_side * d;
    const double cutoff = 1.9 * d;
    const double alpha = 3.5 / cutoff;
    const double spacing = 0.5; // PME grid [Å]
    const int order = 6;
    // Soft-core LJ stand-in for Born–Mayer / Pauli repulsion. At the lattice
    // spacing U_LJ is ~0.004 eV/pair (Coulomb dominates the Madelung sum); it
    // only matters when NPT would otherwise collapse bare point charges.
    const double lj_eps = 0.05;  // [eV]
    const double lj_sigma = 1.5; // [Å] ≪ d
    const double lj_cut = 2.5 * lj_sigma;
    const double neigh_cut = std::max(cutoff, lj_cut);

    auto make_nacl = [&](double scale) {
        cpu::Frame frame(N);
        int idx = 0;
        for (int i = 0; i < n_side; ++i)
            for (int j = 0; j < n_side; ++j)
                for (int k = 0; k < n_side; ++k) {
                    frame.span(ATV::kAtom_x)[idx] = (i * d) * scale;
                    frame.span(ATV::kAtom_y)[idx] = (j * d) * scale;
                    frame.span(ATV::kAtom_z)[idx] = (k * d) * scale;
                    frame.span(ATV::kAtom_charge)[idx] =
                        ((i + j + k) % 2 == 0) ? 1.0 : -1.0;
                    // Na / Cl masses (amu) — only the dynamics path needs them.
                    frame.span(ATV::kAtom_mass)[idx] =
                        ((i + j + k) % 2 == 0) ? 22.9898 : 35.453;
                    frame.span(ATV::kAtom_type)[idx] =
                        ((i + j + k) % 2 == 0) ? 0 : 1;
                    frame.span(ATV::kAtom_tag)[idx] = idx + 1;
                    ++idx;
                }
        frame.simbox = cpu::Box::cubic(L * scale);
        return frame;
    };

    // Pure electrostatic single-point (no LJ) — Madelung / scaling oracles.
    auto electrostatic_energy = [&](cpu::Frame& frame) {
        cpu::Evaluate eval;
        eval.set_neighbor(cpu::md::NeighborList<>{cutoff, /*skin=*/0.3});
        eval.on(evaluate::Phase::evaluate).add(cpu::CoulLong{alpha, cutoff});
        eval.on(evaluate::Phase::evaluate)
            .add(cpu::KSpacePME{alpha, spacing, order});
        eval.build();
        eval.run(frame);
        return frame.meta.get(ATV::kEnergy);
    };

    // ── (1) Madelung ───────────────────────────────────────────────────
    auto frame = make_nacl(1.0);
    const double e0 = electrostatic_energy(frame);
    const double e_ref = -0.5 * N * kMadelung * units::coulomb / d;
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
    auto frame_s = make_nacl(s);
    const double e_s = electrostatic_energy(frame_s);
    const double scale_rel = std::fabs(e_s * s / e0 - 1.0);
    std::printf("Uniform scale s=%.2f: E(s)*s/E(1) - 1 = %.3e\n", s, scale_rel);
    if (!(scale_rel <= 5e-4)) {
        std::printf("FAIL: scaling law broken (stale KSpace cell data?)\n");
        return 1;
    }

    // ── (3) Short isotropic NPT: cell moves, H conserved ───────────────
    // Mild expansion: Madelung cohesion + P_target=0 pulls the cell in; the
    // soft LJ core prevents free collapse of bare point charges. A frozen
    // cell (no barostat / stale stress) fails the volume-change check.
    auto dyn = make_nacl(1.05);
    const double V0 = dyn.simbox.volume();

    const double dt = 0.5;  // [fs] — Coulomb forces need a short step
    const double T = 300.0; // [K]
    const double tau_T = 50.0;
    const double tau_P = 500.0;
    const int n_steps = 400;

    cpu::md::MD sim;
    sim.add(cpu::md::InitVelocity{T, /*seed=*/42u});
    sim.set_integrator(cpu::md::NoseHooverNPT{{.T_target = T,
                                               .tau_T = tau_T,
                                               .P_target = 0.0,
                                               .tau_P = tau_P,
                                               .dt = dt}});
    sim.add_potential(cpu::PairLJCut{lj_eps, lj_sigma, lj_cut});
    sim.set_neighbor(cpu::md::NeighborList<>{neigh_cut, /*skin=*/0.5});
    sim.on(md::Phase::force).add(cpu::CoulLong{alpha, cutoff});
    sim.add(cpu::KSpacePME{alpha, spacing, order});
    sim.enable_timing(false);
    sim.build();

    sim.run(dyn, 1);
    const double H0 = dyn.meta.get(ATV::kConserved);
    if (!(std::isfinite(H0))) {
        std::printf(
            "FAIL: extended Hamiltonian is non-finite after first step\n");
        return 1;
    }

    double worst_rel = 0.0;
    double H_last = H0;
    const int n_chunks = 8;
    for (int c = 0; c < n_chunks; ++c) {
        sim.run(dyn, n_steps / n_chunks);
        const double H = dyn.meta.get(ATV::kConserved);
        if (!(std::isfinite(H)) || !(std::isfinite(dyn.meta.get(ATV::kEnergy)))
            || !(std::isfinite(dyn.meta.get(ATV::kKinetic)))) {
            std::printf(
                "FAIL: non-finite thermo at chunk %d (E=%.3e K=%.3e H=%.3e "
                "V=%.3f)\n",
                c, dyn.meta.get(ATV::kEnergy), dyn.meta.get(ATV::kKinetic), H,
                dyn.simbox.volume());
            return 1;
        }
        const double rel = std::fabs(H - H0) / std::max(1.0, std::fabs(H0));
        if (rel > worst_rel)
            worst_rel = rel;
        H_last = H;
    }
    const double V1 = dyn.simbox.volume();
    std::printf(
        "NPT %d steps (dt=%.2f fs): V0=%.3f V1=%.3f  H0=%.6e H_end=%.6e  "
        "max|ΔH|/max(1,|H0|)=%.3e\n",
        n_steps, dt, V0, V1, H0, H_last, worst_rel);

    if (!(std::fabs(V1 - V0) / V0 > 1e-3)) {
        std::printf("FAIL: NPT left the cell frozen (|ΔV|/V0 = %.3e)\n",
                    std::fabs(V1 - V0) / V0);
        return 1;
    }
    // Expanded start + cohesive electrostatics → cell contracts, but soft
    // core keeps a finite density (not free collapse to V→0).
    if (!(V1 < V0)) {
        std::printf(
            "FAIL: expanded cohesive NaCl did not contract (V1 >= V0)\n");
        return 1;
    }
    if (!(V1 > 0.2 * V0)) {
        std::printf("FAIL: cell collapsed under NPT (V1/V0 = %.3e)\n", V1 / V0);
        return 1;
    }
    // Kinetic energy must stay alive (collapse freezes K → 0).
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

    std::printf(
        "PASS: NaCl PME Madelung + scaling + NPT conserved/cell motion\n");
    return 0;
}

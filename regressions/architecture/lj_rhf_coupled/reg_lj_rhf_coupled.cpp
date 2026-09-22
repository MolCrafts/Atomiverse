/// @file regressions/architecture/lj_rhf_coupled/reg_lj_rhf_coupled.cpp
/// @brief END-TO-END: the FIRST case in which a classical (MM) and an
///        electronic-structure (QM) energy source are bound into ONE driver
///        pass and both survive — kEnergy = E_LJ + E_RHF.
///
/// One `cpu::Evaluate` over one four-atom Frame:
///   * atoms 0-1: a He pair under `cpu::PairLJCut` (per-type table, so only
///     He carries an LJ site) — analytic 12-6 closed form;
///   * atoms 2-3: H2 at the RHF/STO-3G equilibrium bond, 30 A away, driven by
///     a nested `cpu::scf::RHF` whose `ghost_atoms` keeps the He pair OUT of
///     the QM Hamiltonian (basis-only sites: no nuclear charge, no electrons).
///
/// Both addends are pinned independently — the LJ term by the closed-form
/// potential at cited parameters, the QM term by the literature RHF/STO-3G H2
/// total energy — so the SUM pins the accumulation contract itself. Before
/// esmm-contract-05 the electronic evaluator OVERWROTE kEnergy and the MM
/// term was silently erased. Public headers only. See REFERENCE.md.

#include <cmath>
#include <cstdio>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

int main() {
    // ── References ───────────────────────────────────────────────────────
    // (1) MM: Lennard-Jones (12,6) helium. eps/k_B = 10.22 K, sigma = 2.556 A
    //     (J. de Boer, A. Michels, Physica 5, 945 (1938); tabulated in
    //     Hirschfelder, Curtiss & Bird, "Molecular Theory of Gases and
    //     Liquids", Wiley 1954, Table I-A). k_B = 8.617333262e-5 eV/K
    //     (CODATA 2018, exact by the 2019 SI redefinition).
    // (2) QM: RHF/STO-3G H2 total energy at its own equilibrium bond length,
    //     E = -1.11751 Hartree at r_e = 0.712 A (NIST CCCBDB, Standard
    //     Reference Database 101, calculated HF/STO-3G energy and bond length
    //     for H2 (1-Sigma-g+); STO-3G from Hehre, Stewart & Pople, J. Chem.
    //     Phys. 51, 2657 (1969)). Quoted to five decimals, hence known to
    //     +-5e-6 Ha = +-1.4e-4 eV -- that quantization, not our SCF, sets the
    //     absolute-energy tolerance below.
    const double kB_ev = 8.617333262e-5; // [eV/K]
    const double eps = 10.22 * kB_ev;    // [eV]   = 8.806914593764e-4
    const double sigma = 2.556;          // [A]
    const double r_he = 2.3;             // [A] He-He separation
    const double cutoff = 2.5 * sigma;   // [A] = 6.39
    const double r_h2 = 0.7122;          // [A] H-H bond
    const double sep = 30.0;             // [A] MM pair <-> QM molecule
    const double L = 120.0;              // [A] cubic box edge

    const double e_rhf_ha = -1.11751; // [Hartree] literature, as quoted
    const double e_rhf_ev =
        units::convert(e_rhf_ha, units::Energy::Hartree, units::Energy::eV);

    // LAMMPS-style shifted 12-6: V(r) - V(r_c). Closed form, no fitting.
    const double s6 = std::pow(sigma / r_he, 6.0);
    const double sc6 = std::pow(sigma / cutoff, 6.0);
    const double e_lj =
        4.0 * eps * (s6 * s6 - s6) - 4.0 * eps * (sc6 * sc6 - sc6);
    // F(r) = (24 eps / r) [2 (sigma/r)^12 - (sigma/r)^6], repulsive at 2.3 A.
    const double f_lj = 24.0 * eps / r_he * (2.0 * s6 * s6 - s6);

    // Two tolerance tiers, because two different things are being asserted.
    //
    // e_tol -- comparisons against the LITERATURE total energy. The quoted
    // -1.11751 Ha is known to +-5e-6 Ha = +-1.4e-4 eV, and our converged SCF
    // sits 1.1e-4 eV from the rounded quotation, so nothing tighter than that
    // quantization can honestly be asserted about an absolute energy. 2.5e-4 eV
    // (2.2x the quantization, 8.2e-6 relative) is the result. The MM term is
    // 5.88e-3 eV = 23.5 x this tolerance, so an erased MM source cannot hide.
    //
    // exact_tol -- comparisons BETWEEN channels of one run, or between legs of
    // this one binary (kEnergy vs kEnergyPair + kEnergyElectronic; co-bound vs
    // the two sole-source passes). No literature precision enters those: they
    // are the same doubles added in a different order, so the accumulation
    // contract itself is pinned at 1e-10 eV = 3e-12 relative -- five orders
    // tighter than the absolute pin, and it is this tier that makes the case
    // a contract witness rather than an energy check.
    const double e_tol = 2.5e-4;      // [eV] literature-limited
    const double f_tol = 1.0e-8;      // [eV/A] (project FP table, force/exact)
    const double exact_tol = 1.0e-10; // [eV] channel/leg arithmetic

    // ── One Frame: 2 MM sites (He, Z=2) + 1 QM molecule (H2) ─────────────
    cpu::Frame frame(4);
    frame.span(kAtom_type)[0] = 2; // He: LJ site, ghost in the QM Hamiltonian
    frame.span(kAtom_type)[1] = 2;
    frame.span(kAtom_type)[2] =
        1; // H: QM, no LJ site (no diagonal in the table)
    frame.span(kAtom_type)[3] = 1;
    frame.span(kAtom_mass)[0] = 4.0026;
    frame.span(kAtom_mass)[1] = 4.0026;
    frame.span(kAtom_mass)[2] = 1.008;
    frame.span(kAtom_mass)[3] = 1.008;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_x)[1] = r_he;
    frame.span(kAtom_x)[2] = sep - 0.5 * r_h2;
    frame.span(kAtom_x)[3] = sep + 0.5 * r_h2;
    for (int i = 0; i < 4; ++i)
        frame.span(kAtom_tag)[i] = i + 1;
    frame.box = cpu::Box::cubic(L);

    // ── Leg 1: the MM source alone ───────────────────────────────────────
    cpu::Evaluate mm_only;
    mm_only.set_evaluator(cpu::PairLJCut{{{2, 2, eps, sigma}}, cutoff});
    mm_only.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3});
    mm_only.set_threads(1);
    mm_only.build();
    mm_only.run(frame);
    const double mm_energy = frame.meta.get(kEnergy);
    const double mm_pair = frame.meta.get(kEnergyPair);
    const double mm_f_he = frame.span(kAtom_fx)[0];
    const double mm_f_h = frame.span(kAtom_fx)[2];

    // ── Leg 2: the QM source alone (He pair ghosted out of the QM system) ─
    cpu::scf::RHF rhf_only;
    rhf_only.basis = cpu::scf::BasisSet::STO3G;
    rhf_only.tol = 1e-10;
    rhf_only.ghost_atoms = {0, 1};
    cpu::Evaluate qm_only;
    qm_only.on(evaluate::Phase::evaluate).add_ref(rhf_only, 0);
    qm_only.set_threads(1);
    qm_only.build();
    qm_only.run(frame);
    const double qm_energy = frame.meta.get(kEnergy);
    const double qm_electronic = frame.meta.get(kEnergyElectronic);
    const double qm_f_he = frame.span(kAtom_fx)[0];
    const double qm_f_h = frame.span(kAtom_fx)[2];
    const double qm_energy_ha = rhf_only.energy();

    // ── Leg 3: both sources, MM first (the order the overwrite erased) ───
    cpu::scf::RHF rhf_after;
    rhf_after.basis = cpu::scf::BasisSet::STO3G;
    rhf_after.tol = 1e-10;
    rhf_after.ghost_atoms = {0, 1};
    cpu::Evaluate both;
    both.set_evaluator(cpu::PairLJCut{{{2, 2, eps, sigma}}, cutoff});
    both.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3});
    both.on(evaluate::Phase::evaluate).add_ref(rhf_after, /*order=*/10);
    both.set_threads(1);
    both.build();
    both.run(frame);
    const double co_energy = frame.meta.get(kEnergy);
    const double co_pair = frame.meta.get(kEnergyPair);
    const double co_electronic = frame.meta.get(kEnergyElectronic);
    const double co_f_he0 = frame.span(kAtom_fx)[0];
    const double co_f_he1 = frame.span(kAtom_fx)[1];
    const double co_f_h0 = frame.span(kAtom_fx)[2];
    const double co_f_h1 = frame.span(kAtom_fx)[3];

    // ── Leg 4: both sources, QM first — binding order is not a physics knob
    cpu::scf::RHF rhf_before;
    rhf_before.basis = cpu::scf::BasisSet::STO3G;
    rhf_before.tol = 1e-10;
    rhf_before.ghost_atoms = {0, 1};
    cpu::Evaluate both_swapped;
    both_swapped.set_evaluator(cpu::PairLJCut{{{2, 2, eps, sigma}}, cutoff});
    both_swapped.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3});
    both_swapped.on(evaluate::Phase::evaluate)
        .add_ref(rhf_before, /*order=*/-10);
    both_swapped.set_threads(1);
    both_swapped.build();
    both_swapped.run(frame);
    const double swap_energy = frame.meta.get(kEnergy);
    const double swap_f_he0 = frame.span(kAtom_fx)[0];

    // ── Leg 5: control — the same QM molecule with no MM sites present ────
    // Pins that the ghost basis is inert: the He 1s / H 1s Gaussian overlap at
    // 30 A is exp(-mu R^2) with mu = 0.1098 bohr^-2 and R = 56.69 bohr, i.e.
    // ~1e-153, so the QM energy must be the bare-H2 one to machine precision.
    cpu::Frame bare(2);
    bare.span(kAtom_type)[0] = 1;
    bare.span(kAtom_type)[1] = 1;
    bare.span(kAtom_mass)[0] = 1.008;
    bare.span(kAtom_mass)[1] = 1.008;
    bare.span(kAtom_x)[0] = sep - 0.5 * r_h2;
    bare.span(kAtom_x)[1] = sep + 0.5 * r_h2;
    bare.span(kAtom_tag)[0] = 1;
    bare.span(kAtom_tag)[1] = 2;
    bare.box = cpu::Box::cubic(L);
    cpu::scf::RHF rhf_bare;
    rhf_bare.basis = cpu::scf::BasisSet::STO3G;
    rhf_bare.tol = 1e-10;
    cpu::Evaluate bare_eval;
    bare_eval.on(evaluate::Phase::evaluate).add_ref(rhf_bare, 0);
    bare_eval.set_threads(1);
    bare_eval.build();
    bare_eval.run(bare);
    const double bare_energy = bare.meta.get(kEnergy);
    const double bare_f_h = bare.span(kAtom_fx)[0];

    // ── Report ───────────────────────────────────────────────────────────
    const double want = e_lj + e_rhf_ev;
    std::printf("\n== ES x MM co-binding: one Evaluate pass, two sources ==\n");
    std::printf(
        "MM  He-He LJ(12,6) eps=%.12e eV sigma=%.4f A r=%.3f A rc=%.3f A\n",
        eps, sigma, r_he, cutoff);
    std::printf("QM  RHF/STO-3G H2 r=%.4f A, He pair ghosted (basis-only)\n\n",
                r_h2);
    std::printf("  %-34s %20s %20s %12s\n", "quantity", "got [eV]", "want [eV]",
                "diff");
    std::printf("  %-34s %20.12f %20.12f %12.2e\n", "MM alone, kEnergy",
                mm_energy, e_lj, mm_energy - e_lj);
    std::printf("  %-34s %20.12f %20.12f %12.2e\n", "MM alone, kEnergyPair",
                mm_pair, e_lj, mm_pair - e_lj);
    std::printf("  %-34s %20.12f %20.12f %12.2e\n", "QM alone, kEnergy",
                qm_energy, e_rhf_ev, qm_energy - e_rhf_ev);
    std::printf("  %-34s %20.12f %20.12f %12.2e\n",
                "QM alone, kEnergyElectronic", qm_electronic, e_rhf_ev,
                qm_electronic - e_rhf_ev);
    std::printf("  %-34s %20.12f %20.12f %12.2e\n", "QM control (no MM sites)",
                bare_energy, e_rhf_ev, bare_energy - e_rhf_ev);
    std::printf("  %-34s %20.12f %20.12f %12.2e   <== THE PIN\n",
                "co-bound, kEnergy", co_energy, want, co_energy - want);
    std::printf("  %-34s %20.12f %20.12f %12.2e\n", "co-bound, kEnergyPair",
                co_pair, e_lj, co_pair - e_lj);
    std::printf("  %-34s %20.12f %20.12f %12.2e\n",
                "co-bound, kEnergyElectronic", co_electronic, e_rhf_ev,
                co_electronic - e_rhf_ev);
    std::printf("  %-34s %20.12f %20.12f %12.2e\n",
                "co-bound, QM binding first", swap_energy, want,
                swap_energy - want);
    std::printf("\n  the contract tier (no literature precision enters):\n");
    std::printf("  %-34s %20.12f %20.12f %12.2e\n",
                "kEnergy vs sum of per-term keys", co_energy,
                co_pair + co_electronic, co_energy - co_pair - co_electronic);
    std::printf("  %-34s %20.12f %20.12f %12.2e\n",
                "kEnergy vs sum of sole-source legs", co_energy,
                mm_energy + qm_energy, co_energy - mm_energy - qm_energy);
    std::printf("  %-34s %20.12f %20.12f %12.2e\n", "QM leg [Ha] vs quoted",
                qm_energy_ha, e_rhf_ha, qm_energy_ha - e_rhf_ha);
    std::printf("\n  %-34s %20.12e %20.12e %12.2e\n", "MM force on He [eV/A]",
                -co_f_he0, f_lj, -co_f_he0 - f_lj);
    std::printf("  %-34s %20.12e %20.12e %12.2e\n", "QM force on H [eV/A]",
                co_f_h0, qm_f_h, co_f_h0 - qm_f_h);
    std::printf("  SCF: converged=%d iterations=%d  E=%.12f Ha (quoted %.5f, "
                "known to +-5e-6)\n",
                (int)rhf_after.converged(), rhf_after.iterations(),
                qm_energy_ha, e_rhf_ha);
    std::printf(
        "  Under the pre-esmm-contract-05 overwrite the co-bound total\n"
        "  would have been %.12f eV (the QM term alone): the MM term\n"
        "  %.12e eV = %.1f x the %.1e eV literature tolerance and\n"
        "  %.2e x the %.1e eV contract tolerance would be erased.\n\n",
        qm_energy, e_lj, std::fabs(e_lj) / e_tol, e_tol,
        std::fabs(e_lj) / exact_tol, exact_tol);

    bool ok = true;
    auto check = [&ok](bool cond, const char* what, double got, double wanted,
                       double tol) {
        if (!cond) {
            std::printf("FAIL: %s   got=%.12g want=%.12g diff=%.3g tol=%.3g\n",
                        what, got, wanted, got - wanted, tol);
            ok = false;
        }
    };

    // The pin: the two independently known addends sum in one pass.
    check(std::fabs(co_energy - want) <= e_tol,
          "co-bound kEnergy != E_LJ(analytic) + E_RHF(literature)", co_energy,
          want, e_tol);
    // Each addend, still itself, in the presence of the other.
    check(std::fabs(co_pair - e_lj) <= exact_tol,
          "co-bound kEnergyPair != the analytic LJ energy", co_pair, e_lj,
          exact_tol);
    check(std::fabs(co_electronic - e_rhf_ev) <= e_tol,
          "co-bound kEnergyElectronic != the literature RHF energy",
          co_electronic, e_rhf_ev, e_tol);
    // The decomposition invariant: total == sum of the per-term keys (no Bias
    // here, so there is no unclassified remainder).
    check(std::fabs(co_energy - co_pair - co_electronic) <= exact_tol,
          "kEnergy != kEnergyPair + kEnergyElectronic", co_energy,
          co_pair + co_electronic, exact_tol);
    // Sole-source equivalence: co-binding changed neither source's own answer.
    check(std::fabs(co_pair - mm_pair) <= exact_tol,
          "the MM term changed when the QM source was co-bound", co_pair,
          mm_pair, exact_tol);
    check(std::fabs(co_electronic - qm_electronic) <= exact_tol,
          "the QM term changed when the MM source was co-bound", co_electronic,
          qm_electronic, exact_tol);
    check(std::fabs(co_energy - (mm_energy + qm_energy)) <= exact_tol,
          "co-bound total != sum of the two sole-source totals", co_energy,
          mm_energy + qm_energy, exact_tol);
    // Binding order is a scheduling detail, never physics.
    check(std::fabs(swap_energy - co_energy) <= exact_tol,
          "the total depends on which source was bound first", swap_energy,
          co_energy, exact_tol);
    check(std::fabs(swap_f_he0 - co_f_he0) <= f_tol,
          "the MM force depends on which source was bound first", swap_f_he0,
          co_f_he0, f_tol);
    // The clobber tripwire, stated as physics: the MM term must be present at
    // full size, not merely within the energy tolerance.
    check(std::fabs(co_energy - qm_energy) > 0.5 * std::fabs(e_lj),
          "the co-bound total is the QM term alone (MM source erased)",
          co_energy - qm_energy, e_lj, 0.5 * std::fabs(e_lj));
    // Forces: the same accumulator contract on kAtom_f*. The QM gradient write
    // must not wipe the MM force on the He sites (they are ghosts, so the QM
    // gradient there is exactly zero and an overwrite is a total loss).
    check(std::fabs(-co_f_he0 - f_lj) <= f_tol,
          "MM force on He[0] != the analytic LJ force", -co_f_he0, f_lj, f_tol);
    check(std::fabs(co_f_he1 + co_f_he0) <= f_tol,
          "the MM pair force is not antisymmetric", co_f_he1, -co_f_he0, f_tol);
    check(std::fabs(co_f_h0 - qm_f_h) <= f_tol,
          "QM force on H[0] changed when the MM source was co-bound", co_f_h0,
          qm_f_h, f_tol);
    check(std::fabs(co_f_h1 + co_f_h0) <= f_tol,
          "the QM pair force is not antisymmetric", co_f_h1, -co_f_h0, f_tol);
    check(std::fabs(co_f_he0 + co_f_he1 + co_f_h0 + co_f_h1) <= f_tol,
          "the total force on the system is not zero",
          co_f_he0 + co_f_he1 + co_f_h0 + co_f_h1, 0.0, f_tol);
    // Ghost inertness: the MM sites carry basis functions but no physics.
    check(std::fabs(qm_energy - bare_energy) <= exact_tol,
          "the ghosted MM sites perturbed the QM energy", qm_energy,
          bare_energy, exact_tol);
    check(std::fabs(qm_f_h - bare_f_h) <= f_tol,
          "the ghosted MM sites perturbed the QM force", qm_f_h, bare_f_h,
          f_tol);
    check(std::fabs(qm_f_he) <= f_tol,
          "a ghost site carries a QM force (it has no nucleus)", qm_f_he, 0.0,
          f_tol);
    // The MM source must not reach the QM atoms: H has no LJ site, so the
    // 0.7122 A H-H pair inside the 6.39 A cutoff contributes nothing.
    check(std::fabs(mm_f_h) <= f_tol,
          "the LJ table gave the H atoms a site (no diagonal was declared)",
          mm_f_h, 0.0, f_tol);
    check(std::fabs(mm_energy - mm_pair) <= exact_tol,
          "kEnergy != kEnergyPair for the MM-only pass", mm_energy, mm_pair,
          exact_tol);
    if (!rhf_after.converged()) {
        std::printf("FAIL: the co-bound SCF did not converge\n");
        ok = false;
    }
    if (!(std::fabs(mm_f_he) > 0.0)) {
        std::printf("FAIL: the MM-only pass produced no force at all\n");
        ok = false;
    }

    if (!ok)
        return 1;
    std::printf("PASS: kEnergy = E_LJ(analytic) + E_RHF(literature) in one "
                "Evaluate pass, both terms and both force sets intact\n");
    return 0;
}

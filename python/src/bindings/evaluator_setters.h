/// @file python/src/bindings/evaluator_setters.h
/// @brief The eighteen `set_evaluator_*` methods shared by MD and Minimize.
///
/// A force source is a force source: whatever drives molecular dynamics drives
/// a geometry optimization, so both handles expose the SAME eighteen method
/// names with the same argument types and the same construction — that is what
/// lets `atomiverse/_dispatch.py` be one table keyed by component type rather
/// than two. Registering them from one place is what keeps the two facades from
/// drifting apart when a nineteenth force source lands.
///
/// The `#ifdef` arms live here too, for the same reason: which of the eighteen
/// exist is a property of the build (`ATV_HAS_KSPACE` / `ATV_HAS_SEMIEMPIRICAL`
/// / `ATV_HAS_PW`), never of the driver.
///
/// The requirement on `Handle` is one member template,
/// `set_categorical(const char* setter, const Params&, Component)`, which both
/// `MdHandle` and `MinimizeHandle` satisfy: it registers the component on the
/// driver's plural evaluator slot and records the TaskSpec mirror entry.
///
/// Two docstrings are the caller's, not this header's: AM1 and plane-wave DFT
/// cost one self-consistent-field solve per *step* under MD and per *iteration*
/// under a minimizer, and only an MD run also needs a per-atom mass column, so
/// that guidance cannot be written once. The sixteen classical setters describe
/// pair styles and topology terms, whose story does not change with the loop.
#pragma once

#include <nanobind/nanobind.h>

#ifdef ATV_HAS_KSPACE
#include "atomiverse/cpu/longrange/pme.h"
#endif
#ifdef ATV_HAS_PW
// `copy_pw_params` is the one definition of "the same method, no state".
#include "atomiverse/cpu/planewave/internal/scf_loop.h"
#include "atomiverse/cpu/planewave/pw_evaluator.h"
#endif
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_fene.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/dihedral_charmm.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/improper_harmonic.h"
#include "atomiverse/cpu/potentials/pair_born.h"
#include "atomiverse/cpu/potentials/pair_buck.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_dpd.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/potentials/pair_morse.h"
#include "atomiverse/cpu/potentials/pair_wca.h"
#include "atomiverse/cpu/potentials/pair_yukawa.h"
#ifdef ATV_HAS_SEMIEMPIRICAL
#include "atomiverse/cpu/semiempirical/am1_evaluator.h"
#endif
// The one params-mirror-to-component factory of the catalogue: the Born table
// crosses as parallel vectors and has to be transposed back into coefficient
// rows, identically here and on the launcher replay path.
#include "atomiverse/cpu/task/internal/potential_builders.h"
#include "atomiverse/cpu/task/task_spec.h"

namespace nb = nanobind;

/// @brief Register the eighteen force-source setters on a driver handle class.
///
/// @tparam Handle    `MdHandle` or `MinimizeHandle` — anything exposing
///                   `set_categorical(setter, params, component)`.
/// @param cls        The bound handle class the methods are added to.
/// @param am1_doc    Docstring for `set_evaluator_am1`, written in the caller's
///                   loop vocabulary (step vs iteration). Unused on a build
///                   without `ATV_HAS_SEMIEMPIRICAL`.
/// @param pw_doc     Docstring for `set_evaluator_pw`, likewise. Unused on a
///                   build without `ATV_HAS_PW`.
template <typename Handle>
void bind_evaluator_setters(nb::class_<Handle>& cls,
                            [[maybe_unused]] const char* am1_doc,
                            [[maybe_unused]] const char* pw_doc) {
    cls.def(
        "set_evaluator_pair_lj_cut",
        [](Handle& self, const ATV::cpu::task::PairLJCutParams& p) {
            self.set_categorical(
                "set_evaluator", p,
                ATV::cpu::PairLJCut{p.epsilon, p.sigma, p.cutoff});
        },
        nb::arg("comp"),
        "Append the Lennard-Jones 12-6 pair evaluator. Needs a NeighborList "
        "whose cutoff is >= the LJ cutoff.");

    cls.def(
        "set_evaluator_pair_coul_cut",
        [](Handle& self, const ATV::cpu::task::PairCoulCutParams& p) {
            self.set_categorical("set_evaluator", p,
                                 ATV::cpu::PairCoulCut{p.cutoff});
        },
        nb::arg("comp"),
        "Append the truncated Coulomb pair evaluator. Self-contained — it "
        "takes no reciprocal-space partner, so do NOT combine it with CoulLong "
        "or KSpacePME. Needs a NeighborList whose cutoff is >= the Coulomb "
        "cutoff, and a per-atom 'charge' column [e] on the frame handed to "
        "run().");

    cls.def(
        "set_evaluator_coul_long",
        [](Handle& self, const ATV::cpu::task::CoulLongParams& p) {
            self.set_categorical("set_evaluator", p,
                                 ATV::cpu::CoulLong{p.alpha, p.cutoff});
        },
        nb::arg("comp"),
        "Append the real-space half of an Ewald/PME split. Half a Hamiltonian "
        "on its own: build() requires exactly one reciprocal-space partner "
        "(KSpacePME) at the SAME alpha [1/A]. Needs a NeighborList whose "
        "cutoff is >= the Coulomb cutoff, and a per-atom 'charge' column [e] "
        "on the frame handed to run().");

#ifdef ATV_HAS_KSPACE
    cls.def(
        "set_evaluator_kspace_pme",
        [](Handle& self, const ATV::cpu::task::KSpacePmeParams& p) {
            // Move-only (it owns FFTW plans): the prvalue goes straight into
            // the sink parameter, no named local to move from.
            self.set_categorical(
                "set_evaluator", p,
                ATV::cpu::KSpacePME{p.alpha, p.grid_spacing, p.order});
        },
        nb::arg("comp"),
        "Append the reciprocal-space half of the same Ewald/PME split. build() "
        "requires exactly one real-space partner (CoulLong) at the SAME alpha "
        "[1/A]. Bound only on an extension built with ATV_BUILD_KSPACE=ON.");
#endif

#ifdef ATV_HAS_SEMIEMPIRICAL
    cls.def(
        "set_evaluator_am1",
        [](Handle& self,
           const ATV::cpu::semiempirical::AM1Evaluator& evaluator) {
            // Handed a LIVE component rather than a params mirror: AM1Evaluator
            // is bound for its own single_point / forces surface, so Python
            // already holds the real thing. Read the five SCF parameters back
            // off the owned method — the same five the MPI launcher path
            // copies in bind_semiempirical.cpp — and let set_categorical build
            // the driver's own copy from them, so the wire mirror and the
            // registered component cannot disagree.
            const ATV::cpu::semiempirical::AM1& method = evaluator.method();
            const ATV::cpu::task::AM1EvaluatorParams p{
                method.charge, method.multiplicity, method.max_iter,
                method.density_tol, method.energy_tol};
            self.set_categorical("set_evaluator", p,
                                 ATV::cpu::semiempirical::AM1Evaluator{
                                     p.charge, p.multiplicity, p.max_iter,
                                     p.density_tol, p.energy_tol}
                                     .with_gradient(true));
        },
        nb::arg("comp"), am1_doc);
#endif

#ifdef ATV_HAS_PW
    cls.def(
        "set_evaluator_pw",
        [](Handle& self, const ATV::cpu::pw::PWEvaluator& evaluator) {
            // Handed a LIVE component, like set_evaluator_am1 and for the same
            // reason: PWEvaluator is bound in bind_pw.cpp so Python already
            // holds the real thing.
            const ATV::cpu::pw::PW& method = evaluator.method();
            // The two enum members cross the wire as their integer values:
            // `cpu::task::PwEvaluatorParams` is compiled without the
            // plane-wave headers those enums live in.
            const ATV::cpu::task::PwEvaluatorParams p{
                method.upf_path,
                method.gth_name,
                method.ecut_ry,
                method.k_mesh,
                method.tol,
                method.max_iter,
                method.mix_beta,
                method.pulay_depth,
                static_cast<int>(method.xc),
                method.spin_polarized,
                method.multiplicity,
                static_cast<int>(method.smearing),
                method.degauss,
                method.mp_order,
                method.extra_bands};
            self.set_categorical(
                "set_evaluator", p,
                ATV::cpu::pw::PWEvaluator{
                    ATV::cpu::pw::internal::copy_pw_params(method)}
                    .with_gradient(true));
        },
        nb::arg("comp"), pw_doc);
#endif

    cls.def(
        "set_evaluator_bond_harmonic",
        [](Handle& self, const ATV::cpu::task::BondHarmonicParams& p) {
            self.set_categorical("set_evaluator", p,
                                 ATV::cpu::BondHarmonic{p.K, p.r0});
        },
        nb::arg("comp"),
        "Append the harmonic bond evaluator. Reads the frame's bond topology; "
        "parameters are indexed by bond type.");

    cls.def(
        "set_evaluator_angle_harmonic",
        [](Handle& self, const ATV::cpu::task::AngleHarmonicParams& p) {
            self.set_categorical("set_evaluator", p,
                                 ATV::cpu::AngleHarmonic{p.K, p.theta0});
        },
        nb::arg("comp"),
        "Append the harmonic angle evaluator. Reads the frame's angle "
        "topology; parameters are indexed by angle type.");

    cls.def(
        "set_evaluator_dihedral_multi_harmonic",
        [](Handle& self, const ATV::cpu::task::DihedralMultiHarmonicParams& p) {
            self.set_categorical("set_evaluator", p,
                                 ATV::cpu::DihedralMultiHarmonic{p.Cn});
        },
        nb::arg("comp"),
        "Append the multi-harmonic dihedral evaluator. Reads the frame's "
        "dihedral topology; coefficients are indexed by dihedral type.");

    // ── The widened real-space catalogue ──
    //
    // Six pair styles and three bonded ones. Every one of the six withdraws
    // its NeighborConsumer, so none of them REQUIRES a NeighborList, and every
    // one of them as constructed here walks all pairs O(N^2) — including
    // PairBorn, whose only neighbor-list-aware arm is the single-type
    // constructor this path never takes (it always builds the table arm).
    // Installing a NeighborList beside them changes nothing but the cost.

    cls.def(
        "set_evaluator_pair_morse",
        [](Handle& self, const ATV::cpu::task::PairMorseParams& p) {
            self.set_categorical(
                "set_evaluator", p,
                ATV::cpu::PairMorse{p.D, p.alpha, p.r0, p.cutoff});
        },
        nb::arg("comp"),
        "Append the Morse pair evaluator, V = D(1 - exp(-alpha(r - r0)))^2, "
        "shifted to zero at the cutoff.\n\n"
        "This style is the tree's DEMONSTRATION that a new pair style costs "
        "two files and no registry (cpu/potentials/pair_morse.h), and it is "
        "written to stay readable rather than to be fast: it walks every pair "
        "O(N^2) with no neighbor list at all. Use it on the scale where that "
        "is honest — a dimer, a cluster, a teaching example — not on a bulk "
        "system.");

    cls.def(
        "set_evaluator_pair_buck",
        [](Handle& self, const ATV::cpu::task::PairBuckParams& p) {
            self.set_categorical("set_evaluator", p,
                                 ATV::cpu::PairBuck{p.A, p.rho, p.C, p.cutoff});
        },
        nb::arg("comp"),
        "Append the Buckingham pair evaluator, U = A exp(-r/rho) - C/r^6, "
        "shifted to zero at the cutoff (LAMMPS `pair_style buck` with "
        "`pair_modify shift yes`). Single-type: one (A, rho, C) applies to "
        "every pair. Mind the r -> 0 limit — the Born-Mayer wall is finite "
        "while the -C/r^6 term is not, so Buckingham has a spurious "
        "attractive collapse at very short range that Born's +D/r^8 term "
        "cures.");

    cls.def(
        "set_evaluator_pair_born",
        [](Handle& self, const ATV::cpu::task::PairBornParams& p) {
            self.set_categorical(
                "set_evaluator", p,
                ATV::cpu::PairBorn{ATV::cpu::task::internal::born_table(p),
                                   p.cutoff});
        },
        nb::arg("comp"),
        "Append the Born-Mayer-Huggins pair evaluator, "
        "U = A exp((sigma - r)/rho) - C/r^6 + D/r^8, shifted at the cutoff "
        "(LAMMPS `pair_style born`) — the Fumi-Tosi short-range half of a "
        "molten-salt model; add the Coulomb term as a separate evaluator.\n\n"
        "The table is keyed by ATOMIC NUMBER. The frame's per-atom type "
        "column is filled from the molrs `atomic_number` column, so a "
        "sodium-chloride row is type_i=11, type_j=17 and an argon dimer is "
        "18, 18 — not a 0-based index into anything.\n\n"
        "Every interacting type pair must be listed, and the table is "
        "symmetric in (i, j). An unlisted pair does NOT interact: Fumi-Tosi "
        "parameters are not Lorentz-Berthelot, so there is no mixing rule to "
        "fall back on and inventing one silently would be worse than a "
        "zero.\n\n"
        "The seven sequences are one table seen column-wise and are "
        "transposed into coefficient rows HERE, so a length mismatch raises "
        "ValueError from this call rather than from the PairBorn "
        "constructor.");

    cls.def(
        "set_evaluator_pair_yukawa",
        [](Handle& self, const ATV::cpu::task::PairYukawaParams& p) {
            self.set_categorical("set_evaluator", p,
                                 ATV::cpu::PairYukawa{p.A, p.kappa, p.cutoff});
        },
        nb::arg("comp"),
        "Append the screened-Coulomb (Yukawa) pair evaluator, "
        "U = A exp(-kappa r)/r, shifted at the cutoff. Despite the name it "
        "reads NO per-atom charge column: the amplitude A [eV*A] is the whole "
        "interaction strength, which is what makes it a colloid / "
        "screened-plasma model rather than an electrostatic one. For real "
        "charges use PairCoulCut, or CoulLong + KSpacePME.");

    cls.def(
        "set_evaluator_pair_wca",
        [](Handle& self, const ATV::cpu::task::PairWcaParams& p) {
            self.set_categorical("set_evaluator", p,
                                 ATV::cpu::PairWCA{p.epsilon, p.sigma});
        },
        nb::arg("comp"),
        "Append the Weeks-Chandler-Andersen pair evaluator: Lennard-Jones "
        "12-6 cut and shifted at r_c = 2^(1/6) sigma, so U(r_c) = F(r_c) = 0 "
        "and the style is purely repulsive.\n\n"
        "There is no cutoff argument, deliberately: the cutoff is DERIVED "
        "from sigma (the LJ minimum), not chosen — that is the definition of "
        "the style. Beyond it the contribution is exactly zero, not merely "
        "small.");

    cls.def(
        "set_evaluator_pair_dpd",
        [](Handle& self, const ATV::cpu::task::PairDpdParams& p) {
            self.set_categorical("set_evaluator", p,
                                 ATV::cpu::PairDpd{p.a, p.gamma, p.temperature,
                                                   p.cutoff, p.dt, p.seed});
        },
        nb::arg("comp"),
        "Append the Groot-Warren dissipative-particle-dynamics (DPD) pair "
        "evaluator. DPD is COARSE-GRAINED — a particle stands for a blob of "
        "fluid, not an atom — so the conservative force is soft, "
        "F^C = a w(r) rhat with w = 1 - r/r_c and rhat along r_i - r_j "
        "(reported energy U^C = a r_c w^2 / 2 [eV]), and the degrees of "
        "freedom that were coarsened away come back as a pairwise "
        "dissipative/random thermostat with sigma^2 = 2 gamma k_B T.\n\n"
        "It is a THERMOSTAT as well as a potential — install it instead of, "
        "not beside, a NoseHoover / Langevin / CSVR thermostat, and pass it "
        "the same dt the integrator uses (the random force scales as "
        "1/sqrt(dt)). It does NOT replace the integrator.\n\n"
        "Single-amplitude only: the C++ class's multi-component "
        "a_matrix[type_i][type_j] constructor is not bound, so one `a` "
        "[eV/A] applies to every pair.\n\n"
        "Two consequences worth stating before you read a number off a run:\n"
        "  * the reported stress carries the CONSERVATIVE virial only. The "
        "dissipative and random forces are thermostat terms with zero mean at "
        "equilibrium, and the random one diverges as 1/sqrt(dt), so tallying "
        "them would put timestep-dependent noise into every pressure without "
        "changing its average.\n"
        "  * the random force is REDRAWN every step from the {seed, step, i, "
        "j} stream, where i and j are atom INDICES. A run is reproducible for "
        "a given seed, step budget and atom ordering; a single point is not "
        "reproducible across different steps unless gamma = 0, which switches "
        "both thermostat terms off and leaves the conservative half alone.");

    cls.def(
        "set_evaluator_bond_fene",
        [](Handle& self, const ATV::cpu::task::BondFeneParams& p) {
            self.set_categorical("set_evaluator", p,
                                 ATV::cpu::BondFene{p.K, p.R0});
        },
        nb::arg("comp"),
        "Append the Kremer-Grest FENE bond evaluator, "
        "U = -K R0^2 ln(1 - (r/R0)^2) / 2. Reads the frame's bond topology; "
        "parameters are indexed by bond type.\n\n"
        "Finitely extensible, and it says so: a bond at or beyond R0 raises "
        "RuntimeError from run() rather than reporting an infinite energy. "
        "FENE is normally paired with a repulsive WCA pair style, which is "
        "what keeps chains from crossing through each other.");

    cls.def(
        "set_evaluator_dihedral_charmm",
        [](Handle& self, const ATV::cpu::task::DihedralCharmmParams& p) {
            self.set_categorical("set_evaluator", p,
                                 ATV::cpu::DihedralCharmm{p.K, p.n, p.delta});
        },
        nb::arg("comp"),
        "Append the CHARMM periodic torsion evaluator, "
        "V = K(1 + cos(n phi - delta)). Reads the frame's dihedral topology; "
        "parameters are indexed by dihedral type.\n\n"
        "This is LAMMPS `dihedral_style charmm` WITHOUT the optional 1-4 "
        "LJ/Coulomb weight factors — those belong to the pair styles via "
        "special-bonds weights, not to the torsion. The C++ class also "
        "exposes a stress-skipping performance switch "
        "(`set_compute_stress`); it is not bound and stays at its default, so "
        "the torsion always contributes to the virial.");

    cls.def(
        "set_evaluator_improper_harmonic",
        [](Handle& self, const ATV::cpu::task::ImproperHarmonicParams& p) {
            self.set_categorical("set_evaluator", p,
                                 ATV::cpu::ImproperHarmonic{p.K, p.chi0});
        },
        nb::arg("comp"),
        "Append the harmonic improper-torsion evaluator, V = K(chi - chi0)^2 "
        "(LAMMPS `improper_style harmonic`), where chi is the angle between "
        "the two planes of the 4-tuple. Reads the frame's IMPROPER topology — "
        "a separate block from `dihedrals`, holding the out-of-plane tuples "
        "that keep a group planar (an sp2 centre, an aromatic ring); "
        "parameters are indexed by improper type.");
}

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
/// The plane-wave docstring belongs to the caller: plane-wave DFT
/// cost one self-consistent-field solve per *step* under MD and per *iteration*
/// under a minimizer, and only an MD run also needs a per-atom mass column, so
/// that guidance cannot be written once. The sixteen classical setters describe
/// pair styles and topology terms, whose story does not change with the loop.
#pragma once

#include <nanobind/nanobind.h>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef ATV_HAS_KSPACE
#include "atomiverse/cpu/longrange/pme.h"
#include "atomiverse/cpu/longrange/pppm.h"
#endif
#ifdef ATV_HAS_PW
#include "atomiverse/cpu/planewave/pw_solver.h"
#endif
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_fene.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/dihedral_charmm.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/improper_cvff.h"
#include "atomiverse/cpu/potentials/improper_harmonic.h"
#include "atomiverse/cpu/potentials/pair_born.h"
#include "atomiverse/cpu/potentials/pair_buck.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_dpd.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_coul_long.h"
#include "atomiverse/cpu/potentials/pair_morse.h"
#include "atomiverse/cpu/potentials/pair_wca.h"
#include "atomiverse/cpu/potentials/pair_yukawa.h"
#ifdef ATV_HAS_SEMIEMPIRICAL
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/cpu/semiempirical/mndo_solver.h"
#include "atomiverse/cpu/semiempirical/pm3_solver.h"
#include "atomiverse/cpu/semiempirical/rm1_solver.h"
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
/// @param pw_doc     Docstring for `set_evaluator_pw`, likewise. Unused on a
///                   build without `ATV_HAS_PW`.
template <typename Handle>
void bind_evaluator_setters(nb::class_<Handle>& cls,
                            [[maybe_unused]] const char* pw_doc) {
#ifdef ATV_HAS_SEMIEMPIRICAL
    // Concrete casts preserve the nonzero DriverNode base offset. The driver
    // still borrows the original method and owns the normal nested adapter.
    cls.def(
           "set_evaluator",
           [](Handle& self, ATV::cpu::semiempirical::AM1& method) {
               self.set_local_evaluator(method, "AM1");
           },
           nb::arg("comp"), nb::keep_alive<1, 2>())
        .def(
            "set_evaluator",
            [](Handle& self, ATV::cpu::semiempirical::RM1& method) {
                self.set_local_evaluator(method, "RM1");
            },
            nb::arg("comp"), nb::keep_alive<1, 2>())
        .def(
            "set_evaluator",
            [](Handle& self, ATV::cpu::semiempirical::MNDO& method) {
                self.set_local_evaluator(method, "MNDO");
            },
            nb::arg("comp"), nb::keep_alive<1, 2>())
        .def(
            "set_evaluator",
            [](Handle& self, ATV::cpu::semiempirical::PM3& method) {
                self.set_local_evaluator(method, "PM3");
            },
            nb::arg("comp"), nb::keep_alive<1, 2>());
#endif
    cls.def(
           "set_evaluator",
           [](Handle& self, ATV::cpu::Component& evaluator) {
               self.set_local_evaluator(evaluator, evaluator.name().c_str());
           },
           nb::arg("comp"), nb::keep_alive<1, 2>())
        .def(
            "set_evaluator",
            [](Handle& self, ATV::cpu::DriverNode& evaluator) {
                self.set_local_evaluator(evaluator, "nested electronic method");
            },
            nb::arg("comp"), nb::keep_alive<1, 2>());

    cls.def(
        "set_evaluator_pair_lj_cut",
        [](Handle& self, const ATV::cpu::task::PairLJCutParams& p) {
            self.set_categorical("set_evaluator", p,
                                 ATV::cpu::task::internal::make_pair_lj_cut(p));
        },
        nb::arg("comp"),
        "Append the Lennard-Jones 12-6 pair evaluator. Needs a NeighborList "
        "whose cutoff is >= the LJ cutoff.");

    cls.def(
        "set_evaluator_pair_coul_cut",
        [](Handle& self, const ATV::cpu::task::PairCoulCutParams& p) {
            self.set_categorical(
                "set_evaluator", p,
                ATV::cpu::task::internal::make_pair_coul_cut(p));
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

    cls.def(
        "set_evaluator_pair_lj_cut_coul_long",
        [](Handle& self, const ATV::cpu::task::PairLJCutCoulLongParams& p) {
            const std::size_t rows = p.type_i.size();
            if (p.type_j.size() != rows || p.epsilon.size() != rows
                || p.sigma.size() != rows)
                throw std::invalid_argument(
                    "PairLJCutCoulLong: type_i/type_j/epsilon/sigma are one "
                    "coefficient table seen column-wise and must have the "
                    "same length");
            std::vector<ATV::cpu::LJTypeCoeff> table;
            table.reserve(rows);
            for (std::size_t k = 0; k < rows; ++k)
                table.push_back(ATV::cpu::LJTypeCoeff{
                    p.type_i[k], p.type_j[k], p.epsilon[k], p.sigma[k]});
            // Not a wire alternative: the handle records the name and the
            // charge requirement itself, as for KSpacePPPM.
            self.set_unmirrored_evaluator(
                ATV::cpu::PairLJCutCoulLong{std::move(table), p.alpha, p.cutoff}
                    .set_special_bonds(p.special_lj[0], p.special_lj[1],
                                       p.special_lj[2], p.special_coul[0],
                                       p.special_coul[1], p.special_coul[2])
                    .set_energy_shift(p.energy_shift),
                "PairLJCutCoulLong", true);
        },
        nb::arg("comp"),
        "Append fused multi-type LJ + real-space Ewald Coulomb (LAMMPS "
        "lj/cut/coul/long) with separate LJ and Coulomb special-bond weights. "
        "build() requires exactly one reciprocal partner (KSpacePME / "
        "KSpacePPPM) at the SAME alpha. Needs a NeighborList whose cutoff is "
        ">= this cutoff and a per-atom 'charge' column [e]. Not an MPI "
        "task-wire component: a launcher run that names it is refused.");

    cls.def(
        "set_evaluator_improper_cvff",
        [](Handle& self, const ATV::cpu::task::ImproperCvffParams& p) {
            self.set_unmirrored_evaluator(ATV::cpu::ImproperCvff{p.K, p.d, p.n},
                                          "ImproperCvff", false);
        },
        nb::arg("comp"),
        "Append the Amber/GAFF improper V = K(1 + d cos(n phi)). Reads the "
        "frame's improper topology; its type_id indexes the per-type "
        "vectors. Not an MPI task-wire component.");

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
    cls.def(
        "set_evaluator_kspace_pppm",
        [](Handle& self, const ATV::cpu::task::KSpacePppmParams& p) {
            // Move-only (FFTW plans). Not a wire alternative: the handle
            // records the name and the charge requirement itself.
            self.set_unmirrored_evaluator(
                ATV::cpu::KSpacePPPM{p.alpha, p.grid_spacing, p.order},
                "KSpacePPPM", true);
        },
        nb::arg("comp"),
        "Append ik PPPM, the Hockney-Eastwood reciprocal half. build() "
        "requires exactly one CoulLong at the SAME alpha [1/A]. Not an MPI "
        "task-wire component: a launcher run that names it is refused. "
        "Bound only on an extension built with ATV_BUILD_KSPACE=ON.");
#endif

#ifdef ATV_HAS_PW
    cls.def(
        "set_evaluator_pw",
        [](Handle& self, ATV::cpu::pw::PW& method) {
            self.set_local_evaluator(method, "PW");
        },
        nb::arg("comp"), nb::keep_alive<1, 2>(), pw_doc);
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
        "column is filled from the molpy `atomic_number` column, so a "
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

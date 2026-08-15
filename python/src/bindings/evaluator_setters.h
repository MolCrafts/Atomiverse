/// @file python/src/bindings/evaluator_setters.h
/// @brief The nine `set_evaluator_*` methods shared by MD and Minimize.
///
/// A force source is a force source: whatever drives molecular dynamics drives
/// a geometry optimization, so both handles expose the SAME nine method names
/// with the same argument types and the same construction — that is what lets
/// `atomiverse/_dispatch.py` be one table keyed by component type rather than
/// two. Registering them from one place is what keeps the two facades from
/// drifting apart when a tenth force source lands.
///
/// The `#ifdef` arms live here too, for the same reason: which of the nine
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
/// that guidance cannot be written once. The seven classical setters describe
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
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#ifdef ATV_HAS_SEMIEMPIRICAL
#include "atomiverse/cpu/semiempirical/am1_evaluator.h"
#endif
#include "atomiverse/cpu/task/task_spec.h"

namespace nb = nanobind;

/// @brief Register the nine force-source setters on a driver handle class.
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
}

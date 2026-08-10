/// @file facade/stage.h
/// @brief Phase enums that define the staged execution skeleton for each
///        driver type. CUDA-free header.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"

namespace ATV {

/// @brief MD simulation phases -- LAMMPS-style naming.
namespace md {

/// @brief Phase order for one molecular-dynamics step — LAMMPS-Verlet-granular
///        so the whole timestep is legible from the enum (see
///        `.claude/notes/cpu-md-execution-model.md`). Every phase is always
///        walked even when empty; an empty phase is an explicit "nothing
///        happens here" for this run, not an absence to infer. Within a phase
///        components order by a small integer (negative = infrastructure first,
///        0 = the op, positive = consumers) so a new op slots in without adding
///        a phase.
enum class Phase {
    initial_integrate, ///< First half-step: v += 0.5*dt*f/m, x += dt*v
    post_integrate,    ///< Hook right after kick/drift (usually empty)
    pre_exchange,      ///< Re-home atoms that left this rank (atom migration)
    pre_neighbor,    ///< Build ghost atoms (halo send) before the neighbor list
    neighbor,        ///< (Re)build the cell / pair neighbor list
    pre_force,       ///< Zero force/energy/stress accumulators before eval
    force,           ///< Force/energy/virial accumulation (pair + bonded)
    post_force,      ///< Reverse ghost-force comm, virial finalize, constraints
    final_integrate, ///< Second half-step: v += 0.5*dt*f/m
    end_of_step,     ///< Global reductions, thermostat/barostat, observers, I/O
    COUNT            ///< Sentinel -- total number of phases
};

/// @brief Human-readable names for the MD phases.
/// @param phase_tag Tag value used only for overload selection.
/// @return Phase names in the same order as the enum values, excluding `COUNT`.
inline std::vector<std::string> phase_names(Phase phase_tag) {
    (void)phase_tag;
    return {
        "initial_integrate", "post_integrate", "pre_exchange", "pre_neighbor",
        "neighbor",          "pre_force",      "force",        "post_force",
        "final_integrate",   "end_of_step"};
}

/// @brief Workflow discriminant for `md::Phase` (ADL tag dispatch).
/// @param phase_tag Tag value used only for overload selection.
/// @return `PhaseKind::MD`.
constexpr PhaseKind kind_of(Phase phase_tag) {
    (void)phase_tag;
    return PhaseKind::MD;
}

} // namespace md

/// @brief Energy minimizer phases.
namespace minimizer {

/// @brief Phase order for one minimization iteration.
///
/// @note `line_search` is a reserved slot with no current binder: the
///       built-in optimizers (LBFGS Armijo backtracking) implement their
///       line search inline across successive driver iterations, because
///       each trial energy needs a full `force` pass. The slot stays for
///       future exact / dedicated line-search components.
enum class Phase {
    search_direction,  ///< Compute descent direction
    line_search,       ///< Line search along descent direction (reserved)
    force,             ///< Force evaluation (shared with MD evaluators)
    convergence_check, ///< Check stopping criteria
    end_of_step,       ///< Per-iteration observers, recorders
    COUNT              ///< Sentinel -- total number of phases
};

/// @brief Human-readable names for the minimizer phases.
/// @param phase_tag Tag value used only for overload selection.
/// @return Phase names in the same order as the enum values, excluding `COUNT`.
inline std::vector<std::string> phase_names(Phase phase_tag) {
    (void)phase_tag;
    return {"search_direction", "line_search", "force", "convergence_check",
            "end_of_step"};
}

/// @brief Workflow discriminant for `minimizer::Phase` (ADL tag dispatch).
/// @param phase_tag Tag value used only for overload selection.
/// @return `PhaseKind::Minimizer`.
constexpr PhaseKind kind_of(Phase phase_tag) {
    (void)phase_tag;
    return PhaseKind::Minimizer;
}

} // namespace minimizer

/// @brief One-shot evaluation phases.
///
/// Used by `cpu::Evaluate`. There is no `end_of_step` — Evaluate
/// does not have a step concept. The three phases give callers slots to
/// place pre-evaluation setup (zeroing forces / preparing buffers),
/// the evaluator components themselves, and post-evaluation
/// finalization (recorders, observers).
namespace evaluate {

/// @brief Phase order for one Evaluate pass.
enum class Phase {
    prepare,  ///< Pre-evaluation setup (Zero*, buffer prep)
    evaluate, ///< Evaluator components run here
    finalize, ///< Post-evaluation finalization (recorders, observers)
    COUNT     ///< Sentinel -- total number of phases
};

/// @brief Human-readable names for the Evaluate phases.
/// @param phase_tag Tag value used only for overload selection.
/// @return Phase names in the same order as the enum values, excluding `COUNT`.
inline std::vector<std::string> phase_names(Phase phase_tag) {
    (void)phase_tag;
    return {"prepare", "evaluate", "finalize"};
}

/// @brief Workflow discriminant for `evaluate::Phase` (ADL tag dispatch).
/// @param phase_tag Tag value used only for overload selection.
/// @return `PhaseKind::Evaluate`.
constexpr PhaseKind kind_of(Phase phase_tag) {
    (void)phase_tag;
    return PhaseKind::Evaluate;
}

} // namespace evaluate

/// @brief Self-consistent-field (SCF) iteration phases.
///
/// The old NOTE here concluded there was *deliberately* no `scf::Phase`: SCF
/// was treated as an *inner* solver behind an Evaluator component, so it carried
/// no outer phase enum. **That conclusion is reversed.** A self-consistency loop
/// is a staged loop with a termination criterion — exactly a Driver — so the
/// gaussian-basis SCF iteration becomes a `cpu::Driver<scf::Phase>`, and the
/// concrete methods (`RHF`/`UHF`/`RKS`/`UKS`) supply the components that fill its
/// stages. CUDA SCF free-loop bag folded under RHF (RhfScfCycle); full
/// `cuda::Driver` nesting residual. GPW uses `gpw::Phase` below; plane-wave
/// `cpu::pw::PW` uses `pw::Phase` (restricted fixed-occ path). This enum is
/// CPU-only.
///
/// The phase order reflects what runs within one self-consistent iteration:
/// build the Fock/Hamiltonian matrix (`fock`), extrapolate it in **Fock space**
/// via DIIS (Direct Inversion in the Iterative Subspace; `mix`) — which must
/// precede diagonalization, matching the shipping solver (build F →
/// DIIS-extrapolate F + level shift → diagonalize) — then
/// diagonalize (`diagonalize`), build the new density (`density`), and test
/// convergence (`check`). DIIS in Fock space: Pulay, *Chem. Phys. Lett.* **73**,
/// 393 (1980); *J. Comp. Chem.* **3**, 556 (1982). `COUNT` is the sentinel for
/// `slot_count`.
namespace scf {

/// @brief Phase order for one self-consistent-field iteration.
enum class Phase {
    fock,        ///< Build the Fock / Hamiltonian matrix from the density.
    mix,         ///< DIIS extrapolation in Fock space (precedes diagonalize).
    diagonalize, ///< Solve the generalized eigenproblem for the orbitals.
    density,     ///< Build the new density from the occupied orbitals.
    check,       ///< Convergence test (energy Δ / DIIS residual).
    COUNT        ///< Sentinel -- total number of phases.
};

/// @brief Human-readable names for the SCF phases.
/// @param phase_tag Tag value used only for overload selection.
/// @return Phase names in the same order as the enum values, excluding `COUNT`.
inline std::vector<std::string> phase_names(Phase phase_tag) {
    (void)phase_tag;
    return {"fock", "mix", "diagonalize", "density", "check"};
}

/// @brief Workflow discriminant for `scf::Phase` (ADL tag dispatch).
/// @param phase_tag Tag value used only for overload selection.
/// @return `PhaseKind::SCF`.
constexpr PhaseKind kind_of(Phase phase_tag) {
    (void)phase_tag;
    return PhaseKind::SCF;
}

} // namespace scf

/// @brief Gaussian-and-plane-waves (GPW) KS self-consistency phases.
///
/// Same staged-loop shape as `scf::Phase`: the concrete method
/// `cpu::gpw::GpwDft` is a `cpu::Driver<gpw::Phase>` whose stages build the
/// KS Hamiltonian on the product grid, DIIS-mix in H-space, diagonalize,
/// rebuild the density, and test convergence. Phase indices align with
/// `scf::Phase` so a single mental model covers mean-field SCF and GPW.
/// `kind_of` returns `PhaseKind::SCF` (self-consistency workflow class).
namespace gpw {

/// @brief Phase order for one GPW KS self-consistency iteration.
enum class Phase {
    ks_build,    ///< Collocate ρ → Poisson → XC → integrate → H + energy.
    mix,         ///< DIIS extrapolation in KS-matrix space (precedes diagonalize).
    diagonalize, ///< Solve the generalized eigenproblem for the orbitals.
    density,     ///< Build the new density from the occupied orbitals.
    check,       ///< Convergence test (|ΔE| / DIIS residual).
    COUNT        ///< Sentinel — total number of phases.
};

/// @brief Human-readable names for the GPW phases.
/// @param phase_tag Tag value used only for overload selection.
/// @return Phase names in the same order as the enum values, excluding `COUNT`.
inline std::vector<std::string> phase_names(Phase phase_tag) {
    (void)phase_tag;
    return {"ks_build", "mix", "diagonalize", "density", "check"};
}

/// @brief Workflow discriminant for `gpw::Phase` (ADL tag dispatch).
/// @param phase_tag Tag value used only for overload selection.
/// @return `PhaseKind::SCF` (self-consistency loop class).
constexpr PhaseKind kind_of(Phase phase_tag) {
    (void)phase_tag;
    return PhaseKind::SCF;
}

} // namespace gpw

/// @brief Plane-wave KS self-consistency phases.
///
/// `cpu::pw::PW` is a `cpu::Driver<pw::Phase>`. Unlike Gaussian SCF / GPW
/// (H-space DIIS before diagonalize), plane-wave SCF **density-mixes**
/// (Pulay on ρ), so the stage order is
/// `veff → diagonalize → density → mix → check`. `kind_of` returns
/// `PhaseKind::SCF` (self-consistency workflow class). Restricted
/// fixed-occupation is driven by stages; collinear spin and metallic
/// smearing remain free-loop residual until a path-migration wave.
namespace pw {

/// @brief Phase order for one plane-wave KS self-consistency iteration.
enum class Phase {
    veff,        ///< Poisson + XC + Veff from the current density.
    diagonalize, ///< Per-k Davidson + density accumulation.
    density,     ///< Extract new ρ(G) (pre-mix bookkeeping).
    mix,         ///< Pulay / linear density mix.
    check,       ///< Convergence test (‖Δρ‖₂ residual).
    COUNT        ///< Sentinel — total number of phases.
};

/// @brief Human-readable names for the plane-wave phases.
/// @param phase_tag Tag value used only for overload selection.
/// @return Phase names in the same order as the enum values, excluding `COUNT`.
inline std::vector<std::string> phase_names(Phase phase_tag) {
    (void)phase_tag;
    return {"veff", "diagonalize", "density", "mix", "check"};
}

/// @brief Workflow discriminant for `pw::Phase` (ADL tag dispatch).
/// @param phase_tag Tag value used only for overload selection.
/// @return `PhaseKind::SCF` (self-consistency loop class).
constexpr PhaseKind kind_of(Phase phase_tag) {
    (void)phase_tag;
    return PhaseKind::SCF;
}

} // namespace pw

/// @brief Vibrational-analysis (finite-difference response) phases.
///
/// Used by `cpu::vib::Hessian`, which runs a fixed `6N` displacement budget
/// followed by one finalize pass. Every displacement must complete for the
/// Hessian to be defined, so there is nothing for a criterion to decide and
/// none is attached — the budget IS the termination. The phase
/// ordering reflects what runs within a single displacement pass
/// (`setup` → `displace` → `force` → `accumulate`) plus the
/// finalize-only phases (`diagonalize`, `thermochem`, `end_of_step`).
///
/// @note `force` is the slot where the user-registered force evaluator
///       runs — exactly as AIMD registers an evaluator into `md::Phase::force`.
///       `thermochem` is reserved for the thermochemistry link and
///       carries no component in the numerical-Hessian link.
namespace vib {

/// @brief Phase order for one numerical-Hessian pass.
enum class Phase {
    setup, ///< Cache the equilibrium geometry (ReferenceGeometry).
    displace, ///< Apply +/-h displacement for the current coordinate (DisplaceStep).
    force,      ///< Force evaluation (user-registered evaluator runs here).
    accumulate, ///< Central-difference Hessian-row assembly (HessianAccumulator).
    diagonalize, ///< Symmetrization / future diagonalization (Symmetrize).
    thermochem,  ///< Reserved — thermochemistry link; no component here.
    end_of_step, ///< Per-pass observers / recorders.
    COUNT        ///< Sentinel -- total number of phases.
};

/// @brief Human-readable names for the vibrational-analysis phases.
/// @param phase_tag Tag value used only for overload selection.
/// @return Phase names in the same order as the enum values, excluding `COUNT`.
inline std::vector<std::string> phase_names(Phase phase_tag) {
    (void)phase_tag;
    return {"setup",       "displace",   "force",      "accumulate",
            "diagonalize", "thermochem", "end_of_step"};
}

/// @brief Workflow discriminant for `vib::Phase` (ADL tag dispatch).
/// @param phase_tag Tag value used only for overload selection.
/// @return `PhaseKind::Vib`.
constexpr PhaseKind kind_of(Phase phase_tag) {
    (void)phase_tag;
    return PhaseKind::Vib;
}

} // namespace vib

/// @brief Monte Carlo sampling phases.
///
/// Used by `cpu::mc::MC`, which runs a budget of stochastic trial moves;
/// each sweep walks
/// `propose` → `energy` → `end_of_step`, and every phase is always walked
/// (an empty phase is an explicit "nothing happens here", not an absence to
/// infer). Within a phase, components order by a small integer as elsewhere.
namespace sampling {

/// @brief Phase order for one Monte Carlo sweep (one trial move).
///
/// @note There is deliberately **no** `accept` phase. Accept/reject is not a
///       per-phase component slot: it is an `OutcomePolicy` (`Metropolis`),
///       asked once after the stages have run, and it decides whether the
///       proposed state survives — not whether the loop ends. Exactly as MD's
///       step budget is not a phase. See `cpu/driver/outcome_policy.h`.
enum class Phase {
    propose,     ///< Mover mutates the frame in place and records the changeset
                 ///< into `Context::proposal`.
    energy,      ///< Full-frame energy of the trial state (ZeroEnergy then the
                 ///< registered `Capability::Energy` evaluator).
    end_of_step, ///< Per-sweep observers / recorders / thermo.
    COUNT        ///< Sentinel -- total number of phases.
};

/// @brief Human-readable names for the sampling phases.
/// @param phase_tag Tag value used only for overload selection.
/// @return Phase names in the same order as the enum values, excluding `COUNT`.
inline std::vector<std::string> phase_names(Phase phase_tag) {
    (void)phase_tag;
    return {"propose", "energy", "end_of_step"};
}

/// @brief Workflow discriminant for `sampling::Phase` (ADL tag dispatch).
/// @param phase_tag Tag value used only for overload selection.
/// @return `PhaseKind::Sampling`.
constexpr PhaseKind kind_of(Phase phase_tag) {
    (void)phase_tag;
    return PhaseKind::Sampling;
}

} // namespace sampling

/// @brief Path phases — N coupled replicas advanced together.
///
/// Used by `cpu::PathDriver`, whose default criterion (`cpu::AllReplicasConverged`)
/// stops the loop only when **every** replica of a coupled chain has converged —
/// an AND over the band, judged from the `ConvergenceMetrics::replicas_converged`
/// fraction the driver publishes each iteration.
///
/// @note **These names are deliberately generic, and that is load-bearing.**
///       There is no `tangent` phase, no `spring` phase, and no `climb` phase —
///       naming one would weld this phase enum to NEB, exactly the mistake the
///       architecture exists to prevent (a *method* is never a workflow). The
///       NEB tangent estimator (`cpu::path::NebCoupling`), the spring force and
///       the climbing-image flip are all **components** bound at `couple`, and
///       the band optimizer (`cpu::path::BandLBFGS`) is a component too.
///
///       `couple` is the slot where a replica sees its neighbours in the chain.
///       The same slot hosts PIMD's ring-polymer spring (mathematically the very
///       same k±1 harmonic coupling) and REMD's exchange operator, neither of
///       which is in scope here — but both of which would have needed a new
///       phase enum had this one been named after NEB.
namespace path {

/// @brief Phase order for one Path iteration (one advance of the whole chain).
enum class Phase {
    force,  ///< Per-replica force evaluation (any Capability::Force
            ///< evaluator — the same ones MD and Minimize use).
    couple, ///< Inter-replica coupling: NEB tangent + spring + the
            ///< climbing-image flip; later, PIMD's ring-polymer spring.
    search_direction, ///< Per-replica descent direction from the coupled force.
    line_search,      ///< Reserved — line search along the direction.
    convergence_check, ///< Per-replica stopping criteria → ConvergenceMetrics.
    end_of_step,       ///< Per-iteration observers / recorders.
    COUNT              ///< Sentinel -- total number of phases.
};

/// @brief Human-readable names for the path phases.
/// @param phase_tag Tag value used only for overload selection.
/// @return Phase names in the same order as the enum values, excluding `COUNT`.
inline std::vector<std::string> phase_names(Phase phase_tag) {
    (void)phase_tag;
    return {"force",
            "couple",
            "search_direction",
            "line_search",
            "convergence_check",
            "end_of_step"};
}

/// @brief Workflow discriminant for `path::Phase` (ADL tag dispatch).
/// @param phase_tag Tag value used only for overload selection.
/// @return `PhaseKind::Path`.
constexpr PhaseKind kind_of(Phase phase_tag) {
    (void)phase_tag;
    return PhaseKind::Path;
}

} // namespace path

/// @brief Phase enum → Plan slot index. **The one conversion in the codebase.**
///
/// A Plan is a dense array of slots, so somewhere the typed phase has to become an
/// index. That "somewhere" is here, and nowhere else: every other site — a component
/// asking "is this my stage?", a driver replaying a phase, a binding being declared —
/// goes through a typed entry point that calls this. The rule is enforceable by grep,
/// which is the point: `static_cast<int>` applied to a Phase enum anywhere else is a
/// bug, because the seven Phase enums all start at 0 and an `int` cannot tell them apart.
///
/// @tparam PhaseEnum One of the seven workflow Phase enums in this header.
/// @param p Typed phase value.
/// @return Its dense slot index within that workflow's Plan.
template <typename PhaseEnum>
constexpr int slot_of(PhaseEnum p) {
    return static_cast<int>(p);
}

/// @brief Number of Plan slots a workflow's Phase enum declares (its COUNT sentinel).
/// @tparam PhaseEnum One of the seven workflow Phase enums in this header.
/// @return `PhaseEnum::COUNT` as a slot count.
template <typename PhaseEnum>
constexpr int slot_count() {
    return slot_of(PhaseEnum::COUNT);
}

/// @brief Build a workflow-tagged DefaultBinding from a typed Phase value.
///
/// The single sanctioned way for a component to declare a default binding:
/// the `PhaseKind` discriminant is derived from the enum type via ADL
/// (`kind_of`), so it can never disagree with the phase index — hand-written
/// `{kind, phase, order}` triples would reintroduce the transcription errors
/// this helper exists to prevent.
///
/// The `int phase` that comes out is **the erasure itself**, not a leak:
/// `Component` is not a template, so `default_bindings()` has to return one
/// concrete type for every workflow. The `PhaseKind` travelling beside it is what
/// makes the int safe, and `make_binding` is the only way to make the pair.
///
/// @tparam PhaseEnum One of the seven workflow Phase enums in this header.
/// @param p Typed phase value the component wants to run in.
/// @param order Ordering within the phase (lower runs earlier; negative =
///        infrastructure, 0 = the op, positive = consumers).
/// @return A DefaultBinding carrying the workflow discriminant of `PhaseEnum`.
template <typename PhaseEnum>
constexpr DefaultBinding make_binding(PhaseEnum p, int order = 0) {
    return {kind_of(p), slot_of(p), order};
}

} // namespace ATV

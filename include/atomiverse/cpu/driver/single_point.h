/// @file cpu/driver/single_point.h
/// @brief cpu::single_point — free-function syntactic sugar for one-shot
///        evaluator / method-driver runs.
///
/// **NOT a class.** The spec's binding rules permit a helper function as
/// sugar for the `Evaluate + evaluator` pattern, but explicitly forbid a
/// `SinglePoint` driver class — single-point energy is one outer
/// `Evaluate` task, not its own driver category.
///
/// Binds via `on(phase).add_ref`:
/// - **Method-driver** (`cpu::scf::RHF` etc.): nested driver node, budget
///   defaults to `default_budget()` (= `max_iter`).
/// - **Component evaluator** (`PWEvaluator`, `AM1Evaluator`, …): component
///   node on the evaluate phase.
///
/// Usage:
/// @code
///   cpu::scf::RHF scf{};
///   scf.with_gradient(false);
///   cpu::single_point(frame, scf);
///   // frame.meta.get(kEnergy) populated [eV]; scf.energy() in Hartree;
///   // wavefunction on scf.determinant().
/// @endcode
///
/// The evaluator is borrowed (lvalue ref) so the caller retains
/// ownership and can inspect its result state (`determinant()` /
/// `energy()`-style accessors) after the call. To register multiple
/// components, construct a `cpu::Evaluate` directly and add them yourself.
#pragma once

#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {

/// @brief Run a single-pass evaluation with one evaluator or method-driver.
/// @param frame      System state, updated in place.
/// @param evaluator  Component evaluator or mean-field method-driver
///                   (lvalue — caller retains).
/// @param phase      Phase slot to bind on. Defaults to
///                   `evaluate::Phase::evaluate`.
template <typename Evaluator>
void single_point(cpu::Frame& frame, Evaluator& evaluator,
                  ATV::evaluate::Phase phase = ATV::evaluate::Phase::evaluate) {
    cpu::Evaluate eval;
    eval.on(phase).add_ref(evaluator);
    eval.build();
    eval.run(frame);
}

} // namespace cpu
} // namespace ATV

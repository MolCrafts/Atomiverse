/// @file cuda/driver/single_point.cuh
/// @brief cuda::single_point — free-function syntactic sugar for one-shot
///        CUDA evaluator runs.
///
/// **NOT a class.** The CUDA driver-taxonomy chain permits a helper
/// function as sugar for the `cuda::Evaluate + Evaluator` pattern, but
/// explicitly forbids a `cuda::scf::SinglePoint` driver class —
/// single-point CUDA energy is one outer `cuda::Evaluate` task, not its
/// own driver category. Mirrors `cpu::single_point`.
///
/// Usage:
/// @code
///   cuda::scf::SCF scf{integrals, tol, max_iter};
///   cuda::single_point(frame, scf);
///   // frame.scalars.energy populated.
/// @endcode
///
/// The evaluator is borrowed (lvalue ref) so the caller retains ownership
/// and can inspect its state after the call. To register multiple
/// components, construct a `cuda::Evaluate` directly and add them yourself.
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/driver/evaluate.cuh"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cuda {

/// @brief Run a single-pass CUDA evaluation with one evaluator component.
/// @tparam Evaluator Concrete evaluator component type (deduced).
/// @param frame      Host-resident system state, updated in place.
/// @param evaluator  Evaluator component to run (lvalue — caller retains).
/// @param phase      Phase slot to bind the evaluator on. Defaults to
///                   `evaluate::Phase::evaluate`. Override only if the
///                   evaluator must run during `prepare` or `finalize`.
template <typename Evaluator>
void single_point(cpu::Frame& frame, Evaluator& evaluator,
                  ATV::evaluate::Phase phase = ATV::evaluate::Phase::evaluate) {
    cuda::Evaluate eval;
    eval.on(phase).add(evaluator);
    eval.build();
    eval.run(frame);
}

} // namespace cuda
} // namespace ATV

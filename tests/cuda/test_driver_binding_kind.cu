/// @file tests/cuda/test_driver_binding_kind.cu
/// @brief arch-refactor-01-bindings — CUDA mirror of the registration-time
///        `PhaseKind` gate in `Driver::add()`.
///
/// Covers:
///   - `cuda::md::MD::add(cuda::scf::SCF{...})` throws — the
///     evaluator loses its hardcoded `default_bindings()` and falls onto
///     the existing "no default bindings. Use .on(phase).add()." path;
///   - `cuda::Evaluate::add(cuda::md::VelocityVerlet{...})` throws with an
///     error naming BOTH workflows ("MD" and "Evaluate").
///
/// `add()` only registers pending bindings — no device work — so these
/// tests need no CUDA device (mirrors
/// tests/cuda/test_cuda_driver_role_validation.cu).
///
/// Tests target the FINAL API of the spec; on the pre-refactor tree the
/// cross-workflow `add()` calls do NOT throw (silent phase-index misbind),
/// so this file presents RED as assertion failures once compiled on a CUDA
/// node.
///
/// Spec: `.claude/specs/arch-refactor-01-bindings.md` (Files E).

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "atomiverse/cuda/driver/evaluate.cuh"
#include "atomiverse/cuda/md/md.cuh"
#include "atomiverse/cuda/md/velocity_verlet.cuh"
#include "atomiverse/cuda/scf/scf_evaluator.cuh"

namespace {

bool message_mentions(const std::runtime_error& e, const std::string& needle) {
    return std::string(e.what()).find(needle) != std::string::npos;
}

// ─── Violating evaluator: cuda SCF loses default_bindings() ────
//
// Regression target: pre-refactor, `md.add(SCF{...})` silently
// landed the evaluator on `md::Phase::post_integrate` (the erased int `1`
// of `evaluate::Phase::evaluate`). Post-refactor the override is deleted,
// so `add()` takes the existing "has no default bindings" rejection path.

TEST(CudaDriverBindingKind, MdAddScfEvaluatorThrows) {
    ATV::cuda::md::MD md;
    // Synthetic-path constructor (n_occ, alpha, tol, max_iter) — available
    // with or without ATV_HAS_HF.
    EXPECT_THROW(md.add(ATV::cuda::scf::SCF{2, 0.1, 1e-6, 100}),
                 std::runtime_error);
}

// ─── Kind gate: error names BOTH workflows ──────────────────────────────

TEST(CudaDriverBindingKind, EvaluateAddVelocityVerletThrowsWithBothKinds) {
    ATV::cuda::Evaluate ev;
    try {
        // rvalue `add()` branch.
        ev.add(ATV::cuda::md::VelocityVerlet{0.01});
        FAIL() << "expected throw — MD-kind default bindings registered "
                  "into a cuda::Evaluate driver";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "MD")) << e.what();
        EXPECT_TRUE(message_mentions(e, "Evaluate")) << e.what();
    }
}

} // namespace

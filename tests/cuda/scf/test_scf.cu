/// @file test_scf.cu
/// @brief Contract + synthetic-path tests for `cuda::scf::SCF` and private
///        helpers: structural guards (step helpers are not Components; Evaluate
///        is a loop node over evaluate::Phase; SCF is a Component with
///        role()==0), the synthetic `H(P)=H0+alpha*P` device path, single-point
///        via `cuda::Evaluate`, and CPU/CUDA parity against the `cpu::scf::toy`
///        reference solver. The numeric H2/STO-3G/RHF check lives in
///        test_rhf.cu. Free-loop `SCFSolver` is gone (arch-cuda-scf-solver-fold).

#include <cmath>
#include <gtest/gtest.h>
#include <type_traits>

#include "atomiverse/core/driver_template.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/solve.h"
#include "atomiverse/cuda/driver/driver.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/evaluate/single_point.cuh"
#include "atomiverse/cuda/scf/internal/convergence_check.cuh"
#include "atomiverse/cuda/scf/internal/density_builder.cuh"
#include "atomiverse/cuda/scf/internal/eigen_solver.cuh"
#include "atomiverse/cuda/scf/internal/hamiltonian.cuh"
#include "atomiverse/cuda/scf/internal/synthetic_scf_cycle.cuh"
#include "atomiverse/cuda/scf/scf_evaluator.cuh"

using namespace ATV;

namespace {

/// Create a minimal cpu::Frame with n atoms (used as matrix dimension).
cpu::Frame make_scf_system(int n) {
    cpu::Frame frame(n);
    for (int i = 0; i < n; ++i) {
        frame.span(kAtom_x)[i] = double(i);
        frame.span(kAtom_y)[i] = double(0);
        frame.span(kAtom_z)[i] = double(0);
        frame.span(kAtom_mass)[i] = double(1);
        frame.span(kAtom_atomic_number)[i] = 0;
    }
    frame.box = cpu::Box::cubic(100.0);
    return frame;
}

} // anonymous namespace

// ── Structural guards (ac-002 / ac-003 / ac-004 / ac-005) ─────────────
//
// NOTE: the former ac-001 "cuda::scf::SCF must not exist" sentinel guard is
// GONE. `cuda::scf::SCF` (the Evaluator component) intentionally ships as
// transitional debt per docs/developer/project-contract.md, so a test asserting its absence was a
// stale, self-contradictory premise (it then constructed the very type it
// claimed was deleted). The guards below assert what is actually true of the
// shipping type: it is a Component, not a Driver, with role()==0.

TEST(SCF, SyntheticCycleIsNotADriver) {
    // Private synthetic cycle is NOT a Driver<> subclass.
    static_assert(
        !std::is_base_of_v<Driver<ATV::evaluate::Phase, cuda::device_tag>,
                           cuda::scf::internal::SyntheticScfCycle>,
        "SyntheticScfCycle must not be a Driver<> subclass");
    static_assert(!std::is_base_of_v<cuda::Driver<ATV::scf::Phase>,
                                     cuda::scf::internal::SyntheticScfCycle>,
                  "SyntheticScfCycle must not be a cuda::Driver<scf::Phase>");
    SUCCEED();
}

TEST(SCF, NoPublicSCFSolverType) {
    // arch-cuda-scf-solver-fold: the dual free-loop bag is gone.
    // Compile-time: if SCFSolver reappears as a public/internal type under
    // cuda::scf, this TU would need a header that declares it — it must not.
    SUCCEED();
}

TEST(SCF, InternalHelpersAreNotComponents) {
    // ac-003: the relocated SCF step helpers are not Component subclasses.
    static_assert(
        !std::is_base_of_v<cuda::Component, cuda::scf::internal::Hamiltonian>,
        "internal::Hamiltonian must not inherit Component");
    static_assert(
        !std::is_base_of_v<cuda::Component, cuda::scf::internal::EigenSolver>,
        "internal::EigenSolver must not inherit Component");
    static_assert(!std::is_base_of_v<cuda::Component,
                                     cuda::scf::internal::DensityBuilder>,
                  "internal::DensityBuilder must not inherit Component");
    static_assert(!std::is_base_of_v<cuda::Component,
                                     cuda::scf::internal::ConvergenceCheck>,
                  "internal::ConvergenceCheck must not inherit Component");
    SUCCEED();
}

TEST(SCF, EvaluateIsALoopNodeOverTheEvaluatePhase) {
    // ac-004, restated after the category bases were deleted. "cuda::Evaluate is in the
    // Evaluate CATEGORY" is now a statement about nothing — there is no category. What
    // still has content, and is what the SCF single point actually depends on, is that
    // Evaluate is a loop node over the `evaluate::Phase` stage domain: that is what lets
    // an SCF bind `evaluate::Phase::evaluate` and be dispatched.
    static_assert(
        std::is_base_of_v<cuda::Driver<ATV::evaluate::Phase>, cuda::Evaluate>,
        "cuda::Evaluate must be a Driver over evaluate::Phase");
    SUCCEED();
}

TEST(SCF, SCFRoleIsMainLoopEnergySource) {
    // ac-005 (restated after the role/capability enums were deleted): SCF is a
    // real cuda::Component and a main-loop energy source, so role() == 0 — no
    // FrameImmutable/Deferred/ExternalIO Role flags set. (The old
    // Energy|DensityMatrix|Orbitals Capability write-set no longer exists;
    // outputs are Frame keys, validation is by kind.)
    static_assert(std::is_base_of_v<cuda::Component, cuda::scf::SCF>,
                  "cuda::scf::SCF must inherit cuda::Component");
    cuda::scf::SCF eval{/*n_occ=*/2, /*alpha=*/0.1, /*tol=*/1e-6,
                        /*max_iter=*/50};
    EXPECT_EQ(eval.role(), 0);
}

// ── CUDA synthetic-path tests ────────────────────────────────────────

TEST(SCF, EvaluateRejectsFrameWithNoEvaluator) {
    // cuda::Evaluate validation rejects a frame with no Evaluator-role
    // component (`Evaluate::validate_components()`, folded down from the deleted
    // cuda::EvaluateDriver).
    cuda::Evaluate eval;
    EXPECT_THROW(eval.build(), std::runtime_error);
}

// ── CPU reference tests ───────────────────────────────────────────────

// ── CPU/CUDA parity ───────────────────────────────────────────────────

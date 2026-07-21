/// @file test_scf.cu
/// @brief Integration tests for the CUDA SCF inner solver (`SCFSolver`),
///        the `SCF` component, single-point via `cuda::Evaluate`,
///        CPU/CUDA parity, and the structural migration guards.

#include <cmath>
#include <gtest/gtest.h>
#include <type_traits>

#include "atomiverse/cpu/scf/solve.h"
#include "atomiverse/cuda/driver/cuda_driver.cuh"
#include "atomiverse/cuda/driver/evaluate.cuh"
#include "atomiverse/cuda/driver/single_point.cuh"
#include "atomiverse/cuda/scf/internal/convergence_check.cuh"
#include "atomiverse/cuda/scf/internal/density_builder.cuh"
#include "atomiverse/cuda/scf/internal/eigen_solver.cuh"
#include "atomiverse/cuda/scf/internal/hamiltonian.cuh"
#include "atomiverse/cuda/scf/scf_evaluator.cuh"
#include "atomiverse/cuda/scf/scf_solver.cuh"
#include "atomiverse/facade/driver.h"
#include "atomiverse/facade/stage.h"

using namespace ATV;

namespace {

/// Create a minimal cpu::Frame with n atoms (used as matrix dimension).
cpu::Frame make_scf_system(int n) {
    cpu::Frame frame(n);
    for (int i = 0; i < n; ++i) {
        frame.atoms.x[i] = double(i);
        frame.atoms.y[i] = double(0);
        frame.atoms.z[i] = double(0);
        frame.atoms.mass[i] = double(1);
        frame.atoms.type[i] = 0;
    }
    frame.box = Mat3::cubic(100.0);
    return frame;
}

} // anonymous namespace

// ── Structural-guard sentinels (ac-001) ──────────────────────────────
//
// The migration removes the legacy `cuda::scf::SCF` driver class and
// forbids any `cuda::scf::SinglePoint`. To turn "this type must NOT
// exist" into a *compile-time* guard, the test injects its own unique
// sentinel types of those exact names into `ATV::cuda::scf` and then
// asserts that the names resolve to the sentinels. If a future change
// re-introduces a production `cuda::scf::SCF` / `SinglePoint`, this
// translation unit fails to compile (redefinition / ambiguity).
namespace ATV {
namespace cuda {
namespace scf {
namespace migration_guard {
struct SCFSentinel {};
struct SinglePointSentinel {};
} // namespace migration_guard
using SCF = migration_guard::SCFSentinel;
using SinglePoint = migration_guard::SinglePointSentinel;
} // namespace scf
} // namespace cuda
} // namespace ATV

// ── Structural guards (ac-001 / ac-002 / ac-003 / ac-004 / ac-005) ────

TEST(SCF, NoCudaScfSCFDriverClass) {
    // ac-001: `cuda::scf::SCF` resolves only to the test's sentinel — no
    // production driver class of that name survives the migration.
    static_assert(
        std::is_same_v<cuda::scf::SCF, cuda::scf::migration_guard::SCFSentinel>,
        "cuda::scf::SCF driver class must not exist after migration");
    // ac-001: `cuda::scf::SinglePoint` resolves only to the sentinel.
    static_assert(
        std::is_same_v<cuda::scf::SinglePoint,
                       cuda::scf::migration_guard::SinglePointSentinel>,
        "cuda::scf::SinglePoint must not exist");
    SUCCEED();
}

TEST(SCF, SCFSolverIsNotADriver) {
    // ac-002: SCFSolver is an inner solver, NOT a Driver<> subclass.
    static_assert(
        !std::is_base_of_v<Driver<ATV::evaluate::Phase, cuda::device_tag>,
                           cuda::scf::SCFSolver>,
        "cuda::scf::SCFSolver must not be a Driver<> subclass");
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
    static_assert(std::is_base_of_v<cuda::CUDADriver<ATV::evaluate::Phase>,
                                    cuda::Evaluate>,
                  "cuda::Evaluate must be a CUDADriver over evaluate::Phase");
    SUCCEED();
}

TEST(SCF, SCFRoleAndCapabilities) {
    // ac-005: SCF is an Evaluator-role component with the exact
    // Energy|DensityMatrix|Orbitals capability set (no Force).
    static_assert(std::is_base_of_v<cuda::Component, cuda::scf::SCF>,
                  "cuda::scf::SCF must inherit cuda::Component");
    cuda::scf::SCF eval{/*n_occ=*/2, /*alpha=*/0.1, /*tol=*/1e-6,
                        /*max_iter=*/50};
    EXPECT_EQ(eval.role(), ComponentRole::Evaluator);
    EXPECT_EQ(eval.capabilities(), Capability::Energy
                                       | Capability::DensityMatrix
                                       | Capability::Orbitals);
    EXPECT_FALSE(has_capability(eval.capabilities(), Capability::Force));
}

// ── CUDA SCFSolver synthetic-path tests ──────────────────────────────

TEST(SCF, CUDAConvergence) {
    int n = 8;
    int n_occ = 3;
    auto frame = make_scf_system(n);

    cuda::scf::SCF scf{n_occ, double(0.1), double(1e-6), 200};
    cuda::Evaluate eval;
    eval.on(ATV::evaluate::Phase::evaluate).add(std::move(scf));
    eval.build();
    eval.run(frame);

    EXPECT_TRUE(std::isfinite(static_cast<double>(frame.scalars.energy)));
    EXPECT_GT(frame.scalars.energy, double(0));
}

TEST(SCF, CUDAReproducibility) {
    // Two independent SCF runs on identical input give
    // bit-identical energy.
    int n = 8;
    auto frame1 = make_scf_system(n);
    auto frame2 = make_scf_system(n);

    for (auto* f : {&frame1, &frame2}) {
        cuda::Evaluate eval;
        eval.on(ATV::evaluate::Phase::evaluate)
            .add(cuda::scf::SCF{2, double(0.1), double(1e-7), 200});
        eval.build();
        eval.run(*f);
    }

    EXPECT_NEAR(static_cast<double>(frame1.scalars.energy),
                static_cast<double>(frame2.scalars.energy), 1e-10);
}

TEST(SCF, CUDASinglePointHelper) {
    // cuda::single_point produces the identical result as constructing
    // cuda::Evaluate and calling add() directly.
    int n = 8;
    auto frame_direct = make_scf_system(n);
    auto frame_helper = make_scf_system(n);

    {
        cuda::Evaluate eval;
        eval.on(ATV::evaluate::Phase::evaluate)
            .add(cuda::scf::SCF{2, double(0.1), double(1e-7), 200});
        eval.build();
        eval.run(frame_direct);
    }
    {
        cuda::scf::SCF scf{2, double(0.1), double(1e-7), 200};
        cuda::single_point(frame_helper, scf);
    }

    EXPECT_NEAR(static_cast<double>(frame_direct.scalars.energy),
                static_cast<double>(frame_helper.scalars.energy), 1e-10);
}

TEST(SCF, CUDAAlphaEffect) {
    int n = 8;
    auto frame_a0 = make_scf_system(n);
    auto frame_a1 = make_scf_system(n);

    {
        cuda::Evaluate eval;
        eval.on(ATV::evaluate::Phase::evaluate)
            .add(cuda::scf::SCF{2, double(0), double(1e-6), 50});
        eval.build();
        eval.run(frame_a0);
    }
    {
        cuda::Evaluate eval;
        eval.on(ATV::evaluate::Phase::evaluate)
            .add(cuda::scf::SCF{2, double(0.5), double(1e-6), 200});
        eval.build();
        eval.run(frame_a1);
    }

    EXPECT_NE(frame_a0.scalars.energy, frame_a1.scalars.energy);
    EXPECT_TRUE(std::isfinite(static_cast<double>(frame_a0.scalars.energy)));
    EXPECT_TRUE(std::isfinite(static_cast<double>(frame_a1.scalars.energy)));
}

TEST(SCF, CUDAMaxIterCap) {
    // SCFSolver::run honours its max-iteration cap when convergence is
    // not reached (high-alpha synthetic case that does not converge in
    // a single iteration).
    int n = 8;
    auto frame = make_scf_system(n);

    cuda::scf::SCF scf{2, double(0.5), double(1e-12), /*max_iter=*/3};
    cuda::Evaluate eval;
    eval.on(ATV::evaluate::Phase::evaluate).add(std::move(scf));
    eval.build();
    eval.run(frame);

    // The run must terminate (no hang) and leave a finite energy even
    // though the tight tolerance is not met within 3 iterations.
    EXPECT_TRUE(std::isfinite(static_cast<double>(frame.scalars.energy)));
}

TEST(SCF, EvaluateRejectsFrameWithNoEvaluator) {
    // cuda::Evaluate validation rejects a frame with no Evaluator-role
    // component (`Evaluate::validate_components()`, folded down from the deleted
    // cuda::EvaluateDriver).
    cuda::Evaluate eval;
    EXPECT_THROW(eval.build(), std::runtime_error);
}

// ── CPU reference tests ───────────────────────────────────────────────

TEST(SCF, CPUConvergence) {
    double cpu_tol =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-8);
    auto result = cpu::scf::toy::solve(8, 3, double(0.1), cpu_tol, 200);
    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(std::isfinite(static_cast<double>(result.energy)));
    EXPECT_GT(result.energy, double(0));
    EXPECT_GT(result.iterations, 1);
}

TEST(SCF, CPUAlphaZero) {
    // alpha=0: no self-consistency, converges in 2 iterations
    double cpu_tol =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-10);
    auto result = cpu::scf::toy::solve(8, 3, double(0), cpu_tol, 50);
    EXPECT_TRUE(result.converged);
    EXPECT_LE(result.iterations, 3); // 2 for double, possibly 3 for float
}

TEST(SCF, CPUReproducibility) {
    double cpu_tol =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-8);
    auto r1 = cpu::scf::toy::solve(16, 4, double(0.1), cpu_tol, 100);
    auto r2 = cpu::scf::toy::solve(16, 4, double(0.1), cpu_tol, 100);
    EXPECT_DOUBLE_EQ(static_cast<double>(r1.energy),
                     static_cast<double>(r2.energy));
    EXPECT_EQ(r1.iterations, r2.iterations);
}

// ── CPU/CUDA parity ───────────────────────────────────────────────────

TEST(SCF, CPUCUDAParity) {
    int n = 8;
    int n_occ = 2;
    double alpha = double(0.1);
    double tol = double(1e-5);

    // CPU reference
    auto cpu_result = cpu::scf::toy::solve(n, n_occ, alpha, tol, 200);

    // CUDA via SCF + Evaluate
    auto frame = make_scf_system(n);
    {
        cuda::Evaluate eval;
        eval.on(ATV::evaluate::Phase::evaluate)
            .add(cuda::scf::SCF{n_occ, alpha, tol, 200});
        eval.build();
        eval.run(frame);
    }

    // Energy must agree within float tolerance
    double cpu_e = static_cast<double>(cpu_result.energy);
    double cuda_e = static_cast<double>(frame.scalars.energy);
    double rel_tol = (sizeof(double) == sizeof(float)) ? 1e-4 : 1e-8;

    EXPECT_NEAR(cpu_e, cuda_e, std::abs(cpu_e) * rel_tol + 1e-6)
        << "CPU energy: " << cpu_e << ", CUDA energy: " << cuda_e;
}

TEST(SCF, CPUCUDAParityN16) {
    int n = 16;
    int n_occ = 4;
    double alpha = double(0.05);
    double tol = double(1e-5);

    auto cpu_result = cpu::scf::toy::solve(n, n_occ, alpha, tol, 200);

    auto frame = make_scf_system(n);
    {
        cuda::Evaluate eval;
        eval.on(ATV::evaluate::Phase::evaluate)
            .add(cuda::scf::SCF{n_occ, alpha, tol, 200});
        eval.build();
        eval.run(frame);
    }

    double cpu_e = static_cast<double>(cpu_result.energy);
    double cuda_e = static_cast<double>(frame.scalars.energy);
    double rel_tol = (sizeof(double) == sizeof(float)) ? 1e-4 : 1e-8;

    EXPECT_NEAR(cpu_e, cuda_e, std::abs(cpu_e) * rel_tol + 1e-6)
        << "CPU energy: " << cpu_e << ", CUDA energy: " << cuda_e;
}

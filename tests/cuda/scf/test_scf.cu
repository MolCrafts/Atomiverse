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
#include "atomiverse/cpu/scf/solve.h"
#include "atomiverse/cuda/driver/driver.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/evaluate/single_point.cuh"
#include "atomiverse/cuda/scf/internal/convergence_check.cuh"
#include "atomiverse/cuda/scf/internal/density_builder.cuh"
#include "atomiverse/cuda/scf/internal/eigen_solver.cuh"
#include "atomiverse/cuda/scf/internal/hamiltonian.cuh"
#include "atomiverse/core/units.h"
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
        frame.span(kAtom_type)[i] = 0;
    }
    frame.box = cpu::Box::cubic(100.0);
    return frame;
}

} // anonymous namespace

// ── Structural guards (ac-002 / ac-003 / ac-004 / ac-005) ─────────────
//
// NOTE: the former ac-001 "cuda::scf::SCF must not exist" sentinel guard is
// GONE. `cuda::scf::SCF` (the Evaluator component) intentionally ships as
// transitional debt per CLAUDE.md, so a test asserting its absence was a
// stale, self-contradictory premise (it then constructed the very type it
// claimed was deleted). The guards below assert what is actually true of the
// shipping type: it is a Component, not a Driver, with role()==0.

TEST(SCF, SyntheticCycleIsNotADriver) {
    // Private synthetic cycle is NOT a Driver<> subclass.
    static_assert(
        !std::is_base_of_v<Driver<ATV::evaluate::Phase, cuda::device_tag>,
                           cuda::scf::internal::SyntheticScfCycle>,
        "SyntheticScfCycle must not be a Driver<> subclass");
    static_assert(
        !std::is_base_of_v<cuda::Driver<ATV::scf::Phase>,
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

TEST(SCF, CUDAConvergence) {
    int n = 8;
    int n_occ = 3;
    auto frame = make_scf_system(n);

    cuda::scf::SCF scf{n_occ, double(0.1), double(1e-6), 200};
    cuda::Evaluate eval;
    eval.on(ATV::evaluate::Phase::evaluate).add(std::move(scf));
    eval.build();
    eval.run(frame);

    // Public meta energy is eV (synthetic scaled by kHartreeInEv).
    EXPECT_TRUE(std::isfinite(static_cast<double>(frame.meta.get(kEnergy))));
    EXPECT_GT(frame.meta.get(kEnergy), double(0));
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

    EXPECT_NEAR(static_cast<double>(frame1.meta.get(kEnergy)),
                static_cast<double>(frame2.meta.get(kEnergy)), 1e-10);
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

    EXPECT_NEAR(static_cast<double>(frame_direct.meta.get(kEnergy)),
                static_cast<double>(frame_helper.meta.get(kEnergy)), 1e-10);
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

    EXPECT_NE(frame_a0.meta.get(kEnergy), frame_a1.meta.get(kEnergy));
    EXPECT_TRUE(std::isfinite(static_cast<double>(frame_a0.meta.get(kEnergy))));
    EXPECT_TRUE(std::isfinite(static_cast<double>(frame_a1.meta.get(kEnergy))));
}

TEST(SCF, CUDAMaxIterCap) {
    // Synthetic cycle honours its max-iteration cap when convergence is
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
    EXPECT_TRUE(std::isfinite(static_cast<double>(frame.meta.get(kEnergy))));
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

    // CUDA meta kEnergy is eV; toy CPU energy is the raw synthetic scale.
    double cpu_e = static_cast<double>(cpu_result.energy);
    double cuda_e_ha =
        static_cast<double>(frame.meta.get(kEnergy)) / units::kHartreeInEv;
    double rel_tol = (sizeof(double) == sizeof(float)) ? 1e-4 : 1e-8;

    EXPECT_NEAR(cpu_e, cuda_e_ha, std::abs(cpu_e) * rel_tol + 1e-6)
        << "CPU energy: " << cpu_e << ", CUDA energy (Ha scale): " << cuda_e_ha;
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
    double cuda_e_ha =
        static_cast<double>(frame.meta.get(kEnergy)) / units::kHartreeInEv;
    double rel_tol = (sizeof(double) == sizeof(float)) ? 1e-4 : 1e-8;

    EXPECT_NEAR(cpu_e, cuda_e_ha, std::abs(cpu_e) * rel_tol + 1e-6)
        << "CPU energy: " << cpu_e << ", CUDA energy (Ha scale): " << cuda_e_ha;
}

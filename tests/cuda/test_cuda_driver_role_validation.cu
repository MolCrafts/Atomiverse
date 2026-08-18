/// @file tests/cuda/test_cuda_driver_role_validation.cu
/// @brief Assert that each CUDA preset — cuda::Evaluate / cuda::md::MD /
///        cuda::minimizer::Minimize — throws at build() time when its required role +
///        capability set is not satisfied, and builds successfully when it is.
///
/// The three category bases (`cuda::EvaluateDriver` / `DynamicsDriver` /
/// `OptimizeDriver`) are deleted: termination is a `Criterion`, not a class, and with
/// termination gone the bases had nothing left to be. Their `validate_components()`
/// contracts were folded DOWN into the presets, and this file drives them through the
/// preset's `build()` — which is the only way a user ever reached them anyway (nobody
/// instantiated a bare category base outside this test).
///
/// The presets bind their own infrastructure in their constructors (Zero* etc.), and
/// `Minimize` attaches a default `ForceConverged` handler in `install_defaults()`. None
/// of that satisfies a role requirement, so the gates below still fire on exactly the
/// same missing components as before.
///
/// Mirrors tests/cpu/test_cpu_driver_role_validation.cpp.
///
/// `build()` only validates the component set and compiles the Plan — it
/// performs no device work — so these tests need no CUDA device.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/component/optimizer.cuh"
#include "atomiverse/cuda/component/potential.cuh"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/minimize/minimize.cuh"

namespace {

bool message_mentions(const std::runtime_error& e, const std::string& needle) {
    return std::string(e.what()).find(needle) != std::string::npos;
}

// ─── CUDA component stubs — current Role/kind model ─────────────────
//
// The old per-component role enum and the write-set bitset are both
// DELETED. A CUDA Component now carries only a `Role` bitset (`RoleFlag`s from
// core/role.h): a frame-mutating main-loop component has `role() == 0`,
// a frame-immutable structural component sets `RoleFlag::FrameImmutable`
// (plus `Deferred`). "Which KIND is it" (Integrator / Optimizer / force
// source) is decided by the concrete component and detected in the preset's
// `validate_components()` — the CUDA presets do this by the integrator/
// optimizer name token plus the `role()` flags (CUDA kind bases are not yet
// split; see cuda/dynamics/md.cuh + cuda/minimize/minimize.cuh). These stubs
// mirror the kind-derived helpers in tests/cpu/driver_test_helpers.h.

/// @brief Integrator stub — reuses the real `cuda::md::VelocityVerlet`, the
///        integrator `cuda::md::MD::validate_components()` accepts (name
///        "velocity_verlet"; `role() == 0`). Bound explicitly in each test.
class StubIntegrator : public ATV::cuda::md::VelocityVerlet {
  public:
    StubIntegrator() : ATV::cuda::md::VelocityVerlet(/*dt [fs]=*/1.0) {}
};

/// @brief Hot-path force source — `role() == 0` (main-loop, frame-mutating),
///        not an integrator/optimizer name, not a NeighborConsumer.
class StubForceEvaluator : public ATV::cuda::Potential {
  public:
    std::string name() const override { return "stub_force"; }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

/// @brief Hot-path energy source — `role() == 0`. The current CUDA
///        `validate_components()` has no Energy-vs-Force distinction (that
///        lived on the now-deleted component write-set), so this is a
///        generic frame-mutating source, structurally identical to
///        StubForceEvaluator.
class StubEnergyEvaluator : public ATV::cuda::Potential {
  public:
    std::string name() const override { return "stub_energy"; }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

/// @brief Optimizer stub — `name()` carries the "opt" token
///        `cuda::minimizer::Minimize::validate_components()` matches on;
///        `role() == 0` (an optimizer mutates x).
class StubOptimizer : public ATV::cuda::Optimizer {
  public:
    std::string name() const override { return "stub_optimizer"; }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

/// @brief Structural-only stub — `RoleFlag::FrameImmutable | Deferred`, so it
///        is never counted as a hot-path energy/force source. Mirrors
///        `atv_test::StubBareInfra`.
class StubBareInfra : public ATV::cuda::Component {
  public:
    std::string name() const override { return "stub_bare_infra"; }
    ATV::Role role() const override {
        return ATV::RoleFlag::FrameImmutable | ATV::RoleFlag::Deferred;
    }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

} // namespace

// ─── cuda::md::MD (was: DynamicsDriver) ─────────────────────────────

TEST(CUDADriverRoleValidation, MDThrowsWithoutIntegrator) {
    using namespace ATV;
    cuda::md::MD d;
    StubForceEvaluator force;
    d.on(md::Phase::force).add_ref(force);
    try {
        d.build();
        FAIL() << "expected throw — MD missing Integrator";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "Integrator")) << e.what();
    }
}

TEST(CUDADriverRoleValidation, MDThrowsWithoutForceEvaluator) {
    using namespace ATV;
    cuda::md::MD d;
    StubIntegrator integ;
    StubBareInfra bare; // non-force component: MD still needs a Potential/HF
    d.on(md::Phase::initial_integrate).add_ref(integ);
    d.on(md::Phase::force).add_ref(bare);
    try {
        d.build();
        FAIL() << "expected throw — MD missing Force-capable evaluator";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "force source")) << e.what();
    }
}

TEST(CUDADriverRoleValidation, MDBuildsWithIntegratorAndForce) {
    using namespace ATV;
    cuda::md::MD d;
    StubIntegrator integ;
    StubForceEvaluator force;
    d.on(md::Phase::initial_integrate).add_ref(integ);
    d.on(md::Phase::force).add_ref(force);
    EXPECT_NO_THROW(d.build());
}

/// A time budget is not a policy: MD attaches no default handler. (Minimize does —
/// see below — and that asymmetry is the point of a preset.)
TEST(CUDADriverRoleValidation, MDAttachesNoDefaultHandler) {
    using namespace ATV;
    cuda::md::MD d;
    StubIntegrator integ;
    StubForceEvaluator force;
    d.on(md::Phase::initial_integrate).add_ref(integ);
    d.on(md::Phase::force).add_ref(force);
    d.build();
    EXPECT_FALSE(d.has_handler());
}

// ─── cuda::minimizer::Minimize (was: OptimizeDriver) ────────────────

TEST(CUDADriverRoleValidation, MinimizeThrowsWithoutOptimizer) {
    using namespace ATV;
    cuda::minimizer::Minimize o;
    StubForceEvaluator force;
    o.on(minimizer::Phase::force).add_ref(force);
    try {
        o.build();
        FAIL() << "expected throw — Minimize missing Optimizer";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "Optimizer")) << e.what();
    }
}

TEST(CUDADriverRoleValidation, MinimizeThrowsWithoutEnergyOrForceEvaluator) {
    using namespace ATV;
    cuda::minimizer::Minimize o;
    StubOptimizer opt;
    StubBareInfra infra;
    o.on(minimizer::Phase::search_direction).add_ref(opt);
    o.on(minimizer::Phase::end_of_step).add_ref(infra);
    try {
        o.build();
        FAIL() << "expected throw — Minimize missing Energy/Force evaluator";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "energy or force source")) << e.what();
    }
}

TEST(CUDADriverRoleValidation, MinimizeBuildsWithOptimizerAndEnergy) {
    using namespace ATV;
    cuda::minimizer::Minimize o;
    StubOptimizer opt;
    StubEnergyEvaluator energy;
    o.on(minimizer::Phase::search_direction).add_ref(opt);
    o.on(minimizer::Phase::force).add_ref(energy);
    EXPECT_NO_THROW(o.build());
}

/// The preset supplies the policy it is a preset for: a Minimize with no user handler
/// gets `ForceConverged` — otherwise the optimizer would publish metrics and nobody
/// would ever judge them, and the loop could only ever end on its budget.
TEST(CUDADriverRoleValidation, MinimizeAttachesDefaultForceConvergedHandler) {
    using namespace ATV;
    cuda::minimizer::Minimize o;
    StubOptimizer opt;
    StubEnergyEvaluator energy;
    o.on(minimizer::Phase::search_direction).add_ref(opt);
    o.on(minimizer::Phase::force).add_ref(energy);
    EXPECT_FALSE(o.has_handler());
    o.build();
    EXPECT_TRUE(o.has_handler());
}

/// …and a user handler suppresses the default rather than stacking on top of it.
TEST(CUDADriverRoleValidation, MinimizeKeepsTheUserHandler) {
    using namespace ATV;
    cuda::minimizer::Minimize o;
    StubOptimizer opt;
    StubEnergyEvaluator energy;
    o.add_criterion(cuda::ForceConverged{/*ftol=*/1e-4});
    o.on(minimizer::Phase::search_direction).add_ref(opt);
    o.on(minimizer::Phase::force).add_ref(energy);
    EXPECT_TRUE(o.has_handler());
    EXPECT_NO_THROW(o.build());
}

// ─── cuda::Evaluate (was: EvaluateDriver) ───────────────────────────

TEST(CUDADriverRoleValidation, EvaluateThrowsWithoutEvaluator) {
    using namespace ATV;
    cuda::Evaluate e;
    StubBareInfra infra;
    e.on(evaluate::Phase::prepare).add_ref(infra);
    try {
        e.build();
        FAIL() << "expected throw — Evaluate missing Evaluator";
    } catch (const std::runtime_error& ex) {
        EXPECT_TRUE(message_mentions(ex, "energy or force source"))
            << ex.what();
    }
}

TEST(CUDADriverRoleValidation, EvaluateBuildsWithEnergyEvaluator) {
    using namespace ATV;
    cuda::Evaluate e;
    StubEnergyEvaluator energy;
    e.on(evaluate::Phase::evaluate).add_ref(energy);
    EXPECT_NO_THROW(e.build());
}

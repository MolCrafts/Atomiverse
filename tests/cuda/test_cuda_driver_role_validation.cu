/// @file tests/cuda/test_cuda_driver_role_validation.cu
/// @brief Assert that each CUDA preset — cuda::Evaluate / cuda::md::MD /
///        cuda::minimizer::Minimize — throws at build() time when its required role +
///        capability set is not satisfied, and builds successfully when it is.
///
/// The three category bases (`cuda::EvaluateDriver` / `DynamicsDriver` /
/// `OptimizeDriver`) are deleted: termination is a `LoopHandler`, not a class, and with
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

#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/driver/evaluate.cuh"
#include "atomiverse/cuda/md/md.cuh"
#include "atomiverse/cuda/minimizer/minimize.cuh"
#include "atomiverse/facade/component_role.h"
#include "atomiverse/facade/stage.h"

namespace {

bool message_mentions(const std::runtime_error& e, const std::string& needle) {
    return std::string(e.what()).find(needle) != std::string::npos;
}

// ─── CUDA component stubs ───────────────────────────────────────────

/// @brief Integrator-role stub. No default bindings — bound explicitly.
class StubIntegrator : public ATV::cuda::Component {
  public:
    std::string name() const override { return "stub_integrator"; }
    ATV::ComponentRole role() const override {
        return ATV::ComponentRole::Integrator;
    }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

/// @brief Evaluator-role stub advertising `Energy | Force`.
class StubForceEvaluator : public ATV::cuda::Component {
  public:
    std::string name() const override { return "stub_force"; }
    ATV::ComponentRole role() const override {
        return ATV::ComponentRole::Evaluator;
    }
    ATV::Capability capabilities() const override {
        return ATV::Capability::Energy | ATV::Capability::Force;
    }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

/// @brief Evaluator-role stub advertising only `Energy` (no `Force`).
class StubEnergyEvaluator : public ATV::cuda::Component {
  public:
    std::string name() const override { return "stub_energy"; }
    ATV::ComponentRole role() const override {
        return ATV::ComponentRole::Evaluator;
    }
    ATV::Capability capabilities() const override {
        return ATV::Capability::Energy;
    }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

/// @brief Optimizer-role stub.
class StubOptimizer : public ATV::cuda::Component {
  public:
    std::string name() const override { return "stub_optimizer"; }
    ATV::ComponentRole role() const override {
        return ATV::ComponentRole::Optimizer;
    }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

/// @brief `role()` defaults to `None` — structural-only stub.
class StubBareInfra : public ATV::cuda::Component {
  public:
    std::string name() const override { return "stub_bare"; }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

} // namespace

// ─── cuda::md::MD (was: DynamicsDriver) ─────────────────────────────

TEST(CUDADriverRoleValidation, MDThrowsWithoutIntegrator) {
    using namespace ATV;
    cuda::md::MD d;
    StubForceEvaluator force;
    d.on(md::Phase::force).add(force);
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
    StubEnergyEvaluator energy_only;
    d.on(md::Phase::initial_integrate).add(integ);
    d.on(md::Phase::force).add(energy_only);
    try {
        d.build();
        FAIL() << "expected throw — MD missing Force-capable evaluator";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "Force")) << e.what();
    }
}

TEST(CUDADriverRoleValidation, MDBuildsWithIntegratorAndForce) {
    using namespace ATV;
    cuda::md::MD d;
    StubIntegrator integ;
    StubForceEvaluator force;
    d.on(md::Phase::initial_integrate).add(integ);
    d.on(md::Phase::force).add(force);
    EXPECT_NO_THROW(d.build());
}

/// A time budget is not a policy: MD attaches no default handler. (Minimize does —
/// see below — and that asymmetry is the point of a preset.)
TEST(CUDADriverRoleValidation, MDAttachesNoDefaultHandler) {
    using namespace ATV;
    cuda::md::MD d;
    StubIntegrator integ;
    StubForceEvaluator force;
    d.on(md::Phase::initial_integrate).add(integ);
    d.on(md::Phase::force).add(force);
    d.build();
    EXPECT_FALSE(d.has_handler());
}

// ─── cuda::minimizer::Minimize (was: OptimizeDriver) ────────────────

TEST(CUDADriverRoleValidation, MinimizeThrowsWithoutOptimizer) {
    using namespace ATV;
    cuda::minimizer::Minimize o;
    StubForceEvaluator force;
    o.on(minimizer::Phase::force).add(force);
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
    o.on(minimizer::Phase::search_direction).add(opt);
    o.on(minimizer::Phase::end_of_step).add(infra);
    try {
        o.build();
        FAIL() << "expected throw — Minimize missing Energy/Force evaluator";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "Energy")
                    || message_mentions(e, "Force"))
            << e.what();
    }
}

TEST(CUDADriverRoleValidation, MinimizeBuildsWithOptimizerAndEnergy) {
    using namespace ATV;
    cuda::minimizer::Minimize o;
    StubOptimizer opt;
    StubEnergyEvaluator energy;
    o.on(minimizer::Phase::search_direction).add(opt);
    o.on(minimizer::Phase::force).add(energy);
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
    o.on(minimizer::Phase::search_direction).add(opt);
    o.on(minimizer::Phase::force).add(energy);
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
    o.add_handler(cuda::ForceConverged{/*ftol=*/1e-4});
    o.on(minimizer::Phase::search_direction).add(opt);
    o.on(minimizer::Phase::force).add(energy);
    EXPECT_TRUE(o.has_handler());
    EXPECT_NO_THROW(o.build());
}

// ─── cuda::Evaluate (was: EvaluateDriver) ───────────────────────────

TEST(CUDADriverRoleValidation, EvaluateThrowsWithoutEvaluator) {
    using namespace ATV;
    cuda::Evaluate e;
    StubBareInfra infra;
    e.on(evaluate::Phase::prepare).add(infra);
    try {
        e.build();
        FAIL() << "expected throw — Evaluate missing Evaluator";
    } catch (const std::runtime_error& ex) {
        EXPECT_TRUE(message_mentions(ex, "Evaluator")) << ex.what();
    }
}

TEST(CUDADriverRoleValidation, EvaluateBuildsWithEnergyEvaluator) {
    using namespace ATV;
    cuda::Evaluate e;
    StubEnergyEvaluator energy;
    e.on(evaluate::Phase::evaluate).add(energy);
    EXPECT_NO_THROW(e.build());
}

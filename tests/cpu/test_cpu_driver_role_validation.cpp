/// @file tests/cpu/test_cpu_driver_role_validation.cpp
/// @brief Assert that Evaluate / MD / Minimize throw at build() when their
///        required kinds are missing, and build successfully when present.
///
/// Required energy/force kinds: Potential, Bias, PathCoupling, HF,
/// Semiempirical, PlaneWave, GPW. There is no Capability write-set.

#include <gtest/gtest.h>

#include <string>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/minimize/minimize.h"

#include "cpu/driver_test_helpers.h"

using atv_test::StubBareInfra;
using atv_test::StubEnergyEvaluator;
using atv_test::StubForceEvaluator;
using atv_test::StubIntegrator;
using atv_test::StubOptimizer;

namespace {

bool message_mentions(const std::runtime_error& e, const std::string& needle) {
    return std::string(e.what()).find(needle) != std::string::npos;
}

} // namespace

// ─── Dynamics (cpu::md::MD) ──────────────────────────────────────────

TEST(CPUDriverRoleValidation, DynamicsThrowsWithoutIntegrator) {
    using namespace ATV;
    cpu::md::MD d;
    StubForceEvaluator force;
    d.on(md::Phase::force).add_ref(force);
    try {
        d.build();
        FAIL() << "expected throw — MD missing Integrator";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "Integrator")) << e.what();
    }
}

TEST(CPUDriverRoleValidation, DynamicsThrowsWithoutEnergyForceKind) {
    using namespace ATV;
    cpu::md::MD d;
    StubIntegrator integ;
    StubBareInfra bare;
    d.on(md::Phase::initial_integrate).add_ref(integ);
    d.on(md::Phase::force).add_ref(bare);
    try {
        d.build();
        FAIL() << "expected throw — MD missing Potential / Bias / HF / …";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "force source")) << e.what();
    }
}

TEST(CPUDriverRoleValidation, DynamicsBuildsWithIntegratorAndForce) {
    using namespace ATV;
    cpu::md::MD d;
    StubIntegrator integ;
    StubForceEvaluator force;
    d.on(md::Phase::initial_integrate).add_ref(integ);
    d.on(md::Phase::force).add_ref(force);
    EXPECT_NO_THROW(d.build());
}

// ─── Optimize (cpu::minimizer::Minimize) ─────────────────────────────

TEST(CPUDriverRoleValidation, OptimizeThrowsWithoutOptimizer) {
    using namespace ATV;
    cpu::minimizer::Minimize o;
    StubForceEvaluator force;
    o.on(minimizer::Phase::force).add_ref(force);
    try {
        o.build();
        FAIL() << "expected throw — Minimize missing Optimizer";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "Optimizer")) << e.what();
    }
}

TEST(CPUDriverRoleValidation, OptimizeThrowsWithoutEnergyOrForceKind) {
    using namespace ATV;
    cpu::minimizer::Minimize o;
    StubOptimizer opt;
    StubBareInfra infra;
    o.on(minimizer::Phase::search_direction).add_ref(opt);
    o.on(minimizer::Phase::end_of_step).add_ref(infra);
    try {
        o.build();
        FAIL() << "expected throw — Minimize missing Potential / Bias / HF / …";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "energy or force source")) << e.what();
    }
}

TEST(CPUDriverRoleValidation, OptimizeBuildsWithOptimizerAndEnergy) {
    using namespace ATV;
    cpu::minimizer::Minimize o;
    StubOptimizer opt;
    StubEnergyEvaluator energy;
    o.on(minimizer::Phase::search_direction).add_ref(opt);
    o.on(minimizer::Phase::force).add_ref(energy);
    EXPECT_NO_THROW(o.build());
}

// ─── Evaluate (cpu::Evaluate) ────────────────────────────────────────

TEST(CPUDriverRoleValidation, EvaluateThrowsWithoutEnergyKind) {
    using namespace ATV;
    cpu::Evaluate e;
    StubBareInfra infra;
    e.on(evaluate::Phase::prepare).add_ref(infra);
    try {
        e.build();
        FAIL() << "expected throw — Evaluate missing Potential / Bias / HF / …";
    } catch (const std::runtime_error& ex) {
        EXPECT_TRUE(message_mentions(ex, "energy or force source"))
            << ex.what();
    }
}

TEST(CPUDriverRoleValidation, EvaluateBuildsWithEnergyEvaluator) {
    using namespace ATV;
    cpu::Evaluate e;
    StubEnergyEvaluator energy;
    e.on(evaluate::Phase::evaluate).add_ref(energy);
    EXPECT_NO_THROW(e.build());
}

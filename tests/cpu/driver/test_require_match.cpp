/// @file tests/cpu/driver/test_require_match.cpp
/// @brief Single-method gates for Driver::require_* / probe-only energy.

#include <gtest/gtest.h>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/minimize/minimize.h"

#include "cpu/driver_test_helpers.h"

using atv_test::StubIntegrator;
using atv_test::StubOptimizer;
using atv_test::StubProbeEnergy;
using atv_test::StubProbeForce;

TEST(TestRequireMatch, RequireEnergyOrForceAcceptsProbeOnlyComponent) {
    ATV::cpu::Evaluate eval;
    StubProbeEnergy energy;
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(energy);
    EXPECT_NO_THROW(eval.build());
}

TEST(TestRequireMatch, RequireEnergyOrForceRejectsBareComponent) {
    ATV::cpu::Evaluate eval;
    atv_test::StubBareInfra bare;
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(bare);
    try {
        eval.build();
        FAIL() << "expected missing energy or force source";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("energy or force source"),
                  std::string::npos)
            << e.what();
    }
}

TEST(TestRequireMatch, RequireForceAcceptsProbeOnlyForce) {
    ATV::cpu::md::MD md;
    StubIntegrator integ;
    StubProbeForce force;
    md.on(ATV::md::Phase::initial_integrate).add_ref(integ);
    md.on(ATV::md::Phase::force).add_ref(force);
    EXPECT_NO_THROW(md.build());
}

TEST(TestRequireMatch, RequireForceRejectsEnergyOnlyProbe) {
    ATV::cpu::md::MD md;
    StubIntegrator integ;
    StubProbeEnergy energy;
    md.on(ATV::md::Phase::initial_integrate).add_ref(integ);
    md.on(ATV::md::Phase::force).add_ref(energy);
    try {
        md.build();
        FAIL() << "expected missing force source";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("force source"), std::string::npos)
            << e.what();
    }
}

TEST(TestRequireMatch, RequireIntegratorThrows) {
    ATV::cpu::md::MD md;
    StubProbeForce force;
    md.on(ATV::md::Phase::force).add_ref(force);
    try {
        md.build();
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("Integrator"), std::string::npos)
            << e.what();
    }
}

TEST(TestRequireMatch, RequireOptimizerThrows) {
    ATV::cpu::minimizer::Minimize opt;
    StubProbeEnergy energy;
    opt.on(ATV::minimizer::Phase::force).add_ref(energy);
    try {
        opt.build();
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("Optimizer"), std::string::npos)
            << e.what();
    }
}

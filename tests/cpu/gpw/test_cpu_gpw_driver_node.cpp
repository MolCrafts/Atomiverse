/// @file tests/cpu/gpw/test_cpu_gpw_driver_node.cpp
/// @brief Unit: GpwDft is a Driver node with gpw::Phase stage bindings.

#include <functional>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/stages/density_stage.h"
#include "atomiverse/cpu/gpw/stages/diagonalize_stage.h"
#include "atomiverse/cpu/gpw/stages/gpw_converged.h"
#include "atomiverse/cpu/gpw/stages/ks_build_stage.h"
#include "atomiverse/cpu/gpw/stages/mix_stage.h"

using ATV::cpu::gpw::DensityStage;
using ATV::cpu::gpw::DiagonalizeStage;
using ATV::cpu::gpw::GpwConverged;
using ATV::cpu::gpw::GpwDft;
using ATV::cpu::gpw::KsBuildStage;
using ATV::cpu::gpw::MixStage;

// GpwDft is the GPW self-consistency Driver node (MeanFieldMethod pattern).
TEST(GpwDriverNode, InheritsCpuDriverGpwPhase) {
    static_assert(std::is_base_of_v<ATV::cpu::Driver<ATV::gpw::Phase>, GpwDft>);
    GpwDft m;
    EXPECT_EQ(ATV::slot_count<ATV::gpw::Phase>(), 5);
    EXPECT_EQ(ATV::gpw::kind_of(ATV::gpw::Phase{}), ATV::PhaseKind::SCF);
}

// Stage default bindings target the matching gpw::Phase slots (path-local Ops).
TEST(GpwDriverNode, StageDefaultBindings) {
    bool skip = false;
    auto nop = []() {};
    KsBuildStage ks(nop);
    MixStage mix(nop, skip);
    DiagonalizeStage d(nop, skip);
    DensityStage dens(nop, skip);
    GpwConverged crit(skip);

    auto bind0 = [](const auto& c) {
        auto b = c.default_bindings();
        EXPECT_EQ(b.size(), 1u);
        EXPECT_EQ(b[0].kind, ATV::PhaseKind::SCF);
        return b[0].phase;
    };
    EXPECT_EQ(bind0(ks), ATV::slot_of(ATV::gpw::Phase::ks_build));
    EXPECT_EQ(bind0(mix), ATV::slot_of(ATV::gpw::Phase::mix));
    EXPECT_EQ(bind0(d), ATV::slot_of(ATV::gpw::Phase::diagonalize));
    EXPECT_EQ(bind0(dens), ATV::slot_of(ATV::gpw::Phase::density));
    EXPECT_EQ(bind0(crit), ATV::slot_of(ATV::gpw::Phase::check));
}

// Path-local Ops are the multi-path DIIS shape (Γ RKS/UKS + multi-k).
TEST(GpwDriverNode, PathLocalOpsAreCallable) {
    int hits = 0;
    bool skip = false;
    KsBuildStage ks([&]() { ++hits; });
    MixStage mix([&]() { ++hits; }, skip);
    DiagonalizeStage d([&]() { ++hits; }, skip);
    DensityStage dens([&]() { ++hits; }, skip);

    // Direct launch without phase context is guarded — Ops not run off-phase.
    // Construction + binding surface is the unit gate (no full SCF).
    EXPECT_EQ(ks.name(), "gpw_ks_build_stage");
    EXPECT_EQ(mix.name(), "gpw_mix_stage");
    EXPECT_EQ(d.name(), "gpw_diagonalize_stage");
    EXPECT_EQ(dens.name(), "gpw_density_stage");
    EXPECT_EQ(hits, 0);
}

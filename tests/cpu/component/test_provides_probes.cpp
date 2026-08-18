/// @file tests/cpu/component/test_provides_probes.cpp
/// @brief Single-method gates for Component::provides_energy / provides_force.

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

#include "atomiverse/cpu/component/bias.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/gpw.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/component/integrator.h"
#include "atomiverse/cpu/component/path_coupling.h"
#include "atomiverse/cpu/component/plane_wave.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/component/semiempirical.h"

#ifdef ATV_HAS_MD
#include "atomiverse/cpu/dynamics/fix_langevin.h"
#endif
#ifdef ATV_HAS_VIB
#include "atomiverse/cpu/vibration/hessian_accumulator.h"
#include "atomiverse/cpu/vibration/hessian_workspace.h"
#endif

namespace {

template <typename Base>
class ProbeStub : public Base {
  public:
    std::string name() const override { return "probe_stub"; }
    void launch(const ATV::cpu::LaunchContext& ctx) override { (void)ctx; }
};

} // namespace

TEST(TestComponent, ProvidesEnergyDefaultsFalse) {
    EXPECT_FALSE(ProbeStub<ATV::cpu::Component>{}.provides_energy());
}

TEST(TestComponent, ProvidesForceDefaultsFalse) {
    EXPECT_FALSE(ProbeStub<ATV::cpu::Component>{}.provides_force());
}

TEST(TestPotential, ProvidesEnergyTrue) {
    EXPECT_TRUE(ProbeStub<ATV::cpu::Potential>{}.provides_energy());
}

TEST(TestPotential, ProvidesForceTrue) {
    EXPECT_TRUE(ProbeStub<ATV::cpu::Potential>{}.provides_force());
}

TEST(TestBias, ProvidesEnergyTrue) {
    EXPECT_TRUE(ProbeStub<ATV::cpu::Bias>{}.provides_energy());
}

TEST(TestBias, ProvidesForceTrue) {
    EXPECT_TRUE(ProbeStub<ATV::cpu::Bias>{}.provides_force());
}

TEST(TestPathCoupling, ProvidesEnergyTrue) {
    EXPECT_TRUE(ProbeStub<ATV::cpu::PathCoupling>{}.provides_energy());
}

TEST(TestPathCoupling, ProvidesForceTrue) {
    EXPECT_TRUE(ProbeStub<ATV::cpu::PathCoupling>{}.provides_force());
}

TEST(TestSemiempirical, ProvidesEnergyTrue) {
    EXPECT_TRUE(ProbeStub<ATV::cpu::Semiempirical>{}.provides_energy());
}

TEST(TestSemiempirical, ProvidesForceTrue) {
    EXPECT_TRUE(ProbeStub<ATV::cpu::Semiempirical>{}.provides_force());
}

TEST(TestPlaneWave, ProvidesEnergyTrue) {
    EXPECT_TRUE(ProbeStub<ATV::cpu::PlaneWave>{}.provides_energy());
}

TEST(TestPlaneWave, ProvidesForceTrue) {
    EXPECT_TRUE(ProbeStub<ATV::cpu::PlaneWave>{}.provides_force());
}

TEST(TestGPW, ProvidesEnergyTrue) {
    EXPECT_TRUE(ProbeStub<ATV::cpu::GPW>{}.provides_energy());
}

TEST(TestGPW, ProvidesForceTrue) {
    EXPECT_TRUE(ProbeStub<ATV::cpu::GPW>{}.provides_force());
}

TEST(TestHF, IsNotAComponent) {
    static_assert(!std::is_base_of_v<ATV::cpu::Component, ATV::cpu::HF>,
                  "HF is a tag, not a Component");
    SUCCEED();
}

TEST(TestHF, ProvidesEnergyTrue) {
    struct Tag : ATV::cpu::HF {};
    EXPECT_TRUE(Tag{}.provides_energy());
}

TEST(TestHF, ProvidesForceTrue) {
    struct Tag : ATV::cpu::HF {};
    EXPECT_TRUE(Tag{}.provides_force());
}

TEST(TestIntegrator, ProvidesEnergyStaysFalse) {
    EXPECT_FALSE(ProbeStub<ATV::cpu::Integrator>{}.provides_energy());
}

#ifdef ATV_HAS_VIB
TEST(TestHessianAccumulator, ProvidesHessianTrue) {
    ATV::cpu::vib::HessianWorkspace ws;
    ATV::cpu::vib::HessianAccumulator acc{&ws};
    EXPECT_TRUE(acc.provides_hessian());
}

TEST(TestComponent, ProvidesHessianDefaultsFalse) {
    EXPECT_FALSE(ProbeStub<ATV::cpu::Component>{}.provides_hessian());
}
#endif

#ifdef ATV_HAS_MD
TEST(TestFixLangevin, ProvidesForceTrue) {
    ATV::cpu::md::FixLangevin langevin{300.0, 300.0, 100.0, 1.0, 1};
    EXPECT_TRUE(langevin.provides_force());
}

TEST(TestFixLangevin, ProvidesEnergyFalse) {
    ATV::cpu::md::FixLangevin langevin{300.0, 300.0, 100.0, 1.0, 1};
    EXPECT_FALSE(langevin.provides_energy());
}
#endif

/// @file tests/cuda/component/test_provides_probes.cu
/// @brief Single-method gates for cuda::Component::provides_energy / provides_force.
///
/// Instantiates kinds only — no device work. Needs the CUDA language
/// (box.cuh uses __host__/__device__).

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

#include "atomiverse/core/role.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/component/hf.cuh"
#include "atomiverse/cuda/component/integrator.cuh"
#include "atomiverse/cuda/component/observer.cuh"
#include "atomiverse/cuda/component/potential.cuh"
#include "atomiverse/cuda/component/recorder.cuh"
#include "atomiverse/cuda/dynamics/fix_langevin.cuh"

namespace {

template <typename Base>
class ProbeStub : public Base {
  public:
    std::string name() const override { return "probe_stub"; }
    void launch(const ATV::cuda::LaunchContext& ctx) override { (void)ctx; }
};

} // namespace

TEST(TestCudaComponent, ProvidesEnergyDefaultsFalse) {
    EXPECT_FALSE(ProbeStub<ATV::cuda::Component>{}.provides_energy());
}

TEST(TestCudaComponent, ProvidesForceDefaultsFalse) {
    EXPECT_FALSE(ProbeStub<ATV::cuda::Component>{}.provides_force());
}

TEST(TestCudaPotential, ProvidesEnergyTrue) {
    EXPECT_TRUE(ProbeStub<ATV::cuda::Potential>{}.provides_energy());
}

TEST(TestCudaPotential, ProvidesForceTrue) {
    EXPECT_TRUE(ProbeStub<ATV::cuda::Potential>{}.provides_force());
}

TEST(TestCudaHF, IsNotAComponent) {
    static_assert(!std::is_base_of_v<ATV::cuda::Component, ATV::cuda::HF>,
                  "HF is a tag, not a Component");
    SUCCEED();
}

TEST(TestCudaHF, ProvidesEnergyTrue) {
    struct Tag : ATV::cuda::HF {};
    EXPECT_TRUE(Tag{}.provides_energy());
}

TEST(TestCudaHF, ProvidesForceTrue) {
    struct Tag : ATV::cuda::HF {};
    EXPECT_TRUE(Tag{}.provides_force());
}

TEST(TestCudaIntegrator, ProvidesEnergyStaysFalse) {
    EXPECT_FALSE(ProbeStub<ATV::cuda::Integrator>{}.provides_energy());
}

TEST(TestCudaObserver, RoleIsFrameImmutableDeferred) {
    const ATV::Role r = ProbeStub<ATV::cuda::Observer>{}.role();
    EXPECT_EQ(r, ATV::RoleFlag::FrameImmutable | ATV::RoleFlag::Deferred);
}

TEST(TestCudaRecorder, RoleIncludesExternalIO) {
    const ATV::Role r = ProbeStub<ATV::cuda::Recorder>{}.role();
    EXPECT_EQ(r, ATV::RoleFlag::FrameImmutable | ATV::RoleFlag::Deferred
                     | ATV::RoleFlag::ExternalIO);
}

TEST(TestCudaComponent, ProvidesHessianDefaultsFalse) {
    EXPECT_FALSE(ProbeStub<ATV::cuda::Component>{}.provides_hessian());
}

TEST(TestCudaFixLangevin, ProvidesForceTrue) {
    ATV::cuda::md::FixLangevin langevin{300.0, 300.0, 100.0, 1.0, 1};
    EXPECT_TRUE(langevin.provides_force());
}

TEST(TestCudaFixLangevin, ProvidesEnergyFalse) {
    ATV::cuda::md::FixLangevin langevin{300.0, 300.0, 100.0, 1.0, 1};
    EXPECT_FALSE(langevin.provides_energy());
}

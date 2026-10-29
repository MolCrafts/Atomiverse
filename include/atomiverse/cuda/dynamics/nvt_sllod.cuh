/// @file cuda/dynamics/nvt_sllod.cuh
/// @brief CUDA NVT SLLOD. Same step as `cpu::md::NvtSllod`.
///
/// Replaces `VelocityVerlet` and thermostats peculiar velocities itself.
/// `changes_box` is true only at `end_of_step`, when the cell edge or tilt
/// is written. CUDA-free header.
#pragma once

#include "atomiverse/cuda/component/integrator.cuh"

#include <memory>
#include <string>
#include <vector>

#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {
namespace md {

class NvtSllod : public cuda::Integrator {
  public:
    NvtSllod(double temperature_K, double tau_fs, double dt_fs,
             double rate_per_fs, int direction);
    ~NvtSllod() override;
    NvtSllod(NvtSllod&&) noexcept;
    NvtSllod& operator=(NvtSllod&&) noexcept;

    bool changes_box(int stage) const override {
        return stage == slot_of(ATV::md::Phase::end_of_step);
    }
    bool requires_system_stress() const override { return true; }
    std::string name() const override { return "nvt_sllod"; }
    Role role() const override { return 0; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, 0),
                ATV::make_binding(ATV::md::Phase::end_of_step, 0)};
    }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  private:
    double T_, tau_, dt_, rate_;
    int dir_;
    bool armed_ = false;
    bool kicked_ = false;
    long long step_ = 0;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace md
} // namespace cuda
} // namespace ATV

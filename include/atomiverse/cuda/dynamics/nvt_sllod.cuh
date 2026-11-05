/// @file cuda/dynamics/nvt_sllod.cuh
/// @brief CUDA NVT SLLOD. Same step as `cpu::md::NvtSllod`.
///
/// Replaces `VelocityVerlet` and thermostats peculiar velocities itself.
/// The flow (cell, strain clock, flip decision) and the Nosé–Hoover chain
/// live in device memory: end_of_step deforms the cell and decides a box
/// flip on the device; the next pre_exchange executes it on the device
/// (LAMMPS `fix deform` `flip yes`), and the cell motion makes the neighbor
/// gate rebuild. The host never reads the cell back during a run.
/// `changes_box` is true at `end_of_step` and, when a flip is possible, at
/// `pre_exchange`. CUDA-free header.
#pragma once

#include "atomiverse/cuda/component/integrator.cuh"
#include "atomiverse/cuda/component/timestep_protocol.cuh"

#include <memory>
#include <string>
#include <vector>

#include "atomiverse/core/role.h"
#include "atomiverse/core/sllod_flow.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {
namespace md {

/// @brief Reversible NVT SLLOD with one ERATE direction and `remap v`.
///
/// Parameters as `cpu::md::NvtSllod`: temperature [K], `Tdamp` [fs],
/// timestep [fs], engineering rate [1/fs], Voigt direction 0..5
/// (`x y z yz xz xy`) and the Nosé–Hoover chain length `tchain`
/// (LAMMPS `fix nvt/sllod` default 1).
class NvtSllod : public cuda::Integrator, public cuda::TimestepProvider {
  public:
    /// @brief Publishes the step to `cuda::md::MD` (timestep protocol).
    const cuda::TimestepProvider* as_timestep_provider() const override {
        return this;
    }
    /// @return The constructor `dt_fs` [fs].
    double timestep() const override { return dt_; }
    /// @param temperature_K Target temperature [K].
    /// @param tau_fs Thermostat damping time (LAMMPS `Tdamp`) [fs].
    /// @param dt_fs Timestep [fs].
    /// @param rate_per_fs Engineering strain rate [1/fs].
    /// @param direction Voigt index 0..5: x y z yz xz xy.
    /// @param tchain Nosé–Hoover chain length (>= 1; LAMMPS `tchain`).
    NvtSllod(double temperature_K, double tau_fs, double dt_fs,
             double rate_per_fs, int direction, int tchain = 1);
    ~NvtSllod() override;
    NvtSllod(NvtSllod&&) noexcept;
    NvtSllod& operator=(NvtSllod&&) noexcept;

    bool changes_box(int stage) const override {
        return stage == slot_of(ATV::md::Phase::end_of_step)
               || (can_flip_ && stage == slot_of(ATV::md::Phase::pre_exchange));
    }
    /// @brief A shear direction (yz, xz, xy) writes a tilt factor, so the
    ///        plan must dispatch general-cell kernels from the first step.
    bool tilts_box() const override { return dir_ >= 3; }
    bool requires_system_stress() const override { return true; }
    std::string name() const override { return "nvt_sllod"; }
    Role role() const override { return 0; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 0),
                ATV::make_binding(ATV::md::Phase::pre_exchange, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, 0),
                ATV::make_binding(ATV::md::Phase::end_of_step, 0)};
    }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

    /// @brief Current Nosé–Hoover chain (LAMMPS `eta`, `eta_dot`).
    ///
    /// Diagnostic: synchronizes the device and copies the chain back, so it
    /// is for after a run (or tests), never inside a step.
    /// @return Chain positions [1] and velocities [1/fs], innermost first.
    ATV::md::SllodChainState thermostat_chain() const;

  private:
    double T_, tau_, dt_, rate_;
    int dir_;
    int tchain_;
    bool armed_ = false;
    bool kicked_ = false;
    bool measured_ = false;
    bool can_flip_ = false;
    long long step_ = 0;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace md
} // namespace cuda
} // namespace ATV

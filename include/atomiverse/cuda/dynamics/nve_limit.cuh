/// @file cuda/dynamics/nve_limit.cuh
/// @brief CUDA Velocity-Verlet integrator with a per-step displacement cap
///        (LAMMPS-equivalent `fix nve/limit`).
#pragma once

#include "atomiverse/cuda/component/integrator.cuh"
#include "atomiverse/cuda/component/timestep_protocol.cuh"

#include <string>
#include <vector>

#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/cuda/frame/types.cuh"

#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace md {

/// Raw launch wrappers for the NVE/limit half-steps, callable outside the
/// Component/LaunchContext path (used by the multi-GPU PE driver, which owns
/// raw per-subdomain SoA pointers rather than a single Frame). `xmax` is the
/// per-step displacement cap [Å]; `vmax = xmax/dt` is computed internally.
/// These mirror LAMMPS `fix nve/limit` exactly (ATV::units::md_ftm2v units, Euclidean speed
/// cap, orthorhombic PBC wrap on the position update).
/// @brief First NVE/limit half-step on rows [0, n) of `frame`.
/// @param frame Device frame (positions, velocities, fixed-point forces).
/// @param n Number of integrated (owned) rows.
/// @param box Cell kernel argument (resolved in stream order).
/// @param dt Time step [fs].
/// @param xmax Maximum displacement per step [Angstrom].
/// @param step Step index stamped on a force fault.
/// @param stream CUDA stream.
/// @param step_ptr Device step copy read instead of `step` (null: `step`).
void launch_nve_limit_initial(Frame& frame, int n, const cuda::Box& box,
                              mixed_t dt, mixed_t xmax, long long step,
                              cudaStream_t stream = 0,
                              const long long* step_ptr = nullptr);
/// @brief Second NVE/limit half-step on rows [0, n) of `frame`.
/// @param frame Device frame (velocities, fixed-point forces).
/// @param n Number of integrated (owned) rows.
/// @param dt Time step [fs].
/// @param xmax Maximum displacement per step [Angstrom].
/// @param step Step index stamped on a force fault.
/// @param stream CUDA stream.
/// @param step_ptr Device step copy read instead of `step` (null: `step`).
void launch_nve_limit_final(Frame& frame, int n, mixed_t dt, mixed_t xmax,
                            long long step, cudaStream_t stream = 0,
                            const long long* step_ptr = nullptr);

class NVELimit : public cuda::Integrator, public cuda::TimestepProvider {
  public:
    /// @brief Publishes the step to `cuda::md::MD` (timestep protocol).
    const cuda::TimestepProvider* as_timestep_provider() const override {
        return this;
    }
    /// @return The constructor `dt` [fs].
    double timestep() const override { return dt_; }
    /// @brief Construct NVE/limit integrator.
    /// @param dt   Timestep [fs].
    /// @param xmax Maximum distance an atom may move in one step [Å].
    ///             Enforces `|v|·dt <= xmax`.
    NVELimit(double dt, double xmax) : dt_(dt), xmax_(xmax) {}

    std::string name() const override { return "nve_limit"; }
    /* role via Integrator base */
    /// @brief Write-set: velocity Verlet with a per-step displacement cap.

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, 0)};
    }

    /// @brief The fault stamp reads the device step: replayable.
    bool graph_capturable() const override { return true; }

    void launch(const LaunchContext& ctx) override;

  private:
    double dt_;
    double xmax_;
};

} // namespace md
} // namespace cuda
} // namespace ATV

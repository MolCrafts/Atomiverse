/// @file cuda/dynamics/nve_limit.cuh
/// @brief CUDA Velocity-Verlet integrator with a per-step displacement cap
///        (LAMMPS-equivalent `fix nve/limit`).
#pragma once

#include "atomiverse/cuda/component/integrator.cuh"

#include <string>
#include <vector>

#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/types.cuh"

#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace md {

/// Raw launch wrappers for the NVE/limit half-steps, callable outside the
/// Component/LaunchContext path (used by the multi-GPU PE driver, which owns
/// raw per-subdomain SoA pointers rather than a single Frame). `xmax` is the
/// per-step displacement cap [Å]; `vmax = xmax/dt` is computed internally.
/// These mirror LAMMPS `fix nve/limit` exactly (ATV::units::force_to_accel units, Euclidean speed
/// cap, orthorhombic PBC wrap on the position update).
void launch_nve_limit_initial(real_t* x, real_t* y, real_t* z, real_t* vx,
                              real_t* vy, real_t* vz, const real_t* fx,
                              const real_t* fy, const real_t* fz,
                              const real_t* mass, cuda::Box box, work_t dt,
                              work_t xmax, int n, cudaStream_t stream = 0);
void launch_nve_limit_final(real_t* vx, real_t* vy, real_t* vz,
                            const real_t* fx, const real_t* fy,
                            const real_t* fz, const real_t* mass, work_t dt,
                            work_t xmax, int n, cudaStream_t stream = 0);

/// @brief CUDA sibling of `cpu::md::NVELimit` — velocity Verlet with a
///        per-step displacement cap. Mirrors LAMMPS `fix nve/limit xmax`.
///
/// After every half-step kick, each atom's speed `|v|` is capped so the
/// step displacement satisfies `|v|·dt <= xmax`. Equivalent formulation:
/// `vmax = xmax/dt`, `vmax_sq = vmax²`; if `vsq > vmax_sq`, rescale
/// `v *= vmax / sqrt(vsq)` (direction-preserving). Applied in BOTH
/// `initial_integrate` and `final_integrate` halves, exactly matching
/// LAMMPS `fix_nve_limit.cpp` (lines 105–112, 168–175).
///
/// ## LAMMPS correspondence (audit: ALIGNED)
///
/// Cap norm is Euclidean (vector L2), not per-component; rescale
/// `vmax / sqrt(vsq)` is algebraically identical to LAMMPS
/// `sqrt(vlimitsq/vsq)`. PBC wrap is orthorhombic (diagonal `box[0,4,8]`
/// only) — same as `cuda::md::VelocityVerlet`. Triclinic wrap is
/// deferred (not used by MSU PE benchmark).
///
/// ## Use case
///
/// Stage 1 of the MSU PE 5-stage protocol starts from an unrelaxed
/// amorphous configuration with `E_pair ~ 3.4e8 kcal/mol`; plain
/// Velocity-Verlet diverges within one step. `NVELimit{dt=1.0 fs,
/// xmax=0.05 Å}` caps the per-step displacement so the integrator
/// survives transient force spikes long enough for `cuda::md::FixLangevin`
/// to dissipate them. Once the system relaxes the cap is rarely hit
/// and behaviour collapses to plain VV.
///
/// Bound at `initial_integrate @ 0` + `final_integrate @ 0` — replaces
/// (do not combine with) `cuda::md::VelocityVerlet`.
///
/// @see cpu::md::NVELimit for the CPU reference + LAMMPS audit table
/// @see cuda::md::VelocityVerlet for the unconstrained sibling
class NVELimit : public cuda::Integrator {
  public:
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

    void launch(const LaunchContext& ctx) override;

  private:
    double dt_;
    double xmax_;
};

} // namespace md
} // namespace cuda
} // namespace ATV

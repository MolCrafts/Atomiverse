/// @file cuda/md/fix_langevin.cuh
/// @brief CUDA Langevin thermostat — LAMMPS-equivalent `fix langevin`.
#pragma once

#include "atomiverse/cuda/component/thermostat.cuh"

#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {
namespace md {

/// @brief CUDA sibling of `cpu::md::FixLangevin` — Schneider-Stoll
///        Langevin thermostat applied as an additive `post_force` kick.
///        Mirrors LAMMPS `fix langevin Tstart Tstop damp seed`.
///
/// Two forces are added per atom each step:
///
/// 1. Friction (Stokes drag):  `F_drag = -m / (damp · ftm2v) · v`
/// 2. Random kick (FDT-consistent):
///    `F_rand = sqrt(2 m kB T_live / (damp · dt · mvv2e)) / ftm2v · ξ`,
///    `ξ ~ N(0, 1)` per Cartesian component, per atom, per step.
///
/// Live setpoint `T_live = T_start + clamp((step-nfirst)/(nlast-nfirst),
/// 0, 1) · (T_stop - T_start)` (LAMMPS `fix_langevin.cpp::compute_target`
/// lines 794–802, 661). With the default `T_stop < 0` sentinel the
/// ramp folds to `T_start` (constant-T thermostat).
///
/// ## RNG design — `(seed, step, global atom tag)` splitmix Box-Muller
///
/// The kernel uses no persistent device RNG state (no `curand_init` /
/// `curandState*`). Each per-atom call generates 6 fresh `u01` samples
/// from a counter-based splitmix64 stream keyed by the 64-bit triple
/// `hash(seed, step, frame.atoms.tag[i])`, then Box-Muller transforms
/// the 6 uniforms into 3 standard normals `rx, ry, rz`. This makes
/// the random force reproducible across rank counts (the seed depends
/// on the global `tag`, not on the atom's index in any rank-local
/// SoA) and saves the per-atom state buffer that `curandState*` would
/// require.
///
/// Per ensemble-average contract (see `msu-pe-align-03` block-average
/// rule): individual stochastic trajectories may differ from the CPU
/// sibling at a given seed because the splitmix here and the CPU
/// `splitmix` use the same algorithm but the per-atom dispatch order
/// is different (per-thread vs per-OMP-stride). Ensemble averages
/// ⟨T⟩, ⟨P⟩, ⟨ρ⟩ converge to the LAMMPS / CPU values within the
/// block-average tolerance.
///
/// Bound at `post_force @ +500` — after all physical-force evaluators
/// have accumulated, before any user-defined post-force corrections.
///
/// @see cpu::md::FixLangevin for the CPU reference + LAMMPS audit
///      verdict table.
class FixLangevin : public cuda::Thermostat {
  public:
    /// @brief Construct Langevin thermostat.
    /// @param T_start Ramp start temperature [K].
    /// @param T_stop  Ramp end temperature [K]. `< 0` ⇒ no ramp,
    ///                folds to `T_start`.
    /// @param damp    Damping time constant [fs]. γ = 1/damp.
    /// @param dt      Timestep [fs].
    /// @param seed    RNG seed (reproducible across rank counts).
    FixLangevin(double T_start, double T_stop, double damp, double dt,
                std::uint32_t seed)
        : T_start_(T_start), T_stop_(T_stop < 0 ? T_start : T_stop),
          damp_(damp), dt_(dt), seed_(seed), T_live_(T_start) {}

    /// @brief Set the temperature-ramp run span `[nfirst, nlast]`
    ///        (MD-step indices), mirroring LAMMPS
    ///        `update->beginstep / endstep`.
    /// @param nfirst First MD step index of the run.
    /// @param nlast  Last MD step index of the run.
    /// @return `*this`, chainable.
    FixLangevin& set_run_span(std::int64_t nfirst, std::int64_t nlast) {
        nfirst_ = nfirst;
        nlast_ = nlast;
        return *this;
    }

    /// @return Live thermostat setpoint [K], updated each `launch()`.
    double T_target() const noexcept { return T_live_; }

    std::string name() const override { return "fix_langevin"; }
    Role role() const override { return 0; }
    /* role via Thermostat base */
    /// @brief Writes drag + random forces; the Thermostat kind stays
    ///        energy/force-false so only this concrete overrides.
    bool provides_force() const override { return true; }
    /// @brief Langevin's role is Thermostat, but it functionally writes
    /// drag + random forces into fx/fy/fz — so it advertises
    /// `provides_force()`. This lets a pure Langevin run (no LJ / bonded
    /// evaluator) satisfy `cuda::md::MD::validate_components()`'s "something
    /// writes forces" contract.

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::post_force, +500)};
    }

    void launch(const LaunchContext& ctx) override;

  private:
    double T_start_;
    double T_stop_;
    double damp_;
    double dt_;
    std::uint32_t seed_;
    std::int64_t nfirst_ = 0;
    std::int64_t nlast_ = 0;
    double T_live_;
};

} // namespace md
} // namespace cuda
} // namespace ATV

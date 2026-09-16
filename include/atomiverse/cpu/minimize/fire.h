/// @file cpu/minimize/fire.h
/// @brief minimizer::FIRE — Bitzek FIRE 1.0 Optimizer for Minimize.
///
/// Fast Inertial Relaxation Engine (unit mass, symplectic Euler, no line
/// search). Pluggable alternative to `LBFGS` / `CG` on
/// `cpu::minimizer::Minimize`. Publishes `ConvergenceMetrics`; never
/// declares convergence (`ForceConverged` judges).
///
/// Private pseudo-velocity is component-owned — Frame `kAtom_v*` is never
/// written. MPI/DD: owned-row loops + `global_sum`/`global_max`; migration
/// generation bump restarts inertial state (like CG).
///
/// References:
///   Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006),
///     DOI 10.1103/PhysRevLett.97.170201
///   Guénolé et al., Comput. Mater. Sci. 175, 109584 (2020),
///     DOI 10.1016/j.commatsci.2020.109584 (FIRE 2.0 context; not this class)
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/optimizer.h"
#include "atomiverse/cpu/minimize/lbfgs.h" // OptRecord

namespace ATV {
namespace cpu {
namespace minimizer {

/// @brief Bitzek FIRE 1.0 geometry Optimizer (unit mass).
///
/// Algorithm knobs are public members with ASE-aligned FIRE-1.0 defaults.
/// Mutate them before `opt.build()`. No `*Config` bag and no factory.
class FIRE : public cpu::Optimizer {
  public:
    double dt0 = 0.1;    ///< Initial pseudo-timestep [fs].
    double dt_max = 1.0; ///< Maximum pseudo-timestep [fs].
    int n_min = 5;       ///< Steps with P>0 before growing Δt (Bitzek N_min).
    double f_inc = 1.1;  ///< Δt growth factor when accelerating.
    double f_dec = 0.5;  ///< Δt shrink factor when P ≤ 0.
    double alpha0 = 0.1; ///< Initial mixing parameter α_start (dimensionless).
    double f_alpha = 0.99; ///< α decay factor when accelerating.
    double max_step = 0.2; ///< Max per-atom |Δx| per step [Å].

    /// @brief Construct with ASE-aligned FIRE 1.0 defaults (see members).
    FIRE() = default;

    std::string name() const override { return "fire"; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::minimizer::Phase::search_direction, 0),
                ATV::make_binding(ATV::minimizer::Phase::convergence_check, 0)};
    }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Latest published max force magnitude [eV/Å].
    double max_force() const { return max_force_; }

    /// @brief Per-step diagnostic records (energy, forces, displacements).
    const std::vector<OptRecord>& trace() const { return trace_; }

  private:
    // Runtime FIRE state (reset in build / on migration).
    double dt_ = 0.1;    ///< Current pseudo-timestep [fs].
    double alpha_ = 0.1; ///< Current mixing α (dimensionless).
    int n_pos_ = 0;      ///< Consecutive steps with P > 0.
    int step_ = 0;       ///< Completed search_direction count.

    std::vector<double> v_;      ///< Private pseudo-velocity [Å/fs], 3N.
    std::vector<double> x_prev_; ///< Geometry at previous step [Å], 3N.
    bool has_prev_ = false;
    bool x_prev_valid_ = false;
    std::uint64_t last_generation_ = ~std::uint64_t{0};

    double max_force_ = 0;
    double energy_prev_ = 0;
    std::vector<OptRecord> trace_;
};

} // namespace minimizer
} // namespace cpu
} // namespace ATV

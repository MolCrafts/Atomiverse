/// @file cuda/dynamics/fix_csvr.cuh
/// @brief CUDA CSVR (Bussi) thermostat — LAMMPS `fix temp/csvr`, optionally
///        on a type group and with the `temp/com` bias. CUDA-free header.
///
/// Port reference: LAMMPS `src/EXTRA-FIX/fix_temp_csvr.cpp` and
/// `src/compute_temp_com.cpp`, e8c4c278a8.
/// Copyright (2003) Sandia Corporation. Under Contract DE-AC04-94AL85000,
/// the U.S. Government retains certain rights. GPL-2.0-only.
#pragma once

#include "atomiverse/core/csvr.h"
#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/component/thermostat.cuh"
#include "atomiverse/cuda/component/timestep_protocol.cuh"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ATV {
namespace cuda {
namespace md {

/// @brief CUDA sibling of `cpu::md::FixCSVR`: same law, same group and bias
///        semantics, same per-step random stream (`core/csvr.h`), so a CPU
///        and a CUDA run from one state draw identical randoms.
///
/// Per step (end_of_step +100) one kernel reduces the group's mass,
/// momentum and Σ m v² (and Σ m v² of every atom) and, in its last block,
/// draws the step's randoms from the device step, forms α and the COM
/// shift; a second kernel rescales the group, `v ← α v + (1 − α) v_cm`
/// (`v_cm = 0` without bias). No host synchronization: the run is graph
/// capturable. The Frame's `thermostat_energy` scalar accumulates −ΔK and
/// `conserved = energy + K_total + thermostat_energy` is written (valid on
/// steps whose energy was evaluated — the steps a thermo or recorder reads).
///
/// `N_f` is fixed at `build()`: the whole system's `n_dof_global`, or for a
/// type group `3 N_g − 3 − Σ_{t∈g} n_constraints_by_type[t]` (LAMMPS
/// `compute temp(/com)::dof_compute`, `FixShake::dof(igroup)`). Single GPU.
class FixCSVR : public cuda::Thermostat, public cuda::TimestepConsumer {
  public:
    /// @brief Whole-system thermostat, no bias.
    /// @param T_start Target / ramp-start temperature [K].
    /// @param tau     Coupling time constant [fs] (LAMMPS `Tdamp`).
    /// @param dt      Time step [fs]; replaced by the integrator's dt.
    /// @param seed    RNG seed.
    FixCSVR(double T_start, double tau, double dt, std::uint32_t seed);
    /// @brief Group thermostat.
    /// @param types Force-field types of the group; empty = every atom.
    /// @param bias  Velocity bias removed before measuring/rescaling.
    /// @throws std::invalid_argument for a negative type or a non-finite /
    ///         non-positive T_start, tau or dt.
    FixCSVR(double T_start, double tau, double dt, std::uint32_t seed,
            std::vector<int> types, ATV::md::TemperatureBias bias);
    ~FixCSVR() override;
    FixCSVR(FixCSVR&&) noexcept;
    FixCSVR& operator=(FixCSVR&&) noexcept;

    cuda::TimestepConsumer* as_timestep_consumer() override { return this; }
    bool timestep_required() const override { return false; }
    /// @throws std::invalid_argument when `dt_fs` is not finite and > 0.
    void set_timestep(double dt_fs) override;

    /// @brief Ramp end temperature [K] over the span of `set_run_span`.
    FixCSVR& set_T_stop(double T_stop) {
        T_stop_ = T_stop;
        ++epoch_;
        return *this;
    }
    /// @brief Ramp run span `[nfirst, nlast]` (MD step indices).
    FixCSVR& set_run_span(std::int64_t nfirst, std::int64_t nlast) {
        nfirst_ = nfirst;
        nlast_ = nlast;
        ++epoch_;
        return *this;
    }

    std::string name() const override { return "fix_csvr"; }
    Role role() const override { return 0; }
    bool graph_capturable() const override { return true; }
    std::uint64_t graph_epoch() const override { return epoch_; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +100)};
    }
    void build(const BuildContext& ctx) override;
    void before_launch(const LaunchContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

    /// @return Live setpoint [K] (host mirror of the device ramp).
    double T_target() const noexcept { return T_live_; }
    /// @return Group degrees of freedom `N_f` (after `build`).
    std::int64_t n_dof() const noexcept { return n_dof_; }
    /// @return Group force-field types (empty = every atom).
    const std::vector<int>& types() const noexcept { return types_; }
    /// @return Velocity bias.
    ATV::md::TemperatureBias bias() const noexcept { return bias_; }

  private:
    double T_start_, T_stop_, tau_, dt_;
    std::uint32_t seed_;
    std::vector<int> types_;
    std::vector<unsigned char> member_; ///< member_[type] != 0 ⇔ in group.
    ATV::md::TemperatureBias bias_ = ATV::md::TemperatureBias::None;
    std::int64_t n_dof_ = 0;
    std::int64_t nfirst_ = 0;
    std::int64_t nlast_ = 0;
    double T_live_ = 0.0;
    std::uint64_t epoch_ = 0;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace md
} // namespace cuda
} // namespace ATV

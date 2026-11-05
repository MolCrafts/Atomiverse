/// @file cuda/dynamics/nose_hoover_npt.cuh
/// @brief CUDA Martyna-Tobias-Klein NPT integrator (LAMMPS-equivalent
///        `fix npt`).
///
/// CUDA sibling of `cpu::md::NoseHooverNPT`. Same algorithm (Tuckerman
/// 2006 / Shinoda 2004 MTK) and Trotter ordering (P1–P11), same
/// LAMMPS-aligned `mtk_term1` / `mtk_term2`. The barostat is a single
/// chain over the active `ε̇_α` set with per-axis `couple_axis[]`
/// guards; non-coupled axes have `ε̇_α ≡ 0`. Pressure-coupling convention
/// matches LAMMPS `fix npt`: positive ⇔ compressive; per-axis driving
/// pressure `P_α = (σ_αα + 2·KE_α)/V` averaged over coupled axes when
/// `pdim==3` (the legacy isotropic case is bit-identical at
/// equipartition).
///
/// **Replaces (do not add alongside)** `cuda::md::VelocityVerlet` +
/// `cuda::md::NoseHoover` + `cuda::md::Barostat`. Rewrites
/// `frame.scalars->kinetic` and `frame.scalars->conserved` at
/// `end_of_step` so downstream observers (`Thermo`, recorders) see
/// post-NPT values.
///
/// Single-GPU scope (`msu-pe-validation` fixed-scope #1). Pressure-sign
/// is LAMMPS-standard (same as `cuda::Virial` /
/// `frame.scalars->stress` and Berendsen). Trotter ordering across the three
/// MD phases mirrors `cpu::md::NoseHooverNPT::launch`:
///
/// - `initial_integrate` (P1–P6): nhc_baro_half → nhc_atoms_half →
///   update_eps_dot → scale_v_by_baro → kick_v_by_force → propagate_x.
/// - `final_integrate` (P7): kick_v_by_force.
/// - `end_of_step` (P8–P11, after Reduce): recompute KE_global →
///   scale_v_by_baro → update_eps_dot → nhc_atoms_half (reversed
///   atoms-then-baro) → nhc_baro_half → write kinetic + conserved.
///
/// @see cpu::md::NoseHooverNPT for the CPU reference + full algorithm
///      documentation block.
/// @see LAMMPS `fix_nh.cpp` lines 2541-2569 (`mtk_term1` / `mtk_term2`).
#pragma once

#include "atomiverse/cuda/component/integrator.cuh"
#include "atomiverse/cuda/component/timestep_protocol.cuh"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {
namespace md {

/// @brief CUDA MTK NPT integrator. Same API surface and Trotter ordering
///        as `cpu::md::NoseHooverNPT`.
class NoseHooverNPT : public cuda::Integrator, public cuda::TimestepProvider {
  public:
    /// @brief Publishes the step to `cuda::md::MD` (timestep protocol).
    const cuda::TimestepProvider* as_timestep_provider() const override {
        return this;
    }
    /// @return The constructor `dt` [fs].
    double timestep() const override { return dt_; }
    /// @brief Construct MTK NPT (mirror of cpu::md::NoseHooverNPT; no Config bag).
    NoseHooverNPT(double T_target, double tau_T, double P_target, double tau_P,
                  double dt, double drag = 0.0, int mtchain = 3,
                  int mpchain = 3,
                  std::array<bool, 3> couple_axis = {true, true, true},
                  double T_stop = -1.0);

    /// @brief Set the temperature-ramp run span `[nfirst, nlast]`.
    /// @return `*this`, chainable.
    NoseHooverNPT& set_run_span(std::int64_t nfirst, std::int64_t nlast) {
        nfirst_ = nfirst;
        nlast_ = nlast;
        return *this;
    }

    /// @return Live thermostat setpoint [K] — for observers.
    double T_target() const noexcept { return T_live_; }

    bool changes_box(int stage) const override {
        return stage == slot_of(ATV::md::Phase::initial_integrate);
    }
    std::string name() const override { return "nose_hoover_npt"; }
    /// @brief Pair/bonded virial must be on — otherwise P = 2KE/V only
    ///        and the cell explodes (GH200 1335298, +28% in 80 steps).
    bool requires_system_stress() const override { return true; }
    /* role via Integrator base */
    /// @brief Write-set: MTK: propagates x and v, and scales the cell.
    /// @brief Reads — consumes `Stress` to drive ε̇ via `update_eps_dot_`.
    ///        Triggers `cuda::md::MD::install_infra_` to auto-inject
    ///        `cuda::Virial` if none is registered. The Integrator
    ///        role alone could not express this dependency.

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, 0),
                ATV::make_binding(ATV::md::Phase::end_of_step, -40)};
    }

    ~NoseHooverNPT() override;
    NoseHooverNPT(NoseHooverNPT&&) noexcept;
    NoseHooverNPT& operator=(NoseHooverNPT&&) noexcept;
    void initialize_forces(const LaunchContext& ctx) override;
    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  private:
    // Configuration
    double T_target_, T_stop_;
    double tau_T_, P_target_, tau_P_, dt_, drag_;
    int mtchain_, mpchain_, pdim_ = 3;
    std::array<bool, 3> couple_axis_;
    std::int64_t nfirst_ = 0, nlast_ = 0;

    double T_live_ = 0;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace md
} // namespace cuda
} // namespace ATV

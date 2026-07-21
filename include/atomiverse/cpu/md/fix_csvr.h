/// @file cpu/md/fix_csvr.h
/// @brief CPU CSVR (Bussi) stochastic-velocity-rescaling thermostat.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/component/thermostat.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief CSVR thermostat — canonical sampling through velocity rescaling.
///
/// Bussi, Donadio, Parrinello, "Canonical sampling through velocity
/// rescaling", J. Chem. Phys. **126**, 014101 (2007), DOI 10.1063/1.2408420.
/// CP2K's `&THERMOSTAT TYPE CSVR`.
///
/// Once per MD step the global kinetic energy `K` is stochastically driven
/// toward the canonical target `K̄ = ½ N_f k_B T` and every velocity is
/// rescaled by a single factor `α` (Bussi Eq. A7):
///
///     α² = e^{−Δt/τ}
///        + (K̄/(N_f K)) (1 − e^{−Δt/τ}) (R₁² + Σ_{i=2}^{N_f} R_i²)
///        + 2 e^{−Δt/2τ} √((K̄/(N_f K))(1 − e^{−Δt/τ})) R₁
///
/// with `R₁ ~ N(0,1)` and `Σ_{i=2}^{N_f} R_i²` drawn directly from a
/// `χ²(N_f−1) = Gamma((N_f−1)/2, scale 2)` deviate (not a sum of N_f−1
/// gaussians). Unlike Berendsen, CSVR injects the correct canonical
/// fluctuations and carries a conserved quantity: the running thermostat
/// energy `H̃_therm = −Σ ΔK` (Eq. 15), accumulated so that `E_pot + E_kin +
/// H̃_therm` (published to `frame.meta.get(kConserved)` [eV]) is drift-free.
///
/// Reproducibility: the two per-step randoms are GLOBAL (seeded only from
/// `(seed, step)`), so the trajectory is identical across MPI rank counts.
///
/// `N_f = 3N − 3` (COM removed), cached at `build()`. `τ` is the coupling
/// time [fs] (CP2K `TIMECON`). Supports the same `T_start`→`T_stop` linear
/// ramp as `cpu::md::NoseHoover` via `set_T_stop` + `set_run_span`.
class FixCSVR : public cpu::Thermostat {
  public:
    /// @param T_start Target / ramp-start temperature [K].
    /// @param tau     Coupling time constant [fs] (CP2K `TIMECON`).
    /// @param dt      Timestep [fs].
    /// @param seed    RNG seed; reproducible across rank counts.
    FixCSVR(double T_start, double tau, double dt, std::uint32_t seed);

    std::string name() const override { return "fix_csvr"; }
    /* role via Thermostat base */
    /// @brief Write-set: stochastic velocity rescale.
    /// @brief Applied once per step, after the velocity-Verlet step.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +100)};
    }
    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Set the ramp end temperature [K]; enables the linear ramp
    ///        over the span set by `set_run_span`. Default `= T_start`.
    /// @return `*this`, for chaining before registration.
    FixCSVR& set_T_stop(double T_stop) {
        T_stop_ = T_stop;
        return *this;
    }
    /// @brief Set the ramp run span `[nfirst, nlast]` in MD step indices.
    /// @return `*this`, for chaining before registration.
    FixCSVR& set_run_span(std::int64_t nfirst, std::int64_t nlast) {
        nfirst_ = nfirst;
        nlast_ = nlast;
        return *this;
    }
    /// @return Live thermostat setpoint [K] — for diagnostics / observers.
    double T_target() const noexcept { return T_live_; }
    /// @brief Accumulated thermostat energy `H̃_therm = −Σ ΔK` [eV]; the
    ///        Bussi conserved-quantity bookkeeping term (Eq. 15).
    double thermostat_energy() const noexcept { return thermostat_energy_; }

  private:
    /// @brief Interpolate the live setpoint for `step` (no-ramp default
    ///        returns `T_start_`), mirroring `NoseHoover::compute_temp_target`.
    double compute_temp_target(std::int64_t step) const;

    double T_start_;       ///< Ramp start / target temperature [K].
    double T_stop_;        ///< Ramp end temperature [K].
    double tau_;           ///< Coupling time constant [fs] (CP2K `TIMECON`).
    double dt_;            ///< Timestep [fs].
    std::uint32_t seed_;   ///< RNG seed (rank-count invariant).
    int n_dof_global_ = 0; ///< 3N − 3, cached at build().
    double thermostat_energy_ = 0; ///< Accumulated −Σ ΔK [eV] (Eq. 15).

    std::int64_t nfirst_ = 0; ///< First MD step of the run (ramp span).
    std::int64_t nlast_ = 0;  ///< Last MD step of the run (ramp span).
    double T_live_ = 0.0;     ///< Live setpoint [K], interpolated each step.
};

} // namespace md
} // namespace cpu
} // namespace ATV

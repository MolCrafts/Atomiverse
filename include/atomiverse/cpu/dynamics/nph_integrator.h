/// @file cpu/dynamics/nph_integrator.h
/// @brief CPU thermostat-free isotropic NPH (isoenthalpic–isobaric) integrator.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/integrator.h"
#include "atomiverse/cpu/component/timestep_protocol.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Isotropic NPH integrator — barostat without atom thermostat.
///
/// Ensemble \(N,P,H\): no Nosé–Hoover atom chain. The single isotropic
/// barostat DOF \(\dot{\varepsilon}\) is integrated with the MTK volume
/// force (Martyna et al. 1994) and no barostat thermostat either.
///
/// Conserved extended enthalpy written to `frame.meta[kConserved]`:
///
/// \f[
///   H = E_\mathrm{kin} + E_\mathrm{pot} + P_\mathrm{target}\,V
///       + \tfrac12 W \dot{\varepsilon}^2
/// \f]
///
/// There is **no** temperature setpoint. `tau_P` only sets the barostat
/// mass via \(W = (N_f+3)\,k_B T_\mathrm{ref}\,\tau_P^2\) with a fixed
/// reference temperature of 300 K used solely for dimensional mass
/// (not as a thermostat target). Pass `W_cell` explicitly to override.
///
/// Phase bindings match `NoseHooverNPT` / `ParrinelloRahmanNPT`.
/// Requires system stress. v1 is **isotropic volume only**; full-cell
/// NPH-PR is out of scope.
///
/// Units: \(P\) [eV/Å³], \(\tau_P\) [fs], \(dt\) [fs], \(W\) [eV·fs²].
///
/// Reference: Martyna, Tobias, Klein, *J. Chem. Phys.* **101**, 4177 (1994).
class NphIntegrator : public cpu::Integrator, public cpu::TimestepProvider {
  public:
    /// @param P_target Target pressure [eV/Å³], LAMMPS-standard sign
    ///                 (positive ⇔ compressive).
    /// @param tau_P    Barostat coupling time [fs] (sets default mass).
    /// @param dt       Timestep [fs].
    /// @param W_cell   Optional barostat mass [eV·fs²]. Negative (default)
    ///                 uses MTK mass at \(T_\mathrm{ref}=300\,\mathrm{K}\).
    NphIntegrator(double P_target, double tau_P, double dt,
                  double W_cell = -1.0);

    bool requires_system_stress() const override { return true; }
    std::string name() const override { return "nph_integrator"; }
    const cpu::TimestepProvider* as_timestep_provider() const override {
        return this;
    }
    /// @brief Published nuclear timestep [fs] (the constructor `dt`).
    double timestep() const override { return dt_; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, 0),
                ATV::make_binding(ATV::md::Phase::end_of_step, -40)};
    }
    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Instantaneous log-volume strain rate [1/fs] (isotropic).
    double eps_dot() const noexcept { return eps_dot_; }

  private:
    double P_target_;
    double tau_P_;
    double dt_;
    double W_override_;

    int n_dof_ = 0;
    double W_ = 0.0;
    double eps_dot_ = 0.0;
    double KE_global_ = 0.0;

    double compute_global_KE(const cpu::LaunchContext& ctx) const;
    std::array<double, 3> compute_axis_KE(const cpu::LaunchContext& ctx) const;
    void update_eps_dot(const cpu::LaunchContext& ctx, double half_dt);
    void scale_v_by_baro(const cpu::LaunchContext& ctx, double half_dt);
    void kick_v_by_force(const cpu::LaunchContext& ctx, double half_dt);
    void propagate_x_and_box(const cpu::LaunchContext& ctx);
    double extended_enthalpy(const cpu::LaunchContext& ctx) const;
};

} // namespace md
} // namespace cpu
} // namespace ATV

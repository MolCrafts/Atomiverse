/// @file cpu/dynamics/parrinello_rahman_npt.h
/// @brief CPU MTK–Parrinello–Rahman NPT integrator with full cell-matrix DOFs.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/integrator.h"
#include "atomiverse/cpu/component/timestep_protocol.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Martyna–Tobias–Klein NPT with full cell-matrix (Parrinello–Rahman) DOFs.
///
/// Extends the diagonal MTK barostat of @ref NoseHooverNPT to a symmetric
/// strain-rate tensor \(\dot{\varepsilon}\) (six independent components:
/// xx, yy, zz, xy, xz, yz). The cell updates as
/// \(\mathbf{h}\leftarrow\exp(\dot{\varepsilon}\,dt)\,\mathbf{h}\), so shear
/// (tilt) degrees of freedom evolve under the internal stress tensor.
///
/// External stress is hydrostatic at `P_target` (LAMMPS-standard sign:
/// positive \(\Leftrightarrow\) compressive). On a cubic isotropic start the
/// three diagonal rates stay near-equal and shear rates remain noise-bounded —
/// recovering the isotropic MTK limit of `NoseHooverNPT` statistically at the
/// same \((T,P,\tau)\).
///
/// Phase binding matches `NoseHooverNPT`:
///
///   * `initial_integrate @ 0`  — front Trotter half
///   * `final_integrate   @ 0`  — second velocity kick
///   * `end_of_step      @ -40` — rear Trotter half (after `Reduce`)
///
/// **Replaces** (do not add alongside) `VelocityVerlet`, `NoseHoover`, and
/// `Barostat`. Writes the MTK extended Hamiltonian to `kConserved` each step.
///
/// References: Parrinello & Rahman, *J. Appl. Phys.* **52**, 7182 (1981);
/// Martyna, Tobias, Klein, *J. Chem. Phys.* **101**, 4177 (1994);
/// Shinoda, Shiga, Mikami, *Phys. Rev. B* **69**, 134103 (2004).
class ParrinelloRahmanNPT : public cpu::Integrator,
                            public cpu::TimestepProvider {
  public:
    /// @brief Construct a flexible-cell MTK NPT integrator.
    /// @param T_target Target temperature [K].
    /// @param P_target Target hydrostatic pressure [eV/Å³], LAMMPS-standard
    ///                 sign (`+trace(σ)/(3V)`, positive ⇔ compressive).
    ///                 1 bar ≈ 6.2415e-7 eV/Å³.
    /// @param tau_T    Thermostat coupling time [fs] (LAMMPS `Tdamp`).
    /// @param tau_P    Barostat coupling time [fs] (LAMMPS `Pdamp`).
    /// @param dt       Timestep [fs].
    /// @param W_cell   Optional barostat mass \(W\) [eV·fs²]. If ≤0 (default),
    ///                 uses the MTK mass \((N_f+3)\,kT\,\tau_P^2\).
    ParrinelloRahmanNPT(double T_target, double P_target, double tau_T,
                        double tau_P, double dt, double W_cell = -1.0);

    /// @brief Optional empirical drag on NHC momenta (LAMMPS `drag`).
    ParrinelloRahmanNPT& set_drag(double drag) {
        drag_ = drag;
        return *this;
    }

    /// @brief Atom Nosé–Hoover chain length (LAMMPS `mtchain`, default 3).
    ParrinelloRahmanNPT& set_mtchain(int m) {
        mtchain_ = m;
        return *this;
    }

    /// @brief Barostat Nosé–Hoover chain length (LAMMPS `mpchain`, default 3).
    ParrinelloRahmanNPT& set_mpchain(int m) {
        mpchain_ = m;
        return *this;
    }

    bool requires_system_stress() const override { return true; }
    std::string name() const override { return "parrinello_rahman_npt"; }
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

    /// @brief Live strain-rate tensor \(\dot{\varepsilon}\) [1/fs] (diagnostic).
    const Mat3& strain_rate() const { return eps_dot_; }
    /// @brief Alias of @ref strain_rate.
    const Mat3& cell_velocity() const { return eps_dot_; }

  private:
    double T_target_;
    double P_target_;
    double tau_T_;
    double tau_P_;
    double dt_;
    double W_override_ = -1.0;
    double drag_ = 0.0;
    int mtchain_ = 3;
    int mpchain_ = 3;

    double tdrag_factor_ = 1.0;
    double pdrag_factor_ = 1.0;

    int n_dof_ = 0;
    double kT_ = 0.0;
    std::vector<double> Q_t_;
    std::vector<double> Q_b_;
    double W_ = 0.0; ///< Barostat mass [eV·fs²] (isotropic total).

    std::vector<double> eta_t_dot_;
    std::vector<double> eta_t_;
    std::vector<double> eta_b_dot_;
    std::vector<double> eta_b_;
    /// @brief Symmetric strain-rate tensor [1/fs].
    Mat3 eps_dot_{};

    double KE_global_ = 0.0;

    double compute_global_KE(const cpu::LaunchContext& ctx) const;
    std::array<double, 3> compute_axis_KE(const cpu::LaunchContext& ctx) const;
    void nhc_atoms_half(const cpu::LaunchContext& ctx, double half_dt);
    void nhc_baro_half(double half_dt);
    void update_eps_dot(const cpu::LaunchContext& ctx, double half_dt);
    void scale_v_by_baro(const cpu::LaunchContext& ctx, double half_dt);
    void kick_v_by_force(const cpu::LaunchContext& ctx, double half_dt);
    void propagate_x_and_box(const cpu::LaunchContext& ctx);
    double extended_hamiltonian(const cpu::LaunchContext& ctx) const;
};

} // namespace md
} // namespace cpu
} // namespace ATV

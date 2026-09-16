/// @file cpu/dynamics/fix_oscillating_efield.h
/// @brief CPU oscillating (AC) uniform external electric field — LAMMPS variable `fix efield`.
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/bias.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Apply a uniform **time-dependent** electric field to charged atoms.
///
/// Mirrors LAMMPS `fix efield` with equal-style variable components
/// (`lammps/src/fix_efield.cpp`, charge interactions):
///
///   \[
///     \mathbf{E}(t) = \mathbf{E}_0\cos\!\bigl(2\pi\, t / T + \varphi\bigr),
///     \quad t = n_{\mathrm{step}}\,\Delta t
///   \]
///   \[
///     \mathbf{F}_i \mathrel{+}= q_i \mathbf{E}(t),\qquad
///     U \mathrel{+}= -q_i\,\mathbf{E}(t)\cdot\mathbf{r}_i^{\mathrm{u}}
///   \]
///
/// with \(\mathbf{E}_0\) in V/Å, \(T\) and \(\Delta t\) in fs, \(\varphi\) in
/// rad, \(q\) in elementary charge, \(\mathbf{F}\) in eV/Å, \(U\) in eV.
/// Metal-unit `qe2f = 1`. Unwrapped coordinates use orthorhombic image flags
/// (same foothold as `FixEfield`; triclinic throws).
///
/// Instantaneous energy uses the constant-path formula with \(\mathbf{E}\)
/// frozen at the step value (LAMMPS treats variable fields as constant over
/// a timestep; pin energy via `potential` atom-style variable matching
/// \(\phi=-\mathbf{E}\cdot\mathbf{r}\)).
///
/// Bound by default at `md::Phase::post_force` (LAMMPS POST_FORCE mask).
///
/// ── Not in scope (see spec `oscillating-efield-md`) ───────────────────
/// atom-style spatial fields, region keyword, dipole torque, arbitrary
/// equal-style expression engine beyond AC cosine.
class FixOscillatingEfield : public cpu::Bias {
  public:
    /// @param Ex0 Amplitude of \(E_x\) [V/Å].
    /// @param Ey0 Amplitude of \(E_y\) [V/Å].
    /// @param Ez0 Amplitude of \(E_z\) [V/Å].
    /// @param period Oscillation period \(T\) [fs] (must be \(> 0\)).
    /// @param dt Timestep \(\Delta t\) [fs] (must be \(> 0\)); \(t = n\cdot\Delta t\).
    /// @param phase Phase offset \(\varphi\) [rad] (default 0).
    FixOscillatingEfield(double Ex0, double Ey0, double Ez0, double period,
                         double dt, double phase = 0.0) noexcept
        : Ex0(Ex0), Ey0(Ey0), Ez0(Ez0), period(period), dt(dt), phase(phase) {}

    /// Amplitudes [V/Å] — public members (no Config bag).
    double Ex0 = 0.0;
    double Ey0 = 0.0;
    double Ez0 = 0.0;
    /// Period \(T\) [fs].
    double period = 1.0;
    /// Timestep \(\Delta t\) [fs] used to form \(t = n\cdot\Delta t\).
    double dt = 1.0;
    /// Phase \(\varphi\) [rad].
    double phase = 0.0;

    std::string name() const override { return "fix_oscillating_efield"; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::post_force, +100)};
    }

    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Instantaneous \(E_x(n)\) [V/Å] at MD step \(n\).
    double field_x(int step) const noexcept {
        return Ex0 * std::cos(omega_t(step));
    }
    /// @brief Instantaneous \(E_y(n)\) [V/Å] at MD step \(n\).
    double field_y(int step) const noexcept {
        return Ey0 * std::cos(omega_t(step));
    }
    /// @brief Instantaneous \(E_z(n)\) [V/Å] at MD step \(n\).
    double field_z(int step) const noexcept {
        return Ez0 * std::cos(omega_t(step));
    }

    /// @return Last field energy contribution written this launch [eV].
    double last_energy() const noexcept { return last_u_; }

    /// @return Last instantaneous field components applied [V/Å].
    double last_Ex() const noexcept { return last_Ex_; }
    double last_Ey() const noexcept { return last_Ey_; }
    double last_Ez() const noexcept { return last_Ez_; }

  private:
    /// \(2\pi t/T + \varphi\) with \(t = n\cdot\Delta t\).
    double omega_t(int step) const noexcept {
        const double t = static_cast<double>(step) * dt;
        return 2.0 * M_PI * t / period + phase;
    }

    double last_u_ = 0.0;
    double last_Ex_ = 0.0;
    double last_Ey_ = 0.0;
    double last_Ez_ = 0.0;
};

} // namespace md
} // namespace cpu
} // namespace ATV

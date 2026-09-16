/// @file cpu/dynamics/init.h
/// @brief CPU MD infrastructure components: KineticEnergy, InitVelocity,
///        ZeroMomentum.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/observer.h"
#include "atomiverse/cpu/component/zero.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Compute kinetic energy at end_of_step phase.
///
/// KE = 0.5 * units::mass_v2_to_energy * sum(m * v^2), with OpenMP reduction.
class KineticEnergy : public cpu::Observer {
  public:
    std::string name() const override { return "kinetic_energy"; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, -100)};
    }
    void launch(const cpu::LaunchContext& ctx) override;
};

/// @brief Remove net linear COM momentum every `freq` steps.
///
/// LAMMPS `fix momentum N linear 1 1 1` equivalent. Prevents flying-ice-cube
/// drift that accumulates under Langevin + NPT coupling (the stochastic
/// Langevin kicks plus barostat scaling slowly bias the COM away from 0,
/// and fix_npt alone doesn't zero it).  Bound late in `end_of_step` so
/// the freshly-integrated velocities are zeroed before the next step.
class ZeroMomentum : public cpu::Component {
  public:
    /// @param freq Remove momentum every `freq` steps (LAMMPS default 1).
    explicit ZeroMomentum(int freq = 1) : freq_(freq) {}

    std::string name() const override { return "zero_linear_momentum"; }
    /// @brief Write-set: subtracts the drift from every velocity.
    std::vector<DefaultBinding> default_bindings() const override {
        // +25 = after KineticEnergy (@-100) / Reduce (@-50) /
        // NoseHooverNPT (@-40) so we only touch post-integration
        // velocities.  ThermoPrinter sits at +20, also behind us.
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +25)};
    }
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    int freq_;
    long long step_ = 0;
};

/// @brief Velocity-draw distribution for `InitVelocity` (LAMMPS `dist`).
///
/// Mirrors LAMMPS `velocity create ... dist` (`velocity.cpp:38`,
/// `enum { UNIFORM, GAUSSIAN }`):
///  - `uniform`  — each component drawn from `U(-0.5, 0.5)`
///                 (`random->uniform() - 0.5`, velocity.cpp:276), then
///                 scaled by `1/sqrt(m)`.
///  - `gaussian` — each component drawn from `N(0, 1)`
///                 (`random->gaussian()`, velocity.cpp:280), then scaled
///                 by `1/sqrt(m)`.
///
/// After either draw, the velocities are COM-corrected and rescaled so the
/// measured temperature exactly equals the requested `T`, so the choice of
/// `dist` only affects the *shape* of the per-atom distribution, never the
/// aggregate temperature.
enum class VelocityDist {
    uniform,  ///< U(-0.5, 0.5) per component (LAMMPS `dist uniform`).
    gaussian, ///< N(0, 1) per component (LAMMPS `dist gaussian`).
};

/// @brief Maxwell-Boltzmann velocity initialization (build-time only).
///
/// ## LAMMPS correspondence (`velocity create T seed`)
///
/// `InitVelocity` mirrors the LAMMPS `velocity create` command. Audited
/// against `lammps/src/velocity.cpp` (`stable` branch) —
/// `Velocity::create()` (velocity.cpp:160).
///
/// | LAMMPS keyword / behaviour    | `InitVelocity`              | Default            | Verdict |
/// |-------------------------------|-----------------------------|--------------------|---------|
/// | `velocity create T seed`      | `(temperature, seed)` ctor  | —                  | aligned (semantic parameter order) |
/// | `dist uniform` / `dist gaussian` | `VelocityDist` ctor arg  | `gaussian`         | **documented divergence**: LAMMPS defaults to `dist uniform` (velocity.cpp:86, `dist_flag = UNIFORM`). Atomiverse keeps `gaussian` as its default because a Gaussian draw *is* the physically-correct Maxwell-Boltzmann distribution; LAMMPS's `uniform` default is a historical convenience and is rescaled to the same temperature regardless. Both modes are exposed; only the default differs. |
/// | `mom yes` (zero linear momentum) | always removes COM velocity | always (`mom yes`) | aligned — `Velocity::zero_momentum()` (velocity.cpp:738) subtracts `Vcm`; `InitVelocity` does the same. A `mom no` opt-out is not exposed (no use case in the Atomiverse pipeline). |
/// | `rot no` (zero angular momentum) | not done                  | `rot no`           | aligned — LAMMPS default is `rot no` (velocity.cpp:89, `rotation_flag = 0`); `rot yes` is deferred. |
/// | `seed` (positive int)         | `unsigned seed` ctor arg    | —                  | aligned (semantic); the RNG itself differs — Atomiverse uses `std::mt19937`, LAMMPS uses `RanPark`. The seed→stream mapping is therefore not bit-identical (documented RNG divergence). |
/// | exact temperature             | post-COM-removal rescale    | always             | aligned — `Velocity::create()` calls `rescale(t, t_desired)` (velocity.cpp:368, `factor = sqrt(t_new/t_old)`) *after* `zero_momentum()`. `InitVelocity` replicates this: after the COM subtraction it measures the kinetic temperature (`3N − 3` DOF) and scales all velocities by `sqrt(T_target / T_measured)`, so the measured temperature equals the requested `T` exactly. |
///
/// Generates velocities from the chosen `dist` scaled by `1/sqrt(m)`,
/// removes the COM velocity, then rescales to the exact target
/// temperature. No-op during launch().
class InitVelocity : public cpu::Component {
  public:
    /// @param temperature Target temperature [K].
    /// @param seed Random number generator seed.
    /// @param dist Velocity-draw distribution (LAMMPS `dist`; Atomiverse
    ///        default `gaussian` — see the correspondence table for the
    ///        documented divergence from LAMMPS `dist uniform`).
    InitVelocity(double temperature, unsigned seed,
                 VelocityDist dist = VelocityDist::gaussian);

    std::string name() const override { return "init_velocity"; }
    /// @brief Write-set: seeds the Maxwell distribution into v (at build).
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, -999)};
    }
    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override {}

    /// @brief Velocity-draw distribution (LAMMPS `dist`).
    VelocityDist dist() const { return dist_; }

  private:
    double temperature_;
    unsigned seed_;
    VelocityDist dist_;
};

} // namespace md
} // namespace cpu
} // namespace ATV

/// @file cpu/dynamics/gear_pc.h
/// @brief Fifth-order Gear predictor–corrector integrator (CPU).
///
/// Replaces `VelocityVerlet`.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/integrator.h"
#include "atomiverse/cpu/component/timestep_protocol.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Six-value Gear predictor–corrector on the first-order equations of
///        motion \(\dot r = v\), \(\dot v = F/m\).
///
/// Each Cartesian position and velocity component carries a scaled Nordsieck
/// vector through its fifth derivative (`core/gear_pc.h`). The step is
///  - `initial_integrate`: predict every vector (positions and velocities
///    in the frame are the predicted values afterwards);
///  - `force`: the forces at the predicted positions (one evaluation per
///    step);
///  - `final_integrate`: evaluate the right-hand side at the predicted state
///    and correct with Gear's first-order coefficients.
///
/// The motion is adiabatic (Petravic & Delhommelle eq. 8 with
/// \(\alpha = 0\)): an external field such as `FixEfield` heats the system.
///
/// **Start-up.** The first launch after `build()` (or after the caller
/// changed positions, velocities or the cell between runs) sets
/// \(z_1 = h\,\dot y\) for every variable, \(z_2 = \tfrac{h^2}{2}\ddot r\)
/// for positions, and the remaining derivatives (which need force
/// derivatives) to zero, the customary Gear start. The method is sixth-order
/// accurate from an exact start; this start leaves a one-time transient that
/// persists as an \(O(h^2)\) offset of the trajectory (a harmonic oscillator
/// shows it), harmless for steady-state averages started from an
/// equilibrated state.
///
/// **Bookkeeping.** The Nordsieck history (30 doubles per atom) follows the
/// atoms by tag, so spatial sorting may reorder rows between steps. The
/// integrator needs every atom on the rank (`requires_full_frame`), and it
/// refuses holonomic constraints (\(N_\mathrm{dof}\) must be \(3N-3\)). The forces at
/// the corrected positions are never evaluated: thermo output at
/// `end_of_step` pairs corrected velocities with the forces and energy of
/// the predicted positions, as in every Gear predictor–corrector.
class GearPC : public cpu::Integrator, public cpu::TimestepProvider {
  public:
    /// @brief Adiabatic Gear PC.
    /// @param dt_fs Timestep in the run's time unit (ps for `metal`; > 0).
    explicit GearPC(double dt_fs);

    std::string name() const override { return "gear_pc"; }
    const cpu::TimestepProvider* as_timestep_provider() const override {
        return this;
    }
    double timestep() const override { return dt_; }
    bool requires_full_frame() const override { return true; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, 0)};
    }

    /// @brief Positions, velocities or the cell changed between runs: the
    ///        history no longer describes the trajectory, so the next step
    ///        starts afresh.
    void invalidate_inputs(InputChange changes,
                           const BuildContext& ctx) override;

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    void start_(const cpu::LaunchContext& ctx);
    void follow_rows_(const cpu::LaunchContext& ctx);
    void load_state_(const cpu::LaunchContext& ctx);
    void rates_(const cpu::LaunchContext& ctx);
    void predict_(const cpu::LaunchContext& ctx);
    void correct_(const cpu::LaunchContext& ctx);
    double* hist_(int var, int k) {
        return hist_store_.data() + (5 * var + (k - 1)) * n_;
    }

    double dt_;

    bool started_ = false;
    int n_ = 0;
    std::vector<double> hist_store_; ///< z1..z5 of x y z vx vy vz, 30·n
    std::vector<TagInt> row_tag_;    ///< tag of each history row
    std::vector<double> state_;      ///< x y z vx vy vz of the frame, 6·n
    std::vector<double> mass_;       ///< per row [mass]
    std::vector<double> rate_;       ///< ẋ ẏ ż v̇x v̇y v̇z, 6·n
};

} // namespace md
} // namespace cpu
} // namespace ATV

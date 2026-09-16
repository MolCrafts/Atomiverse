/// @file cpu/dynamics/fix_move.h
/// @brief Constant-velocity SMD / LAMMPS-like `fix move` (linear foothold).
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Impose \(\mathbf{r}(t)=\mathbf{r}_0+\mathbf{v}t\) on atoms of one type.
///
/// Mirrors the constant-velocity path of LAMMPS `fix move` / SMD (no spring).
/// Overwrites position each `initial_integrate` after free drift is unwanted:
/// binds at `initial_integrate @ +200` (after VelocityVerlet half-kick/drift).
/// Also zeros force on the group so NVE does not double-count (v fixed).
///
/// Parameters are ctor args / members — no Config bag.
class FixMove : public cpu::Component {
  public:
    /// @param vx Velocity x [Å/fs]
    /// @param vy Velocity y [Å/fs]
    /// @param vz Velocity z [Å/fs]
    /// @param type_id Atom type to move
    /// @param dt Timestep [fs] used to advance the imposed trajectory
    FixMove(double vx, double vy, double vz, int type_id, double dt) noexcept
        : vx(vx), vy(vy), vz(vz), type_id(type_id), dt(dt) {}

    double vx = 0.0, vy = 0.0, vz = 0.0;
    int type_id = 0;
    double dt = 0.001; ///< [fs]

    std::string name() const override { return "fix_move"; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, +200),
                ATV::make_binding(ATV::md::Phase::post_force, +150)};
    }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext&) override;

    /// @brief Unwrapped z of first selected atom after last integrate [Å].
    double last_z() const { return last_z_; }

  private:
    std::vector<int> atoms_;
    std::vector<double> x0_, y0_, z0_;
    int step0_ = 0;
    bool origin_set_ = false;
    double last_z_ = 0.0;
};

} // namespace md
} // namespace cpu
} // namespace ATV

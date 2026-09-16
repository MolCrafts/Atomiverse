/// @file cpu/dynamics/fix_freeze.h
/// @brief Immobile atom subset (electrode / wall freeze) — LAMMPS `fix setforce 0`.
#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/constraint.h"
#include "atomiverse/cpu/component/constraint_protocol.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Freeze atoms of one type: zero force, zero velocity, restore positions.
///
/// ## LAMMPS correspondence — `fix setforce 0.0 0.0 0.0` (audit: ALIGNED)
///
/// Mirrors LAMMPS `fix setforce fx fy fz` with all three components zero on a
/// wall / electrode group (`lammps/src/fix_setforce.cpp`): after forces are
/// accumulated, selected atoms have \(F \leftarrow 0\). Combined with initial
/// `velocity wall set 0 0 0`, velocity-Verlet then leaves those atoms fixed.
///
/// Atomiverse additionally restores build-time positions after the position
/// half-step and re-zeros velocities after both integrate halves so residual
/// round-off (or a thermostat force that landed before this component) cannot
/// creep the wall. That is a strict *superset* of setforce-0 immobility and
/// matches MetalWalls `mobile = false` electrode practice.
///
/// | Aspect | LAMMPS | `FixFreeze` | Verdict |
/// |---|---|---|---|
/// | Selection | `group` (type/region/id) | single atom `type` ctor arg | ALIGNED for electrode type groups |
/// | Force zero | `post_force`: \(F=0\) | `post_force @ +1000` | ALIGNED — after physical + Langevin (`+500`) |
/// | Velocity | caller zeros once | zero after integrate halves + at build | superset (safe) |
/// | Position | fixed if \(v=F=0\) | restore build-time \(x\) after drift | superset |
/// | DOF | group does not auto-adjust Ndof | `ConstraintProvider`: 3 per frozen atom | needed for T |
///
/// Bindings:
/// - `post_force @ +1000` — zero forces on frozen type
/// - `post_integrate @ 0` — restore \(x\), zero \(v\) after initial half
/// - `final_integrate @ +200` — zero \(v\) after final half
///
/// @note Electrode / constant-potential walls (E1): freeze the electrode type
///       so charge solve may update \(Q\) without integrating nuclear motion.
class FixFreeze : public cpu::Constraint, public cpu::ConstraintProvider {
  public:
    /// @param type_id Atom type to freeze (matches `frame.span(kAtom_type)`).
    ///                Electrode / wall atoms share one type id in textbook cases.
    explicit FixFreeze(int type_id);

    /// @return Frozen atom type id.
    int type_id() const noexcept { return type_id_; }

    std::string name() const override { return "fix_freeze"; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::post_force, +1000),
                ATV::make_binding(ATV::md::Phase::post_integrate, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, +200)};
    }

    const cpu::ConstraintProvider* as_constraint() const override {
        return this;
    }

    /// @brief Three scalar constraints per frozen atom (\(v_x=v_y=v_z=0\)).
    int n_constraints(const cpu::Context& ctx,
                      const cpu::Frame& frame) const override;

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    void resolve_rows_(const cpu::Frame& f, int n_owned);
    void zero_force_(cpu::Frame& f) const;
    void zero_velocity_(cpu::Frame& f) const;
    void restore_positions_(cpu::Frame& f) const;
    void snapshot_positions_(const cpu::Frame& f);

    int type_id_;
    /// Local owned rows of frozen atoms (rebuilt each launch under migration).
    std::vector<int> rows_;
    /// Build-time positions keyed by stable global tag [Å].
    std::unordered_map<TagInt, std::array<StorageReal, 3>> x0_by_tag_;
    int n_frozen_global_ = 0;
};

} // namespace md
} // namespace cpu
} // namespace ATV

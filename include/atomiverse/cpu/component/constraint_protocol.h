/// @file cpu/component/constraint_protocol.h
/// @brief ConstraintProvider — opt-in protocol for components that remove degrees
///        of freedom. CUDA-free header.
///
/// Follows the `NeighborConsumer` precedent: `Component`
/// stays minimal, and a component that participates inherits the protocol and
/// overrides the matching `as_*` accessor. The driver probes the typed accessor —
/// no `dynamic_cast`, no `name()` matching.
#pragma once

#include "atomiverse/core/precision.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace ATV {
namespace cpu {

class Frame;
struct Context;

/// @brief Add `n` constraints owned by an atom of type `type` to a per-type
///        tally (the `count_constraints_by_type` accumulator), growing it.
/// @throws std::invalid_argument for a negative type id.
inline void tally_constraints(std::vector<BigInt>& by_type, int type,
                              BigInt n) {
    if (type < 0)
        throw std::invalid_argument(
            "count_constraints_by_type: negative atom type id "
            + std::to_string(type));
    if (by_type.size() <= static_cast<std::size_t>(type))
        by_type.resize(static_cast<std::size_t>(type) + 1, 0);
    by_type[static_cast<std::size_t>(type)] += n;
}

/// @brief A component that constrains the system, and therefore removes degrees
///        of freedom from it.
///
/// **Why the engine has to know.** Temperature is `T = 2 KE / (Ndof kB)`. A rigid
/// water molecule has 6 degrees of freedom, not 9 — SETTLE removes 3 (two O-H bond
/// lengths and the H-O-H angle). If the engine keeps assuming `3N - 3`, a rigid-water
/// run reports a temperature ~33 % too low, AND every thermostat drives the system to
/// the wrong setpoint, because the thermostat's target kinetic energy is
/// `Ndof kB T / 2`. The bug is silent: nothing crashes, the trajectory just
/// equilibrates to the wrong temperature.
///
/// The count is per-frame (it scales with the number of constrained molecules), so
/// the driver asks for it after the frame is bound and before any component builds —
/// thermostats cache `Ndof` in their own `build()`.
class ConstraintProvider {
  public:
    virtual ~ConstraintProvider() = default;

    /// @brief Number of scalar constraints this component imposes, GLOBALLY.
    ///
    /// The count must be the whole system's, not this rank's share: `Ndof` is a
    /// global quantity, and the thermostats that consume it are driving a global
    /// kinetic target. The Context is passed because under domain decomposition the
    /// implementation has to know which rows of the frame are actually this rank's
    /// (`frame.nlocal()` is NOT the owner count — the tail is ghost slack) and
    /// which communicator to reduce over.
    ///
    /// @param ctx Execution context (decomposition, MPI size).
    /// @param frame The bound system state.
    /// @return Count of independent scalar constraints (e.g. 3 per rigid water).
    virtual int n_constraints(const cpu::Context& ctx,
                              const cpu::Frame& frame) const = 0;

    /// @brief Add this rank's constraints to `by_type`, keyed by the
    ///        force-field type (`kAtom_type`) of the atom that owns each
    ///        constraint — the row holding the bond, the angle centre, the
    ///        water oxygen, the frozen atom.
    ///
    /// A group thermostat counts its degrees of freedom as LAMMPS
    /// `FixShake::dof(igroup)` does: a constraint belongs to a group when
    /// its owning (central) atom does. Only OWNED rows are scanned and
    /// `by_type` is grown as needed; the driver sums the ranks once
    /// (`ctx.md.n_constraints_by_type`), so the per-type total equals
    /// `n_constraints` summed over the types.
    /// @param ctx Execution context.
    /// @param frame The bound system state.
    /// @param by_type Per-type counts, indexed by type id; accumulated into.
    virtual void
    count_constraints_by_type(const cpu::Context& ctx, const cpu::Frame& frame,
                              std::vector<BigInt>& by_type) const = 0;
};

} // namespace cpu
} // namespace ATV

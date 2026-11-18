/// @file cpu/driver/dof.h
/// @brief The system's degree-of-freedom count — computed once, owned by one place.
#pragma once

#include "atomiverse/core/precision.h"

#include <vector>

namespace ATV {
namespace cpu {

class Frame;
struct Context;

/// @brief Compute the global DOF count `3N - 3 - n_constraints` and publish it on
///        `ctx.md.n_dof_global`.
///
/// **Single owner.** Five sites used to hardcode `3N - 3` for themselves (Thermo,
/// InitVelocity, NoseHoover, NoseHooverNPT, FixCSVR). That formula stops being true
/// the moment anything is constrained, and the failure is silent: rigid water would
/// report a temperature ~33 % low, and every thermostat would drive the system to
/// the wrong setpoint, because a thermostat's target kinetic energy is
/// `Ndof kB T / 2`.
///
/// Called by `Driver::setup_pass()` between `set_frame()` and
/// `build_components()`. That ordering is load-bearing — thermostats cache Ndof in
/// their own `build()`, so a count published any later would be ignored by exactly
/// the components that most need it.
///
/// Lives in a `.cpp` rather than in the `Driver` template so the driver header
/// stays free of `<mpi.h>`.
///
/// @param ctx Execution context; `ctx.md.n_dof_global` is written.
/// @param frame Bound system state (supplies the local atom count).
/// @param n_constraints Total scalar constraints imposed by Constraint components.
/// @throws std::runtime_error if no degrees of freedom remain.
void publish_global_dof(Context& ctx, const Frame& frame, int n_constraints);

/// @brief Sum this rank's per-type constraint tallies over the ranks (in
///        place) and publish a view of them on `ctx.md`.
/// @param ctx Execution context; `md.n_constraints_by_type`,
///        `md.n_constraint_types` and `md.constraints_by_type_known` are
///        written.
/// @param tallies This rank's tallies (owned rows), indexed by type id;
///        replaced by the global sums. The caller owns it and keeps it alive
///        while `ctx` is used (the view points into it).
/// @param known False when the tallies could not be formed (no `type`
///        column with constraints present); published as unknown.
void publish_constraints_by_type(Context& ctx, std::vector<BigInt>& tallies,
                                 bool known);

} // namespace cpu
} // namespace ATV

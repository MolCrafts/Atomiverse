/// @file cuda/component/constraint_protocol.cuh
/// @brief ConstraintProvider — opt-in protocol for CUDA components that
///        remove degrees of freedom (mirrors cpu::ConstraintProvider).
///        CUDA-free header.
#pragma once

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/component/constraint_protocol.h" // tally_constraints

#include <vector>

namespace ATV {
namespace cpu {
class Frame;
} // namespace cpu
namespace cuda {

/// @brief A component that constrains the system, and therefore removes
///        degrees of freedom from it.
///
/// `T = 2 KE / (Ndof kB)` and every thermostat's target `Ndof kB T / 2` use
/// `Ndof = 3N - 3 - n_constraints`; a rigid water has 6 degrees of freedom,
/// not 9. The backend asks every provider of the loaded plan for its count
/// when it materializes the host frame, before any component builds, because
/// thermostats cache `Ndof` in their own `build()`.
class ConstraintProvider {
  public:
    virtual ~ConstraintProvider() = default;

    /// @brief Number of scalar constraints this component imposes on `frame`.
    /// @param frame The host frame being materialized.
    /// @return Count of independent scalar constraints (3 per rigid water).
    virtual BigInt n_constraints(const cpu::Frame& frame) const = 0;

    /// @brief Add the constraints to `by_type`, keyed by the force-field
    ///        type of the atom owning each one (bond owner row, angle
    ///        centre, water oxygen) — LAMMPS `FixShake::dof(igroup)`: a
    ///        constraint belongs to a group when its central atom does.
    ///        Mirrors `cpu::ConstraintProvider::count_constraints_by_type`.
    /// @param frame The host frame being materialized.
    /// @param by_type Per-type counts, indexed by type id; grown as needed.
    virtual void
    count_constraints_by_type(const cpu::Frame& frame,
                              std::vector<BigInt>& by_type) const = 0;
};

} // namespace cuda
} // namespace ATV

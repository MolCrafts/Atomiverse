/// @file cpu/dynamics/fix_lincs.h
/// @brief cpu::md::FixLincs — LINCS distance constraints.
#pragma once
#include "atomiverse/cpu/dynamics/distance_constraints.h"
namespace ATV::cpu::md {
/// @brief LINCS (Hess et al., J. Comput. Chem. 18, 1463 (1997)): RATTLE's
///        position and velocity stages solved by a truncated matrix
///        expansion instead of a Gauss–Seidel sweep.
///
/// The expansion converges only for weakly coupled constraints (bonds, or
/// bonds plus isolated angles). Strongly coupled angle constraints can make
/// it diverge; the component then throws and FixRattle or FixSettle is the
/// applicable solver.
class FixLincs : public DistanceConstraints {
  public:
    /// @param dt Timestep, in the run's time unit.
    /// @param tolerance Relative tolerance on `|r|^2/d^2 - 1`.
    /// @param max_iter Cap on correction sweeps per stage; exceeding it
    ///        throws.
    /// @param order Matrix-expansion order [dimensionless], >= 1.
    explicit FixLincs(double dt, double tolerance = default_tolerance(),
                      int max_iter = 500, int order = 4)
        : DistanceConstraints(dt, tolerance, max_iter, Algorithm::Lincs,
                              order) {}
};
} // namespace ATV::cpu::md

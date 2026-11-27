/// Force-level SHAKE: predict the next constrained positions and add the
/// resulting constraint forces before the final Verlet half-kick.
#pragma once
#include "atomiverse/cpu/dynamics/distance_constraints.h"
namespace ATV::cpu::md {
class FixShake : public DistanceConstraints {
  public:
    explicit FixShake(double dt, double tolerance = default_tolerance(),
                      int max_iter = 500)
        : DistanceConstraints(dt, tolerance, max_iter, Algorithm::Shake) {}
};
} // namespace ATV::cpu::md

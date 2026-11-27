/// SHAKE/RATTLE share the distance constraint solver and virial tally.
#pragma once
#include "atomiverse/cpu/dynamics/distance_constraints.h"
namespace ATV::cpu::md {
class FixRattle : public DistanceConstraints {
  public:
    explicit FixRattle(double dt, double tolerance = default_tolerance(),
                       int max_iter = 500)
        : DistanceConstraints(dt, tolerance, max_iter, Algorithm::Rattle) {}
};
} // namespace ATV::cpu::md

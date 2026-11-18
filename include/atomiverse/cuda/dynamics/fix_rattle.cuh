/// CUDA distance constraints with pressure-consistent reaction virial.
#pragma once
#include "atomiverse/cuda/dynamics/distance_constraints.cuh"
namespace ATV::cuda::md {
class FixRattle : public DistanceConstraints {
  public:
    explicit FixRattle(double dt, double tolerance = default_tolerance(),
                       int max_iter = 500)
        : DistanceConstraints(dt, tolerance, max_iter, true) {}
};
} // namespace ATV::cuda::md

/// CUDA distance constraints with pressure-consistent reaction virial.
#pragma once
#include "atomiverse/cuda/dynamics/distance_constraints.cuh"
namespace ATV::cuda::md {
class FixShake : public DistanceConstraints {
  public:
    explicit FixShake(double dt, double tolerance = default_tolerance(),
                      int max_iter = 500)
        : DistanceConstraints(dt, tolerance, max_iter, false) {}
};
} // namespace ATV::cuda::md

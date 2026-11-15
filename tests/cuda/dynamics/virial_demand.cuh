/// @file tests/cuda/dynamics/virial_demand.cuh
/// @brief Test-side consumer of the CUDA virial for the CPU-parity fixtures.
#pragma once

#include <string>

#include "atomiverse/cuda/component/observer.cuh"

namespace cuda_parity {

/// @brief Demands the virial on every step (LAMMPS `ev_set` vflag), so the
///        CUDA run fills `kStress` for the comparison with the CPU twin.
///        Nothing in the compared plans reads the stress, and CUDA force
///        kernels add the virial only on demand.
class VirialDemand final : public ATV::cuda::Observer {
  public:
    std::string name() const override { return "virial_demand"; }
    bool graph_capturable() const override { return true; }
    ATV::cuda::EvaluationDemand demand() const override {
        return {false, true, 1};
    }
    void launch(const ATV::cuda::LaunchContext& /*ctx*/) override {}
};

} // namespace cuda_parity

/// @file cuda/gpw/internal/gpw_access.cuh
/// @brief Private algorithm/unit access; not part of the Frame method API.
#pragma once
#include "atomiverse/cuda/gpw/gpw_solver.cuh"

namespace ATV::cuda::gpw::internal {
struct GpwDftAccess {
    static std::vector<std::array<double, 3>>
    forces_from_state(GpwDft& method,
                      const std::vector<cpu::gpw::internal::GpwAtom>& atoms,
                      GpwResult& scf, cudaStream_t stream = nullptr) {
        return method.forces_from_state_(atoms, scf, stream);
    }
    static std::array<double, 6>
    stress_from_state(const GpwDft& method,
                      const std::vector<cpu::gpw::internal::GpwAtom>& atoms,
                      const GpwResult& scf) {
        return method.stress_from_state_(atoms, scf);
    }
};
} // namespace ATV::cuda::gpw::internal

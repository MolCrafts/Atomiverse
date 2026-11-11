/// @file cuda/gpw/internal/gpw_access.cuh
/// @brief Private algorithm/unit access; not part of the Frame method API.
#pragma once
#include "atomiverse/cpu/gpw/internal/gpw_geometry.h"
#include "atomiverse/cuda/gpw/gpw_solver.cuh"

namespace ATV::cuda::gpw::internal {
struct GpwDftAccess {
    static cpu::gpw::internal::GpwGeometry
    assemble_geometry(const GpwDft& method,
                      std::vector<cpu::gpw::internal::GpwAtom> atoms) {
        return method.assemble_geometry_(std::move(atoms));
    }

    static std::vector<std::array<double, 3>> forces_from_state(
        GpwDft& method, const cpu::gpw::internal::GpwGeometry& geometry,
        const GpwResult& scf, const cpu::gpw::internal::GpwWorkspace& workspace,
        cudaStream_t stream = nullptr) {
        return method.forces_from_state_(geometry, scf, workspace, stream);
    }
    static std::array<double, 6>
    stress_from_state(const GpwDft& method,
                      const cpu::gpw::internal::GpwGeometry& geometry,
                      const GpwResult& scf,
                      const cpu::gpw::internal::GpwWorkspace& workspace) {
        return method.stress_from_state_(geometry, scf, workspace);
    }
};
} // namespace ATV::cuda::gpw::internal

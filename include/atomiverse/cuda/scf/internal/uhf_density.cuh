/// @file cuda/scf/internal/uhf_density.cuh
/// @brief internal::UHFDensity — back-transform, P^σ, UHF energy.
///
/// PySCF `uhf.make_rdm1` (occupation 1) and `uhf.energy_elec`:
///   E = ½ Tr[(H+Fα)Pα] + ½ Tr[(H+Fβ)Pβ] + V_nuc
/// which equals e1 + e2 for F^σ = H + V^σ. Fully on device.
#pragma once

#include "atomiverse/cuda/scf/internal/solver_context.cuh"
#include "atomiverse/cuda/scf/internal/uhf_scf_cycle.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {

#ifdef ATV_HAS_HF

class UHFDensity {
  public:
    UHFDensity() = default;
    void build(const SolverContext& ctx, SCFWorkspace& ws);
    void step(const SolverContext& ctx, SCFWorkspace& ws, UhfBeta& beta);
    void destroy(const SolverContext& ctx, SCFWorkspace& ws);

  private:
    void* cublas_handle_ = nullptr;
    double* d_HplusF_ = nullptr;
};

#endif

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV

/// @file cuda/scf/rhf_density.cu
/// @brief internal::RHFDensity: C = X*C', P = 2*C_occ*C_occ^T,
///        E = 0.5*Tr[(Hcore+F)*P]+V_nuc.

#ifdef ATV_HAS_HF

#include "atomiverse/cuda/scf/internal/rhf_density.cuh"
#include "cublas_dispatch.cuh"

#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

namespace ATV {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace scf {
namespace internal {

void RHFDensity::build(const SolverContext& ctx, SCFWorkspace& ws) {
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto& scf = ws;
    int n = scf.n;

    // Store n_occ in workspace
    scf.n_occ = n_occ_;

    // Allocate Hcore+F temporary
    cudaMalloc(&d_HplusF_, n * n * sizeof(prec_t));

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    cublas_handle_ = handle;
}

void RHFDensity::step(const SolverContext& ctx, SCFWorkspace& ws) {
    auto& scf = ws;
    if (scf.converged)
        return;

    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto handle = static_cast<cublasHandle_t>(cublas_handle_);
    int n = scf.n;
    int nn = n * n;

    // After EigenSolver: scf.C contains eigenvectors of F' (in orthogonal basis),
    // scf.eig contains eigenvalues. scf.work contains the real Fock matrix F.

    // Snapshot the orthonormal-basis eigenvectors C' for the NEXT
    // iteration's level-shift projector (hf-robust-03). scf.C still holds
    // C' here (EigenSolver's output), before the back-transform below
    // overwrites it with X·C'.
    cudaMemcpyAsync(scf.C_prime, scf.C, nn * sizeof(prec_t),
                    cudaMemcpyDeviceToDevice, stream);

    // Step 1: Back-transform C = X * C'
    // C' is in scf.C (from EigenSolver). We need C = X * C'.
    // Use d_HplusF_ as temp to hold C' before overwriting scf.C.
    cudaMemcpyAsync(d_HplusF_, scf.C, nn * sizeof(prec_t),
                    cudaMemcpyDeviceToDevice, stream);
    prec_t one = prec_t(1);
    prec_t zero = prec_t(0);
    detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, scf.X, n,
                 d_HplusF_, n, &zero, scf.C, n);

    // Step 2: P_old = P; P = 2 * C_occ * C_occ^T
    cudaMemcpyAsync(scf.P_old, scf.P, nn * sizeof(prec_t),
                    cudaMemcpyDeviceToDevice, stream);
    prec_t two = prec_t(2);
    detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n_occ_, &two, scf.C, n,
                 scf.C, n, &zero, scf.P, n);

    // Step 3: Energy = 0.5 * Tr[(Hcore + F) * P] + V_nuc
    // scf.work = F (saved by RHFFock). Hcore = scf.Hcore.
    // d_HplusF_ = Hcore + F
    cudaMemcpyAsync(d_HplusF_, scf.Hcore, nn * sizeof(prec_t),
                    cudaMemcpyDeviceToDevice, stream);
    detail::axpy(handle, nn, &one, scf.work, 1, d_HplusF_, 1);

    // E_elec = 0.5 * dot(HplusF, P)
    prec_t dot_val = 0;
    detail::dot(handle, nn, d_HplusF_, 1, scf.P, 1, &dot_val);
    prec_t energy = prec_t(0.5) * dot_val + static_cast<prec_t>(scf.V_nuc);

    // Write to device energy scalar
    cudaMemcpyAsync(&ctx.frame.scalars->energy, &energy, sizeof(prec_t),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
}

void RHFDensity::destroy(const SolverContext& /*ctx*/, SCFWorkspace& /*ws*/) {
    if (cublas_handle_) {
        cublasDestroy(static_cast<cublasHandle_t>(cublas_handle_));
        cublas_handle_ = nullptr;
    }
    cudaFree(d_HplusF_);
    d_HplusF_ = nullptr;
}

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV

#endif // ATV_HAS_HF

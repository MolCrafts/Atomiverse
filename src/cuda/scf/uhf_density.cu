/// @file cuda/scf/uhf_density.cu
/// @brief UHF density and energy on device.
///
/// `pyscf.scf.uhf.make_rdm1`: P^σ = C_occ^σ (C_occ^σ)^T.
/// `pyscf.scf.uhf.energy_elec`: E = e1 + e2 + V_nuc with
/// e1 + e2 = ½ Σ_σ Tr[(H + F^σ) P^σ].

#ifdef ATV_HAS_HF

#include "atomiverse/cuda/scf/internal/uhf_density.cuh"

#include "cublas_dispatch.cuh"

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/types.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {
namespace {

__global__ void uhf_write_energy_ev_kernel(double* __restrict__ energy_ev,
                                           const prec_t* __restrict__ dot_a,
                                           const prec_t* __restrict__ dot_b,
                                           const double* __restrict__ d_V_nuc,
                                           prec_t ha2ev) {
    const prec_t e_ha =
        prec_t(0.5) * ((*dot_a) + (*dot_b)) + static_cast<prec_t>(*d_V_nuc);
    *energy_ev = e_ha * ha2ev;
}

void rebuild_spin(cublasHandle_t handle, cudaStream_t stream, int n, int n_occ,
                  prec_t* X, prec_t* C, prec_t* C_prime, prec_t* P,
                  prec_t* P_old, prec_t* Hcore, prec_t* F, prec_t* HplusF,
                  prec_t* dot_out) {
    const int nn = n * n;
    const size_t bytes = static_cast<size_t>(nn) * sizeof(prec_t);
    cudaMemcpyAsync(C_prime, C, bytes, cudaMemcpyDeviceToDevice, stream);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    const prec_t one = prec_t(1);
    const prec_t zero = prec_t(0);
    detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, X, n, C_prime,
                 n, &zero, C, n);
    cudaMemcpyAsync(P_old, P, bytes, cudaMemcpyDeviceToDevice, stream);
    if (n_occ <= 0) {
        cudaMemsetAsync(P, 0, bytes, stream);
    } else {
        detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n_occ, &one, C, n,
                     C, n, &zero, P, n);
    }
    cudaMemcpyAsync(HplusF, Hcore, bytes, cudaMemcpyDeviceToDevice, stream);
    detail::axpy(handle, nn, &one, F, 1, HplusF, 1);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    detail::dot(handle, nn, HplusF, 1, P, 1, dot_out);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
}

} // namespace

void UHFDensity::build(const SolverContext& ctx, SCFWorkspace& ws) {
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    cudaMalloc(&d_HplusF_, static_cast<size_t>(ws.n * ws.n) * sizeof(prec_t));
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    cublas_handle_ = handle;
}

void UHFDensity::step(const SolverContext& ctx, SCFWorkspace& ws,
                      UhfBeta& beta) {
    if (ws.converged)
        return;
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto handle = static_cast<cublasHandle_t>(cublas_handle_);
    cublasSetStream(handle, stream);
    rebuild_spin(handle, stream, ws.n, ws.n_occ, ws.X, ws.C, ws.C_prime, ws.P,
                 ws.P_old, ws.Hcore, ws.work, d_HplusF_, ws.d_scalar_scratch);
    rebuild_spin(handle, stream, ws.n, beta.n_occ, ws.X, beta.C, beta.C_prime,
                 beta.P, beta.P_old, ws.Hcore, beta.work, d_HplusF_,
                 ws.d_scalar_scratch + 1);
    if (ctx.frame.scalars) {
        uhf_write_energy_ev_kernel<<<1, 1, 0, stream>>>(
            &ctx.frame.scalars->energy, ws.d_scalar_scratch,
            ws.d_scalar_scratch + 1, ws.d_V_nuc,
            static_cast<prec_t>(units::kHartreeInEv));
        ATV_CUDA_CHECK(cudaGetLastError());
    }
}

void UHFDensity::destroy(const SolverContext&, SCFWorkspace&) {
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

#endif

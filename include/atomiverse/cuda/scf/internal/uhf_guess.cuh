/// @file cuda/scf/internal/uhf_guess.cuh
/// @brief Device MINAO projection; no scientific host state or fallback.
#pragma once

#include "atomiverse/cuda/scf/internal/uhf_guess_data.h"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace ATV::cuda::scf::internal {

/// Owns setup constants and bounded O(nAO^2+nAO*nMINAO) projection scratch.
class UhfMinaoGuess {
  public:
    UhfMinaoGuess(const int* atomic_numbers, int natm);
    ~UhfMinaoGuess();
    UhfMinaoGuess(const UhfMinaoGuess&) = delete;
    UhfMinaoGuess& operator=(const UhfMinaoGuess&) = delete;

    /// Allocate/upload only geometry-independent constants, before iter0.
    void build(const SCFWorkspace& ws, cudaStream_t stream);
    /// Pa=Pb=P_MINAO/2, retaining neutral atomic occupations for all charges.
    /// With an unsupported element leaves the caller's zero densities intact.
    void seed(const SCFWorkspace& ws, prec_t* beta_density,
              cudaStream_t stream);

  private:
    void clear_() noexcept;
    UhfMinaoData table_;
    int natm_ = 0, n_ = 0, m_ = 0, potrf_size_ = 0;
    cublasHandle_t blas_ = nullptr;
    cusolverDnHandle_t solver_ = nullptr;
    double *alpha_ = nullptr, *coefficient_ = nullptr, *occupation_ = nullptr;
    int *offset_ = nullptr, *angular_ = nullptr, *atom_ = nullptr;
    int *ao_shell_ = nullptr, *ao_cart_ = nullptr;
    double *factor_ = nullptr, *projected_ = nullptr, *scaled_ = nullptr;
    double* potrf_work_ = nullptr;
    int *h_status_ = nullptr, *d_status_ = nullptr;
};

} // namespace ATV::cuda::scf::internal

/// @file cuda/linalg/pulay.cuh
/// @brief Device Pulay coefficient solve used by CUDA CDIIS.
///
/// The bordered system is the same one as `cpu::linalg::pulay_coefficients`:
/// [Gram, -1; -1, 0] c = [0; -1], factored with cuSOLVER LU. A singular or
/// nonfinite factorization writes a one-hot coefficient on `latest` so the
/// mixer keeps that history vector and nothing else.
#pragma once

#include <cusolverDn.h>

#include "atomiverse/cuda/frame/types.cuh"

namespace ATV::cuda::linalg {

void solve_pulay(cusolverDnHandle_t solver, cudaStream_t stream,
                 const prec_t* gram, prec_t* augmented, prec_t* coefficients,
                 prec_t* workspace, int* pivots, int* info, int m, int latest);

} // namespace ATV::cuda::linalg

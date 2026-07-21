/// @file cuda/frame/types.cuh
/// @brief CUDA backend mixed-precision types.
///
/// Three precision tiers for device data:
///   real_t  -- per-atom storage (N-scaled, bandwidth-bound). Default: float.
///   work_t  -- kernel computation, shared memory, per-type params. Default: double.
///   prec_t  -- accumulators, box, SCF workspace (precision-critical). Default: double.
///
/// Constraint: sizeof(real_t) <= sizeof(work_t) <= sizeof(prec_t).
/// Host code uses plain double. These types are CUDA-backend-only.
///
/// Configure via CMake:
///   -DATV_CUDA_REAL=float|double     (default: float)
///   -DATV_CUDA_WORK=float|double     (default: double)
///   -DATV_CUDA_PREC=float|double     (default: double)
#pragma once

#include <type_traits>

namespace ATV {
namespace cuda {

#ifndef ATV_CUDA_REAL_TYPE
#define ATV_CUDA_REAL_TYPE float
#endif

#ifndef ATV_CUDA_WORK_TYPE
#define ATV_CUDA_WORK_TYPE double
#endif

#ifndef ATV_CUDA_PREC_TYPE
#define ATV_CUDA_PREC_TYPE double
#endif

using real_t = ATV_CUDA_REAL_TYPE;
using work_t = ATV_CUDA_WORK_TYPE;
using prec_t = ATV_CUDA_PREC_TYPE;

static_assert(sizeof(real_t) <= sizeof(work_t),
              "real_t must not exceed work_t");
static_assert(sizeof(work_t) <= sizeof(prec_t),
              "work_t must not exceed prec_t");

} // namespace cuda
} // namespace ATV

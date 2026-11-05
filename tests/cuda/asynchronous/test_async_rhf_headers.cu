/// @file tests/cuda/asynchronous/test_async_rhf_headers.cu
/// @brief Compile-smoke for the RHF-gated asynchronous consumers.
///
/// `cuda::MullikenObs` (observe/mulliken.cuh) and `cuda::DensityRecorder`
/// (record/density_recorder.cuh) are header-only and, outside the benchmark
/// suite, are not included by any other compiled translation unit. Their
/// port onto `cuda::AsyncConsumer` therefore needs an explicit compile site
/// so a type error in `consume` / `payload_source` / `payload_bytes` is
/// caught.
/// Including the headers here type-checks the class bodies under ATV_HAS_HF;
/// the `sizeof` references force the complete types. No GPU work is performed.

#include <gtest/gtest.h>

#include "atomiverse/cuda/observe/mulliken.cuh"
#include "atomiverse/cuda/record/density_recorder.cuh"

TEST(CudaAsyncRhfHeaders, Compile) {
#ifdef ATV_HAS_HF
    EXPECT_GT(sizeof(ATV::cuda::MullikenObs), 0u);
    EXPECT_GT(sizeof(ATV::cuda::DensityRecorder), 0u);
#endif
    SUCCEED();
}

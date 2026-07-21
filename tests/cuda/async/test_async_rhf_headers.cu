/// @file tests/cuda/async/test_async_rhf_headers.cu
/// @brief Compile-smoke for the RHF-gated AsyncBridge subclasses.
///
/// `cuda::MullikenObs` (observer/mulliken.cuh) and `cuda::DensityRecorder`
/// (recorder/density_recorder.cuh) are header-only and, outside the benchmark
/// suite, are not included by any other compiled translation unit. Their
/// device-ring freeze retrofit therefore needs an explicit compile site so a
/// type error in `freeze_slot` / `d2h_slot` / `on_data_ready(slot)` is caught.
/// Including the headers here type-checks the class bodies under ATV_HAS_HF;
/// the `sizeof` references force the complete types. No GPU work is performed.

#include <gtest/gtest.h>

#include "atomiverse/cuda/observer/mulliken.cuh"
#include "atomiverse/cuda/recorder/density_recorder.cuh"

TEST(CudaAsyncRhfHeaders, Compile) {
#ifdef ATV_HAS_HF
    EXPECT_GT(sizeof(ATV::cuda::MullikenObs), 0u);
    EXPECT_GT(sizeof(ATV::cuda::DensityRecorder), 0u);
#endif
    SUCCEED();
}

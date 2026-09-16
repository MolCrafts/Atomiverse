/// @file cuda/asynchronous/sync_guard.cuh
/// @brief `cuda::stream_synchronize` wrapper + opt-in hot-path sync counter
///        used to prove the `AsyncRecordBase` contract.
///
/// This header provides a thin wrapper `cuda::stream_synchronize(stream)`
/// that components route through instead of calling `cudaStreamSynchronize`
/// directly. Under `-DATV_BUILD_TESTS=ON` it increments an atomic counter
/// that the chain link 07 tests poll to verify the hot-path zero-sync
/// invariant. Under a release build the wrapper compiles down to a direct
/// `cudaStreamSynchronize` call with zero counter overhead — there is no
/// atomic increment, no dependency on test infrastructure, and the
/// preprocessed source contains exactly one runtime call (the bare CUDA
/// API). See `tests/cuda/async/test_hot_path_no_sync.cu` for the gate.
///
/// The counter is **opt-in**: only sync calls that go through this wrapper
/// are counted. Backend-level slot-boundary syncs in `src/cuda/driver/driver.cu`
/// continue to call `cudaStreamSynchronize` directly and are deliberately
/// excluded from the count — they are not user-component code and are not
/// what the spec is gating.
///
/// @see cuda::AsyncRecordBase for the recorder base class whose contract
///      this counter protects.
#pragma once

#include <cuda_runtime.h>

namespace ATV {
namespace cuda {

namespace detail {

#ifdef ATV_BUILD_TESTS

/// @brief Read the count of `cuda::stream_synchronize` calls since the
///        last `reset_sync_count()`.
/// @return Non-negative call count; only defined in `ATV_BUILD_TESTS=ON`
///         builds.
unsigned long long sync_count() noexcept;

/// @brief Reset the sync call counter to zero. Tests call this at the
///        start of each step and assert `sync_count() == 0` at end.
void reset_sync_count() noexcept;

/// @brief Atomically increment the sync counter. Internal helper invoked
///        by `cuda::stream_synchronize`; not part of the public API.
void inc_sync_count() noexcept;

#endif // ATV_BUILD_TESTS

} // namespace detail

/// @brief Thin wrapper around `cudaStreamSynchronize` that increments the
///        opt-in sync counter under `ATV_BUILD_TESTS=ON`.
/// @param stream CUDA stream to synchronise (non-owning handle).
/// @return Underlying `cudaError_t` from `cudaStreamSynchronize`.
///
/// Release builds reduce to a single call to `cudaStreamSynchronize`.
inline cudaError_t stream_synchronize(cudaStream_t stream) {
#ifdef ATV_BUILD_TESTS
    detail::inc_sync_count();
#endif
    return cudaStreamSynchronize(stream);
}

} // namespace cuda
} // namespace ATV

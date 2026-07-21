/// @file cuda/device/log_callback.cpp
/// @brief cuLogs callback implementation -- compiled by host compiler (not nvcc).
///
/// Registers a callback with cuLogsRegisterCallback (CUDA Driver API, 12.8+)
/// that forwards CUDA internal log/error messages to spdlog.
/// Guarded by CUDA_VERSION; compiles to no-ops on older toolkits.

#include "atomiverse/cuda/device/log_callback.h"
#include "atomiverse/core/log.h"

#include <cuda.h>

// cuLogs API requires CUDA 12.8+
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080

static CUlogsCallbackHandle g_log_handle = nullptr;

static void cuda_log_callback(void* /*userData*/, CUlogLevel level,
                              char* message, size_t /*length*/) {
    // CUlogLevel values are not yet publicly documented; treat everything
    // at warn-or-above so CUDA internal errors surface visibly.
    (void)level;
    ATV_WARN("[CUDA] {}", message);
}

namespace ATV {
namespace cuda {

void register_log_callback() {
    if (g_log_handle)
        return;
    CUresult res =
        cuLogsRegisterCallback(cuda_log_callback, nullptr, &g_log_handle);
    if (res != CUDA_SUCCESS) {
        ATV_WARN("cuLogsRegisterCallback failed (CUresult={})",
                 static_cast<int>(res));
    } else {
        ATV_DEBUG("CUDA log callback registered");
    }
}

void unregister_log_callback() {
    if (!g_log_handle)
        return;
    cuLogsUnregisterCallback(g_log_handle);
    g_log_handle = nullptr;
}

} // namespace cuda
} // namespace ATV

#else // CUDA < 12.8 -- no-ops

namespace ATV {
namespace cuda {

void register_log_callback() {}
void unregister_log_callback() {}

} // namespace cuda
} // namespace ATV

#endif

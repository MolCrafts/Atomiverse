/// @file python/src/bindings/capture_stats.h
/// @brief `ATV::CaptureStats` → Python dict, shared by the CPU and CUDA
///        driver handles (`MD.capture_stats()`).
#pragma once

#include <nanobind/nanobind.h>

#include "atomiverse/core/capture_buffer.h"

/// @brief The driver's capture-buffer accounting as a dict with the keys of
///        a `*.mrec` store's `status.buffer` section plus `capacity` and
///        `captures`.
inline nanobind::dict capture_stats_dict(const ATV::CaptureStats& s) {
    nanobind::dict d;
    d["budget_bytes"] = s.budget_bytes;
    d["frame_bytes"] = s.frame_bytes;
    d["capacity"] = s.capacity;
    d["captures"] = s.captures;
    d["peak_bytes"] = s.peak_bytes;
    d["blocked_episodes"] = s.blocked_episodes;
    d["blocked_seconds"] = s.blocked_seconds;
    return d;
}

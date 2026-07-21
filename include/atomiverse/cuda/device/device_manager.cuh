#pragma once

/// @file cuda/device/device_manager.cuh
/// @brief DeviceManager — single-process multi-GPU context for the CUDA MD
///        backend. Enumerates the visible GPUs, creates a main + deferrable
///        stream per device, and enables CUDA peer access (P2P over NVLink)
///        between every pair that supports it. Each device hosts one spatial
///        subdomain; inter-subdomain halo/migration use cudaMemcpyPeerAsync.
///
/// Pure host-side resource owner (plain class, not a pydantic-style value) —
/// holds live CUDA handles, so it has an explicit ctor/dtor and is move-only.
/// Single-GPU runs simply construct it with count()==1 (no peer access), so
/// the existing single-device path is unaffected.

#include <memory>

namespace ATV {
namespace cuda {

class DeviceManager {
  public:
    /// @param n_requested  Number of GPUs to use (clamped to the visible
    ///                     count). <= 0 ⇒ use all visible GPUs.
    explicit DeviceManager(int n_requested = -1);
    ~DeviceManager();
    DeviceManager(DeviceManager&&) noexcept;
    DeviceManager& operator=(DeviceManager&&) noexcept;
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    /// Number of GPUs this manager owns.
    int count() const;

    /// CUDA device id (the value passed to cudaSetDevice) for slot `i`.
    int device_id(int i) const;

    /// Per-device main / deferrable streams (cudaStream_t as void*, matching
    /// cuda::Context's stream convention).
    void* stream(int i) const;
    void* defer_stream(int i) const;

    /// True iff peer access slot a -> slot b was successfully enabled (P2P).
    bool peer_enabled(int a, int b) const;

    /// True iff every owned pair has peer access (full NVLink mesh).
    bool all_peers_enabled() const;

    /// cudaSetDevice(device_id(i)); throws on error.
    void set_device(int i) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cuda
} // namespace ATV

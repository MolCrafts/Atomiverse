/// @file cuda/device/device_manager.cu
/// @brief DeviceManager implementation — multi-GPU enumeration, per-device
///        streams, and P2P peer-access enable.

#include "atomiverse/cuda/device/device_manager.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace ATV {
namespace cuda {

namespace {
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("DeviceManager: ") + what + ": "
                                 + cudaGetErrorString(e));
}
} // namespace

struct DeviceManager::Impl {
    int n = 0;
    std::vector<int> ids;            // slot -> cuda device id
    std::vector<cudaStream_t> main;  // per-slot main stream
    std::vector<cudaStream_t> defer; // per-slot deferrable stream
    std::vector<char> peer;          // n*n row-major: peer[a*n+b]
};

DeviceManager::DeviceManager(int n_requested)
    : impl_(std::make_unique<Impl>()) {
    auto& s = *impl_;
    int visible = 0;
    check(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");
    if (visible <= 0)
        throw std::runtime_error("DeviceManager: no CUDA devices visible");
    s.n = (n_requested <= 0) ? visible
                             : (n_requested < visible ? n_requested : visible);
    s.ids.resize(s.n);
    s.main.resize(s.n);
    s.defer.resize(s.n);
    s.peer.assign(static_cast<size_t>(s.n) * s.n, 0);

    for (int i = 0; i < s.n; ++i) {
        s.ids[i] = i; // identity mapping (slot i -> device i)
        check(cudaSetDevice(s.ids[i]), "cudaSetDevice");
        check(cudaStreamCreate(&s.main[i]), "cudaStreamCreate(main)");
        check(cudaStreamCreate(&s.defer[i]), "cudaStreamCreate(defer)");
        s.peer[static_cast<size_t>(i) * s.n + i] = 1; // self
    }

    // Enable peer access for every ordered pair that supports it. Failures
    // are recorded (peer stays 0) rather than thrown — a missing P2P link
    // falls back to staged copies, it is not fatal.
    for (int a = 0; a < s.n; ++a) {
        check(cudaSetDevice(s.ids[a]), "cudaSetDevice(peer)");
        for (int b = 0; b < s.n; ++b) {
            if (a == b)
                continue;
            int can = 0;
            cudaDeviceCanAccessPeer(&can, s.ids[a], s.ids[b]);
            if (!can)
                continue;
            cudaError_t e = cudaDeviceEnablePeerAccess(s.ids[b], 0);
            if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled) {
                s.peer[static_cast<size_t>(a) * s.n + b] = 1;
                if (e == cudaErrorPeerAccessAlreadyEnabled)
                    cudaGetLastError(); // clear sticky
            }
        }
    }
    check(cudaSetDevice(s.ids[0]), "cudaSetDevice(reset)");
}

DeviceManager::~DeviceManager() {
    if (!impl_)
        return;
    auto& s = *impl_;
    for (int i = 0; i < s.n; ++i) {
        cudaSetDevice(s.ids[i]);
        if (s.main[i])
            cudaStreamDestroy(s.main[i]);
        if (s.defer[i])
            cudaStreamDestroy(s.defer[i]);
    }
}

DeviceManager::DeviceManager(DeviceManager&&) noexcept = default;
DeviceManager& DeviceManager::operator=(DeviceManager&&) noexcept = default;

int DeviceManager::count() const { return impl_->n; }
int DeviceManager::device_id(int i) const { return impl_->ids[i]; }
void* DeviceManager::stream(int i) const {
    return static_cast<void*>(impl_->main[i]);
}
void* DeviceManager::defer_stream(int i) const {
    return static_cast<void*>(impl_->defer[i]);
}
bool DeviceManager::peer_enabled(int a, int b) const {
    return impl_->peer[static_cast<size_t>(a) * impl_->n + b] != 0;
}
bool DeviceManager::all_peers_enabled() const {
    const auto& s = *impl_;
    for (int a = 0; a < s.n; ++a)
        for (int b = 0; b < s.n; ++b)
            if (!s.peer[static_cast<size_t>(a) * s.n + b])
                return false;
    return true;
}
void DeviceManager::set_device(int i) const {
    check(cudaSetDevice(impl_->ids[i]), "set_device");
}

} // namespace cuda
} // namespace ATV

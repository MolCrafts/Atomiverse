/// @file tests/cuda/test_device_manager.cu
/// @brief cuda::DeviceManager — enumeration, per-device streams, P2P enable.

#include <gtest/gtest.h>

#include "atomiverse/cuda/device/device_manager.cuh"

using namespace ATV;

TEST(DeviceManager, EnumeratesAndCreatesStreams) {
    cuda::DeviceManager dm; // all visible GPUs
    EXPECT_GE(dm.count(), 1);
    for (int i = 0; i < dm.count(); ++i) {
        EXPECT_EQ(dm.device_id(i), i);
        EXPECT_NE(dm.stream(i), nullptr);
        EXPECT_NE(dm.defer_stream(i), nullptr);
        EXPECT_NO_THROW(dm.set_device(i));
        EXPECT_TRUE(dm.peer_enabled(i, i)); // self is always "enabled"
    }
}

TEST(DeviceManager, RespectsRequestedCount) {
    cuda::DeviceManager dm(1);
    EXPECT_EQ(dm.count(), 1);
}

TEST(DeviceManager, PeerAccessAcrossGpus) {
    cuda::DeviceManager dm;
    if (dm.count() < 2)
        GTEST_SKIP() << "needs >= 2 GPUs for P2P";
    // On the NVLink-mesh GH200 node every ordered pair should peer-access.
    EXPECT_TRUE(dm.all_peers_enabled());
    // And a real peer copy must work between slot 0 and slot 1.
    cuda::DeviceManager& d = dm;
    const size_t N = 1u << 20;
    int *a0 = nullptr, *a1 = nullptr;
    d.set_device(0);
    ASSERT_EQ(cudaMalloc(&a0, N * sizeof(int)), cudaSuccess);
    d.set_device(1);
    ASSERT_EQ(cudaMalloc(&a1, N * sizeof(int)), cudaSuccess);
    std::vector<int> host(N), back(N);
    for (size_t k = 0; k < N; ++k)
        host[k] = static_cast<int>(k * 2654435761u);
    d.set_device(0);
    ASSERT_EQ(
        cudaMemcpy(a0, host.data(), N * sizeof(int), cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpyPeer(a1, 1, a0, 0, N * sizeof(int)), cudaSuccess);
    d.set_device(1);
    ASSERT_EQ(
        cudaMemcpy(back.data(), a1, N * sizeof(int), cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(host, back);
    cudaFree(a0);
    cudaFree(a1);
}

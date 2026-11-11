#include "atomiverse/cuda/scf/internal/nuclear_identity.cuh"

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <vector>

namespace {
using ATV::cuda::scf::internal::NuclearIdentity;

TEST(CudaNuclearIdentity, DetectsSameSizeElementChangesAndPermutations) {
    NuclearIdentity identity;
    const std::vector<int> water{8, 1, 1};
    thrust::device_vector<int> first(water.begin(), water.end());
    const auto* original = thrust::raw_pointer_cast(first.data());
    EXPECT_FALSE(identity.matches(original, 3, nullptr));
    identity.record(original, 3, nullptr);
    EXPECT_TRUE(identity.matches(original, 3, nullptr));

    // A different allocation with identical values is the same identity.
    thrust::device_vector<int> next(first);
    auto* current = thrust::raw_pointer_cast(next.data());
    EXPECT_TRUE(identity.matches(current, 3, nullptr));
    const int reordered[3] = {1, 8, 1};
    ASSERT_EQ(cudaMemcpy(current, reordered, sizeof(reordered),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    // Equal counts and equal sum(Z) must not hide a change in row identity.
    EXPECT_FALSE(identity.matches(current, 3, nullptr));
    identity.record(current, 3, nullptr);
    EXPECT_TRUE(identity.matches(current, 3, nullptr));
    EXPECT_FALSE(identity.matches(original, 3, nullptr));
    const int different[3] = {1, 7, 1};
    ASSERT_EQ(cudaMemcpy(current, different, sizeof(different),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    EXPECT_FALSE(identity.matches(current, 3, nullptr));
}

TEST(CudaNuclearIdentity, RecordsSizeChangesInTheBorrowedStream) {
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    struct StreamCleanup {
        cudaStream_t stream;
        ~StreamCleanup() { cudaStreamDestroy(stream); }
    } cleanup{stream};
    NuclearIdentity identity;
    thrust::device_vector<int> input(4);
    auto* types = thrust::raw_pointer_cast(input.data());
    const int atoms[4] = {6, 1, 1, 1};
    ASSERT_EQ(cudaMemcpyAsync(types, atoms, sizeof(atoms),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    identity.record(types, 4, stream);
    EXPECT_TRUE(identity.matches(types, 4, stream));
    EXPECT_FALSE(identity.matches(types, 2, stream));
    identity.record(types, 2, stream);
    EXPECT_TRUE(identity.matches(types, 2, stream));
    EXPECT_FALSE(identity.matches(types, 4, stream));
    identity.record(types, 4, stream);
    EXPECT_TRUE(identity.matches(types, 4, stream));
}
} // namespace

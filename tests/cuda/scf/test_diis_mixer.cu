/// @file test_diis_mixer.cu
/// @brief Analytic single-step tests of the device Pulay mixer.
#include <array>
#include <gtest/gtest.h>

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/scf/internal/diis_mixer.cuh"

namespace {
using ATV::cuda::prec_t;
using ATV::cuda::scf::internal::DIISMixer;

class CudaDiisMixer : public ::testing::Test {
  protected:
    ATV::cuda::Frame frame;
    ATV::cuda::scf::SCFWorkspace ws;
    cudaStream_t stream = nullptr;
    DIISMixer mixer{2, 0};

    void SetUp() override {
        ATV_CUDA_CHECK(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ws.n = 2;
        for (auto ptr : {&ws.S, &ws.X, &ws.P, &ws.work, &ws.H, &ws.F_prime})
            ATV_CUDA_CHECK(cudaMalloc(ptr, 4 * sizeof(prec_t)));
        const std::array<prec_t, 4> identity{1, 0, 0, 1};
        const std::array<prec_t, 4> density{2, 0, 0, 0};
        ATV_CUDA_CHECK(cudaMemcpyAsync(ws.S, identity.data(), sizeof(identity),
                                       cudaMemcpyHostToDevice, stream));
        ATV_CUDA_CHECK(cudaMemcpyAsync(ws.X, identity.data(), sizeof(identity),
                                       cudaMemcpyHostToDevice, stream));
        ATV_CUDA_CHECK(cudaMemcpyAsync(ws.P, density.data(), sizeof(density),
                                       cudaMemcpyHostToDevice, stream));
        // Setup's stack-backed input values must outlive their async copies.
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        mixer.build({stream, frame, 0}, ws);
    }

    void TearDown() override {
        cudaStreamSynchronize(stream);
        mixer.destroy({stream, frame, 0}, ws);
        for (auto ptr : {ws.S, ws.X, ws.P, ws.work, ws.H, ws.F_prime})
            cudaFree(ptr);
        cudaStreamDestroy(stream);
    }

    std::array<prec_t, 4> step(const std::array<prec_t, 4>& fock,
                               int iteration) {
        ATV_CUDA_CHECK(cudaMemcpyAsync(ws.work, fock.data(), sizeof(fock),
                                       cudaMemcpyHostToDevice, stream));
        mixer.step({stream, frame, iteration}, ws);
        std::array<prec_t, 4> result{};
        ATV_CUDA_CHECK(cudaMemcpyAsync(result.data(), ws.H, sizeof(result),
                                       cudaMemcpyDeviceToHost, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        return result;
    }

    void expect_matrix(const std::array<prec_t, 4>& got,
                       const std::array<prec_t, 4>& want) {
        for (int i = 0; i < 4; ++i)
            EXPECT_NEAR(got[i], want[i], 2e-12) << "element " << i;
    }
};

TEST_F(CudaDiisMixer, AnalyticExtrapolationAndCircularOverwrite) {
    // With P=diag(2,0), the commutator is proportional to F_01.
    // Errors e and 2e give Pulay weights (2,-1), cancelling the error.
    const std::array<prec_t, 4> first{1, 1, 1, 2};
    const std::array<prec_t, 4> second{3, 2, 2, 4};
    expect_matrix(step(first, 0), first);
    expect_matrix(step(second, 1), {-1, 0, 0, 0});
    // The ring now contains 3e and 2e: weights (-2,3) in physical order.
    expect_matrix(step({5, 3, 3, 7}, 2), {-1, 0, 0, -2});
}

TEST_F(CudaDiisMixer, SingularHistoryRetainsLatestFock) {
    step({1, 1, 1, 2}, 0);
    // Equal commutators yield dependent Pulay rows, even though F differs.
    const std::array<prec_t, 4> latest{3, 1, 1, 4};
    expect_matrix(step(latest, 1), latest);
}

TEST_F(CudaDiisMixer, ResetDiscardsHistory) {
    step({1, 1, 1, 2}, 0);
    step({3, 2, 2, 4}, 1);
    mixer.reset();
    const std::array<prec_t, 4> latest{5, 3, 3, 7};
    expect_matrix(step(latest, 0), latest);
}

TEST_F(CudaDiisMixer, RejectsChangedOrderBeforeAccessingWorkspace) {
    ws.n = 3;
    EXPECT_THROW(mixer.step({stream, frame, 0}, ws), std::invalid_argument);
    ws.n = 2;
}

TEST(CudaDiisMixerParameters, RejectsNonpositiveHistory) {
    EXPECT_THROW(DIISMixer(0), std::invalid_argument);
    EXPECT_THROW(DIISMixer(-1), std::invalid_argument);
}

TEST_F(CudaDiisMixer, ZeroCommutatorRetainsFiniteFock) {
    step({1, 0, 0, 2}, 0);
    const std::array<prec_t, 4> latest{3, 0, 0, 4};
    expect_matrix(step(latest, 1), latest);
}

TEST_F(CudaDiisMixer, CanonicalizePreservesPhysicalFockAndHistory) {
    step({1, 1, 1, 2}, 0);
    const std::array<prec_t, 4> physical{7, 4, 4, 9};
    ATV_CUDA_CHECK(cudaMemcpyAsync(ws.work, physical.data(), sizeof(physical),
                                   cudaMemcpyHostToDevice, stream));
    mixer.canonicalize({stream, frame, 1}, ws);
    std::array<prec_t, 4> transformed{}, retained{}, error{};
    ATV_CUDA_CHECK(cudaMemcpyAsync(transformed.data(), ws.H,
                                   sizeof(transformed), cudaMemcpyDeviceToHost,
                                   stream));
    ATV_CUDA_CHECK(cudaMemcpyAsync(retained.data(), ws.work, sizeof(retained),
                                   cudaMemcpyDeviceToHost, stream));
    ATV_CUDA_CHECK(cudaMemcpyAsync(error.data(), mixer.error(), sizeof(error),
                                   cudaMemcpyDeviceToHost, stream));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    expect_matrix(transformed, physical);
    expect_matrix(retained, physical);
    expect_matrix(error, {0, 8, -8, 0});
    // Canonicalization did not push its 4e error into the two-entry history:
    // the following ordinary step still combines the original e with 2e.
    expect_matrix(step({3, 2, 2, 4}, 2), {-1, 0, 0, 0});
}

TEST_F(CudaDiisMixer, CanonicalizeNeverAppliesVirtualLevelShift) {
    DIISMixer shifted{2, 0, 0.5};
    const std::array<prec_t, 4> identity{1, 0, 0, 1};
    const std::array<prec_t, 4> physical{1, 0, 0, 2};
    ws.n_occ = 1;
    ATV_CUDA_CHECK(cudaMalloc(&ws.C_prime, sizeof(identity)));
    ATV_CUDA_CHECK(cudaMemcpyAsync(ws.C_prime, identity.data(),
                                   sizeof(identity), cudaMemcpyHostToDevice,
                                   stream));
    ATV_CUDA_CHECK(cudaMemcpyAsync(ws.work, physical.data(), sizeof(physical),
                                   cudaMemcpyHostToDevice, stream));
    shifted.build({stream, frame, 0}, ws);
    shifted.canonicalize({stream, frame, 1}, ws);
    std::array<prec_t, 4> canonical{};
    ATV_CUDA_CHECK(cudaMemcpyAsync(canonical.data(), ws.H, sizeof(canonical),
                                   cudaMemcpyDeviceToHost, stream));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    expect_matrix(canonical, physical);
    shifted.destroy({stream, frame, 0}, ws);
    cudaFree(ws.C_prime);
    ws.C_prime = nullptr;
}
} // namespace

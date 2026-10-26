/// @file cuda/scf/test_convergence_check.cu
/// @brief Single-step physical convergence checks; no SCF iteration or solve.
#include <array>
#include <gtest/gtest.h>
#include <limits>

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/scf/internal/convergence_check.cuh"

namespace {

class CudaHfConvergence : public ::testing::Test {
  protected:
    static constexpr double tolerance = 1e-10;
    ATV::cuda::Frame frame;
    ATV::cuda::scf::SCFWorkspace ws;
    ATV::cuda::scf::internal::ConvergenceCheck check{tolerance, false};
    cudaStream_t stream = nullptr;
    double* alpha_error = nullptr;
    double* beta_error = nullptr;
    double* beta_density = nullptr;
    double* beta_old = nullptr;

    void SetUp() override {
        ATV_CUDA_CHECK(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ws.n = 2;
        for (auto ptr : {&ws.P, &ws.P_old, &alpha_error, &beta_error,
                         &beta_density, &beta_old}) {
            ATV_CUDA_CHECK(cudaMalloc(ptr, 4 * sizeof(double)));
            ATV_CUDA_CHECK(
                cudaMemsetAsync(*ptr, 0, 4 * sizeof(double), stream));
        }
        ATV_CUDA_CHECK(cudaMalloc(&ws.d_residual, sizeof(double)));
        ATV_CUDA_CHECK(cudaMalloc(&ws.d_converged, sizeof(int)));
        ATV_CUDA_CHECK(cudaMalloc(&frame.scalars, sizeof(ATV::SystemScalars)));
        set_energy(-75.0);
        check.build({stream, frame, 0}, ws);
    }

    void TearDown() override {
        cudaStreamSynchronize(stream);
        check.destroy({stream, frame, 0}, ws);
        for (auto ptr : {ws.P, ws.P_old, alpha_error, beta_error, beta_density,
                         beta_old, ws.d_residual})
            cudaFree(ptr);
        cudaFree(ws.d_converged);
        cudaFree(frame.scalars);
        cudaStreamDestroy(stream);
    }

    void set_energy(double hartree) {
        ATV::SystemScalars scalars{};
        scalars.energy = hartree * ATV::units::kHartreeInEv;
        ATV_CUDA_CHECK(cudaMemcpyAsync(frame.scalars, &scalars, sizeof(scalars),
                                       cudaMemcpyHostToDevice, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void set_entry(double* destination, double value) {
        std::array<double, 4> values{value, 0.0, 0.0, 0.0};
        ATV_CUDA_CHECK(cudaMemcpyAsync(destination, values.data(),
                                       sizeof(values), cudaMemcpyHostToDevice,
                                       stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    int step(int iteration, bool unrestricted = false) {
        check.step_hf({stream, frame, iteration}, ws, alpha_error,
                      unrestricted ? beta_error : nullptr,
                      unrestricted ? beta_density : nullptr,
                      unrestricted ? beta_old : nullptr);
        int flag = -1;
        ATV_CUDA_CHECK(cudaMemcpyAsync(&flag, ws.d_converged, sizeof(flag),
                                       cudaMemcpyDeviceToHost, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        return flag;
    }
};

TEST_F(CudaHfConvergence, StationaryDensityCannotHidePhysicalOrbitalError) {
    // Delta P is exactly zero, while the pre-extrapolation physical Fock
    // commutator is not. Density-only convergence used to accept this state.
    set_entry(alpha_error, 1e-2);
    EXPECT_EQ(step(0), 0);
    EXPECT_EQ(step(1), 0);
    set_entry(alpha_error, 0.0);
    EXPECT_EQ(step(2), 1);
}

TEST_F(CudaHfConvergence, EnergyChangeUsesHartreeAndMustAlsoConverge) {
    EXPECT_EQ(step(0), 0);
    set_energy(-75.0 + 1e-6);
    EXPECT_EQ(step(1), 0);
    EXPECT_EQ(step(2), 1);
}

TEST_F(CudaHfConvergence, RetainsTheDensityResidualRequirement) {
    EXPECT_EQ(step(0), 0);
    set_entry(ws.P, 1e-6);
    EXPECT_EQ(step(1), 0);
    set_entry(ws.P_old, 1e-6);
    EXPECT_EQ(step(2), 1);
}

TEST_F(CudaHfConvergence, CombinesBothSpinOrbitalErrors) {
    set_entry(alpha_error, 8e-6);
    set_entry(beta_error, 8e-6);
    EXPECT_EQ(step(0, true), 0);
    // Each norm is smaller than sqrt(tol), but their combined norm is not.
    EXPECT_EQ(step(1, true), 0);
    set_entry(alpha_error, 6e-6);
    set_entry(beta_error, 6e-6);
    EXPECT_EQ(step(2, true), 1);
}

TEST_F(CudaHfConvergence, ResetDiscardsAcceptedEnergyHistory) {
    EXPECT_EQ(step(0), 0);
    EXPECT_EQ(step(1), 1);
    check.reset(stream);
    EXPECT_EQ(step(2), 0);
    EXPECT_EQ(step(3), 1);
}

TEST_F(CudaHfConvergence, NonfinitePhysicalErrorCannotConverge) {
    EXPECT_EQ(step(0), 0);
    set_entry(alpha_error, std::numeric_limits<double>::quiet_NaN());
    EXPECT_EQ(step(1), 0);
}

} // namespace

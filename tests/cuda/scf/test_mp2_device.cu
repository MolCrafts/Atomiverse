/// @file cuda/scf/test_mp2_device.cu
/// @brief Single-transform MP2 primitive checks; no SCF or convergence loop.
#ifdef ATV_HAS_HF

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/scf/internal/mp2_device.cuh"

namespace {

struct Input {
    ATV::cuda::scf::SCFWorkspace ws;
    ATV::cuda::Frame frame;
    cudaStream_t stream = nullptr;
    std::vector<double> coefficients, eri, energies;

    Input(int n, int occupied)
        : coefficients(n * n), eri(n * n * n * n), energies(n) {
        ws.n = n;
        ws.n_occ = occupied;
        for (int p = 0; p < n; ++p) {
            energies[p] = p < occupied ? -1.0 + p * 0.2 : 0.4 + p * 0.3;
            for (int mu = 0; mu < n; ++mu)
                coefficients[mu + p * n] =
                    0.2 * std::sin(1.0 + mu + 2.0 * p) + (mu == p ? 0.8 : 0.0);
        }
        for (int s = 0; s < n; ++s)
            for (int l = 0; l < n; ++l)
                for (int v = 0; v < n; ++v)
                    for (int m = 0; m < n; ++m)
                        eri[m + n * (v + n * (l + n * s))] =
                            0.005
                            * (1 + m + 2 * v + 3 * l + 5 * s + 0.1 * m * s);
        ATV_CUDA_CHECK(cudaStreamCreate(&stream));
        ATV_CUDA_CHECK(cudaMalloc(&ws.C, coefficients.size() * sizeof(double)));
        ATV_CUDA_CHECK(cudaMalloc(&ws.ERI, eri.size() * sizeof(double)));
        ATV_CUDA_CHECK(cudaMalloc(&ws.eig, energies.size() * sizeof(double)));
        ATV_CUDA_CHECK(cudaMalloc(&frame.scalars, sizeof(ATV::SystemScalars)));
        ATV_CUDA_CHECK(cudaMemcpyAsync(ws.C, coefficients.data(),
                                       coefficients.size() * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
        ATV_CUDA_CHECK(cudaMemcpyAsync(ws.ERI, eri.data(),
                                       eri.size() * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
        upload_energies();
        set_energy(2.0);
    }

    ~Input() {
        cudaStreamSynchronize(stream);
        cudaFree(ws.C);
        cudaFree(ws.ERI);
        cudaFree(ws.eig);
        cudaFree(frame.scalars);
        cudaStreamDestroy(stream);
    }

    void upload_energies() {
        ATV_CUDA_CHECK(cudaMemcpyAsync(ws.eig, energies.data(),
                                       energies.size() * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
    }

    void set_energy(double energy_ev) {
        ATV::SystemScalars scalars{};
        scalars.energy = energy_ev;
        ATV_CUDA_CHECK(cudaMemcpyAsync(frame.scalars, &scalars, sizeof(scalars),
                                       cudaMemcpyHostToDevice, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    double integral(int i, int a, int j, int b) const {
        double value = 0.0;
        const int n = ws.n;
        for (int s = 0; s < n; ++s)
            for (int l = 0; l < n; ++l)
                for (int v = 0; v < n; ++v)
                    for (int m = 0; m < n; ++m)
                        value +=
                            eri[m + n * (v + n * (l + n * s))]
                            * coefficients[m + n * i] * coefficients[v + n * a]
                            * coefficients[l + n * j] * coefficients[s + n * b];
        return value;
    }
};

TEST(CudaMp2Device, TransformAndSpinChannelsMatchDirectFourIndexSum) {
    Input input(5, 3);
    constexpr int frozen = 1;
    constexpr double os_scale = 1.2, ss_scale = 1.0 / 3.0;
    ATV::cuda::scf::internal::MP2Device mp2;
    mp2.build(input.ws, frozen, os_scale, ss_scale, 1u << 20, input.frame, true,
              input.stream);
    constexpr double hf = -3.0;
    input.set_energy(hf * ATV::units::kHartreeInEv);
    mp2.evaluate(input.ws, input.frame, input.stream);
    double reference = 0.0;
    const auto got = mp2.result(input.stream, reference);
    double os = 0.0, ss = 0.0;
    for (int i = frozen; i < input.ws.n_occ; ++i)
        for (int j = frozen; j < input.ws.n_occ; ++j)
            for (int a = input.ws.n_occ; a < input.ws.n; ++a)
                for (int b = input.ws.n_occ; b < input.ws.n; ++b) {
                    const double direct = input.integral(i, a, j, b);
                    const double exchange = input.integral(i, b, j, a);
                    const double denom = input.energies[i] + input.energies[j]
                                         - input.energies[a]
                                         - input.energies[b];
                    os += direct * direct / denom;
                    ss += direct * (direct - exchange) / denom;
                }
    EXPECT_NEAR(got.e_corr_os, os, 1e-11);
    EXPECT_NEAR(got.e_corr_ss, ss, 1e-11);
    EXPECT_NEAR(got.e_corr, os_scale * os + ss_scale * ss, 1e-11);
    EXPECT_NEAR(got.e_tot, hf + got.e_corr, 1e-11);
    EXPECT_NEAR(reference, hf, 1e-14);
    EXPECT_EQ(got.frozen, frozen);
    EXPECT_FALSE(got.ri);
    ATV::SystemScalars scalars{};
    ATV_CUDA_CHECK(cudaMemcpyAsync(&scalars, input.frame.scalars,
                                   sizeof(scalars), cudaMemcpyDeviceToHost,
                                   input.stream));
    ATV_CUDA_CHECK(cudaStreamSynchronize(input.stream));
    EXPECT_NEAR(scalars.energy, 2.0 + got.e_tot * ATV::units::kHartreeInEv,
                1e-11);
}

TEST(CudaMp2Device, EmptyCorrelationSpacesReturnZero) {
    for (int occupied : {2, 3}) {
        Input input(3, occupied);
        ATV::cuda::scf::internal::MP2Device mp2;
        const int frozen = occupied == 2 ? occupied : 0;
        mp2.build(input.ws, frozen, 1.0, 1.0, 1u << 20, input.frame, false,
                  input.stream);
        input.set_energy(-2.0 * ATV::units::kHartreeInEv);
        mp2.evaluate(input.ws, input.frame, input.stream);
        double reference = 0;
        const auto got = mp2.result(input.stream, reference);
        EXPECT_EQ(got.e_corr, 0.0);
        EXPECT_EQ(got.e_corr_os, 0.0);
        EXPECT_EQ(got.e_corr_ss, 0.0);
        EXPECT_NEAR(got.e_tot, -2.0, 1e-14);
    }
}

TEST(CudaMp2Device, InvalidDenominatorRejectsResult) {
    Input input(4, 2);
    for (double& energy : input.energies)
        energy = 0.0;
    input.upload_energies();
    ATV::cuda::scf::internal::MP2Device mp2;
    mp2.build(input.ws, 0, 1.0, 1.0, 1u << 20, input.frame, false,
              input.stream);
    mp2.evaluate(input.ws, input.frame, input.stream);
    double reference = 0.0;
    EXPECT_THROW(mp2.result(input.stream, reference), std::invalid_argument);
}

TEST(CudaMp2Device, RejectsInvalidFrozenCountAndInsufficientBudget) {
    Input input(4, 2);
    ATV::cuda::scf::internal::MP2Device mp2;
    EXPECT_THROW(mp2.build(input.ws, 3, 1.0, 1.0, 1u << 20, input.frame, false,
                           input.stream),
                 std::invalid_argument);
    EXPECT_THROW(
        mp2.build(input.ws, 0, 1.0, 1.0, 1, input.frame, false, input.stream),
        std::length_error);
}

} // namespace
#endif // ATV_HAS_HF

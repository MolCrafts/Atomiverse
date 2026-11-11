/// @file tests/cuda/semiempirical/test_nddo_fock.cu
/// @brief Single-kernel NDDO checks; no SCF or full molecular solve.
#include "atomiverse/cuda/semiempirical/internal/nddo_fock.cuh"

#include "atomiverse/cpu/semiempirical/internal/nddo_system.h"
#include "atomiverse/cpu/semiempirical/nddo_fock.h"
#include "atomiverse/cpu/semiempirical/nddo_hcore.h"
#include "atomiverse/semiempirical/am1_parameters.h"
#include "atomiverse/semiempirical/rm1_parameters.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
namespace gpu = ATV::cuda::semiempirical::internal;
namespace cpu = ATV::cpu::semiempirical;

class CudaNDDOOneCenter : public ::testing::TestWithParam<bool> {
  protected:
    static constexpr int n = 13;
    static constexpr int nn = n * n;
    cpu::NDDOSystem system;
    std::vector<double> p, spin, h, packed_p, packed_spin, packed_h;
    cudaStream_t stream = nullptr;
    gpu::NDDOAtom* atoms = nullptr;
    double* data = nullptr;
    double *dp = nullptr, *ds = nullptr, *dh = nullptr, *df = nullptr,
           *dq = nullptr, *de = nullptr;
    void* scratch = nullptr;
    std::size_t scratch_bytes = 0;

    void SetUp() override {
        const std::array<int, 4> zs{1, 6, 7, 8};
        std::array<gpu::NDDOAtom, 4> descriptors{};
        int first = 0;
        for (std::size_t i = 0; i < zs.size(); ++i) {
            const auto* param = GetParam()
                                    ? ATV::semiempirical::rm1_parameter(zs[i])
                                    : ATV::semiempirical::am1_parameter(zs[i]);
            ASSERT_NE(param, nullptr);
            cpu::NDDOAtom atom;
            atom.atomic_number = zs[i];
            atom.first_ao = first;
            atom.n_orbitals = param->n_orbitals;
            atom.last_ao = first + param->n_orbitals - 1;
            atom.core_charge = param->core_charge;
            atom.params = param;
            system.atoms.push_back(atom);
            descriptors[i] = {first,      param->n_orbitals, param->core_charge,
                              param->uss, param->upp,        param->gss,
                              param->gsp, param->gpp,        param->gp2,
                              param->hsp};
            first += param->n_orbitals;
        }
        ASSERT_EQ(first, n);
        system.n_orbitals = n;
        p.resize(nn);
        spin.resize(nn);
        h.resize(nn);
        packed_p.resize(n * (n + 1) / 2);
        packed_spin.resize(packed_p.size());
        packed_h.resize(packed_p.size());
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= j; ++i) {
                const double pv = i == j ? 0.8 + 0.02 * i
                                         : 0.07 * std::sin(1.3 * i + 0.9 * j);
                const double sv = pv * (0.4 + 0.01 * (i + j));
                const double hv =
                    i == j ? -5.0 - i : 0.1 * std::cos(i + 2.0 * j);
                p[j * n + i] = p[i * n + j] = pv;
                spin[j * n + i] = spin[i * n + j] = sv;
                h[j * n + i] = h[i * n + j] = hv;
                const auto ij = cpu::packed_index(i, j);
                packed_p[ij] = pv;
                packed_spin[ij] = sv;
                packed_h[ij] = hv;
            }
        }
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                  cudaSuccess);
        ASSERT_EQ(cudaMalloc(&atoms, sizeof(descriptors)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&data, (4 * nn + 5) * sizeof(double)),
                  cudaSuccess);
        dp = data;
        ds = dp + nn;
        dh = ds + nn;
        df = dh + nn;
        dq = df + nn;
        de = dq + 4;
        ASSERT_EQ(cudaMemcpyAsync(atoms, descriptors.data(),
                                  sizeof(descriptors), cudaMemcpyHostToDevice,
                                  stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(dp, p.data(), nn * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(ds, spin.data(), nn * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(dh, h.data(), nn * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        // descriptors is stack storage; complete its one-time upload now.
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        scratch_bytes = gpu::electronic_energy_workspace_bytes(n);
        ASSERT_EQ(cudaMalloc(&scratch, scratch_bytes), cudaSuccess);
    }

    void TearDown() override {
        if (stream)
            EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(cudaFree(scratch), cudaSuccess);
        EXPECT_EQ(cudaFree(data), cudaSuccess);
        EXPECT_EQ(cudaFree(atoms), cudaSuccess);
        if (stream)
            EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
};

TEST_P(CudaNDDOOneCenter, HcoreMatchesCpu) {
    const auto expected = cpu::NDDOHCoreBuilder{}.build_one_center(system);
    gpu::one_center_hcore(4, n, atoms, df, stream);
    std::vector<double> actual(nn);
    ASSERT_EQ(cudaMemcpyAsync(actual.data(), df, nn * sizeof(double),
                              cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            EXPECT_DOUBLE_EQ(actual[j * n + i],
                             expected.packed_hcore[cpu::packed_index(i, j)]);
}

TEST_P(CudaNDDOOneCenter, FockMatchesCpuForIndependentSpinDensity) {
    const auto expected = cpu::NDDOFockBuilder{}.add_one_center(
        system, packed_h, packed_p, packed_spin);
    // Distinct output and in-place Hcore are both part of this primitive's
    // contract; the same allocated storage is reused for the second call.
    for (const bool inplace : {false, true}) {
        double* output = inplace ? dh : df;
        gpu::one_center_fock(4, n, atoms, dh, dp, ds, output, stream);
        std::vector<double> actual(nn);
        ASSERT_EQ(cudaMemcpyAsync(actual.data(), output, nn * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                EXPECT_NEAR(actual[j * n + i],
                            expected[cpu::packed_index(i, j)], 2e-13)
                    << "i=" << i << " j=" << j << " inplace=" << inplace;
    }
}

TEST_P(CudaNDDOOneCenter, ChargesMatchCpu) {
    const auto expected =
        cpu::atomic_charges_from_packed_density(system, packed_p);
    gpu::atomic_charges(4, n, atoms, dp, dq, stream);
    std::array<double, 4> actual{};
    ASSERT_EQ(cudaMemcpyAsync(actual.data(), dq, sizeof(actual),
                              cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[i], 2e-15);
}

TEST_P(CudaNDDOOneCenter, EnergyUsesDeviceReductionAndReusableScratch) {
    const auto expected_fock = cpu::NDDOFockBuilder{}.add_one_center(
        system, packed_h, packed_p, packed_spin);
    const double expected =
        cpu::electronic_energy_ev(n, packed_p, packed_h, expected_fock);
    gpu::one_center_fock(4, n, atoms, dh, dp, ds, df, stream);
    for (int repeat = 0; repeat < 2; ++repeat) {
        gpu::electronic_energy(n, dp, dh, df, de, scratch, scratch_bytes,
                               stream);
        double actual = 0.0;
        ASSERT_EQ(cudaMemcpyAsync(&actual, de, sizeof(actual),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_NEAR(actual, expected, 1e-11);
    }
}

TEST_P(CudaNDDOOneCenter, RestrictedSpinScalingAvoidsDensityCopy) {
    for (int i = 0; i < nn; ++i)
        spin[i] = 0.5 * p[i];
    ASSERT_EQ(cudaMemcpyAsync(ds, spin.data(), nn * sizeof(double),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    gpu::one_center_fock(4, n, atoms, dh, dp, ds, df, stream);
    std::vector<double> expected(nn), actual(nn);
    ASSERT_EQ(cudaMemcpyAsync(expected.data(), df, nn * sizeof(double),
                              cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    gpu::one_center_fock(4, n, atoms, dh, dp, dp, df, stream, 0.5);
    ASSERT_EQ(cudaMemcpyAsync(actual.data(), df, nn * sizeof(double),
                              cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (int i = 0; i < nn; ++i)
        EXPECT_NEAR(actual[i], expected[i], 1e-13);
}

INSTANTIATE_TEST_SUITE_P(AM1AndRM1, CudaNDDOOneCenter, ::testing::Bool());

TEST(CudaNDDODimensions, RejectsInvalidExtentBeforeDeviceAccess) {
    EXPECT_THROW(gpu::electronic_energy_workspace_bytes(0),
                 std::invalid_argument);
    EXPECT_THROW(gpu::electronic_energy_workspace_bytes(-1),
                 std::invalid_argument);
    EXPECT_THROW(
        gpu::electronic_energy_workspace_bytes(std::numeric_limits<int>::max()),
        std::invalid_argument);
    EXPECT_THROW(gpu::one_center_hcore(1, 1, nullptr, nullptr, nullptr),
                 std::invalid_argument);
}
} // namespace

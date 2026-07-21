/// @file tests/cuda/pw/test_xc_lda.cu
/// @brief Verifies cuda::pw::internal::xc_lda_grid against the CPU per-point XC
///        cpu::pw::PW_LDA uses: evaluate_xc(PBE, rho/2, rho/2, sigma=0), with
///        the rho<1e-12 cutoff.
#include "atomiverse/cpu/scf/xc_functional.h"
#include "atomiverse/cuda/pw/internal/xc_lda.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using ATV::cpu::scf::evaluate_xc;
using ATV::cpu::scf::XCType;
using ATV::cuda::pw::internal::xc_lda_grid;

namespace {
void* dmalloc(size_t bytes) {
    void* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess);
    return p;
}

// Host reference: exactly what cpu::pw::PW_LDA::launch computes per point.
void host_ref(double rho, double& vxc, double& exc) {
    if (rho < 1e-12) {
        vxc = 0.0;
        exc = 0.0;
        return;
    }
    auto out = evaluate_xc(XCType::PBE, rho / 2.0, rho / 2.0);
    vxc = out.vrho_alpha;
    exc = out.exc;
}
} // namespace

TEST(CudaPWXcLda, MatchesCpuEvaluateXc) {
    // Density sweep: sub-cutoff, tiny, and a realistic range up to ~10.
    std::vector<double> rho;
    rho.push_back(0.0);
    rho.push_back(1e-14); // below cutoff → zero
    rho.push_back(5e-13); // below cutoff
    rho.push_back(1e-12); // at cutoff (not < 1e-12 → evaluated)
    for (int i = 0; i < 200; ++i) {
        // log-spaced 1e-10 … 10
        double t = static_cast<double>(i) / 199.0;
        rho.push_back(std::pow(10.0, -10.0 + 11.0 * t));
    }
    const int nr = static_cast<int>(rho.size());

    auto* d_rho = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_vxc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_exc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    ASSERT_EQ(cudaMemcpy(d_rho, rho.data(), nr * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    xc_lda_grid(d_rho, d_vxc, d_exc, nr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> vxc(nr), exc(nr);
    ASSERT_EQ(cudaMemcpy(vxc.data(), d_vxc, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(exc.data(), d_exc, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    double max_vxc_err = 0, max_exc_err = 0;
    for (int i = 0; i < nr; ++i) {
        double ref_vxc, ref_exc;
        host_ref(rho[i], ref_vxc, ref_exc);
        // Relative-aware tolerance: values range ~1e-3 … few Hartree.
        double tv = 1e-12 * std::fmax(1.0, std::abs(ref_vxc));
        double te = 1e-12 * std::fmax(1.0, std::abs(ref_exc));
        max_vxc_err = std::max(max_vxc_err, std::abs(vxc[i] - ref_vxc) - tv);
        max_exc_err = std::max(max_exc_err, std::abs(exc[i] - ref_exc) - te);
        EXPECT_NEAR(vxc[i], ref_vxc, tv) << "vxc mismatch at rho=" << rho[i];
        EXPECT_NEAR(exc[i], ref_exc, te) << "exc mismatch at rho=" << rho[i];
    }
    // Aggregate margins (negative ⇒ all points within tolerance).
    EXPECT_LE(max_vxc_err, 0.0);
    EXPECT_LE(max_exc_err, 0.0);

    cudaFree(d_rho);
    cudaFree(d_vxc);
    cudaFree(d_exc);
}

// E_xc = dV·Σ ρ·exc matches a host Kahan-summed reference (relative tol covers
// the device atomic-reduction summation order).
TEST(CudaPWXcLda, EnergyReduction) {
    const int nr = 5000;
    std::vector<double> rho(nr);
    std::mt19937 rng(2024);
    std::uniform_real_distribution<double> dist(1e-3, 5.0);
    for (int i = 0; i < nr; ++i)
        rho[i] = dist(rng);
    const double dV = 0.0137; // arbitrary Ω/nr

    auto* d_rho = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_vxc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_exc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_E = static_cast<double*>(dmalloc(sizeof(double)));
    cudaMemcpy(d_rho, rho.data(), nr * sizeof(double), cudaMemcpyHostToDevice);

    xc_lda_grid(d_rho, d_vxc, d_exc, nr);
    ATV::cuda::pw::internal::xc_energy(d_rho, d_exc, nr, dV, d_E);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    double E_dev = 0;
    cudaMemcpy(&E_dev, d_E, sizeof(double), cudaMemcpyDeviceToHost);

    // Host reference: same exc per point (via CPU evaluate_xc), Kahan sum.
    double sum = 0, c = 0;
    for (int i = 0; i < nr; ++i) {
        double vxc, exc;
        host_ref(rho[i], vxc, exc);
        double y = rho[i] * exc - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    double E_ref = sum * dV;

    EXPECT_NEAR(E_dev, E_ref, 1e-10 * std::abs(E_ref));

    cudaFree(d_rho);
    cudaFree(d_vxc);
    cudaFree(d_exc);
    cudaFree(d_E);
}

// The sub-cutoff points must be exactly zero (cutoff branch).
TEST(CudaPWXcLda, SubCutoffIsZero) {
    std::vector<double> rho = {0.0, 1e-14, 5e-13, 9.99e-13};
    const int nr = static_cast<int>(rho.size());
    auto* d_rho = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_vxc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_exc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    cudaMemcpy(d_rho, rho.data(), nr * sizeof(double), cudaMemcpyHostToDevice);

    xc_lda_grid(d_rho, d_vxc, d_exc, nr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> vxc(nr), exc(nr);
    cudaMemcpy(vxc.data(), d_vxc, nr * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(exc.data(), d_exc, nr * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nr; ++i) {
        EXPECT_EQ(vxc[i], 0.0);
        EXPECT_EQ(exc[i], 0.0);
    }
    cudaFree(d_rho);
    cudaFree(d_vxc);
    cudaFree(d_exc);
}

/// @file tests/cuda/gpw/test_xc_lda.cu
/// @brief CUDA GPW Padé LDA XC foothold vs same-settings CPU oracle.
///
/// Unit only: per-point match to `cpu::xc::pade_lsd` (and the closed-shell
/// `xc_grid_pade` reduction shape). Not a full GPW SCF product.
#include "atomiverse/cpu/xc/pade.h"
#include "atomiverse/cuda/gpw/internal/xc_lda.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using ATV::cpu::xc::pade_lsd;
using ATV::cuda::gpw::internal::xc_energy;
using ATV::cuda::gpw::internal::xc_energy_lsd;
using ATV::cuda::gpw::internal::xc_lda_grid;
using ATV::cuda::gpw::internal::xc_lda_grid_lsd;

namespace {

void* dmalloc(size_t bytes) {
    void* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess);
    return p;
}

// Host reference for closed-shell: exactly cpu::gpw::xc_grid_pade per point.
void host_ref_closed(double rho, double& vxc, double& exc) {
    auto out = pade_lsd(0.5 * rho, 0.5 * rho);
    vxc = out.vrho_a;
    exc = out.exc;
}

} // namespace

// ── Unpolarized grid vs pade_lsd(ρ/2, ρ/2) ───────────────────────────────
TEST(CudaGpwXcLda, MatchesCpuPadeClosedShell) {
    std::vector<double> rho;
    rho.push_back(0.0);
    rho.push_back(1e-14); // below cutoff → zero
    rho.push_back(5e-13); // below cutoff
    rho.push_back(
        1e-12); // at cutoff (rhoab < 1e-12 ⇒ zero; 0.5+0.5 = 1e-12 not <)
    for (int i = 0; i < 200; ++i) {
        const double t = static_cast<double>(i) / 199.0;
        rho.push_back(std::pow(10.0, -10.0 + 11.0 * t)); // 1e-10 … 10
    }
    // High-density GPW-ish points.
    rho.push_back(50.0);
    rho.push_back(100.0);
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
        host_ref_closed(rho[i], ref_vxc, ref_exc);
        // Relative-aware tol: device pow/cbrt vs host — ~1e-12 class.
        const double tv = 1e-12 * std::fmax(1.0, std::abs(ref_vxc));
        const double te = 1e-12 * std::fmax(1.0, std::abs(ref_exc));
        max_vxc_err = std::max(max_vxc_err, std::abs(vxc[i] - ref_vxc) - tv);
        max_exc_err = std::max(max_exc_err, std::abs(exc[i] - ref_exc) - te);
        EXPECT_NEAR(vxc[i], ref_vxc, tv) << "vxc mismatch at rho=" << rho[i];
        EXPECT_NEAR(exc[i], ref_exc, te) << "exc mismatch at rho=" << rho[i];
    }
    EXPECT_LE(max_vxc_err, 0.0);
    EXPECT_LE(max_exc_err, 0.0);

    cudaFree(d_rho);
    cudaFree(d_vxc);
    cudaFree(d_exc);
}

// ── Spin-resolved vs pade_lsd(ρ_a, ρ_b) ───────────────────────────────────
TEST(CudaGpwXcLda, MatchesCpuPadeLsd) {
    struct Pt {
        double ra, rb;
    };
    std::vector<Pt> pts = {
        {0.0, 0.0},   {1e-14, 1e-14}, {0.5, 0.5},  {0.8, 0.2}, {1.0, 0.05},
        {1e-3, 5e-4}, {50.0, 30.0},   {10.0, 0.0}, // fully polarized edge
    };
    // Random polarized samples in a realistic GPW density range.
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(1e-4, 20.0);
    std::uniform_real_distribution<double> spin(0.0, 1.0);
    for (int i = 0; i < 300; ++i) {
        const double tot = dist(rng);
        const double f = spin(rng);
        pts.push_back({f * tot, (1.0 - f) * tot});
    }
    const int nr = static_cast<int>(pts.size());
    std::vector<double> ra(nr), rb(nr);
    for (int i = 0; i < nr; ++i) {
        ra[i] = pts[i].ra;
        rb[i] = pts[i].rb;
    }

    auto* d_ra = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_rb = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_va = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_vb = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_exc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    cudaMemcpy(d_ra, ra.data(), nr * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rb, rb.data(), nr * sizeof(double), cudaMemcpyHostToDevice);

    xc_lda_grid_lsd(d_ra, d_rb, d_va, d_vb, d_exc, nr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> va(nr), vb(nr), exc(nr);
    cudaMemcpy(va.data(), d_va, nr * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vb.data(), d_vb, nr * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(exc.data(), d_exc, nr * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < nr; ++i) {
        const auto ref = pade_lsd(ra[i], rb[i]);
        const double t = 1e-12 * std::fmax(1.0, std::abs(ref.exc));
        const double ta = 1e-12 * std::fmax(1.0, std::abs(ref.vrho_a));
        const double tb = 1e-12 * std::fmax(1.0, std::abs(ref.vrho_b));
        EXPECT_NEAR(exc[i], ref.exc, t)
            << "exc at ra=" << ra[i] << " rb=" << rb[i];
        EXPECT_NEAR(va[i], ref.vrho_a, ta)
            << "vrho_a at ra=" << ra[i] << " rb=" << rb[i];
        EXPECT_NEAR(vb[i], ref.vrho_b, tb)
            << "vrho_b at ra=" << ra[i] << " rb=" << rb[i];
    }

    cudaFree(d_ra);
    cudaFree(d_rb);
    cudaFree(d_va);
    cudaFree(d_vb);
    cudaFree(d_exc);
}

// ── Energy reduction vs host Kahan sum (xc_grid_pade shape) ───────────────
TEST(CudaGpwXcLda, EnergyReductionClosedShell) {
    const int nr = 5000;
    std::vector<double> rho(nr);
    std::mt19937 rng(2024);
    std::uniform_real_distribution<double> dist(1e-3, 5.0);
    for (int i = 0; i < nr; ++i)
        rho[i] = dist(rng);
    const double w = 0.0137; // arbitrary V_cell/npts

    auto* d_rho = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_vxc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_exc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_E = static_cast<double*>(dmalloc(sizeof(double)));
    cudaMemcpy(d_rho, rho.data(), nr * sizeof(double), cudaMemcpyHostToDevice);

    xc_lda_grid(d_rho, d_vxc, d_exc, nr);
    xc_energy(d_rho, d_exc, nr, w, d_E);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    double E_dev = 0;
    cudaMemcpy(&E_dev, d_E, sizeof(double), cudaMemcpyDeviceToHost);

    double sum = 0, c = 0;
    for (int i = 0; i < nr; ++i) {
        double vxc, exc;
        host_ref_closed(rho[i], vxc, exc);
        const double y = rho[i] * exc - c;
        const double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    const double E_ref = sum * w;

    // Atomic reduction order: relative 1e-10 class (same as PW xc_lda).
    EXPECT_NEAR(E_dev, E_ref, 1e-10 * std::abs(E_ref));

    cudaFree(d_rho);
    cudaFree(d_vxc);
    cudaFree(d_exc);
    cudaFree(d_E);
}

TEST(CudaGpwXcLda, EnergyReductionLsd) {
    const int nr = 3000;
    std::vector<double> ra(nr), rb(nr);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(1e-3, 3.0);
    for (int i = 0; i < nr; ++i) {
        ra[i] = dist(rng);
        rb[i] = dist(rng);
    }
    const double w = 0.02;

    auto* d_ra = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_rb = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_va = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_vb = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_exc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_E = static_cast<double*>(dmalloc(sizeof(double)));
    cudaMemcpy(d_ra, ra.data(), nr * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rb, rb.data(), nr * sizeof(double), cudaMemcpyHostToDevice);

    xc_lda_grid_lsd(d_ra, d_rb, d_va, d_vb, d_exc, nr);
    xc_energy_lsd(d_ra, d_rb, d_exc, nr, w, d_E);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    double E_dev = 0;
    cudaMemcpy(&E_dev, d_E, sizeof(double), cudaMemcpyDeviceToHost);

    double sum = 0, c = 0;
    for (int i = 0; i < nr; ++i) {
        const auto ref = pade_lsd(ra[i], rb[i]);
        const double y = (ra[i] + rb[i]) * ref.exc - c;
        const double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    const double E_ref = sum * w;
    EXPECT_NEAR(E_dev, E_ref, 1e-10 * std::abs(E_ref));

    cudaFree(d_ra);
    cudaFree(d_rb);
    cudaFree(d_va);
    cudaFree(d_vb);
    cudaFree(d_exc);
    cudaFree(d_E);
}

// Sub-cutoff points must be exactly zero (cutoff branch).
TEST(CudaGpwXcLda, SubCutoffIsZero) {
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

// Closed-shell unpolarized: vrho_a == vrho_b on device (via lsd path).
TEST(CudaGpwXcLda, UnpolarizedSymmetryLsdPath) {
    std::vector<double> ra = {0.05, 0.5, 5.0};
    std::vector<double> rb = ra;
    const int nr = static_cast<int>(ra.size());
    auto* d_ra = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_rb = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_va = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_vb = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_exc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    cudaMemcpy(d_ra, ra.data(), nr * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rb, rb.data(), nr * sizeof(double), cudaMemcpyHostToDevice);

    xc_lda_grid_lsd(d_ra, d_rb, d_va, d_vb, d_exc, nr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> va(nr), vb(nr);
    cudaMemcpy(va.data(), d_va, nr * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vb.data(), d_vb, nr * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nr; ++i)
        EXPECT_DOUBLE_EQ(va[i], vb[i]);

    cudaFree(d_ra);
    cudaFree(d_rb);
    cudaFree(d_va);
    cudaFree(d_vb);
    cudaFree(d_exc);
}

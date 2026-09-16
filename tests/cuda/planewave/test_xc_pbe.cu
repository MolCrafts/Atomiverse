/// @file tests/cuda/planewave/test_xc_pbe.cu
/// @brief Verifies cuda::pw::internal::xc_pbe_grid against the CPU per-point
///        XC that `cpu::pw::internal::XCBuilder::step(ws, XCType::PBE)`
///        evaluates: evaluate_xc(PBE, rho/2, rho/2, sigma/4, sigma/4,
///        sigma/4) with sigma = |grad rho|^2, plus the flux assembly
///        flux_d = (vsigma_aa + 0.5 vsigma_ab) * grad_d — with the
///        rho < 1e-12 cutoff. Tolerances mirror the LDA parity test:
///        1e-12 * max(1, |ref|) absorbing the libdevice-vs-libm
///        transcendental spread (PBE chains ~10 pow/exp/log per point).
#include "atomiverse/cpu/scf/xc_functional.h"
#include "atomiverse/cuda/planewave/internal/xc_pbe.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using ATV::cpu::scf::evaluate_xc;
using ATV::cpu::scf::XCType;
using ATV::cuda::pw::internal::xc_pbe_grid;

namespace {

void* dmalloc(size_t bytes) {
    void* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess);
    return p;
}

/// Host reference: exactly what XCBuilder::step(ws, PBE)'s grid loop
/// computes per point from rho and the spectral gradient.
void host_ref(double rho, double gx, double gy, double gz, double& vxc,
              double& exc, double& fx, double& fy, double& fz) {
    if (rho < 1e-12) {
        vxc = exc = fx = fy = fz = 0.0;
        return;
    }
    const double sigma = gx * gx + gy * gy + gz * gz;
    const double s4 = 0.25 * sigma;
    const auto out = evaluate_xc(XCType::PBE, rho / 2.0, rho / 2.0, s4, s4, s4);
    vxc = out.vrho_alpha;
    exc = out.exc;
    const double vsig_eff = out.vsigma_aa + 0.5 * out.vsigma_ab;
    fx = vsig_eff * gx;
    fy = vsig_eff * gy;
    fz = vsig_eff * gz;
}

/// Density × gradient sweep shared by the tests: sub-cutoff points, the
/// sigma = 0 (LDA-limit) column, and gradients from weak to dominating.
void make_sweep(std::vector<double>& rho, std::vector<double>& gx,
                std::vector<double>& gy, std::vector<double>& gz) {
    const double rho_vals[] = {0.0,   1e-14, 5e-13, 1e-12, 1e-8, 1e-4,
                               0.005, 0.05,  0.5,   2.0,   10.0};
    // |grad rho| magnitudes; components split anisotropically so all three
    // flux lanes are exercised with distinct values.
    const double g_vals[] = {0.0, 1e-6, 0.01, 0.3, 1.0, 8.0};
    for (const double r : rho_vals)
        for (const double g : g_vals) {
            rho.push_back(r);
            gx.push_back(0.6 * g);
            gy.push_back(-0.8 * g);
            gz.push_back(0.3 * g);
        }
}

} // namespace

TEST(CudaPWXcPbe, MatchesCpuEvaluateXc) {
    std::vector<double> rho, gx, gy, gz;
    make_sweep(rho, gx, gy, gz);
    const int nr = static_cast<int>(rho.size());

    auto* d_rho = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_gx = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_gy = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_gz = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_vxc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_exc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    ASSERT_EQ(cudaMemcpy(d_rho, rho.data(), nr * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_gx, gx.data(), nr * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_gy, gy.data(), nr * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_gz, gz.data(), nr * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    xc_pbe_grid(d_rho, d_gx, d_gy, d_gz, d_vxc, d_exc, nr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> vxc(nr), exc(nr), fx(nr), fy(nr), fz(nr);
    ASSERT_EQ(cudaMemcpy(vxc.data(), d_vxc, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(exc.data(), d_exc, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    // The kernel overwrites the gradient arrays with the flux.
    ASSERT_EQ(cudaMemcpy(fx.data(), d_gx, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(fy.data(), d_gy, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(fz.data(), d_gz, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (int i = 0; i < nr; ++i) {
        double rv, re, rfx, rfy, rfz;
        host_ref(rho[i], gx[i], gy[i], gz[i], rv, re, rfx, rfy, rfz);
        // Same relative-aware bound as the LDA kernel test (1e-12·max(1,|·|)):
        // the residual is the libdevice/libm spread over the chained
        // pow/exp/log, ≲ 1 ulp each.
        const auto tol = [](double ref) {
            return 1e-12 * std::fmax(1.0, std::abs(ref));
        };
        EXPECT_NEAR(vxc[i], rv, tol(rv))
            << "vxc mismatch at rho=" << rho[i] << " |g|x=" << gx[i];
        EXPECT_NEAR(exc[i], re, tol(re))
            << "exc mismatch at rho=" << rho[i] << " |g|x=" << gx[i];
        EXPECT_NEAR(fx[i], rfx, tol(rfx))
            << "flux_x mismatch at rho=" << rho[i] << " |g|x=" << gx[i];
        EXPECT_NEAR(fy[i], rfy, tol(rfy))
            << "flux_y mismatch at rho=" << rho[i] << " |g|x=" << gx[i];
        EXPECT_NEAR(fz[i], rfz, tol(rfz))
            << "flux_z mismatch at rho=" << rho[i] << " |g|x=" << gx[i];
    }

    cudaFree(d_rho);
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_gz);
    cudaFree(d_vxc);
    cudaFree(d_exc);
}

// At sigma = 0 the device kernel must be the PW92-based PBE limit — NOT the
// VWN5 LSDA that xc_lsda_grid implements. The two correlation bases disagree
// by ~1e-4 Ha at valence densities, so a kernel that silently reused the
// LSDA path would sail past a loose tolerance; this pins the base.
TEST(CudaPWXcPbe, SigmaZeroIsPw92BaseNotVwn5) {
    const std::vector<double> rho = {0.005, 0.05, 0.5, 2.0};
    const int nr = static_cast<int>(rho.size());
    std::vector<double> zeros(nr, 0.0);

    auto* d_rho = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_gx = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_gy = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_gz = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_vxc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_exc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    cudaMemcpy(d_rho, rho.data(), nr * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gx, zeros.data(), nr * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_gy, zeros.data(), nr * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_gz, zeros.data(), nr * sizeof(double),
               cudaMemcpyHostToDevice);

    xc_pbe_grid(d_rho, d_gx, d_gy, d_gz, d_vxc, d_exc, nr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> exc(nr);
    cudaMemcpy(exc.data(), d_exc, nr * sizeof(double), cudaMemcpyDeviceToHost);

    double max_pbe_gap = 0, max_vwn5_gap = 1e300;
    for (int i = 0; i < nr; ++i) {
        const double exc_pbe0 =
            evaluate_xc(XCType::PBE, rho[i] / 2.0, rho[i] / 2.0).exc;
        const double exc_lsda =
            evaluate_xc(XCType::LSDA, rho[i] / 2.0, rho[i] / 2.0).exc;
        max_pbe_gap = std::max(max_pbe_gap, std::abs(exc[i] - exc_pbe0));
        max_vwn5_gap = std::min(max_vwn5_gap, std::abs(exc[i] - exc_lsda));
    }
    EXPECT_LE(max_pbe_gap, 1e-12);
    EXPECT_GT(max_vwn5_gap, 1e-6) << "device eps_xc at sigma=0 is "
                                     "indistinguishable from VWN5 LSDA";

    cudaFree(d_rho);
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_gz);
    cudaFree(d_vxc);
    cudaFree(d_exc);
}

// The sub-cutoff points must be exactly zero (cutoff branch), including the
// flux the kernel writes over the gradient arrays.
TEST(CudaPWXcPbe, SubCutoffIsZero) {
    std::vector<double> rho = {0.0, 1e-14, 5e-13, 9.99e-13};
    const int nr = static_cast<int>(rho.size());
    std::vector<double> g(nr, 0.7); // nonzero gradient: must still zero out

    auto* d_rho = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_gx = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_gy = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_gz = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_vxc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_exc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    cudaMemcpy(d_rho, rho.data(), nr * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gx, g.data(), nr * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gy, g.data(), nr * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gz, g.data(), nr * sizeof(double), cudaMemcpyHostToDevice);

    xc_pbe_grid(d_rho, d_gx, d_gy, d_gz, d_vxc, d_exc, nr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> vxc(nr), exc(nr), fx(nr);
    cudaMemcpy(vxc.data(), d_vxc, nr * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(exc.data(), d_exc, nr * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(fx.data(), d_gx, nr * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nr; ++i) {
        EXPECT_EQ(vxc[i], 0.0);
        EXPECT_EQ(exc[i], 0.0);
        EXPECT_EQ(fx[i], 0.0);
    }
    cudaFree(d_rho);
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_gz);
    cudaFree(d_vxc);
    cudaFree(d_exc);
}

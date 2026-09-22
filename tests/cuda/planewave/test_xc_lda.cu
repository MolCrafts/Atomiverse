/// @file tests/cuda/planewave/test_xc_lda.cu
/// @brief Verifies cuda::pw::internal::xc_lsda_grid against the CPU per-point
///        XC that `cpu::pw::internal::XCBuilder::step(ws, XCType::LSDA)`
///        evaluates: evaluate_xc(LSDA, rho/2, rho/2) — Slater exchange +
///        VWN5 correlation — with the rho < 1e-12 cutoff.
#include "atomiverse/cpu/scf/xc_functional.h"
#include "atomiverse/cuda/planewave/internal/xc_lda.cuh"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using ATV::cpu::scf::evaluate_xc;
using ATV::cpu::scf::XCType;
using ATV::cuda::pw::internal::kXcEnergyPartials;
using ATV::cuda::pw::internal::xc_lsda_grid;

namespace {
void* dmalloc(size_t bytes) {
    void* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess);
    return p;
}

// Host reference: exactly what XCBuilder::step(ws, LSDA) computes per point.
void host_ref(double rho, double& vxc, double& exc) {
    if (rho < 1e-12) {
        vxc = 0.0;
        exc = 0.0;
        return;
    }
    auto out = evaluate_xc(XCType::LSDA, rho / 2.0, rho / 2.0);
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

    xc_lsda_grid(d_rho, d_vxc, d_exc, nr);
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
        // Relative-aware tolerance: values range ~1e-3 … few Hartree. The
        // residual spread is libdevice vs libm log/atan2/pow, ≲ 1 ulp each.
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

// The device kernel must be VWN5, not the PW92 that PBE uses as its LSDA
// base. The two disagree by ~1e-4 Ha at valence densities, so a regression to
// PW92 would sail past a 1e-12 test written against evaluate_xc(PBE, ...).
TEST(CudaPWXcLda, IsVwn5NotPw92) {
    const std::vector<double> rho = {0.005, 0.05, 0.5, 2.0};
    const int nr = static_cast<int>(rho.size());
    auto* d_rho = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_vxc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_exc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    cudaMemcpy(d_rho, rho.data(), nr * sizeof(double), cudaMemcpyHostToDevice);

    xc_lsda_grid(d_rho, d_vxc, d_exc, nr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> exc(nr);
    cudaMemcpy(exc.data(), d_exc, nr * sizeof(double), cudaMemcpyDeviceToHost);

    double max_gap = 0;
    for (int i = 0; i < nr; ++i) {
        // PBE at sigma = 0 is Slater + PW92 — the functional this kernel used
        // to implement.
        const double exc_pw92 =
            evaluate_xc(XCType::PBE, rho[i] / 2.0, rho[i] / 2.0).exc;
        max_gap = std::max(max_gap, std::abs(exc[i] - exc_pw92));
    }
    EXPECT_GT(max_gap, 1e-6) << "device eps_xc is indistinguishable from the "
                                "PW92 (PBE sigma=0) branch";
}

// E_xc = dV·Σ ρ·exc matches a host Kahan-summed reference. The device
// reduction is a fixed-geometry tree, so the only spread is tree-vs-serial
// summation of identical per-point values.
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
    auto* d_scratch =
        static_cast<double*>(dmalloc(kXcEnergyPartials * sizeof(double)));
    cudaMemcpy(d_rho, rho.data(), nr * sizeof(double), cudaMemcpyHostToDevice);

    xc_lsda_grid(d_rho, d_vxc, d_exc, nr);
    ATV::cuda::pw::internal::xc_energy(d_rho, d_exc, nr, dV, d_E, d_scratch);
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

    // Tightened from 1e-10 now that the reduction is a deterministic tree
    // rather than an atomicAdd race.
    EXPECT_NEAR(E_dev, E_ref, 1e-13 * std::abs(E_ref));

    // Bitwise reproducibility: same input, same fixed grid, same bits.
    double E_again = 0;
    ATV::cuda::pw::internal::xc_energy(d_rho, d_exc, nr, dV, d_E, d_scratch);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaMemcpy(&E_again, d_E, sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_EQ(E_dev, E_again);

    cudaFree(d_rho);
    cudaFree(d_vxc);
    cudaFree(d_exc);
    cudaFree(d_E);
    cudaFree(d_scratch);
}

// Collinear LSDA: device vs evaluate_xc(LSDA, ρ_α, ρ_β) over a (ρ, ζ) grid.
TEST(CudaPWXcLda, SpinMatchesCpuEvaluateXc) {
    std::vector<double> rho_a, rho_b;
    // Sub-cutoff total
    rho_a.push_back(0.0);
    rho_b.push_back(0.0);
    rho_a.push_back(1e-14);
    rho_b.push_back(1e-14);
    // Polarized and unpolarized valence densities
    const double rhos[] = {1e-4, 1e-3, 0.01, 0.05, 0.2, 0.5, 1.0, 2.0};
    const double zetas[] = {0.0, 0.2, 0.5, 0.8, 0.99, -0.3, -0.7};
    for (double r : rhos) {
        for (double z : zetas) {
            // ζ = (ρa − ρb)/ρ  ⇒  ρa = ρ(1+ζ)/2, ρb = ρ(1−ζ)/2
            rho_a.push_back(r * (1.0 + z) * 0.5);
            rho_b.push_back(r * (1.0 - z) * 0.5);
        }
    }
    // One fully polarized limit (β floored by correlation)
    rho_a.push_back(0.4);
    rho_b.push_back(0.0);

    const int nr = static_cast<int>(rho_a.size());
    auto* d_ra = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_rb = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_va = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_vb = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_exc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    ASSERT_EQ(cudaMemcpy(d_ra, rho_a.data(), nr * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_rb, rho_b.data(), nr * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    ATV::cuda::pw::internal::xc_lsda_spin_grid(d_ra, d_rb, d_va, d_vb, d_exc,
                                               nr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> va(nr), vb(nr), exc(nr);
    cudaMemcpy(va.data(), d_va, nr * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vb.data(), d_vb, nr * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(exc.data(), d_exc, nr * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < nr; ++i) {
        double ra = std::max(0.0, rho_a[i]);
        double rb = std::max(0.0, rho_b[i]);
        double ref_va = 0, ref_vb = 0, ref_exc = 0;
        if (ra + rb >= 1e-12) {
            auto out = evaluate_xc(XCType::LSDA, ra, rb);
            ref_va = out.vrho_alpha;
            ref_vb = out.vrho_beta;
            ref_exc = out.exc;
        }
        const double t = 1e-12 * std::fmax(1.0, std::abs(ref_exc));
        const double ta = 1e-12 * std::fmax(1.0, std::abs(ref_va));
        const double tb = 1e-12 * std::fmax(1.0, std::abs(ref_vb));
        EXPECT_NEAR(exc[i], ref_exc, t)
            << "exc at ra=" << rho_a[i] << " rb=" << rho_b[i];
        EXPECT_NEAR(va[i], ref_va, ta)
            << "vxc_a at ra=" << rho_a[i] << " rb=" << rho_b[i];
        EXPECT_NEAR(vb[i], ref_vb, tb)
            << "vxc_b at ra=" << rho_a[i] << " rb=" << rho_b[i];
    }

    // ζ = 0 must reproduce the restricted kernel (v_a == v_b).
    for (int i = 0; i < nr; ++i) {
        if (std::abs(rho_a[i] - rho_b[i]) < 1e-18
            && rho_a[i] + rho_b[i] > 1e-12)
            EXPECT_NEAR(va[i], vb[i], 1e-14);
    }

    cudaFree(d_ra);
    cudaFree(d_rb);
    cudaFree(d_va);
    cudaFree(d_vb);
    cudaFree(d_exc);
}

// The sub-cutoff points must be exactly zero (cutoff branch).
TEST(CudaPWXcLda, SubCutoffIsZero) {
    std::vector<double> rho = {0.0, 1e-14, 5e-13, 9.99e-13};
    const int nr = static_cast<int>(rho.size());
    auto* d_rho = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_vxc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_exc = static_cast<double*>(dmalloc(nr * sizeof(double)));
    cudaMemcpy(d_rho, rho.data(), nr * sizeof(double), cudaMemcpyHostToDevice);

    xc_lsda_grid(d_rho, d_vxc, d_exc, nr);
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

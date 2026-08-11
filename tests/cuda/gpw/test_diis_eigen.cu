/// @file tests/cuda/gpw/test_diis_eigen.cu
/// @brief Unit: device AO DIIS residual + eigen/density vs host oracles.
///
/// Single-function checks for `internal::AoDiisEigenDevice` (no full SCF).
/// Host oracles: `gpw_diis_error`, `diagonalize_fock`, `build_density_matrix`.

#include "atomiverse/cpu/gpw/internal/diis.h"
#include "atomiverse/cpu/linalg/density.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/linalg/matrix.h"
#include "atomiverse/cuda/gpw/internal/ao_diis_eigen.cuh"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using ATV::cpu::linalg::CanonicalOrth;
using ATV::cpu::linalg::RealMatrix;
using ATV::cpu::linalg::build_density_matrix;
using ATV::cpu::linalg::diagonalize_fock;
using ATV::cuda::gpw::internal::AoDiisEigenDevice;

namespace {

/// Small well-conditioned SPD S and symmetric H for n=3.
void make_toy(RealMatrix& S, RealMatrix& H, RealMatrix& P, int& n_occ) {
    const int n = 3;
    n_occ = 1;
    S = xt::zeros<double>({n, n});
    H = xt::zeros<double>({n, n});
    // S = I + small off-diag (SPD)
    S(0, 0) = 1.0;
    S(1, 1) = 1.0;
    S(2, 2) = 1.0;
    S(0, 1) = S(1, 0) = 0.15;
    S(0, 2) = S(2, 0) = 0.05;
    S(1, 2) = S(2, 1) = 0.10;
    // Symmetric H
    H(0, 0) = -0.9;
    H(1, 1) = -0.4;
    H(2, 2) = 0.2;
    H(0, 1) = H(1, 0) = -0.12;
    H(0, 2) = H(2, 0) = 0.03;
    H(1, 2) = H(2, 1) = -0.07;
    CanonicalOrth orth(S, 1e-10);
    auto [eig, C] = diagonalize_fock(H, orth.X);
    (void)eig;
    P = build_density_matrix(C, n_occ, n, 2.0);
}

double max_abs_diff(const double* a, const double* b, int n) {
    double m = 0.0;
    for (int i = 0; i < n; ++i)
        m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

} // namespace

TEST(CudaGpwDiisEigen, ErrorMatchesHost) {
    RealMatrix S, H, P;
    int n_occ = 0;
    make_toy(S, H, P, n_occ);
    const int n = 3;
    CanonicalOrth orth(S, 1e-10);
    const int m = static_cast<int>(orth.X.shape(1));
    ASSERT_EQ(m, n);

    const RealMatrix e_host =
        ATV::cpu::gpw::internal::gpw_diis_error(H, P, S, orth.X);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    AoDiisEigenDevice dev;
    dev.build(stream, n, m, /*max_vectors=*/4, n_occ, /*occ=*/2.0,
              /*level_shift=*/0.0);
    dev.upload_SX(S.data(), orth.X.data());
    dev.upload_P(P.data());

    std::vector<double> e_dev(static_cast<size_t>(m * m));
    dev.error_to_host(H.data(), e_dev.data());

    const double d =
        max_abs_diff(e_host.data(), e_dev.data(), m * m);
    EXPECT_LE(d, 1e-10) << "max|e_dev-e_host|=" << d;

    cudaStreamDestroy(stream);
}

TEST(CudaGpwDiisEigen, EigenAndDensityMatchHost) {
    RealMatrix S, H, P0;
    int n_occ = 0;
    make_toy(S, H, P0, n_occ);
    const int n = 3;
    CanonicalOrth orth(S, 1e-10);
    const int m = static_cast<int>(orth.X.shape(1));

    auto [eig_h, C_h] = diagonalize_fock(H, orth.X);
    RealMatrix P_h = build_density_matrix(C_h, n_occ, n, 2.0);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    AoDiisEigenDevice dev;
    dev.build(stream, n, m, /*max_vectors=*/0, n_occ, 2.0, 0.0);
    dev.upload_SX(S.data(), orth.X.data());
    dev.upload_P(P0.data());

    std::vector<double> eig_d(static_cast<size_t>(m));
    std::vector<double> C_d(static_cast<size_t>(n * m));
    dev.eigen_to_host(H.data(), eig_d.data(), C_d.data());

    double de = 0.0;
    for (int i = 0; i < m; ++i)
        de = std::max(de, std::abs(eig_d[static_cast<size_t>(i)] - eig_h(i)));
    EXPECT_LE(de, 1e-9) << "max|eig_dev-eig_host|=" << de;

    // Density is phase-invariant; compare P, not C.
    RealMatrix P_d = xt::zeros<double>({n, n});
    // Rebuild P from device C (or download after eigen_to_host which sets d_P_)
    // eigen_to_host runs diagonalize_and_density_ so d_P_ is live.
    dev.download_P(P_d.data());
    const double dP = max_abs_diff(P_h.data(), P_d.data(), n * n);
    EXPECT_LE(dP, 1e-10) << "max|P_dev-P_host|=" << dP;

    cudaStreamDestroy(stream);
}

TEST(CudaGpwDiisEigen, DiisStepAdvancesHistory) {
    // Two steps with DIIS depth 4: second advance must not throw and P stays
    // symmetric positive-ish (trace ≈ 2 * n_occ for RKS factor 2).
    RealMatrix S, H, P;
    int n_occ = 0;
    make_toy(S, H, P, n_occ);
    const int n = 3;
    CanonicalOrth orth(S, 1e-10);
    const int m = static_cast<int>(orth.X.shape(1));

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    AoDiisEigenDevice dev;
    dev.build(stream, n, m, /*max_vectors=*/4, n_occ, 2.0, 0.0);
    dev.upload_SX(S.data(), orth.X.data());
    dev.upload_P(P.data());
    dev.reset();

    const double e0 = dev.step(H.data());
    EXPECT_GE(e0, 0.0);

    // Perturb H slightly for a second residual.
    RealMatrix H2 = H;
    H2(0, 0) += 0.01;
    const double e1 = dev.step(H2.data());
    EXPECT_GE(e1, 0.0);

    RealMatrix P_out = xt::zeros<double>({n, n});
    dev.download_P(P_out.data());
    // Closed-shell: Tr(P S) = 2 * n_occ (not Tr(P) when S ≠ I).
    double trPS = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            trPS += P_out(i, j) * S(j, i);
    EXPECT_NEAR(trPS, 2.0 * n_occ, 1e-8);

    cudaStreamDestroy(stream);
}

// ── UKS single-channel (occ_factor=1; cuda-gpw-uks-device-diis) ────────────

TEST(CudaGpwDiisEigen, UksChannelErrorAndDensityMatchHost) {
    RealMatrix S, H, P_rks;
    int n_occ_rks = 0;
    make_toy(S, H, P_rks, n_occ_rks);
    (void)P_rks;
    const int n = 3;
    const int n_alpha = 2; // open-shell-ish channel on toy
    CanonicalOrth orth(S, 1e-10);
    const int m = static_cast<int>(orth.X.shape(1));
    auto [eig_h, C_h] = diagonalize_fock(H, orth.X);
    (void)eig_h;
    RealMatrix P_a = build_density_matrix(C_h, n_alpha, n, 1.0);

    const RealMatrix e_host =
        ATV::cpu::gpw::internal::gpw_diis_error(H, P_a, S, orth.X);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    AoDiisEigenDevice dev;
    dev.build(stream, n, m, /*max_vectors=*/4, n_alpha, /*occ=*/1.0,
              /*level_shift=*/0.0);
    dev.upload_SX(S.data(), orth.X.data());
    dev.upload_P(P_a.data());

    std::vector<double> e_dev(static_cast<size_t>(m * m));
    dev.error_to_host(H.data(), e_dev.data());
    const double de =
        max_abs_diff(e_host.data(), e_dev.data(), m * m);
    EXPECT_LE(de, 1e-10) << "max|e_dev-e_host| UKS=" << de;

    // Eigen + density with occ_factor=1.
    RealMatrix P_host = build_density_matrix(C_h, n_alpha, n, 1.0);
    std::vector<double> eig_d(static_cast<size_t>(m));
    std::vector<double> C_d(static_cast<size_t>(n * m));
    // Fresh device for eigen-only path (no DIIS).
    AoDiisEigenDevice eigen_dev;
    eigen_dev.build(stream, n, m, /*max_vectors=*/0, n_alpha, 1.0, 0.0);
    eigen_dev.upload_SX(S.data(), orth.X.data());
    eigen_dev.upload_P(P_a.data());
    eigen_dev.eigen_to_host(H.data(), eig_d.data(), C_d.data());
    RealMatrix P_d = xt::zeros<double>({n, n});
    eigen_dev.download_P(P_d.data());
    const double dP = max_abs_diff(P_host.data(), P_d.data(), n * n);
    EXPECT_LE(dP, 1e-10) << "max|P_dev-P_host| UKS factor-1=" << dP;

    double trPS = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            trPS += P_d(i, j) * S(j, i);
    EXPECT_NEAR(trPS, 1.0 * n_alpha, 1e-8);

    cudaStreamDestroy(stream);
}

TEST(CudaGpwDiisEigen, UksEmptyBetaChannelZeroDensity) {
    // High-spin channel with n_occ=0 must yield P=0 and zero residual.
    RealMatrix S, H, P_rks;
    int n_occ_rks = 0;
    make_toy(S, H, P_rks, n_occ_rks);
    const int n = 3;
    CanonicalOrth orth(S, 1e-10);
    const int m = static_cast<int>(orth.X.shape(1));
    RealMatrix P0 = xt::zeros<double>({n, n});

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    AoDiisEigenDevice beta;
    beta.build(stream, n, m, /*max_vectors=*/4, /*n_occ=*/0, 1.0, 0.0);
    beta.upload_SX(S.data(), orth.X.data());
    beta.upload_P(P0.data());
    beta.reset();

    const double err = beta.step(H.data());
    EXPECT_NEAR(err, 0.0, 1e-12);

    RealMatrix P_out = xt::zeros<double>({n, n});
    beta.download_P(P_out.data());
    for (int i = 0; i < n * n; ++i)
        EXPECT_NEAR(P_out.data()[i], 0.0, 1e-14);

    cudaStreamDestroy(stream);
}

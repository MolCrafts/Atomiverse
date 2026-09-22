/// @file tests/cuda/gpw/test_kpoint_diis_eigen.cu
/// @brief Unit: multi-k device DIIS residual + eigen/density vs host oracles.
///
/// Single-function checks for `internal::KpointDiisEigenDevice` (no full SCF).
/// Host oracles: `k_diis_error`, `diagonalize_k`, closed-shell density.

#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/linalg/matrix.h"
#include "atomiverse/cuda/gpw/internal/kpoint_diis_eigen.cuh"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

using ATV::cpu::gpw::internal::CplxMatrix;
using ATV::cpu::gpw::internal::diagonalize_k;
using ATV::cpu::gpw::internal::k_diis_error;
using ATV::cpu::gpw::internal::KOrth;
using ATV::cuda::gpw::internal::KpointDiisEigenDevice;
using Cplx = std::complex<double>;

namespace {

/// Small Hermitian complex S (SPD) and H for n=2, one k.
void make_toy(CplxMatrix& S, CplxMatrix& H, CplxMatrix& P, int& n_occ) {
    const int n = 2;
    n_occ = 1;
    S = xt::zeros<Cplx>({n, n});
    H = xt::zeros<Cplx>({n, n});
    S(0, 0) = 1.0;
    S(1, 1) = 1.0;
    S(0, 1) = Cplx(0.12, 0.03);
    S(1, 0) = std::conj(S(0, 1));
    H(0, 0) = -0.8;
    H(1, 1) = -0.3;
    H(0, 1) = Cplx(-0.10, 0.05);
    H(1, 0) = std::conj(H(0, 1));
    KOrth orth(S, 1e-10);
    auto [eps, C] = diagonalize_k(H, orth);
    (void)eps;
    P = xt::zeros<Cplx>({n, n});
    for (int o = 0; o < n_occ; ++o)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                P(i, j) += 2.0 * C(i, o) * std::conj(C(j, o));
}

double max_abs_cplx(const Cplx* a, const Cplx* b, int n) {
    double m = 0.0;
    for (int i = 0; i < n; ++i)
        m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

} // namespace

TEST(CudaGpwKpointDiisEigen, ErrorMatchesHost) {
    CplxMatrix S, H, P;
    int n_occ = 0;
    make_toy(S, H, P, n_occ);
    const int n = 2;
    KOrth orth(S, 1e-10);
    const int m = static_cast<int>(orth.X.shape(1));
    ASSERT_EQ(m, n);

    const CplxMatrix e_host = k_diis_error(H, P, S, orth.X);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    KpointDiisEigenDevice dev;
    const std::vector<double> w = {1.0};
    dev.build(stream, /*nk=*/1, n, m, /*max_vectors=*/4, n_occ, 2.0, w);
    dev.upload_SX_k(0, S.data(), orth.X.data(), m);
    dev.upload_P_k(0, P.data());

    std::vector<Cplx> e_dev(static_cast<size_t>(m * m));
    dev.error_to_host(0, H.data(), e_dev.data());

    const double d = max_abs_cplx(e_host.data(), e_dev.data(), m * m);
    EXPECT_LE(d, 1e-10) << "max|e_dev-e_host|=" << d;

    cudaStreamDestroy(stream);
}

TEST(CudaGpwKpointDiisEigen, EigenAndDensityMatchHost) {
    CplxMatrix S, H, P0;
    int n_occ = 0;
    make_toy(S, H, P0, n_occ);
    const int n = 2;
    KOrth orth(S, 1e-10);
    const int m = static_cast<int>(orth.X.shape(1));

    auto [eig_h, C_h] = diagonalize_k(H, orth);
    CplxMatrix P_h = xt::zeros<Cplx>({n, n});
    for (int o = 0; o < n_occ; ++o)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                P_h(i, j) += 2.0 * C_h(i, o) * std::conj(C_h(j, o));

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    KpointDiisEigenDevice dev;
    const std::vector<double> w = {1.0};
    // max_vectors=0: eigen only, no DIIS history.
    dev.build(stream, 1, n, m, /*max_vectors=*/0, n_occ, 2.0, w);
    dev.upload_SX_k(0, S.data(), orth.X.data(), m);
    dev.upload_P_k(0, P0.data());

    std::vector<double> eig_d(static_cast<size_t>(m));
    std::vector<Cplx> C_d(static_cast<size_t>(n * m));
    dev.eigen_to_host(0, H.data(), eig_d.data(), C_d.data());

    double de = 0.0;
    for (int i = 0; i < m; ++i)
        de = std::max(de, std::abs(eig_d[static_cast<size_t>(i)]
                                   - eig_h[static_cast<size_t>(i)]));
    EXPECT_LE(de, 1e-9) << "max|eig_dev-eig_host|=" << de;

    // Density is U(1) phase-invariant — compare P, not C.
    CplxMatrix P_d = xt::zeros<Cplx>({n, n});
    dev.download_P_k(0, P_d.data());
    const double dp = max_abs_cplx(P_h.data(), P_d.data(), n * n);
    EXPECT_LE(dp, 1e-9) << "max|P_dev-P_host|=" << dp;

    cudaStreamDestroy(stream);
}

TEST(CudaGpwKpointDiisEigen, TwoKSharedDiisAdvances) {
    // Smoke: two identical k weights, residual+advance does not throw and
    // updates P (history depth 4, two steps).
    CplxMatrix S, H, P;
    int n_occ = 0;
    make_toy(S, H, P, n_occ);
    const int n = 2;
    KOrth orth(S, 1e-10);
    const int m = static_cast<int>(orth.X.shape(1));

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    KpointDiisEigenDevice dev;
    const std::vector<double> w = {0.5, 0.5};
    dev.build(stream, /*nk=*/2, n, m, /*max_vectors=*/4, n_occ, 2.0, w);
    for (int ik = 0; ik < 2; ++ik) {
        dev.upload_SX_k(ik, S.data(), orth.X.data(), m);
        dev.upload_P_k(ik, P.data());
    }
    std::vector<const Cplx*> Hs = {H.data(), H.data()};
    const double e0 = dev.residual(Hs);
    EXPECT_GE(e0, 0.0);
    dev.advance();
    CplxMatrix P1 = xt::zeros<Cplx>({n, n});
    dev.download_P_k(0, P1.data());
    // Fresh density from the same H should match host diagonalize density.
    auto [eps, C] = diagonalize_k(H, orth);
    (void)eps;
    CplxMatrix P_ref = xt::zeros<Cplx>({n, n});
    for (int o = 0; o < n_occ; ++o)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                P_ref(i, j) += 2.0 * C(i, o) * std::conj(C(j, o));
    // First advance with mh=1 is warm-up (no extrap) → pure eigen of H.
    const double dp = max_abs_cplx(P_ref.data(), P1.data(), n * n);
    EXPECT_LE(dp, 1e-9) << "warm-up density max|dP|=" << dp;

    cudaStreamDestroy(stream);
}

TEST(CudaGpwKpointDiisEigen, ResidualFromDeviceMatchesHost) {
    // cuda-gpw-kpoint-h-residency: device RM H → ingest → residual_from_device
    // must match host residual(H) path (no full H D2H required in product).
    CplxMatrix S, H, P;
    int n_occ = 0;
    make_toy(S, H, P, n_occ);
    const int n = 2;
    KOrth orth(S, 1e-10);
    const int m = static_cast<int>(orth.X.shape(1));

    // Slightly different second-k H so multi-k residual is non-degenerate.
    CplxMatrix H2 = H;
    H2(0, 0) = -0.75;
    H2(1, 1) = -0.35;
    H2(0, 1) = Cplx(-0.08, 0.04);
    H2(1, 0) = std::conj(H2(0, 1));

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    KpointDiisEigenDevice host_path;
    const std::vector<double> w = {0.4, 0.6};
    host_path.build(stream, /*nk=*/2, n, m, /*max_vectors=*/4, n_occ, 2.0, w);
    for (int ik = 0; ik < 2; ++ik) {
        host_path.upload_SX_k(ik, S.data(), orth.X.data(), m);
        host_path.upload_P_k(ik, P.data());
    }
    std::vector<const Cplx*> Hs = {H.data(), H2.data()};
    const double err_host = host_path.residual(Hs);

    KpointDiisEigenDevice dev_path;
    dev_path.build(stream, /*nk=*/2, n, m, /*max_vectors=*/4, n_occ, 2.0, w);
    for (int ik = 0; ik < 2; ++ik) {
        dev_path.upload_SX_k(ik, S.data(), orth.X.data(), m);
        dev_path.upload_P_k(ik, P.data());
    }
    // Upload row-major H to device and ingest (simulate assemble_H d_H).
    Cplx* d_Hrm[2] = {nullptr, nullptr};
    const size_t bytes = static_cast<size_t>(n * n) * sizeof(Cplx);
    for (int ik = 0; ik < 2; ++ik) {
        ASSERT_EQ(cudaMalloc(&d_Hrm[ik], bytes), cudaSuccess);
        const Cplx* src = (ik == 0) ? H.data() : H2.data();
        ASSERT_EQ(cudaMemcpyAsync(d_Hrm[ik], src, bytes, cudaMemcpyHostToDevice,
                                  stream),
                  cudaSuccess);
        dev_path.ingest_H_k_device_rm(ik, d_Hrm[ik]);
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    const double err_dev = dev_path.residual_from_device();

    EXPECT_NEAR(err_dev, err_host, 1e-12)
        << "residual_from_device vs residual(host) |derr|="
        << std::abs(err_dev - err_host);

    for (int ik = 0; ik < 2; ++ik)
        cudaFree(d_Hrm[ik]);
    cudaStreamDestroy(stream);
}

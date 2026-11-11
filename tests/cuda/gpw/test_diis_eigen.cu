/// @file tests/cuda/gpw/test_diis_eigen.cu
/// @brief Unit: device AO DIIS residual + eigen/density vs host oracles.
///
/// Single-function checks for `internal::AoDiisEigenDevice` (no full SCF).
/// Host oracles: `gpw_diis_error`, `diagonalize_fock`, `build_density_matrix`.

#include "atomiverse/cpu/linalg/density.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/linalg/internal/diis_mixer.h"
#include "atomiverse/cpu/linalg/matrix.h"
#include "atomiverse/cpu/linalg/pulay.h"
#include "atomiverse/cuda/gpw/internal/ao_diis_eigen.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using ATV::cpu::linalg::build_density_matrix;
using ATV::cpu::linalg::CanonicalOrth;
using ATV::cpu::linalg::diagonalize_fock;
using ATV::cpu::linalg::RealMatrix;
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

    const RealMatrix e_host = ATV::cpu::linalg::diis_error(H, P, S, orth.X);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    AoDiisEigenDevice dev;
    dev.build(stream, n, m, /*max_vectors=*/4, n_occ, /*occ=*/2.0,
              /*level_shift=*/0.0);
    dev.upload_SX(S.data(), orth.X.data());
    dev.upload_P(P.data());

    std::vector<double> e_dev(static_cast<size_t>(m * m));
    dev.error_to_host(H.data(), e_dev.data());

    const double d = max_abs_diff(e_host.data(), e_dev.data(), m * m);
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

/// Device-resident H → eigen payload (cuda-gpw-device-payload). No H2D of H.
TEST(CudaGpwDiisEigen, EigenFromDeviceMatchesHost) {
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

    // Upload H once to a device buffer (setup boundary), then eigen_from_device.
    double* d_H = nullptr;
    ASSERT_EQ(cudaMalloc(&d_H, static_cast<size_t>(n * n) * sizeof(double)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_H, H.data(),
                              static_cast<size_t>(n * n) * sizeof(double),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);

    std::vector<double> eig_d(static_cast<size_t>(m));
    std::vector<double> C_d(static_cast<size_t>(n * m));
    dev.eigen_from_device(d_H, eig_d.data(), C_d.data());

    double de = 0.0;
    for (int i = 0; i < m; ++i)
        de = std::max(de, std::abs(eig_d[static_cast<size_t>(i)] - eig_h(i)));
    EXPECT_LE(de, 1e-9) << "max|eig_from_device-host|=" << de;

    RealMatrix P_d = xt::zeros<double>({n, n});
    dev.download_P(P_d.data());
    const double dP = max_abs_diff(P_h.data(), P_d.data(), n * n);
    EXPECT_LE(dP, 1e-10) << "max|P_from_device-host|=" << dP;

    // download_eig / download_C boundary helpers match eigen_from_device buffers.
    std::vector<double> eig2(static_cast<size_t>(m));
    std::vector<double> C2(static_cast<size_t>(n * m));
    dev.download_eig(eig2.data());
    dev.download_C(C2.data());
    EXPECT_LE(max_abs_diff(eig_d.data(), eig2.data(), m), 0.0);
    EXPECT_LE(max_abs_diff(C_d.data(), C2.data(), n * m), 0.0);

    cudaFree(d_H);
    cudaStreamDestroy(stream);
}

TEST(CudaGpwDiisEigen, PhysicalForceWeightPreservesAcceptedDensity) {
    RealMatrix S, H, accepted;
    int occupied = 0;
    make_toy(S, H, accepted, occupied);
    CanonicalOrth orth(S, 1e-10);
    const auto [seed_eig, seed_C] = diagonalize_fock(H, orth.X);
    H(0, 0) += .37;
    H(0, 2) = H(2, 0) = -.21;
    const auto [physical_eig, physical_C] = diagonalize_fock(H, orth.X);
    const auto canonical = build_density_matrix(physical_C, occupied, 3, 2.0);
    ASSERT_GT(max_abs_diff(canonical.data(), accepted.data(), 9), .1)
        << "accepted P must differ from the raw-H canonical density";
    RealMatrix expected_W = xt::zeros<double>({3, 3});
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int o = 0; o < occupied; ++o)
                expected_W(i, j) +=
                    2 * physical_eig(o) * physical_C(i, o) * physical_C(j, o);

    struct DeviceBoundary {
        cudaStream_t stream = nullptr;
        double* H = nullptr;
        ~DeviceBoundary() {
            cudaFree(H);
            cudaStreamDestroy(stream);
        }
    } boundary;
    ASSERT_EQ(cudaStreamCreate(&boundary.stream), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&boundary.H, 9 * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(boundary.H, H.data(), 9 * sizeof(double),
                              cudaMemcpyHostToDevice, boundary.stream),
              cudaSuccess);
    AoDiisEigenDevice device;
    device.build(boundary.stream, 3, 3, 4, occupied, 2, .7);
    device.upload_SX(S.data(), orth.X.data());
    device.upload_P(accepted.data());
    device.upload_C(seed_C.data());
    // Seed both DIIS and the shifted eigenspace. The force state must instead
    // use the supplied raw physical H, with no shift or history extrapolation.
    device.step(H.data());
    device.upload_P(accepted.data());
    device.reserve_force_state();
    for (int repeat = 0; repeat < 2; ++repeat) {
        device.finalize_physical_force_state(boundary.H);
        std::array<double, 9> got_P{}, got_W{};
        std::array<double, 3> got_eig{};
        device.download_P(got_P.data());
        device.download_eig(got_eig.data());
        ASSERT_EQ(cudaMemcpyAsync(got_W.data(), device.d_W(), sizeof(got_W),
                                  cudaMemcpyDeviceToHost, boundary.stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(boundary.stream), cudaSuccess);
        EXPECT_EQ(max_abs_diff(got_P.data(), accepted.data(), 9), 0);
        EXPECT_LE(max_abs_diff(got_W.data(), expected_W.data(), 9), 1e-10);
        EXPECT_LE(max_abs_diff(got_eig.data(), physical_eig.data(), 3), 1e-10);
    }
}

TEST(CudaGpwDiisEigen, EmptyChannelPhysicalForceWeightIsZero) {
    RealMatrix S, H, unused;
    int ignored = 0;
    make_toy(S, H, unused, ignored);
    CanonicalOrth orth(S, 1e-10);
    RealMatrix zero = xt::zeros<double>({3, 3});
    struct DeviceBoundary {
        cudaStream_t stream = nullptr;
        double* H = nullptr;
        ~DeviceBoundary() {
            cudaFree(H);
            cudaStreamDestroy(stream);
        }
    } boundary;
    ASSERT_EQ(cudaStreamCreate(&boundary.stream), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&boundary.H, 9 * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(boundary.H, H.data(), 9 * sizeof(double),
                              cudaMemcpyHostToDevice, boundary.stream),
              cudaSuccess);
    AoDiisEigenDevice device;
    device.build(boundary.stream, 3, 3, 0, 0, 1, .7);
    device.upload_SX(S.data(), orth.X.data());
    device.upload_P(zero.data());
    EXPECT_THROW(device.finalize_physical_force_state(boundary.H),
                 std::logic_error);
    device.reserve_force_state();
    device.finalize_physical_force_state(boundary.H);
    std::array<double, 9> got_W{}, got_P{};
    device.download_P(got_P.data());
    ASSERT_EQ(cudaMemcpyAsync(got_W.data(), device.d_W(), sizeof(got_W),
                              cudaMemcpyDeviceToHost, boundary.stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(boundary.stream), cudaSuccess);
    for (int i = 0; i < 9; ++i) {
        EXPECT_EQ(got_P[i], 0);
        EXPECT_EQ(got_W[i], 0);
    }
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

    const RealMatrix e_host = ATV::cpu::linalg::diis_error(H, P_a, S, orth.X);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    AoDiisEigenDevice dev;
    dev.build(stream, n, m, /*max_vectors=*/4, n_alpha, /*occ=*/1.0,
              /*level_shift=*/0.0);
    dev.upload_SX(S.data(), orth.X.data());
    dev.upload_P(P_a.data());

    std::vector<double> e_dev(static_cast<size_t>(m * m));
    dev.error_to_host(H.data(), e_dev.data());
    const double de = max_abs_diff(e_host.data(), e_dev.data(), m * m);
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

namespace ATV::cuda::gpw::internal {
struct PairedDiisProbe {
    static std::array<int, 2> cursor(const AoDiisEigenDevice& channel) {
        return {channel.hist_count_, channel.hist_start_};
    }
    static RealMatrix P(const AoDiisEigenDevice& channel) {
        RealMatrix result = xt::zeros<double>({channel.n_, channel.n_});
        channel.download_P(result.data());
        return result;
    }

    static RealMatrix H(const AoDiisEigenDevice& channel) {
        RealMatrix result = xt::zeros<double>({channel.n_, channel.n_});
        if (cudaMemcpyAsync(result.data(), channel.d_H_,
                            result.size() * sizeof(double),
                            cudaMemcpyDeviceToHost, channel.stream_)
                != cudaSuccess
            || cudaStreamSynchronize(channel.stream_) != cudaSuccess)
            throw std::runtime_error("probe H download");
        return result;
    }
};
} // namespace ATV::cuda::gpw::internal
namespace {
void paired_require(bool ok, const char* label) {
    if (!ok)
        throw std::runtime_error(label);
}
double paired_difference(const RealMatrix& a, const RealMatrix& b) {
    double result = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        result = std::max(result, std::abs(a.data()[i] - b.data()[i]));
    return result;
}
RealMatrix paired_field(int n, int step, bool beta, bool near) {
    RealMatrix h = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            double value =
                i == j ? -.8 + .33 * i : .03 * std::sin((i + 1) * (j + 2));
            const double scale = near ? 1e-3 * step : 1.;
            value += scale
                     * (.04
                            * std::sin((i + 1) * (j + 1) + .7 * step
                                       + (beta ? .4 : 0))
                        + .025
                              * std::cos(.31 * step * (i + j + 1)
                                         + (beta ? .7 : 0)));
            h(i, j) = h(j, i) = value;
        }
    return h;
}
struct CoupledStream {
    cudaStream_t stream = nullptr;
    CoupledStream() {
        paired_require(cudaStreamCreate(&stream) == cudaSuccess,
                       "paired stream create");
    }
    ~CoupledStream() {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
};
void check_coupled_oracle(const char* name, bool empty, bool near,
                          bool singular) {
    constexpr int n = 4;
    RealMatrix S = xt::eye<double>(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j)
            S(i, j) = S(j, i) = .03 / (1 + i - j);
    ATV::cpu::linalg::CanonicalOrth orth(S, 1e-12);
    const auto initial = paired_field(n, 1, false, false);
    auto [epsilon, C] = ATV::cpu::linalg::diagonalize_fock(initial, orth.X);
    const auto Pa = ATV::cpu::linalg::build_density_matrix(C, 2, n, 1.);
    const auto Pb =
        ATV::cpu::linalg::build_density_matrix(C, empty ? 0 : 1, n, 1.);
    CoupledStream execution;
    const auto stream = execution.stream;
    {
        const int depth = near ? 2 : 3;
        AoDiisEigenDevice a, b;
        a.build(stream, n, n, depth, 2, 1., 0);
        b.build(stream, n, n, depth, empty ? 0 : 1, 1., 0);
        a.upload_SX(S.data(), orth.X.data());
        b.upload_SX(S.data(), orth.X.data());
        ATV::cpu::linalg::internal::DIISMixer cpu(depth), independent_a(depth),
            independent_b(depth);
        double maxH = 0, maxP = 0, counterfactual = 0;
        for (int round = 0; round < 2; ++round) {
            a.reset();
            b.reset();
            cpu.reset();
            independent_a.reset();
            independent_b.reset();
            const int steps = near ? 5 : 9;
            for (int step = 0; step < steps; ++step) {
                const auto Ha = paired_field(n, singular ? 2 : step + 2, false,
                                             near),
                           Hb = paired_field(n, singular ? 2 : step + 2, true,
                                             near);
                a.upload_P(Pa.data());
                b.upload_P(Pb.data());
                a.residual(Ha.data());
                b.residual(Hb.data());
                const auto ea = ATV::cpu::linalg::diis_error(Ha, Pa, S, orth.X),
                           eb = ATV::cpu::linalg::diis_error(Hb, Pb, S, orth.X);
                const auto [mixeda, mixedb] = cpu.step(Ha, ea, Hb, eb);
                const auto own_a = independent_a.step(Ha, ea),
                           own_b = independent_b.step(Hb, eb);
                counterfactual =
                    std::max({counterfactual, paired_difference(mixeda, own_a),
                              paired_difference(mixedb, own_b)});
                a.advance_paired(b);
                const auto actuala =
                               ATV::cuda::gpw::internal::PairedDiisProbe::H(a),
                           actualb =
                               ATV::cuda::gpw::internal::PairedDiisProbe::H(b);
                maxH = std::max({maxH, paired_difference(actuala, mixeda),
                                 paired_difference(actualb, mixedb)});
                const auto [epsa, Ca] =
                    ATV::cpu::linalg::diagonalize_fock(mixeda, orth.X);
                const auto [epsb, Cb] =
                    ATV::cpu::linalg::diagonalize_fock(mixedb, orth.X);
                const auto expecteda = ATV::cpu::linalg::build_density_matrix(
                               Ca, 2, n, 1.),
                           expectedb = ATV::cpu::linalg::build_density_matrix(
                               Cb, empty ? 0 : 1, n, 1.);
                RealMatrix actualPa = xt::zeros<double>({n, n}),
                           actualPb = actualPa;
                a.download_P(actualPa.data());
                b.download_P(actualPb.data());
                maxP = std::max({maxP, paired_difference(actualPa, expecteda),
                                 paired_difference(actualPb, expectedb)});
                if (empty)
                    paired_require(std::all_of(actualPb.begin(), actualPb.end(),
                                               [](double x) { return x == 0; }),
                                   "empty beta density");
            }
        }
        std::printf("PAIRED_COMPONENT %s max_H=%.17g max_P=%.17g "
                    "independent_coeff_counterfactual=%.17g empty=%d depth=%d "
                    "reset_rounds=2\n",
                    name, maxH, maxP, counterfactual, empty, depth);
        paired_require(maxH < 1e-8 && maxP < 1e-8,
                       "coupled CPU mixer/eigen oracle gate");
        if (!singular && !empty)
            paired_require(
                counterfactual > 1e-6,
                "oracle must distinguish independent spin coefficients");
    }
}

} // namespace
TEST(CudaGpwDiisEigen, CoupledSpinMatchesCpuWithHistoryWrapAndReset) {
    check_coupled_oracle("wrapped", false, false, false);
}
TEST(CudaGpwDiisEigen, CoupledEmptyBetaMatchesCpuWithHistoryWrapAndReset) {
    check_coupled_oracle("empty-beta-wrapped", true, false, false);
}
TEST(CudaGpwDiisEigen, CoupledNearRankDeficientHistoryMatchesCpu) {
    check_coupled_oracle("near-rankdef", false, true, false);
}
TEST(CudaGpwDiisEigen, CoupledSingularHistoryKeepsBothLatestPhysicalOperators) {
    check_coupled_oracle("singular", false, false, true);
}

TEST(CudaGpwDiisEigen, CoupledRejectsMismatchedBindingsBeforeMutation) {
    using Probe = ATV::cuda::gpw::internal::PairedDiisProbe;
    // n, orthonormal m, borrowed stream, history depth, count, cursor, self.
    for (int mismatch = 0; mismatch < 7; ++mismatch) {
        SCOPED_TRACE(mismatch);
        CoupledStream execution, another;
        AoDiisEigenDevice a, b;
        auto initialize = [](AoDiisEigenDevice& channel, int n, int m,
                             int depth, cudaStream_t stream) {
            channel.build(stream, n, m, depth, 1, 1., 0.);
            RealMatrix S = xt::eye<double>(n);
            RealMatrix X = xt::zeros<double>({n, m});
            RealMatrix P = xt::zeros<double>({n, n});
            for (int i = 0; i < m; ++i)
                X(i, i) = 1;
            P(0, 0) = 1;
            const auto H = paired_field(n, 2, false, false);
            channel.upload_SX(S.data(), X.data());
            channel.upload_P(P.data());
            channel.residual(
                H.data()); // Also fences all initialization uploads.
        };
        initialize(a, 4, 4, 3, execution.stream);
        initialize(b, mismatch == 0 ? 3 : 4,
                   mismatch == 0 || mismatch == 1 ? 3 : 4,
                   mismatch == 3 ? 2 : 3,
                   mismatch == 2 ? another.stream : execution.stream);
        if (mismatch == 4)
            a.advance();
        if (mismatch == 5) {
            // Both have full count three, with distinct circular start slots.
            for (int i = 0; i < 4; ++i)
                a.advance();
            for (int i = 0; i < 3; ++i)
                b.advance();
        }
        const auto before_a = Probe::cursor(a), before_b = Probe::cursor(b);
        const auto Ha = Probe::H(a), Hb = Probe::H(b);
        const auto Pa = Probe::P(a), Pb = Probe::P(b);
        EXPECT_THROW(a.advance_paired(mismatch == 6 ? a : b),
                     std::invalid_argument);
        EXPECT_EQ(Probe::cursor(a), before_a);
        EXPECT_EQ(Probe::cursor(b), before_b);
        EXPECT_DOUBLE_EQ(paired_difference(Probe::H(a), Ha), 0);
        EXPECT_DOUBLE_EQ(paired_difference(Probe::H(b), Hb), 0);
        EXPECT_DOUBLE_EQ(paired_difference(Probe::P(a), Pa), 0);
        EXPECT_DOUBLE_EQ(paired_difference(Probe::P(b), Pb), 0);
    }
}

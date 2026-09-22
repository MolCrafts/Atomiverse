/// @file test_device_integrals.cu
/// @brief GPU integral kernel validation against CPU native (ground truth: libcint).
///
/// Launches GPU kernels, copies results to host, compares element-by-element
/// against CPU native integrals.

#ifdef ATV_HAS_HF

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/basis_data/accessors.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cuda/scf/device_integrals.cuh"

using namespace ATV;

static constexpr double kA2B = ATV::units::kAngstromInBohr;

// ── Boys function ─────────────────────────────────────────────────────

TEST(DeviceIntegrals, BoysF0) {
    // Generate test values: 0, small, medium, large
    std::vector<double> h_x = {0,           double(1e-15), double(0.01),
                               double(0.1), double(0.5),   double(1),
                               double(2),   double(5),     double(10),
                               double(50),  double(100)};
    int n = static_cast<int>(h_x.size());

    double *d_x, *d_out;
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMalloc(&d_out, n * sizeof(double));
    cudaMemcpy(d_x, h_x.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    cuda::scf::launch_boys_f0_test(d_x, d_out, n, nullptr);
    cudaDeviceSynchronize();

    std::vector<double> h_out(n);
    cudaMemcpy(h_out.data(), d_out, n * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        double cpu_val = cpu::scf::native::boys_f0(static_cast<double>(h_x[i]));
        EXPECT_NEAR(static_cast<double>(h_out[i]), cpu_val, 1e-6)
            << "Boys F_0 mismatch at x=" << h_x[i];
    }

    cudaFree(d_x);
    cudaFree(d_out);
}

// ── Helper: upload basis data to device ───────────────────────────────

struct DeviceBasis {
    double* d_alpha = nullptr;
    double* d_coeff = nullptr;
    double* d_centers = nullptr;
    int* d_Z = nullptr;
    double* d_pos = nullptr;
    int n_shells, n_prim, natm;

    // Host-side CPU reference
    cpu::scf::internal::IntegralWorkspace ref;

    DeviceBasis() : n_shells(2), natm(2) {
        // Contraction from basis_data::sto3g() — do not retype exponents.
        const auto& h1s = cpu::scf::basis_data::pure_s_sto3g_shell(/*Z=*/1);
        n_prim = static_cast<int>(h1s.alpha.size());

        // Compute CPU reference
        cpu::Frame frame(2);
        frame.span(kAtom_x)[0] = double(0);
        frame.span(kAtom_y)[0] = double(0);
        frame.span(kAtom_z)[0] = double(0);
        frame.span(kAtom_x)[1] = double(0.74);
        frame.span(kAtom_y)[1] = double(0);
        frame.span(kAtom_z)[1] = double(0);
        frame.span(kAtom_mass)[0] = double(1.008);
        frame.span(kAtom_mass)[1] = double(1.008);
        frame.span(kAtom_type)[0] = 1;
        frame.span(kAtom_type)[1] = 1;
        frame.box = cpu::Box::cubic(100.0);
        // compute_eri=true: the ERI test reads ref.ERI directly.
        ref = cpu::scf::internal::IntegralWorkspace{
            frame, cpu::scf::BasisSet::STO3G, 0, true};

        // Build host arrays with PHYSICAL normalization (for GPU kernels).
        std::vector<double> h_alpha(
            static_cast<std::size_t>(n_shells * n_prim));
        std::vector<double> h_coeff(
            static_cast<std::size_t>(n_shells * n_prim));
        double h_centers[6];
        int h_Z[2] = {1, 1};
        double h_pos[6];

        double cx[2] = {0, 0.74 * kA2B};
        for (int s = 0; s < 2; ++s) {
            for (int p = 0; p < n_prim; ++p) {
                h_alpha[static_cast<std::size_t>(s * n_prim + p)] =
                    h1s.alpha[static_cast<std::size_t>(p)];
                h_coeff[static_cast<std::size_t>(s * n_prim + p)] =
                    h1s.coeff[static_cast<std::size_t>(p)]
                    * cpu::scf::native::phys_norm(
                        h1s.alpha[static_cast<std::size_t>(p)]);
            }
            h_centers[s * 3 + 0] = double(cx[s]);
            h_centers[s * 3 + 1] = double(0);
            h_centers[s * 3 + 2] = double(0);
            h_pos[s * 3 + 0] = double(cx[s]);
            h_pos[s * 3 + 1] = double(0);
            h_pos[s * 3 + 2] = double(0);
        }

        const std::size_t n_ac = static_cast<std::size_t>(n_shells)
                                 * static_cast<std::size_t>(n_prim);
        cudaMalloc(&d_alpha, n_ac * sizeof(double));
        cudaMalloc(&d_coeff, n_ac * sizeof(double));
        cudaMalloc(&d_centers, 6 * sizeof(double));
        cudaMalloc(&d_Z, 2 * sizeof(int));
        cudaMalloc(&d_pos, 6 * sizeof(double));

        cudaMemcpy(d_alpha, h_alpha.data(), n_ac * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_coeff, h_coeff.data(), n_ac * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_centers, h_centers, 6 * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_Z, h_Z, 2 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos, h_pos, 6 * sizeof(double), cudaMemcpyHostToDevice);
    }

    ~DeviceBasis() {
        cudaFree(d_alpha);
        cudaFree(d_coeff);
        cudaFree(d_centers);
        cudaFree(d_Z);
        cudaFree(d_pos);
    }
};

// ── 1e integrals: overlap ─────────────────────────────────────────────

TEST(DeviceIntegrals, Overlap) {
    DeviceBasis db;
    int nn = db.n_shells * db.n_shells;

    double *d_S, *d_T, *d_V;
    cudaMalloc(&d_S, nn * sizeof(double));
    cudaMalloc(&d_T, nn * sizeof(double));
    cudaMalloc(&d_V, nn * sizeof(double));

    cuda::scf::launch_1e_integrals(d_S, d_T, d_V, db.d_alpha, db.d_coeff,
                                   db.d_centers, db.d_Z, db.d_pos, db.n_shells,
                                   db.n_prim, db.natm, nullptr);
    cudaDeviceSynchronize();

    std::vector<double> h_S(nn);
    cudaMemcpy(h_S.data(), d_S, nn * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(static_cast<double>(h_S[i + j * 2]),
                        static_cast<double>(db.ref.S(i, j)), 5e-7)
                << "S(" << i << "," << j << ") GPU vs CPU mismatch";

    cudaFree(d_S);
    cudaFree(d_T);
    cudaFree(d_V);
}

// ── 1e integrals: kinetic ─────────────────────────────────────────────

TEST(DeviceIntegrals, Kinetic) {
    DeviceBasis db;
    int nn = db.n_shells * db.n_shells;

    double *d_S, *d_T, *d_V;
    cudaMalloc(&d_S, nn * sizeof(double));
    cudaMalloc(&d_T, nn * sizeof(double));
    cudaMalloc(&d_V, nn * sizeof(double));

    cuda::scf::launch_1e_integrals(d_S, d_T, d_V, db.d_alpha, db.d_coeff,
                                   db.d_centers, db.d_Z, db.d_pos, db.n_shells,
                                   db.n_prim, db.natm, nullptr);
    cudaDeviceSynchronize();

    std::vector<double> h_T(nn);
    cudaMemcpy(h_T.data(), d_T, nn * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(static_cast<double>(h_T[i + j * 2]),
                        static_cast<double>(db.ref.T(i, j)), 5e-7)
                << "T(" << i << "," << j << ") GPU vs CPU mismatch";

    cudaFree(d_S);
    cudaFree(d_T);
    cudaFree(d_V);
}

// ── 1e integrals: nuclear ─────────────────────────────────────────────

TEST(DeviceIntegrals, Nuclear) {
    DeviceBasis db;
    int nn = db.n_shells * db.n_shells;

    double *d_S, *d_T, *d_V;
    cudaMalloc(&d_S, nn * sizeof(double));
    cudaMalloc(&d_T, nn * sizeof(double));
    cudaMalloc(&d_V, nn * sizeof(double));

    cuda::scf::launch_1e_integrals(d_S, d_T, d_V, db.d_alpha, db.d_coeff,
                                   db.d_centers, db.d_Z, db.d_pos, db.n_shells,
                                   db.n_prim, db.natm, nullptr);
    cudaDeviceSynchronize();

    std::vector<double> h_V(nn);
    cudaMemcpy(h_V.data(), d_V, nn * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(static_cast<double>(h_V[i + j * 2]),
                        static_cast<double>(db.ref.V(i, j)), 5e-7)
                << "V(" << i << "," << j << ") GPU vs CPU mismatch";

    cudaFree(d_S);
    cudaFree(d_T);
    cudaFree(d_V);
}

// ── 2e integrals: ERI ─────────────────────────────────────────────────

TEST(DeviceIntegrals, ERI) {
    DeviceBasis db;
    int n4 = 16;

    double* d_ERI;
    cudaMalloc(&d_ERI, n4 * sizeof(double));

    cuda::scf::launch_eri(d_ERI, db.d_alpha, db.d_coeff, db.d_centers,
                          db.n_shells, db.n_prim, nullptr);
    cudaDeviceSynchronize();

    std::vector<double> h_ERI(n4);
    cudaMemcpy(h_ERI.data(), d_ERI, n4 * sizeof(double),
               cudaMemcpyDeviceToHost);

    int n = 2;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l) {
                    int idx = i + j * n + k * n * n + l * n * n * n;
                    EXPECT_NEAR(static_cast<double>(h_ERI[idx]),
                                static_cast<double>(db.ref.ERI(i, j, k, l)),
                                5e-7)
                        << "ERI(" << i << "," << j << "," << k << "," << l
                        << ") GPU vs CPU mismatch";
                }

    cudaFree(d_ERI);
}

#else

#include <gtest/gtest.h>
TEST(DeviceIntegrals, Disabled) {
    GTEST_SKIP() << "Device integral tests require -DATV_BUILD_HF=ON";
}

#endif

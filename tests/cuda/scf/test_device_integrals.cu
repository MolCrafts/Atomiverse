/// @file test_device_integrals.cu
/// @brief GPU integral kernel validation against CPU native (ground truth: libcint).
///
/// Launches GPU kernels, copies results to host, compares element-by-element
/// against CPU native integrals.

#ifdef ATV_HAS_HF

#include <array>
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
#include "atomiverse/cpu/scf/rys_eri.h"
#include "atomiverse/cuda/device/error.cuh"
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

namespace {

// Four noncollinear, single-primitive shells. The full Cartesian AO maps
// exercise the production s/p launcher, without an SCF or a new test kernel.
struct PrimitiveQuartet {
    std::array<double, 4> alpha{0.43, 1.13, 0.71, 1.67}, coefficient;
    std::array<double, 12> center{0.13,  -0.31, 0.27, 1.22, 0.48,  -0.53,
                                  -0.76, 1.37,  0.64, 0.42, -0.91, 1.51};
    std::array<int, 4> angular, start;
    std::vector<int> ao_shell, ao_cart;
    std::vector<void*> owned;
    cudaStream_t stream = nullptr;
    int n = 0;
    double *d_alpha, *d_coefficient, *d_center, *d_eri;
    int *d_angular, *d_start, *d_ao_shell, *d_ao_cart;

    explicit PrimitiveQuartet(int mask) {
        ATV_CUDA_CHECK(cudaStreamCreate(&stream));
        const double amplitude[4] = {0.83, -0.61, 1.21, 0.47};
        for (int s = 0; s < 4; ++s) {
            angular[s] = (mask >> s) & 1;
            start[s] = n;
            coefficient[s] =
                amplitude[s]
                * (angular[s] ? cpu::scf::native::phys_norm_p(alpha[s])
                              : cpu::scf::native::phys_norm(alpha[s]));
            for (int cart = 0; cart < (angular[s] ? 3 : 1); ++cart) {
                ao_shell.push_back(s);
                ao_cart.push_back(cart);
                ++n;
            }
        }
        d_alpha = copy(alpha);
        d_coefficient = copy(coefficient);
        d_center = copy(center);
        d_angular = copy(angular);
        d_start = copy(start);
        d_ao_shell = copy(ao_shell);
        d_ao_cart = copy(ao_cart);
        allocate(d_eri, n * n * n * n);
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    ~PrimitiveQuartet() {
        cudaStreamSynchronize(stream);
        for (void* p : owned)
            cudaFree(p);
        cudaStreamDestroy(stream);
    }

    template <class T>
    void allocate(T*& p, std::size_t count) {
        ATV_CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
        owned.push_back(p);
    }

    template <class Values>
    auto copy(const Values& values) -> typename Values::value_type* {
        using T = typename Values::value_type;
        T* p = nullptr;
        allocate(p, values.size());
        ATV_CUDA_CHECK(cudaMemcpyAsync(p, values.data(),
                                       values.size() * sizeof(T),
                                       cudaMemcpyHostToDevice, stream));
        return p;
    }

    cpu::scf::native::ShellBlock block(int s) const {
        return {1,
                &alpha[s],
                &coefficient[s],
                {center[s * 3], center[s * 3 + 1], center[s * 3 + 2]},
                angular[s],
                angular[s] ? 3 : 1,
                start[s]};
    }
};

} // namespace

TEST(DeviceIntegrals, SpPrimitiveHermiteOrdersZeroThroughFour) {
    for (int mask = 0; mask < 16; ++mask) {
        PrimitiveQuartet input(mask);
        cuda::scf::launch_eri_sp(
            input.d_eri, input.d_alpha, input.d_coefficient, input.d_center,
            input.d_angular, input.d_start, input.d_ao_shell, input.d_ao_cart,
            4, input.n, 1, input.stream);
        ATV_CUDA_CHECK(cudaGetLastError());
        std::vector<double> got(input.n * input.n * input.n * input.n);
        ATV_CUDA_CHECK(cudaMemcpyAsync(got.data(), input.d_eri,
                                       got.size() * sizeof(double),
                                       cudaMemcpyDeviceToHost, input.stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(input.stream));
        for (const auto& shells : {std::array<int, 4>{0, 1, 2, 3},
                                   {0, 0, 0, 0},
                                   {3, 3, 3, 3},
                                   {0, 3, 0, 3},
                                   {0, 3, 1, 3}}) {
            const auto a = input.block(shells[0]), b = input.block(shells[1]),
                       c = input.block(shells[2]), d = input.block(shells[3]);
            std::array<double, 81> reference{};
            cpu::scf::native::eri_block(a, b, c, d, reference.data());
            int index = 0;
            for (int ia = 0; ia < a.n_cart; ++ia)
                for (int ib = 0; ib < b.n_cart; ++ib)
                    for (int ic = 0; ic < c.n_cart; ++ic)
                        for (int id = 0; id < d.n_cart; ++id) {
                            const int mu = a.ao_start + ia,
                                      nu = b.ao_start + ib,
                                      lam = c.ao_start + ic,
                                      sig = d.ao_start + id;
                            const int k =
                                mu
                                + input.n
                                      * (nu + input.n * (lam + input.n * sig));
                            EXPECT_NEAR(got[k], reference[index++], 2e-12)
                                << "s/p mask=" << mask << " tuple=" << mu << ','
                                << nu << ',' << lam << ',' << sig;
                        }
        }
    }
}

TEST(DeviceIntegrals, SpPrimitiveRaisedHermiteOrderFive) {
    PrimitiveQuartet input(15);
    const int n = input.n;
    const int component[4] = {0, 1, 2, 0};
    const double weight[4] = {0.7, -0.4, 0.9, 0.3};
    std::vector<double> density(n * n, 0), zero(n * n, 0);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            density[input.start[i] + component[i]
                    + n * (input.start[j] + component[j])] =
                weight[i] * weight[j];
    const std::array<int, 4> atom{0, 1, 2, 3}, nuclear_charge{};
    int* d_atom = input.copy(atom);
    int* d_charge = input.copy(nuclear_charge);
    double* d_W = input.copy(zero);
    double* d_P = input.copy(density);
    double* d_force = nullptr;
    input.allocate(d_force, 12);
    std::array<double, 12> even_force{};
    for (int sign : {1, -1}) {
        std::vector<double> signed_density(density);
        for (double& value : signed_density)
            value *= sign;
        ATV_CUDA_CHECK(cudaMemcpyAsync(d_P, signed_density.data(),
                                       signed_density.size() * sizeof(double),
                                       cudaMemcpyHostToDevice, input.stream));
        cuda::scf::launch_rhf_forces_sp(
            d_force, d_force + 4, d_force + 8, d_P, d_W, input.d_alpha,
            input.d_coefficient, input.d_center, d_atom, input.d_angular,
            input.d_ao_shell, input.d_ao_cart, d_charge, input.d_center, n, 1,
            4, input.stream);
        ATV_CUDA_CHECK(cudaGetLastError());
        std::array<double, 12> got;
        ATV_CUDA_CHECK(cudaMemcpyAsync(got.data(), d_force, sizeof(got),
                                       cudaMemcpyDeviceToHost, input.stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(input.stream));
        // The 1e contribution is odd in P; averaging +/-P isolates the
        // even 2e gradient without adding a special production launcher.
        for (int a = 0; a < 4; ++a)
            for (int axis = 0; axis < 3; ++axis)
                even_force[a * 3 + axis] += 0.5 * got[a + axis * 4];
    }
    std::array<double, 12> gradient{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                for (int l = 0; l < 4; ++l) {
                    const int shell[4] = {i, j, k, l};
                    const auto a = input.block(i), b = input.block(j),
                               c = input.block(k), d = input.block(l);
                    const int index =
                        ((component[i] * 3 + component[j]) * 3 + component[k])
                            * 3
                        + component[l];
                    const double factor =
                        0.25 * weight[i] * weight[j] * weight[k] * weight[l];
                    for (int which = 0; which < 4; ++which) {
                        std::array<double, 3 * 81> derivative{};
                        cpu::scf::native::rys_eri_block_deriv(
                            a, b, c, d, which, derivative.data());
                        for (int axis = 0; axis < 3; ++axis)
                            gradient[shell[which] * 3 + axis] +=
                                factor * derivative[axis * 81 + index];
                    }
                }
    for (int a = 0; a < 4; ++a)
        for (int axis = 0; axis < 3; ++axis)
            EXPECT_NEAR(even_force[a * 3 + axis],
                        -gradient[a * 3 + axis] * units::kHartreeInEv
                            * units::kAngstromInBohr,
                        2e-11)
                << "raised p quartet atom=" << a << " axis=" << axis;
}

#else

#include <gtest/gtest.h>
TEST(DeviceIntegrals, Disabled) {
    GTEST_SKIP() << "Device integral tests require -DATV_BUILD_HF=ON";
}

#endif

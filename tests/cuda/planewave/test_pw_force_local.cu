/// @file tests/cuda/planewave/test_pw_forces.cu
/// @brief Unit tests for CUDA plane-wave analytic force kernels.
///
/// UNIT SCOPE: local Hellmann–Feynman kernel vs a host twin on synthetic
/// ρ(G); public-API single-atom symmetry. Full Si-dimer CUDA↔CPU dual-gate
/// lives in regressions/pw/rks — never under ctest.

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/planewave/internal/pw_forces.cuh"
#include "atomiverse/cuda/planewave/pw_solver.cuh"
#include <memory>

#include <cmath>
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ATV;

namespace {

constexpr int kBlock = 256;

/// Host twin of the device local-force kernel (same formula as CPU
/// `accumulate_local_forces`).
void host_local_forces(const std::vector<std::complex<double>>& rho,
                       const std::vector<double>& gx,
                       const std::vector<double>& gy,
                       const std::vector<double>& gz,
                       const std::vector<double>& vg,
                       const std::vector<double>& pos,
                       const std::vector<int>& species, int n_atoms,
                       double omega, std::vector<double>& F) {
    const int ng = static_cast<int>(rho.size());
    F.assign(static_cast<std::size_t>(3 * n_atoms), 0.0);
    for (int ig = 0; ig < ng; ++ig) {
        const double Gx = gx[static_cast<std::size_t>(ig)];
        const double Gy = gy[static_cast<std::size_t>(ig)];
        const double Gz = gz[static_cast<std::size_t>(ig)];
        const double g2 = Gx * Gx + Gy * Gy + Gz * Gz;
        if (g2 < 1e-16)
            continue;
        const double rr = rho[static_cast<std::size_t>(ig)].real();
        const double ri = rho[static_cast<std::size_t>(ig)].imag();
        for (int ia = 0; ia < n_atoms; ++ia) {
            const double v = vg[species[ia] * ng + ig];
            const double ax = pos[static_cast<std::size_t>(3 * ia + 0)];
            const double ay = pos[static_cast<std::size_t>(3 * ia + 1)];
            const double az = pos[static_cast<std::size_t>(3 * ia + 2)];
            const double phase = Gx * ax + Gy * ay + Gz * az;
            const double scale =
                omega * v * (rr * std::sin(phase) + ri * std::cos(phase));
            F[static_cast<std::size_t>(3 * ia + 0)] += scale * Gx;
            F[static_cast<std::size_t>(3 * ia + 1)] += scale * Gy;
            F[static_cast<std::size_t>(3 * ia + 2)] += scale * Gz;
        }
    }
}

template <typename T>
T* d_alloc(std::size_t n) {
    T* p = nullptr;
    ATV_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

} // namespace

TEST(CudaPWForces, LocalKernelMatchesHostForTwoSpecies) {
    // Synthetic density sphere: a few G shells, two atoms, nonzero ρ.
    constexpr int ng = 8;
    constexpr int n_atoms = 2;
    constexpr double omega = 100.0;

    std::vector<std::complex<double>> rho(ng);
    std::vector<double> gx(ng), gy(ng), gz(ng), vg(2 * ng);
    for (int i = 0; i < ng; ++i) {
        const double t = 0.3 * static_cast<double>(i + 1);
        gx[static_cast<std::size_t>(i)] = (i == 0) ? 0.0 : t;
        gy[static_cast<std::size_t>(i)] = (i == 0) ? 0.0 : 0.1 * t;
        gz[static_cast<std::size_t>(i)] = (i == 0) ? 0.0 : -0.2 * t;
        vg[static_cast<std::size_t>(i)] = -0.5 / (1.0 + t);
        vg[ng + i] = -0.25 / (1 + t * t);
        rho[static_cast<std::size_t>(i)] = {0.02 * std::cos(t),
                                            0.01 * std::sin(t)};
    }
    std::vector<double> pos = {0.0, 0.0, 0.0, 1.2, 0.3, -0.4};

    const std::vector<int> species{1, 0};
    std::vector<double> F_host;
    host_local_forces(rho, gx, gy, gz, vg, pos, species, n_atoms, omega,
                      F_host);

    cufftDoubleComplex* d_rho = d_alloc<cufftDoubleComplex>(ng);
    double* d_gx = d_alloc<double>(ng);
    double* d_gy = d_alloc<double>(ng);
    double* d_gz = d_alloc<double>(ng);
    double* d_vg = d_alloc<double>(vg.size());
    int* d_species = d_alloc<int>(species.size());
    double* d_pos = d_alloc<double>(3 * n_atoms);
    double* d_F = d_alloc<double>(3 * n_atoms);

    ATV_CUDA_CHECK(cudaMemcpy(d_rho, rho.data(),
                              ng * sizeof(cufftDoubleComplex),
                              cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemcpy(d_gx, gx.data(), ng * sizeof(double),
                              cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemcpy(d_gy, gy.data(), ng * sizeof(double),
                              cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemcpy(d_gz, gz.data(), ng * sizeof(double),
                              cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemcpy(d_vg, vg.data(), vg.size() * sizeof(double),
                              cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemcpy(d_pos, pos.data(), 3 * n_atoms * sizeof(double),
                              cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemcpy(d_species, species.data(),
                              species.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemset(d_F, 0, 3 * n_atoms * sizeof(double)));

    cuda::pw::internal::accumulate_local_forces_device(d_rho, d_gx, d_gy, d_gz,
                                                       d_vg, d_pos, d_species,
                                                       ng, n_atoms, omega, d_F,
                                                       /*stream=*/nullptr);
    ATV_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> F_dev(3 * n_atoms);
    ATV_CUDA_CHECK(cudaMemcpy(F_dev.data(), d_F, 3 * n_atoms * sizeof(double),
                              cudaMemcpyDeviceToHost));

    double max_diff = 0.0;
    for (int i = 0; i < 3 * n_atoms; ++i) {
        max_diff =
            std::max(max_diff, std::abs(F_dev[static_cast<std::size_t>(i)]
                                        - F_host[static_cast<std::size_t>(i)]));
    }
    // Bit-level for a pure arithmetic kernel (atomicAdd order on 2 atoms
    // is deterministic enough at double for this tiny ng).
    EXPECT_LT(max_diff, 1e-12) << " max|ΔF|=" << max_diff;

    cudaFree(d_rho);
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_gz);
    cudaFree(d_vg);
    cudaFree(d_species);
    cudaFree(d_pos);
    cudaFree(d_F);
    (void)kBlock;
}

TEST(CudaPWForces, PublicApiRejectsEmptyFrame) {
    cpu::Frame f(0);
    cuda::pw::PW method{{std::make_shared<cpu::pw::GthPseudopotential>(
                            molrs::Element::H, "GTH-PBE-q1")},
                        /*ecut_ry=*/4.0,
                        /*k_mesh=*/1,
                        cpu::xc::XCType::LSDA};
    method.tol = 1e-4;
    method.max_iter = 4;
    EXPECT_THROW(method.forces(f), std::invalid_argument);
}

// Spin-PBE forces are on the device whitelist (exchange_correlation_spin
// already runs xc_pbe_spin_grid). Empty-frame still rejects before SCF;
// the message must not claim PBE is unsupported.
TEST(CudaPWForces, PublicApiAcceptsSpinPbeOnEmptyFrameGate) {
    cpu::Frame f(0);
    cuda::pw::PW method{{std::make_shared<cpu::pw::GthPseudopotential>(
                            molrs::Element::H, "GTH-PBE-q1")},
                        /*ecut_ry=*/30.0,
                        /*k_mesh=*/4,
                        cpu::xc::XCType::PBE};
    method.spin_polarized = true;
    method.multiplicity = 2;
    try {
        (void)method.forces(f);
        FAIL() << "expected empty-frame throw";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("nonempty"), std::string::npos) << "msg=" << msg;
        EXPECT_EQ(msg.find("LSDA only"), std::string::npos)
            << "spin-PBE must not hit the XC reject; msg=" << msg;
    }
}

TEST(CudaPWForces, PublicApiAcceptsSpinSmearOnEmptyFrameGate) {
    cpu::Frame f(0);
    cuda::pw::PW method{{std::make_shared<cpu::pw::GthPseudopotential>(
                            molrs::Element::H, "GTH-PBE-q1")},
                        /*ecut_ry=*/30.0,
                        /*k_mesh=*/4,
                        cpu::xc::XCType::LSDA};
    method.spin_polarized = true;
    method.multiplicity = 2;
    method.smearing = ::ATV::electronic::Smearing::fermi_dirac;
    method.degauss = 0.01;
    try {
        (void)method.forces(f);
        FAIL() << "expected empty-frame throw";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("nonempty"), std::string::npos) << "msg=" << msg;
        EXPECT_EQ(msg.find("no force path"), std::string::npos) << msg;
    }
}

// PBE is on the force-path XC whitelist (DeviceDensity + xc_pbe). Empty-frame
// still rejects before SCF; the message must not claim PBE is unsupported.
TEST(CudaPWForces, PublicApiAcceptsPbeOnEmptyFrameGate) {
    cpu::Frame f(0);
    cuda::pw::PW method{{std::make_shared<cpu::pw::GthPseudopotential>(
                            molrs::Element::H, "GTH-PBE-q1")},
                        /*ecut_ry=*/4.0,
                        /*k_mesh=*/1,
                        cpu::xc::XCType::PBE};
    method.tol = 1e-4;
    method.max_iter = 4;
    try {
        (void)method.forces(f);
        FAIL() << "expected empty-frame throw";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("nonempty"), std::string::npos) << "msg=" << msg;
        EXPECT_EQ(msg.find("no device path"), std::string::npos)
            << "PBE must not hit the XC whitelist reject; msg=" << msg;
    }
}

/// @file tests/cuda/gpw/test_device_poisson.cu
/// @brief CUDA GPW Γ Poisson foothold vs same-settings CPU `solve_poisson`.
///
/// Unit only: plane-wave round-trip + water-like multi-center Gaussian density
/// Hartree energy against the CPU oracle. Not a full GPW SCF product.
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/poisson_fft.h"
#include "atomiverse/cuda/gpw/internal/poisson_fft.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

using ATV::cpu::gpw::PoissonResult;
using ATV::cpu::gpw::solve_poisson;
using ATV::cpu::gpw::UniformGrid;
using ATV::cuda::gpw::internal::GpwGridDesc;
using ATV::cuda::gpw::internal::PoissonFftDevice;

namespace {

UniformGrid cubic_grid(int n, double L) {
    UniformGrid g;
    for (int d = 0; d < 3; ++d) {
        g.n[d] = n;
        g.L[d] = L;
        g.origin[d] = 0.0;
    }
    return g;
}

GpwGridDesc to_desc(const UniformGrid& g) {
    GpwGridDesc d;
    for (int i = 0; i < 3; ++i) {
        d.n[i] = g.n[i];
        d.L[i] = g.L[i];
    }
    return d;
}

struct Gvec {
    double g[3];
    double g2;
};
Gvec gvec(const UniformGrid& grid, const int k[3]) {
    Gvec out;
    out.g2 = 0.0;
    for (int d = 0; d < 3; ++d) {
        out.g[d] = 2.0 * M_PI * k[d] / grid.L[d];
        out.g2 += out.g[d] * out.g[d];
    }
    return out;
}

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": "
                                 + cudaGetErrorString(e));
}

void* dmalloc(size_t bytes) {
    void* p = nullptr;
    check_cuda(cudaMalloc(&p, bytes), "cudaMalloc");
    return p;
}

/// Device solve with host V + e_H mirrors. `d_rho` and `d_v` are distinct.
struct DeviceSolve {
    std::vector<double> v;
    double e_hartree = 0.0;
};

DeviceSolve run_device(const UniformGrid& grid, const std::vector<double>& rho) {
    const long ng = grid.npts();
    if (static_cast<long>(rho.size()) != ng)
        throw std::runtime_error("run_device: rho size mismatch");

    auto* d_rho = static_cast<double*>(
        dmalloc(static_cast<size_t>(ng) * sizeof(double)));
    auto* d_v = static_cast<double*>(
        dmalloc(static_cast<size_t>(ng) * sizeof(double)));
    check_cuda(cudaMemcpy(d_rho, rho.data(),
                          static_cast<size_t>(ng) * sizeof(double),
                          cudaMemcpyHostToDevice),
               "H2D rho");

    PoissonFftDevice solver(to_desc(grid));
    solver.solve(d_rho, d_v);
    check_cuda(cudaDeviceSynchronize(), "sync after solve");

    DeviceSolve out;
    out.v.resize(static_cast<std::size_t>(ng));
    check_cuda(cudaMemcpy(out.v.data(), d_v,
                          static_cast<size_t>(ng) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "D2H v");
    check_cuda(cudaMemcpy(&out.e_hartree, solver.last_e_hartree_device(),
                          sizeof(double), cudaMemcpyDeviceToHost),
               "D2H e");

    cudaFree(d_rho);
    cudaFree(d_v);
    return out;
}

} // namespace

// Single plane wave: V=cos(G·r), ρ=(|G|²/4π)V → CUDA matches CPU V and e_H.
TEST(CudaGpwDevicePoisson, PlaneWaveMatchesCpu) {
    UniformGrid grid = cubic_grid(16, 10.0);
    const int k[3] = {1, 0, 0};
    const Gvec G = gvec(grid, k);
    const long ng = grid.npts();
    std::vector<double> rho(static_cast<std::size_t>(ng));
    const double pref = G.g2 / (4.0 * M_PI);
    for (long i = 0; i < ng; ++i) {
        double r[3];
        grid.point(i, r);
        const double phase = G.g[0] * r[0] + G.g[1] * r[1] + G.g[2] * r[2];
        rho[static_cast<std::size_t>(i)] = pref * std::cos(phase);
    }

    PoissonResult cpu = solve_poisson(grid, rho);
    DeviceSolve dev = run_device(grid, rho);

    double max_dv = 0.0;
    for (long i = 0; i < ng; ++i)
        max_dv = std::max(max_dv, std::abs(dev.v[static_cast<std::size_t>(i)]
                                           - cpu.v_r[static_cast<std::size_t>(i)]));
    EXPECT_LT(max_dv, 1e-10) << "max |V_cuda - V_cpu|";
    EXPECT_NEAR(dev.e_hartree, cpu.e_hartree,
                1e-12 * std::max(1.0, std::abs(cpu.e_hartree)));
    EXPECT_GT(dev.e_hartree, 0.0);
}

// Superposition of two wavevectors (different axes) vs CPU.
TEST(CudaGpwDevicePoisson, SuperpositionMatchesCpu) {
    UniformGrid grid = cubic_grid(20, 12.0);
    const int k1[3] = {1, 0, 0};
    const int k2[3] = {0, 2, 0};
    const Gvec G1 = gvec(grid, k1);
    const Gvec G2 = gvec(grid, k2);
    const long ng = grid.npts();
    std::vector<double> rho(static_cast<std::size_t>(ng));
    for (long i = 0; i < ng; ++i) {
        double r[3];
        grid.point(i, r);
        const double p1 = G1.g[0] * r[0] + G1.g[1] * r[1] + G1.g[2] * r[2];
        const double p2 = G2.g[0] * r[0] + G2.g[1] * r[1] + G2.g[2] * r[2];
        rho[static_cast<std::size_t>(i)] =
            (G1.g2 / (4.0 * M_PI)) * std::cos(p1)
            + 0.5 * (G2.g2 / (4.0 * M_PI)) * std::cos(p2);
    }
    PoissonResult cpu = solve_poisson(grid, rho);
    DeviceSolve dev = run_device(grid, rho);
    double max_dv = 0.0;
    for (long i = 0; i < ng; ++i)
        max_dv = std::max(max_dv, std::abs(dev.v[static_cast<std::size_t>(i)]
                                           - cpu.v_r[static_cast<std::size_t>(i)]));
    EXPECT_LT(max_dv, 1e-10);
    EXPECT_NEAR(dev.e_hartree, cpu.e_hartree,
                1e-12 * std::max(1.0, std::abs(cpu.e_hartree)));
}

// Constant density is pure G=0 → V==0, e_H==0 on both backends.
TEST(CudaGpwDevicePoisson, ConstantDensityZero) {
    UniformGrid grid = cubic_grid(12, 8.0);
    std::vector<double> rho(static_cast<std::size_t>(grid.npts()), 0.37);
    PoissonResult cpu = solve_poisson(grid, rho);
    DeviceSolve dev = run_device(grid, rho);
    for (double v : dev.v)
        EXPECT_NEAR(v, 0.0, 1e-10);
    EXPECT_NEAR(dev.e_hartree, 0.0, 1e-10);
    EXPECT_NEAR(dev.e_hartree, cpu.e_hartree, 1e-12);
}

// Water-like three Gaussian centers in a cubic cell — Hartree energy vs CPU.
// Twin: Γ water density class (O + 2 H s-Gaussians), not full GPW SCF energy.
TEST(CudaGpwDevicePoisson, WaterLikeGaussianHartreeMatchesCpu) {
    // Cell ~ water vacuum box class (Bohr); modest grid for unit timeout.
    UniformGrid grid = cubic_grid(24, 12.0);
    const long ng = grid.npts();
    std::vector<double> rho(static_cast<std::size_t>(ng), 0.0);

    // Approximate water geometry (Bohr) shifted into the cell interior.
    struct Center {
        double c[3];
        double alpha;
        double amp;
    };
    const double half_angle = 0.5 * 104.5 * M_PI / 180.0;
    const double rOH = 1.43;
    const Center centers[] = {
        {{6.0, 6.0, 6.0}, 1.0, 8.0}, // O-like
        {{6.0 + rOH * std::cos(half_angle), 6.0 + rOH * std::sin(half_angle),
          6.0},
         0.5, 1.0}, // H
        {{6.0 + rOH * std::cos(-half_angle), 6.0 + rOH * std::sin(-half_angle),
          6.0},
         0.5, 1.0}, // H
    };

    for (long i = 0; i < ng; ++i) {
        double r[3];
        grid.point(i, r);
        double s = 0.0;
        for (const auto& cn : centers) {
            double d2 = 0.0;
            for (int d = 0; d < 3; ++d) {
                double dx = r[d] - cn.c[d];
                dx -= grid.L[d] * std::round(dx / grid.L[d]);
                d2 += dx * dx;
            }
            s += cn.amp * std::exp(-cn.alpha * d2);
        }
        rho[static_cast<std::size_t>(i)] = s;
    }

    // Neutralize mean density so G=0 drop does not dominate the comparison
    // (periodic Poisson convention). Same on both backends.
    double mean = 0.0;
    for (double x : rho)
        mean += x;
    mean /= static_cast<double>(ng);
    for (double& x : rho)
        x -= mean;

    PoissonResult cpu = solve_poisson(grid, rho);
    DeviceSolve dev = run_device(grid, rho);

    double max_dv = 0.0;
    for (long i = 0; i < ng; ++i)
        max_dv = std::max(max_dv, std::abs(dev.v[static_cast<std::size_t>(i)]
                                           - cpu.v_r[static_cast<std::size_t>(i)]));
    // cuFFT vs FFTW: not bit-identical; tight absolute bar for this grid size.
    EXPECT_LT(max_dv, 1e-9) << "max |V_cuda - V_cpu| water-like";
    EXPECT_NEAR(dev.e_hartree, cpu.e_hartree,
                1e-10 * std::max(1.0, std::abs(cpu.e_hartree)));
    EXPECT_GT(std::abs(cpu.e_hartree), 1e-8); // non-trivial twin
}

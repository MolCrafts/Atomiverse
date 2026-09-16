/// @file regressions/gpw/cuda-gpw-ks-multigrid/reg_cuda-gpw-ks-multigrid.cu
/// @brief END-TO-END dual-gate: CUDA multi-grid KS chain vs CPU Γ multigrid
///        fragment (prolong/restrict + ks_grid_build_rks + continuous products).
///
/// Gates:
///   1. CPU path produces finite e_H, e_xc, V_eff, integrate acc;
///   2. CUDA path matches within stated stage tols (Hartree).
///
/// Not a full GPW method. See REFERENCE.md.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/ks_grid_build.h"
#include "atomiverse/cpu/gpw/internal/multigrid.h"
#include "atomiverse/cuda/gpw/internal/ks_chain.cuh"
#include "atomiverse/cuda/gpw/internal/multigrid_transfer.cuh"
#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"

using ATV::cpu::gpw::UniformGrid;
using ATV::cpu::gpw::XcKind;
using ATV::cpu::gpw::internal::ks_grid_build_rks;
using ATV::cpu::gpw::prolong_fft;
using ATV::cpu::gpw::restrict_fft;
using ATV::cuda::gpw::internal::CollocateGridDesc;
using ATV::cuda::gpw::internal::DeviceProduct;
using ATV::cuda::gpw::internal::KsMultiGridChainDevice;
using ATV::cuda::gpw::internal::MultigridDesc;

namespace {

constexpr double kERel = 1.0e-9;
constexpr double kVAbs = 1.0e-11;
constexpr double kVRel = 1.0e-8;
constexpr double kAAbs = 1.0e-11;
constexpr double kARel = 1.0e-8;

void die(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
}

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

UniformGrid cubic_grid(int n, double L) {
    UniformGrid g;
    for (int d = 0; d < 3; ++d) {
        g.n[d] = n;
        g.L[d] = L;
        g.origin[d] = 0.0;
    }
    return g;
}

CollocateGridDesc to_desc(const UniformGrid& g) {
    CollocateGridDesc d;
    for (int i = 0; i < 3; ++i) {
        d.n[i] = g.n[i];
        d.L[i] = g.L[i];
        d.origin[i] = g.origin[i];
    }
    return d;
}

DeviceProduct make_ss(double p, const double pc[3], double weight) {
    DeviceProduct q;
    q.p = p;
    q.weight = weight;
    for (int d = 0; d < 3; ++d) {
        q.pc[d] = pc[d];
        q.ra[d] = pc[d];
        q.rb[d] = pc[d];
        q.la[d] = 0;
        q.lb[d] = 0;
    }
    return q;
}

double host_shape(const DeviceProduct& q, const double r[3]) {
    if (q.weight == 0.0)
        return 0.0;
    const double dx = r[0] - q.pc[0];
    const double dy = r[1] - q.pc[1];
    const double dz = r[2] - q.pc[2];
    return q.weight * std::exp(-q.p * (dx * dx + dy * dy + dz * dz));
}

double host_shape_bare(const DeviceProduct& q, const double r[3]) {
    const double dx = r[0] - q.pc[0];
    const double dy = r[1] - q.pc[1];
    const double dz = r[2] - q.pc[2];
    return std::exp(-q.p * (dx * dx + dy * dy + dz * dz));
}

std::vector<double> host_collocate(const UniformGrid& grid,
                                   const DeviceProduct& q) {
    std::vector<double> rho(static_cast<std::size_t>(grid.npts()), 0.0);
    for (long i = 0; i < grid.npts(); ++i) {
        double r[3];
        grid.point(i, r);
        rho[static_cast<std::size_t>(i)] = host_shape(q, r);
    }
    return rho;
}

double host_integrate(const UniformGrid& grid, const DeviceProduct& q,
                      const std::vector<double>& V) {
    double acc = 0.0;
    for (long i = 0; i < grid.npts(); ++i) {
        double r[3];
        grid.point(i, r);
        acc += V[static_cast<std::size_t>(i)] * host_shape_bare(q, r);
    }
    return acc;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

double max_abs(const std::vector<double>& a) {
    double m = 0.0;
    for (double v : a)
        m = std::max(m, std::abs(v));
    return m;
}

bool near(double a, double b, double rel) {
    return std::abs(a - b) <= rel * std::max(1.0, std::abs(b));
}

} // namespace

int main(int argc, char** argv) {
    bool want_cpu = true;
    bool want_cuda = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device=cpu") == 0) {
            want_cuda = false;
        } else if (std::strcmp(argv[i], "--device=cuda") == 0) {
            want_cpu = false;
        } else if (std::strcmp(argv[i], "--device=both") == 0) {
            want_cpu = want_cuda = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf("usage: %s [--device=both|cpu|cuda]\n", argv[0]);
            return 0;
        }
    }

    const double L = 8.0;
    const UniformGrid fine = cubic_grid(16, L);
    const UniformGrid coarse = cubic_grid(8, L);
    const double pc0[3] = {4.0, 4.0, 4.0};
    const double pc1[3] = {3.5, 4.2, 4.5};
    const DeviceProduct q0 = make_ss(4.0, pc0, 1.1);
    const DeviceProduct q1 = make_ss(0.8, pc1, 0.7);

    const auto rho0 = host_collocate(fine, q0);
    const auto rho1 = host_collocate(coarse, q1);
    const auto up = prolong_fft(coarse, fine, rho1, true);
    std::vector<double> rho = rho0;
    for (std::size_t i = 0; i < rho.size(); ++i)
        rho[i] += up[i];
    std::vector<double> n_ion(rho.size(), 0.0);
    std::vector<double> v_loc;
    const auto cpu =
        ks_grid_build_rks(fine, rho, n_ion, v_loc, XcKind::PADE, false, false,
                          false, false);
    const auto V1 = restrict_fft(fine, coarse, cpu.veff, true);
    const double acc0 = host_integrate(fine, q0, cpu.veff);
    const double acc1 = host_integrate(coarse, q1, V1);

    if (!std::isfinite(cpu.e_hartree) || !std::isfinite(cpu.e_xc))
        die("CPU energies not finite");
    if (max_abs(cpu.veff) < 1e-12)
        die("CPU V_eff near zero");
    if (want_cpu)
        std::printf("CPU  e_H=%.12e e_xc=%.12e acc0=%.12e acc1=%.12e\n",
                    cpu.e_hartree, cpu.e_xc, acc0, acc1);

    if (!want_cuda) {
        std::printf("PASS (cpu-only)\n");
        return 0;
    }

    MultigridDesc mg;
    mg.levels = {to_desc(fine), to_desc(coarse)};
    KsMultiGridChainDevice chain(mg);

    DeviceProduct *d_p0 = nullptr, *d_p1 = nullptr;
    double *d_a0 = nullptr, *d_a1 = nullptr;
    check_cuda(cudaMalloc(&d_p0, sizeof(DeviceProduct)), "malloc p0");
    check_cuda(cudaMalloc(&d_p1, sizeof(DeviceProduct)), "malloc p1");
    check_cuda(cudaMalloc(&d_a0, sizeof(double)), "malloc a0");
    check_cuda(cudaMalloc(&d_a1, sizeof(double)), "malloc a1");
    check_cuda(cudaMemcpy(d_p0, &q0, sizeof(DeviceProduct),
                          cudaMemcpyHostToDevice),
               "H2D p0");
    check_cuda(cudaMemcpy(d_p1, &q1, sizeof(DeviceProduct),
                          cudaMemcpyHostToDevice),
               "H2D p1");
    const DeviceProduct* pp[2] = {d_p0, d_p1};
    const int np[2] = {1, 1};
    double* ap[2] = {d_a0, d_a1};
    const auto e = chain.run(pp, np, nullptr, nullptr, ap);

    std::vector<double> veff(static_cast<std::size_t>(fine.npts()));
    check_cuda(cudaMemcpy(veff.data(), chain.fine_chain().d_veff(),
                          veff.size() * sizeof(double), cudaMemcpyDeviceToHost),
               "D2H veff");
    double a0 = 0.0, a1 = 0.0;
    check_cuda(cudaMemcpy(&a0, d_a0, sizeof(double), cudaMemcpyDeviceToHost),
               "D2H a0");
    check_cuda(cudaMemcpy(&a1, d_a1, sizeof(double), cudaMemcpyDeviceToHost),
               "D2H a1");
    cudaFree(d_p0);
    cudaFree(d_p1);
    cudaFree(d_a0);
    cudaFree(d_a1);

    std::printf("CUDA e_H=%.12e e_xc=%.12e acc0=%.12e acc1=%.12e\n", e.e_hartree,
                e.e_xc, a0, a1);

    if (!near(e.e_hartree, cpu.e_hartree, kERel))
        die("e_hartree mismatch");
    if (!near(e.e_xc, cpu.e_xc, kERel))
        die("e_xc mismatch");
    const double peak_v = max_abs(cpu.veff);
    const double dv = max_abs_diff(veff, cpu.veff);
    if (dv > kVRel * std::max(1.0, peak_v) + kVAbs) {
        std::fprintf(stderr, "FAIL: max|ΔV_eff|=%.3e peak=%.3e\n", dv, peak_v);
        return 1;
    }
    if (std::abs(a0 - acc0) > kARel * std::max(1.0, std::abs(acc0)) + kAAbs)
        die("acc0 mismatch");
    if (std::abs(a1 - acc1) > kARel * std::max(1.0, std::abs(acc1)) + kAAbs)
        die("acc1 mismatch");

    std::printf("PASS max|ΔV_eff|=%.3e\n", dv);
    return 0;
}

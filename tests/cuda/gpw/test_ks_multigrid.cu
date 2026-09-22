/// @file tests/cuda/gpw/test_ks_multigrid.cu
/// @brief CUDA multi-grid KS chain vs CPU Γ multigrid collint fragment.
///
/// Unit only (Γ small grids): products on fine + coarse levels → spectral
/// prolong → Poisson + Padé LDA on fine → spectral restrict → integrate.
/// Continuous product shapes (same footholds as single-grid chain). Not a
/// full GPW SCF product.
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/ks_grid_build.h"
#include "atomiverse/cpu/gpw/internal/multigrid.h"
#include "atomiverse/cuda/gpw/internal/ks_chain.cuh"
#include "atomiverse/cuda/gpw/internal/multigrid_transfer.cuh"
#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using ATV::cpu::gpw::prolong_fft;
using ATV::cpu::gpw::restrict_fft;
using ATV::cpu::gpw::UniformGrid;
using ATV::cpu::gpw::XcKind;
using ATV::cpu::gpw::internal::ks_grid_build_rks;
using ATV::cuda::gpw::internal::CollocateGridDesc;
using ATV::cuda::gpw::internal::DeviceProduct;
using ATV::cuda::gpw::internal::KsChainEnergies;
using ATV::cuda::gpw::internal::KsMultiGridChainDevice;
using ATV::cuda::gpw::internal::MultigridDesc;
using ATV::cuda::gpw::internal::MultigridTransferDevice;

namespace {

void check_cuda(cudaError_t e, const char* what) {
    ASSERT_EQ(e, cudaSuccess) << what << ": " << cudaGetErrorString(e);
}

void* dmalloc(size_t bytes) {
    void* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess);
    return p;
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
    // Continuous full-grid foothold (host_collocate oracle is non-sphere).
    // Production product path sets periodic=true (sphere-window PBC).
    d.periodic = false;
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

double host_product_shape(const DeviceProduct& q, const double r[3]) {
    if (q.weight == 0.0)
        return 0.0;
    const double dx = r[0] - q.pc[0];
    const double dy = r[1] - q.pc[1];
    const double dz = r[2] - q.pc[2];
    double v = q.weight * std::exp(-q.p * (dx * dx + dy * dy + dz * dz));
    for (int d = 0; d < 3; ++d) {
        double da = r[d] - q.ra[d];
        double db = r[d] - q.rb[d];
        for (int i = 0; i < q.la[d]; ++i)
            v *= da;
        for (int i = 0; i < q.lb[d]; ++i)
            v *= db;
    }
    return v;
}

double host_product_shape_bare(const DeviceProduct& q, const double r[3]) {
    const double dx = r[0] - q.pc[0];
    const double dy = r[1] - q.pc[1];
    const double dz = r[2] - q.pc[2];
    double v = std::exp(-q.p * (dx * dx + dy * dy + dz * dz));
    for (int d = 0; d < 3; ++d) {
        double da = r[d] - q.ra[d];
        double db = r[d] - q.rb[d];
        for (int i = 0; i < q.la[d]; ++i)
            v *= da;
        for (int i = 0; i < q.lb[d]; ++i)
            v *= db;
    }
    return v;
}

std::vector<double> host_collocate(const UniformGrid& grid,
                                   const std::vector<DeviceProduct>& prods) {
    const long ng = grid.npts();
    std::vector<double> rho(static_cast<std::size_t>(ng), 0.0);
    for (long i = 0; i < ng; ++i) {
        double r[3];
        grid.point(i, r);
        for (const auto& q : prods)
            rho[static_cast<std::size_t>(i)] += host_product_shape(q, r);
    }
    return rho;
}

std::vector<double> host_integrate(const UniformGrid& grid,
                                   const std::vector<DeviceProduct>& prods,
                                   const std::vector<double>& V) {
    const long ng = grid.npts();
    std::vector<double> acc(prods.size(), 0.0);
    for (long i = 0; i < ng; ++i) {
        double r[3];
        grid.point(i, r);
        const double v = V[static_cast<std::size_t>(i)];
        for (std::size_t q = 0; q < prods.size(); ++q)
            acc[q] += v * host_product_shape_bare(prods[q], r);
    }
    return acc;
}

double max_abs_diff(const std::vector<double>& a,
                    const std::vector<double>& b) {
    EXPECT_EQ(a.size(), b.size());
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

std::vector<double> fill_rand(long n, unsigned seed) {
    std::vector<double> v(static_cast<std::size_t>(n));
    unsigned s = seed;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = (s >> 8) / 16777216.0 - 0.5;
    }
    return v;
}

} // namespace

// ── Spectral transfer vs CPU prolong_fft / restrict_fft ───────────────────
TEST(CudaGpwMultigridTransfer, ProlongRestrictMatchCpu) {
    const UniformGrid fine = cubic_grid(16, 8.0);
    const UniformGrid coarse = cubic_grid(8, 8.0);
    const auto fc = fill_rand(coarse.npts(), 21u);
    const auto vf = fill_rand(fine.npts(), 7u);

    const auto cpu_p = prolong_fft(coarse, fine, fc, /*periodic=*/true);
    const auto cpu_r = restrict_fft(fine, coarse, vf, /*periodic=*/true);

    MultigridTransferDevice xfer(to_desc(coarse), to_desc(fine));
    auto* d_c = static_cast<double*>(
        dmalloc(static_cast<size_t>(coarse.npts()) * sizeof(double)));
    auto* d_f = static_cast<double*>(
        dmalloc(static_cast<size_t>(fine.npts()) * sizeof(double)));
    check_cuda(cudaMemcpy(d_c, fc.data(),
                          static_cast<size_t>(coarse.npts()) * sizeof(double),
                          cudaMemcpyHostToDevice),
               "H2D coarse");
    xfer.prolong(d_c, d_f);
    std::vector<double> dev_p(static_cast<std::size_t>(fine.npts()));
    check_cuda(cudaMemcpy(dev_p.data(), d_f,
                          static_cast<size_t>(fine.npts()) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "D2H prolong");
    check_cuda(cudaMemcpy(d_f, vf.data(),
                          static_cast<size_t>(fine.npts()) * sizeof(double),
                          cudaMemcpyHostToDevice),
               "H2D fine");
    xfer.restrict(d_f, d_c);
    std::vector<double> dev_r(static_cast<std::size_t>(coarse.npts()));
    check_cuda(cudaMemcpy(dev_r.data(), d_c,
                          static_cast<size_t>(coarse.npts()) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "D2H restrict");
    cudaFree(d_c);
    cudaFree(d_f);

    const double peak_p = max_abs(cpu_p);
    const double peak_r = max_abs(cpu_r);
    EXPECT_LE(max_abs_diff(dev_p, cpu_p), 1e-10 * std::max(1.0, peak_p) + 1e-12)
        << "prolong";
    EXPECT_LE(max_abs_diff(dev_r, cpu_r), 1e-10 * std::max(1.0, peak_r) + 1e-12)
        << "restrict";
}

// ── Two-level product MG chain vs CPU fragment ────────────────────────────
TEST(CudaGpwKsMultigrid, TwoLevelMatchesCpuFragment) {
    const double L = 8.0;
    const UniformGrid fine = cubic_grid(16, L);
    const UniformGrid coarse = cubic_grid(8, L);

    // Tight product on fine; diffuse product on coarse.
    const double pc_f[3] = {4.0, 4.0, 4.0};
    const double pc_c[3] = {3.5, 4.2, 4.5};
    std::vector<DeviceProduct> prods0 = {make_ss(/*p=*/4.0, pc_f, /*w=*/1.1)};
    std::vector<DeviceProduct> prods1 = {make_ss(/*p=*/0.8, pc_c, /*w=*/0.7)};

    // CPU: collocate → prolong → KS → restrict → integrate
    const auto rho0 = host_collocate(fine, prods0);
    const auto rho1 = host_collocate(coarse, prods1);
    const auto up = prolong_fft(coarse, fine, rho1, /*periodic=*/true);
    std::vector<double> rho_fine = rho0;
    for (std::size_t i = 0; i < rho_fine.size(); ++i)
        rho_fine[i] += up[i];

    const std::size_t n = rho_fine.size();
    std::vector<double> n_ion(n, 0.0);
    std::vector<double> v_loc;
    const auto cpu =
        ks_grid_build_rks(fine, rho_fine, n_ion, v_loc, XcKind::PADE,
                          /*nonperiodic=*/false, false, false, false);
    const auto V1 = restrict_fft(fine, coarse, cpu.veff, /*periodic=*/true);
    const auto acc0 = host_integrate(fine, prods0, cpu.veff);
    const auto acc1 = host_integrate(coarse, prods1, V1);

    // Device multi-grid chain
    MultigridDesc mg;
    mg.levels = {to_desc(fine), to_desc(coarse)};
    KsMultiGridChainDevice chain(mg);

    auto* d_p0 = static_cast<DeviceProduct*>(dmalloc(sizeof(DeviceProduct)));
    auto* d_p1 = static_cast<DeviceProduct*>(dmalloc(sizeof(DeviceProduct)));
    check_cuda(cudaMemcpy(d_p0, prods0.data(), sizeof(DeviceProduct),
                          cudaMemcpyHostToDevice),
               "H2D p0");
    check_cuda(cudaMemcpy(d_p1, prods1.data(), sizeof(DeviceProduct),
                          cudaMemcpyHostToDevice),
               "H2D p1");
    auto* d_a0 = static_cast<double*>(dmalloc(sizeof(double)));
    auto* d_a1 = static_cast<double*>(dmalloc(sizeof(double)));

    const DeviceProduct* prods_ptr[2] = {d_p0, d_p1};
    const int nprod[2] = {1, 1};
    double* acc_ptr[2] = {d_a0, d_a1};

    const KsChainEnergies e =
        chain.run(prods_ptr, nprod, nullptr, nullptr, acc_ptr);

    std::vector<double> veff(n);
    check_cuda(cudaMemcpy(veff.data(), chain.fine_chain().d_veff(),
                          n * sizeof(double), cudaMemcpyDeviceToHost),
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

    // Energies
    EXPECT_NEAR(e.e_hartree, cpu.e_hartree,
                1e-9 * std::max(1.0, std::abs(cpu.e_hartree)));
    EXPECT_NEAR(e.e_xc, cpu.e_xc, 1e-9 * std::max(1.0, std::abs(cpu.e_xc)));
    EXPECT_DOUBLE_EQ(e.e_locgrid, 0.0);

    // V_eff on fine
    const double peak_v = max_abs(cpu.veff);
    ASSERT_GT(peak_v, 1e-8);
    EXPECT_LE(max_abs_diff(veff, cpu.veff),
              1e-8 * std::max(1.0, peak_v) + 1e-11)
        << "max|ΔV_eff| peak=" << peak_v;

    // Integrate acc
    EXPECT_NEAR(a0, acc0[0], 1e-8 * std::max(1.0, std::abs(acc0[0])) + 1e-11);
    EXPECT_NEAR(a1, acc1[0], 1e-8 * std::max(1.0, std::abs(acc1[0])) + 1e-11);
}

// ── Single-level multigrid reduces to fine-only path ──────────────────────
TEST(CudaGpwKsMultigrid, SingleLevelMatchesFineOnly) {
    const UniformGrid fine = cubic_grid(12, 6.0);
    const double pc[3] = {3.0, 3.0, 3.0};
    std::vector<DeviceProduct> prods = {make_ss(2.5, pc, 1.0)};

    const auto rho = host_collocate(fine, prods);
    std::vector<double> n_ion(rho.size(), 0.0);
    std::vector<double> v_loc;
    const auto cpu = ks_grid_build_rks(fine, rho, n_ion, v_loc, XcKind::PADE,
                                       false, false, false, false);
    const auto acc = host_integrate(fine, prods, cpu.veff);

    MultigridDesc mg;
    mg.levels = {to_desc(fine)};
    KsMultiGridChainDevice chain(mg);

    auto* d_p = static_cast<DeviceProduct*>(dmalloc(sizeof(DeviceProduct)));
    check_cuda(cudaMemcpy(d_p, prods.data(), sizeof(DeviceProduct),
                          cudaMemcpyHostToDevice),
               "H2D");
    auto* d_a = static_cast<double*>(dmalloc(sizeof(double)));
    const DeviceProduct* pp[1] = {d_p};
    const int np[1] = {1};
    double* ap[1] = {d_a};
    const auto e = chain.run(pp, np, nullptr, nullptr, ap);
    double a = 0.0;
    check_cuda(cudaMemcpy(&a, d_a, sizeof(double), cudaMemcpyDeviceToHost),
               "D2H a");
    cudaFree(d_p);
    cudaFree(d_a);

    EXPECT_NEAR(e.e_hartree, cpu.e_hartree,
                1e-10 * std::max(1.0, std::abs(cpu.e_hartree)));
    EXPECT_NEAR(e.e_xc, cpu.e_xc, 1e-10 * std::max(1.0, std::abs(cpu.e_xc)));
    EXPECT_NEAR(a, acc[0], 1e-9 * std::max(1.0, std::abs(acc[0])) + 1e-12);
}

// ── n_ion + V_loc on fine under 2-level MG ────────────────────────────────
TEST(CudaGpwKsMultigrid, TwoLevelWithIonAndVloc) {
    const double L = 10.0;
    const UniformGrid fine = cubic_grid(20, L);
    const UniformGrid coarse = cubic_grid(10, L);

    const double c0[3] = {5.0, 5.0, 5.0};
    const double c1[3] = {4.0, 5.5, 5.0};
    std::vector<DeviceProduct> prods0 = {make_ss(2.0, c0, 0.9)};
    std::vector<DeviceProduct> prods1 = {make_ss(0.6, c1, 0.4)};

    const auto rho0 = host_collocate(fine, prods0);
    const auto rho1 = host_collocate(coarse, prods1);
    const auto up = prolong_fft(coarse, fine, rho1, true);
    std::vector<double> rho = rho0;
    for (std::size_t i = 0; i < rho.size(); ++i)
        rho[i] += up[i];

    std::vector<double> n_ion(rho.size(), 0.0);
    std::vector<double> v_loc(rho.size(), -0.04);
    {
        const double alpha = 2.5;
        const double pref = std::pow(alpha / M_PI, 1.5);
        for (std::size_t i = 0; i < n_ion.size(); ++i) {
            double r[3];
            fine.point(static_cast<long>(i), r);
            const double dx = r[0] - 5.0;
            const double dy = r[1] - 5.0;
            const double dz = r[2] - 5.0;
            n_ion[i] = pref * std::exp(-alpha * (dx * dx + dy * dy + dz * dz));
        }
    }

    const auto cpu = ks_grid_build_rks(fine, rho, n_ion, v_loc, XcKind::PADE,
                                       false, false, false, false);
    const auto V1 = restrict_fft(fine, coarse, cpu.veff, true);
    const auto acc0 = host_integrate(fine, prods0, cpu.veff);
    const auto acc1 = host_integrate(coarse, prods1, V1);

    MultigridDesc mg;
    mg.levels = {to_desc(fine), to_desc(coarse)};
    KsMultiGridChainDevice chain(mg);

    auto* d_p0 = static_cast<DeviceProduct*>(dmalloc(sizeof(DeviceProduct)));
    auto* d_p1 = static_cast<DeviceProduct*>(dmalloc(sizeof(DeviceProduct)));
    check_cuda(cudaMemcpy(d_p0, prods0.data(), sizeof(DeviceProduct),
                          cudaMemcpyHostToDevice),
               "H2D p0");
    check_cuda(cudaMemcpy(d_p1, prods1.data(), sizeof(DeviceProduct),
                          cudaMemcpyHostToDevice),
               "H2D p1");
    auto* d_ion = static_cast<double*>(dmalloc(n_ion.size() * sizeof(double)));
    auto* d_vloc = static_cast<double*>(dmalloc(v_loc.size() * sizeof(double)));
    check_cuda(cudaMemcpy(d_ion, n_ion.data(), n_ion.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "H2D ion");
    check_cuda(cudaMemcpy(d_vloc, v_loc.data(), v_loc.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "H2D vloc");
    auto* d_a0 = static_cast<double*>(dmalloc(sizeof(double)));
    auto* d_a1 = static_cast<double*>(dmalloc(sizeof(double)));
    const DeviceProduct* pp[2] = {d_p0, d_p1};
    const int np[2] = {1, 1};
    double* ap[2] = {d_a0, d_a1};

    const auto e = chain.run(pp, np, d_ion, d_vloc, ap);
    double a0 = 0.0, a1 = 0.0;
    check_cuda(cudaMemcpy(&a0, d_a0, sizeof(double), cudaMemcpyDeviceToHost),
               "a0");
    check_cuda(cudaMemcpy(&a1, d_a1, sizeof(double), cudaMemcpyDeviceToHost),
               "a1");

    cudaFree(d_p0);
    cudaFree(d_p1);
    cudaFree(d_ion);
    cudaFree(d_vloc);
    cudaFree(d_a0);
    cudaFree(d_a1);

    EXPECT_NEAR(e.e_hartree, cpu.e_hartree,
                1e-9 * std::max(1.0, std::abs(cpu.e_hartree)));
    EXPECT_NEAR(e.e_xc, cpu.e_xc, 1e-9 * std::max(1.0, std::abs(cpu.e_xc)));
    EXPECT_NEAR(e.e_locgrid, cpu.e_locgrid,
                1e-9 * std::max(1.0, std::abs(cpu.e_locgrid)));
    EXPECT_NEAR(a0, acc0[0], 1e-8 * std::max(1.0, std::abs(acc0[0])) + 1e-11);
    EXPECT_NEAR(a1, acc1[0], 1e-8 * std::max(1.0, std::abs(acc1[0])) + 1e-11);
}

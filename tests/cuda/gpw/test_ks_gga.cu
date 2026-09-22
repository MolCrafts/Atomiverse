/// @file tests/cuda/gpw/test_ks_gga.cu
/// @brief CUDA GPW closed-shell PBE-GGA KS chain vs CPU `ks_grid_build_rks`.
///
/// Unit only (Γ small grid): product collocate → Poisson → spectral PBE-GGA →
/// V_eff → integrate. Not a full GPW SCF product.
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/grid_fft.h"
#include "atomiverse/cpu/gpw/internal/ks_grid_build.h"
#include "atomiverse/cuda/gpw/internal/ks_chain.cuh"
#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"
#include "atomiverse/cuda/gpw/internal/spectral_fft.cuh"

#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using ATV::cpu::gpw::UniformGrid;
using ATV::cpu::gpw::XcKind;
using ATV::cpu::gpw::internal::ks_grid_build_rks;
using ATV::cuda::gpw::internal::CollocateGridDesc;
using ATV::cuda::gpw::internal::DeviceProduct;
using ATV::cuda::gpw::internal::GpwGridDesc;
using ATV::cuda::gpw::internal::KsChainDevice;
using ATV::cuda::gpw::internal::KsChainEnergies;
using ATV::cuda::gpw::internal::KsXcKind;
using ATV::cuda::gpw::internal::SpectralFftDevice;

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
    return d;
}

GpwGridDesc to_poisson(const UniformGrid& g) {
    GpwGridDesc d;
    for (int i = 0; i < 3; ++i) {
        d.n[i] = g.n[i];
        d.L[i] = g.L[i];
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

struct ChainHost {
    std::vector<double> veff;
    std::vector<double> acc;
    KsChainEnergies e;
};

ChainHost run_device(const UniformGrid& grid,
                     const std::vector<DeviceProduct>& prods,
                     const std::vector<double>& n_ion,
                     const std::vector<double>& v_loc, bool do_integrate,
                     KsXcKind xc) {
    const long ng = grid.npts();
    const int nprod = static_cast<int>(prods.size());
    auto* d_prods = static_cast<DeviceProduct*>(
        dmalloc(static_cast<size_t>(nprod) * sizeof(DeviceProduct)));
    check_cuda(cudaMemcpy(d_prods, prods.data(),
                          static_cast<size_t>(nprod) * sizeof(DeviceProduct),
                          cudaMemcpyHostToDevice),
               "H2D prods");

    double* d_n_ion = nullptr;
    if (!n_ion.empty()) {
        d_n_ion = static_cast<double*>(
            dmalloc(static_cast<size_t>(ng) * sizeof(double)));
        check_cuda(cudaMemcpy(d_n_ion, n_ion.data(),
                              static_cast<size_t>(ng) * sizeof(double),
                              cudaMemcpyHostToDevice),
                   "H2D n_ion");
    }
    double* d_v_loc = nullptr;
    if (!v_loc.empty()) {
        d_v_loc = static_cast<double*>(
            dmalloc(static_cast<size_t>(ng) * sizeof(double)));
        check_cuda(cudaMemcpy(d_v_loc, v_loc.data(),
                              static_cast<size_t>(ng) * sizeof(double),
                              cudaMemcpyHostToDevice),
                   "H2D v_loc");
    }
    double* d_acc = nullptr;
    if (do_integrate) {
        d_acc = static_cast<double*>(
            dmalloc(static_cast<size_t>(nprod) * sizeof(double)));
    }

    KsChainDevice chain(to_desc(grid));
    ChainHost out;
    out.e = chain.run(d_prods, nprod, d_n_ion, d_v_loc, d_acc, xc);

    out.veff.resize(static_cast<std::size_t>(ng));
    check_cuda(cudaMemcpy(out.veff.data(), chain.d_veff(),
                          static_cast<size_t>(ng) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "D2H veff");
    if (do_integrate) {
        out.acc.resize(static_cast<std::size_t>(nprod));
        check_cuda(cudaMemcpy(out.acc.data(), d_acc,
                              static_cast<size_t>(nprod) * sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "D2H acc");
    }

    cudaFree(d_prods);
    if (d_n_ion)
        cudaFree(d_n_ion);
    if (d_v_loc)
        cudaFree(d_v_loc);
    if (d_acc)
        cudaFree(d_acc);
    return out;
}

} // namespace

// ── Spectral ∇ / −∇· match CPU grid_fft on a smooth field ─────────────────
TEST(CudaGpwKsGga, SpectralMatchesCpuGradientDiv) {
    UniformGrid grid = cubic_grid(16, 8.0);
    const long ng = grid.npts();
    std::vector<double> f(static_cast<std::size_t>(ng));
    // Smooth periodic field so spectral derivatives are well-defined.
    for (long i = 0; i < ng; ++i) {
        double r[3];
        grid.point(i, r);
        f[static_cast<std::size_t>(i)] =
            0.3 * std::cos(2.0 * M_PI * r[0] / grid.L[0])
            + 0.2 * std::sin(2.0 * M_PI * r[1] / grid.L[1])
            + 0.1 * std::cos(4.0 * M_PI * r[2] / grid.L[2]);
    }

    const auto g_cpu = ATV::cpu::gpw::gradient(grid, f);
    const std::array<std::vector<double>, 3> v = {g_cpu[0], g_cpu[1], g_cpu[2]};
    const auto div_cpu = ATV::cpu::gpw::divergence(grid, v);

    double* d_f =
        static_cast<double*>(dmalloc(static_cast<size_t>(ng) * sizeof(double)));
    double* d_gx =
        static_cast<double*>(dmalloc(static_cast<size_t>(ng) * sizeof(double)));
    double* d_gy =
        static_cast<double*>(dmalloc(static_cast<size_t>(ng) * sizeof(double)));
    double* d_gz =
        static_cast<double*>(dmalloc(static_cast<size_t>(ng) * sizeof(double)));
    double* d_div =
        static_cast<double*>(dmalloc(static_cast<size_t>(ng) * sizeof(double)));
    check_cuda(cudaMemcpy(d_f, f.data(),
                          static_cast<size_t>(ng) * sizeof(double),
                          cudaMemcpyHostToDevice),
               "H2D f");

    SpectralFftDevice sp(to_poisson(grid));
    sp.gradient(d_f, d_gx, d_gy, d_gz);
    sp.divergence(d_gx, d_gy, d_gz, d_div);
    check_cuda(cudaDeviceSynchronize(), "spectral sync");

    std::vector<double> gx(static_cast<std::size_t>(ng));
    std::vector<double> gy(static_cast<std::size_t>(ng));
    std::vector<double> gz(static_cast<std::size_t>(ng));
    std::vector<double> div(static_cast<std::size_t>(ng));
    check_cuda(cudaMemcpy(gx.data(), d_gx,
                          static_cast<size_t>(ng) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "D2H gx");
    check_cuda(cudaMemcpy(gy.data(), d_gy,
                          static_cast<size_t>(ng) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "D2H gy");
    check_cuda(cudaMemcpy(gz.data(), d_gz,
                          static_cast<size_t>(ng) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "D2H gz");
    check_cuda(cudaMemcpy(div.data(), d_div,
                          static_cast<size_t>(ng) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "D2H div");

    const double peak_g = std::max(
        {max_abs(g_cpu[0]), max_abs(g_cpu[1]), max_abs(g_cpu[2]), 1.0});
    const double tol_g = 1e-10 * peak_g + 1e-12;
    EXPECT_LE(max_abs_diff(gx, g_cpu[0]), tol_g);
    EXPECT_LE(max_abs_diff(gy, g_cpu[1]), tol_g);
    EXPECT_LE(max_abs_diff(gz, g_cpu[2]), tol_g);
    const double peak_d = std::max(1.0, max_abs(div_cpu));
    EXPECT_LE(max_abs_diff(div, div_cpu), 1e-10 * peak_d + 1e-12);

    cudaFree(d_f);
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_gz);
    cudaFree(d_div);
}

// ── Single s–s product: V_eff + energies vs ks_grid_build_rks PBE ──────────
TEST(CudaGpwKsGga, SSMatchesCpuKsGridPbe) {
    UniformGrid grid = cubic_grid(16, 8.0);
    const double pc[3] = {4.0, 4.0, 4.0};
    std::vector<DeviceProduct> prods = {make_ss(/*p=*/2.5, pc, /*w=*/1.3)};

    const auto rho = host_collocate(grid, prods);
    const std::size_t n = rho.size();
    std::vector<double> n_ion(n, 0.0);
    std::vector<double> v_loc;
    const auto cpu =
        ks_grid_build_rks(grid, rho, n_ion, v_loc, XcKind::PBE,
                          /*nonperiodic=*/false, false, false, false);

    const auto dev = run_device(grid, prods, n_ion, v_loc, /*integrate=*/true,
                                KsXcKind::PBE);

    EXPECT_NEAR(dev.e.e_hartree, cpu.e_hartree,
                1e-10 * std::max(1.0, std::abs(cpu.e_hartree)));
    EXPECT_NEAR(dev.e.e_xc, cpu.e_xc,
                1e-10 * std::max(1.0, std::abs(cpu.e_xc)));
    EXPECT_DOUBLE_EQ(dev.e.e_locgrid, 0.0);

    const double peak_v = max_abs(cpu.veff);
    ASSERT_GT(peak_v, 1e-8);
    const double max_dv = max_abs_diff(dev.veff, cpu.veff);
    const double tol_v = 1e-9 * std::max(1.0, peak_v) + 1e-12;
    EXPECT_LE(max_dv, tol_v) << "max|ΔV_eff|=" << max_dv << " peak=" << peak_v;

    const auto acc_host = host_integrate(grid, prods, cpu.veff);
    ASSERT_EQ(dev.acc.size(), acc_host.size());
    const double peak_a = max_abs(acc_host);
    const double max_da = max_abs_diff(dev.acc, acc_host);
    const double tol_a = 1e-9 * std::max(1.0, peak_a) + 1e-12;
    EXPECT_LE(max_da, tol_a) << "max|Δacc|=" << max_da << " peak=" << peak_a;
}

// ── Multi-center + n_ion + V_loc under PBE ─────────────────────────────────
TEST(CudaGpwKsGga, MultiWithIonAndVlocPbe) {
    UniformGrid grid = cubic_grid(20, 10.0);
    std::vector<DeviceProduct> prods;
    {
        const double c0[3] = {5.0, 5.0, 5.0};
        prods.push_back(make_ss(1.8, c0, 0.8));
    }
    {
        const double c1[3] = {4.2, 5.0, 5.6};
        prods.push_back(make_ss(2.2, c1, 0.5));
    }
    {
        const double c2[3] = {5.8, 5.0, 5.6};
        prods.push_back(make_ss(2.2, c2, 0.5));
    }

    const auto rho = host_collocate(grid, prods);
    const std::size_t n = rho.size();
    std::vector<double> n_ion(n, 0.0);
    std::vector<double> v_loc(n, -0.05);
    {
        const double pc[3] = {5.0, 5.0, 5.0};
        const double alpha = 3.0;
        const double pref = std::pow(alpha / M_PI, 1.5);
        for (std::size_t i = 0; i < n; ++i) {
            double r[3];
            grid.point(static_cast<long>(i), r);
            const double dx = r[0] - pc[0];
            const double dy = r[1] - pc[1];
            const double dz = r[2] - pc[2];
            n_ion[i] = pref * std::exp(-alpha * (dx * dx + dy * dy + dz * dz));
        }
    }

    const auto cpu = ks_grid_build_rks(grid, rho, n_ion, v_loc, XcKind::PBE,
                                       false, false, false, false);
    const auto dev = run_device(grid, prods, n_ion, v_loc, /*integrate=*/true,
                                KsXcKind::PBE);

    EXPECT_NEAR(dev.e.e_hartree, cpu.e_hartree,
                1e-10 * std::max(1.0, std::abs(cpu.e_hartree)));
    EXPECT_NEAR(dev.e.e_xc, cpu.e_xc,
                1e-10 * std::max(1.0, std::abs(cpu.e_xc)));
    EXPECT_NEAR(dev.e.e_locgrid, cpu.e_locgrid,
                1e-10 * std::max(1.0, std::abs(cpu.e_locgrid)));

    const double peak_v = max_abs(cpu.veff);
    ASSERT_GT(peak_v, 1e-8);
    const double max_dv = max_abs_diff(dev.veff, cpu.veff);
    const double tol_v = 1e-9 * std::max(1.0, peak_v) + 1e-12;
    EXPECT_LE(max_dv, tol_v) << "max|ΔV_eff|=" << max_dv;

    const auto acc_host = host_integrate(grid, prods, cpu.veff);
    const double peak_a = max_abs(acc_host);
    const double max_da = max_abs_diff(dev.acc, acc_host);
    const double tol_a = 1e-9 * std::max(1.0, peak_a) + 1e-12;
    EXPECT_LE(max_da, tol_a) << "max|Δacc|=" << max_da;
}

// ── Default PADE path still matches (API default) ─────────────────────────
TEST(CudaGpwKsGga, DefaultPadeUnchanged) {
    UniformGrid grid = cubic_grid(12, 6.0);
    const double pc[3] = {3.0, 3.0, 3.0};
    std::vector<DeviceProduct> prods = {make_ss(2.0, pc, 1.0)};
    const auto rho = host_collocate(grid, prods);
    const std::size_t n = rho.size();
    std::vector<double> n_ion(n, 0.0);
    std::vector<double> v_loc;
    const auto cpu = ks_grid_build_rks(grid, rho, n_ion, v_loc, XcKind::PADE,
                                       false, false, false, false);
    const auto dev = run_device(grid, prods, n_ion, v_loc, /*integrate=*/false,
                                KsXcKind::PADE);
    EXPECT_NEAR(dev.e.e_xc, cpu.e_xc,
                1e-10 * std::max(1.0, std::abs(cpu.e_xc)));
    const double peak_v = max_abs(cpu.veff);
    EXPECT_LE(max_abs_diff(dev.veff, cpu.veff),
              1e-9 * std::max(1.0, peak_v) + 1e-12);
}

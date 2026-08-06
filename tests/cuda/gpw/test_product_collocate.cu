/// @file tests/cuda/gpw/test_product_collocate.cu
/// @brief CUDA GPW product ρ collocate foothold vs CPU collint oracle.
///
/// Unit only (Γ small grid): s–s and low-L product Gaussians vs
/// `cpu::gpw::internal::collint_collocate_product`. Not a full GPW SCF product.
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/multigrid.h"
#include "atomiverse/cpu/gpw/internal/product_collint.h"
#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using ATV::cpu::gpw::GridProduct;
using ATV::cpu::gpw::UniformGrid;
using ATV::cpu::gpw::internal::collint_collocate_product;
using ATV::cuda::gpw::internal::CollocateGridDesc;
using ATV::cuda::gpw::internal::DeviceProduct;
using ATV::cuda::gpw::internal::collocate_integrate_rho;
using ATV::cuda::gpw::internal::collocate_product;
using ATV::cuda::gpw::internal::collocate_products;

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

GridProduct to_grid_product(const DeviceProduct& q) {
    GridProduct g;
    g.p = q.p;
    g.k = 1.0; // weight already carries k · P · coeffs
    for (int d = 0; d < 3; ++d) {
        g.pc[d] = q.pc[d];
        g.ra[d] = q.ra[d];
        g.rb[d] = q.rb[d];
        g.la[d] = q.la[d];
        g.lb[d] = q.lb[d];
    }
    return g;
}

/// Host continuous product shape (same formula as the device kernel).
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

std::vector<double> cpu_collint(const UniformGrid& grid,
                                const std::vector<DeviceProduct>& prods,
                                double eps = 1e-14) {
    const long ng = grid.npts();
    std::vector<double> rho(static_cast<std::size_t>(ng), 0.0);
    for (const auto& q : prods) {
        GridProduct gp = to_grid_product(q);
        const double mag = std::abs(q.weight);
        collint_collocate_product(grid, gp, q.weight, mag, eps, rho.data());
    }
    return rho;
}

std::vector<double> run_device(const UniformGrid& grid,
                               const std::vector<DeviceProduct>& prods) {
    const long ng = grid.npts();
    const int nprod = static_cast<int>(prods.size());
    auto* d_prods = static_cast<DeviceProduct*>(
        dmalloc(static_cast<size_t>(nprod) * sizeof(DeviceProduct)));
    auto* d_rho =
        static_cast<double*>(dmalloc(static_cast<size_t>(ng) * sizeof(double)));
    check_cuda(cudaMemcpy(d_prods, prods.data(),
                          static_cast<size_t>(nprod) * sizeof(DeviceProduct),
                          cudaMemcpyHostToDevice),
               "H2D prods");
    check_cuda(cudaMemset(d_rho, 0, static_cast<size_t>(ng) * sizeof(double)),
               "zero rho");

    collocate_products(to_desc(grid), d_prods, nprod, d_rho);
    check_cuda(cudaDeviceSynchronize(), "sync collocate");

    std::vector<double> rho(static_cast<std::size_t>(ng));
    check_cuda(cudaMemcpy(rho.data(), d_rho,
                          static_cast<size_t>(ng) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "D2H rho");
    cudaFree(d_prods);
    cudaFree(d_rho);
    return rho;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
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

} // namespace

// ── s–s product: device matches continuous host + CPU collint ─────────────
TEST(CudaGpwProductCollocate, SSMatchesHostAndCollint) {
    // Small Γ grid; tight Gaussian so collint support covers significant mass.
    UniformGrid grid = cubic_grid(16, 8.0);
    const double pc[3] = {4.0, 4.0, 4.0};
    std::vector<DeviceProduct> prods = {make_ss(/*p=*/2.5, pc, /*w=*/1.3)};

    const auto rho_dev = run_device(grid, prods);
    const auto rho_host = host_collocate(grid, prods);
    const auto rho_cpu = cpu_collint(grid, prods, /*eps=*/1e-16);

    // Continuous formula: device ≡ host (bit-identical arithmetic path, ~1e-15).
    const double max_h = max_abs_diff(rho_dev, rho_host);
    EXPECT_LE(max_h, 1e-14) << "device vs continuous host max|Δρ|=" << max_h;

    // Collint sphere/pol path vs continuous: tol stated for foothold.
    const double peak = max_abs(rho_cpu);
    ASSERT_GT(peak, 1e-6);
    const double max_c = max_abs_diff(rho_dev, rho_cpu);
    // Absolute floor + relative: collint window truncation vs full formula.
    const double tol = 1e-10 * peak + 1e-14;
    EXPECT_LE(max_c, tol) << "device vs collint max|Δρ|=" << max_c
                          << " peak=" << peak << " tol=" << tol;
}

// ── Multi-center s–s products (water-like) ────────────────────────────────
TEST(CudaGpwProductCollocate, MultiSSMatchesCollint) {
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

    const auto rho_dev = run_device(grid, prods);
    const auto rho_cpu = cpu_collint(grid, prods, /*eps=*/1e-16);
    const double peak = max_abs(rho_cpu);
    ASSERT_GT(peak, 1e-6);
    const double max_c = max_abs_diff(rho_dev, rho_cpu);
    const double tol = 1e-10 * peak + 1e-14;
    EXPECT_LE(max_c, tol) << "multi s–s max|Δρ|=" << max_c << " peak=" << peak;
}

// ── p-type monom (la_z=1): collint cab→cxyz path ──────────────────────────
TEST(CudaGpwProductCollocate, PTypeMatchesCollint) {
    UniformGrid grid = cubic_grid(16, 8.0);
    DeviceProduct q;
    q.p = 2.0;
    q.weight = 0.9;
    for (int d = 0; d < 3; ++d) {
        q.pc[d] = 4.0;
        q.ra[d] = 4.0;
        q.rb[d] = 4.0;
        q.la[d] = 0;
        q.lb[d] = 0;
    }
    // (r_z − ra_z)^1 · exp — exercises non-zero angular path.
    q.la[2] = 1;
    // Split centers slightly so expand_axis da/db nonzero path is live.
    q.ra[2] = 3.85;
    q.rb[2] = 4.15;
    // pc along z is product center for α equal → midpoint.
    q.pc[2] = 0.5 * (q.ra[2] + q.rb[2]);

    std::vector<DeviceProduct> prods = {q};
    const auto rho_dev = run_device(grid, prods);
    const auto rho_host = host_collocate(grid, prods);
    const auto rho_cpu = cpu_collint(grid, prods, /*eps=*/1e-16);

    EXPECT_LE(max_abs_diff(rho_dev, rho_host), 1e-14);

    const double peak = max_abs(rho_cpu);
    ASSERT_GT(peak, 1e-6);
    const double max_c = max_abs_diff(rho_dev, rho_cpu);
    const double tol = 1e-9 * peak + 1e-13;
    EXPECT_LE(max_c, tol) << "p-type max|Δρ|=" << max_c << " peak=" << peak;
}

// ── ∫ρ device reduction vs host Kahan (energy-class) ──────────────────────
TEST(CudaGpwProductCollocate, IntegrateRhoMatchesHost) {
    UniformGrid grid = cubic_grid(18, 9.0);
    const double pc[3] = {4.5, 4.5, 4.5};
    std::vector<DeviceProduct> prods = {make_ss(1.5, pc, 1.0)};

    const long ng = grid.npts();
    auto* d_prods =
        static_cast<DeviceProduct*>(dmalloc(sizeof(DeviceProduct)));
    auto* d_rho =
        static_cast<double*>(dmalloc(static_cast<size_t>(ng) * sizeof(double)));
    auto* d_sum = static_cast<double*>(dmalloc(sizeof(double)));
    check_cuda(cudaMemcpy(d_prods, prods.data(), sizeof(DeviceProduct),
                          cudaMemcpyHostToDevice),
               "H2D");
    check_cuda(cudaMemset(d_rho, 0, static_cast<size_t>(ng) * sizeof(double)),
               "zero");
    collocate_product(to_desc(grid), d_prods, d_rho);
    collocate_integrate_rho(d_rho, ng, grid.weight(), d_sum);
    check_cuda(cudaDeviceSynchronize(), "sync");

    double sum_dev = 0.0;
    check_cuda(cudaMemcpy(&sum_dev, d_sum, sizeof(double), cudaMemcpyDeviceToHost),
               "D2H sum");

    const auto rho_host = host_collocate(grid, prods);
    double sum = 0.0, c = 0.0;
    for (double v : rho_host) {
        const double y = v - c;
        const double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    const double sum_host = sum * grid.weight();

    // Atomic reduction order: relative 1e-10 class (same as XC energy foothold).
    EXPECT_NEAR(sum_dev, sum_host, 1e-10 * std::abs(sum_host));

    cudaFree(d_prods);
    cudaFree(d_rho);
    cudaFree(d_sum);
}

// ── Zero weight / empty product list are no-ops ───────────────────────────
TEST(CudaGpwProductCollocate, ZeroWeightNoOp) {
    UniformGrid grid = cubic_grid(8, 6.0);
    const double pc[3] = {3.0, 3.0, 3.0};
    DeviceProduct q = make_ss(1.0, pc, 0.0);
    std::vector<DeviceProduct> prods = {q};

    const auto rho_dev = run_device(grid, prods);
    for (double v : rho_dev)
        EXPECT_DOUBLE_EQ(v, 0.0);

    // nprod = 0 must not touch buffer.
    auto* d_rho = static_cast<double*>(
        dmalloc(static_cast<size_t>(grid.npts()) * sizeof(double)));
    check_cuda(cudaMemset(d_rho, 0xFF,
                          static_cast<size_t>(grid.npts()) * sizeof(double)),
               "poison");
    collocate_products(to_desc(grid), nullptr, 0, d_rho);
    check_cuda(cudaDeviceSynchronize(), "sync empty");
    // Leave poisoned pattern (no write). Just ensure launch succeeds.
    cudaFree(d_rho);
}

/// @file tests/cuda/gpw/test_product_integrate.cu
/// @brief CUDA GPW product V·shape integrate foothold vs CPU collint oracle.
///
/// Unit only (Γ small grid): s–s and low-L product Gaussians vs
/// `cpu::gpw::internal::collint_integrate_product`. Not a full GPW SCF product.
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/multigrid.h"
#include "atomiverse/cpu/gpw/internal/product_collint.h"
#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"
#include "atomiverse/cuda/gpw/internal/product_integrate.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using ATV::cpu::gpw::GridProduct;
using ATV::cpu::gpw::UniformGrid;
using ATV::cpu::gpw::internal::collint_integrate_product;
using ATV::cuda::gpw::internal::CollocateGridDesc;
using ATV::cuda::gpw::internal::DeviceProduct;
using ATV::cuda::gpw::internal::integrate_product;
using ATV::cuda::gpw::internal::integrate_products;

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

DeviceProduct make_ss(double p, const double pc[3]) {
    DeviceProduct q;
    q.p = p;
    q.weight = 0.0; // integrate ignores weight (shape-only)
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
    g.k = 1.0;
    for (int d = 0; d < 3; ++d) {
        g.pc[d] = q.pc[d];
        g.ra[d] = q.ra[d];
        g.rb[d] = q.rb[d];
        g.la[d] = q.la[d];
        g.lb[d] = q.lb[d];
    }
    return g;
}

/// Host continuous bare product shape (same formula as the device kernel).
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

std::vector<double> host_integrate(const UniformGrid& grid,
                                   const std::vector<DeviceProduct>& prods,
                                   const std::vector<double>& V) {
    const long ng = grid.npts();
    std::vector<double> acc(prods.size(), 0.0);
    for (long i = 0; i < ng; ++i) {
        double r[3];
        grid.point(i, r);
        const double Vi = V[static_cast<std::size_t>(i)];
        for (std::size_t q = 0; q < prods.size(); ++q)
            acc[q] += Vi * host_product_shape_bare(prods[q], r);
    }
    return acc;
}

std::vector<double> cpu_collint_integrate(const UniformGrid& grid,
                                          const std::vector<DeviceProduct>& prods,
                                          const std::vector<double>& V,
                                          double eps = 1e-14) {
    std::vector<double> acc(prods.size(), 0.0);
    for (std::size_t q = 0; q < prods.size(); ++q) {
        GridProduct gp = to_grid_product(prods[q]);
        // mag: window radius scale; large enough to keep the Gaussian mass.
        const double mag = 1.0;
        acc[q] = collint_integrate_product(grid, gp, mag, eps, V.data());
    }
    return acc;
}

/// Smooth non-constant potential so integrate is non-trivial.
std::vector<double> make_V(const UniformGrid& grid) {
    const long ng = grid.npts();
    std::vector<double> V(static_cast<std::size_t>(ng));
    for (long i = 0; i < ng; ++i) {
        double r[3];
        grid.point(i, r);
        // Mild quadratic + offset (Hartree-ish scale).
        V[static_cast<std::size_t>(i)] =
            0.35 + 0.02 * ((r[0] - 4.0) * (r[0] - 4.0) + 0.5 * (r[1] - 4.0)
                           + 0.25 * (r[2] - 4.0));
    }
    return V;
}

std::vector<double> run_device(const UniformGrid& grid,
                               const std::vector<DeviceProduct>& prods,
                               const std::vector<double>& V) {
    const long ng = grid.npts();
    const int nprod = static_cast<int>(prods.size());
    auto* d_prods = static_cast<DeviceProduct*>(
        dmalloc(static_cast<size_t>(nprod) * sizeof(DeviceProduct)));
    auto* d_V =
        static_cast<double*>(dmalloc(static_cast<size_t>(ng) * sizeof(double)));
    auto* d_acc =
        static_cast<double*>(dmalloc(static_cast<size_t>(nprod) * sizeof(double)));
    check_cuda(cudaMemcpy(d_prods, prods.data(),
                          static_cast<size_t>(nprod) * sizeof(DeviceProduct),
                          cudaMemcpyHostToDevice),
               "H2D prods");
    check_cuda(cudaMemcpy(d_V, V.data(),
                          static_cast<size_t>(ng) * sizeof(double),
                          cudaMemcpyHostToDevice),
               "H2D V");
    check_cuda(
        cudaMemset(d_acc, 0xFF, static_cast<size_t>(nprod) * sizeof(double)),
        "poison acc");

    integrate_products(to_desc(grid), d_prods, nprod, d_V, d_acc);
    check_cuda(cudaDeviceSynchronize(), "sync integrate");

    std::vector<double> acc(static_cast<std::size_t>(nprod));
    check_cuda(cudaMemcpy(acc.data(), d_acc,
                          static_cast<size_t>(nprod) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "D2H acc");
    cudaFree(d_prods);
    cudaFree(d_V);
    cudaFree(d_acc);
    return acc;
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
TEST(CudaGpwProductIntegrate, SSMatchesHostAndCollint) {
    UniformGrid grid = cubic_grid(16, 8.0);
    const double pc[3] = {4.0, 4.0, 4.0};
    std::vector<DeviceProduct> prods = {make_ss(/*p=*/2.5, pc)};
    const auto V = make_V(grid);

    const auto acc_dev = run_device(grid, prods, V);
    const auto acc_host = host_integrate(grid, prods, V);
    const auto acc_cpu = cpu_collint_integrate(grid, prods, V, /*eps=*/1e-16);

    // Continuous formula: device ≡ host within reduction tree noise.
    // Block reduction vs sequential host: ~1e-12 class on small grids.
    const double max_h = max_abs_diff(acc_dev, acc_host);
    const double peak_h = max_abs(acc_host);
    ASSERT_GT(peak_h, 1e-6);
    EXPECT_LE(max_h, 1e-12 * peak_h + 1e-14)
        << "device vs continuous host max|Δacc|=" << max_h
        << " peak=" << peak_h;

    // Collint sphere/pol path vs continuous: tol stated for foothold.
    const double peak = max_abs(acc_cpu);
    ASSERT_GT(peak, 1e-6);
    const double max_c = max_abs_diff(acc_dev, acc_cpu);
    const double tol = 1e-10 * peak + 1e-14;
    EXPECT_LE(max_c, tol) << "device vs collint max|Δacc|=" << max_c
                          << " peak=" << peak << " tol=" << tol;
}

// ── Multi-center s–s products ─────────────────────────────────────────────
TEST(CudaGpwProductIntegrate, MultiSSMatchesCollint) {
    UniformGrid grid = cubic_grid(20, 10.0);
    std::vector<DeviceProduct> prods;
    {
        const double c0[3] = {5.0, 5.0, 5.0};
        prods.push_back(make_ss(1.8, c0));
    }
    {
        const double c1[3] = {4.2, 5.0, 5.6};
        prods.push_back(make_ss(2.2, c1));
    }
    {
        const double c2[3] = {5.8, 5.0, 5.6};
        prods.push_back(make_ss(2.2, c2));
    }
    // Re-center V around box mid.
    const long ng = grid.npts();
    std::vector<double> V(static_cast<std::size_t>(ng));
    for (long i = 0; i < ng; ++i) {
        double r[3];
        grid.point(i, r);
        V[static_cast<std::size_t>(i)] =
            0.4 + 0.01 * (r[0] + 0.5 * r[1] - r[2]);
    }

    const auto acc_dev = run_device(grid, prods, V);
    const auto acc_host = host_integrate(grid, prods, V);
    const auto acc_cpu = cpu_collint_integrate(grid, prods, V, /*eps=*/1e-16);

    const double peak_h = max_abs(acc_host);
    ASSERT_GT(peak_h, 1e-6);
    EXPECT_LE(max_abs_diff(acc_dev, acc_host), 1e-12 * peak_h + 1e-14);

    const double peak = max_abs(acc_cpu);
    ASSERT_GT(peak, 1e-6);
    const double max_c = max_abs_diff(acc_dev, acc_cpu);
    const double tol = 1e-10 * peak + 1e-14;
    EXPECT_LE(max_c, tol) << "multi s–s max|Δacc|=" << max_c
                          << " peak=" << peak;
}

// ── p-type monom (la_z=1): collint cab→cxyz path ──────────────────────────
TEST(CudaGpwProductIntegrate, PTypeMatchesCollint) {
    UniformGrid grid = cubic_grid(16, 8.0);
    DeviceProduct q;
    q.p = 2.0;
    q.weight = 0.0;
    for (int d = 0; d < 3; ++d) {
        q.pc[d] = 4.0;
        q.ra[d] = 4.0;
        q.rb[d] = 4.0;
        q.la[d] = 0;
        q.lb[d] = 0;
    }
    q.la[2] = 1;
    q.ra[2] = 3.85;
    q.rb[2] = 4.15;
    q.pc[2] = 0.5 * (q.ra[2] + q.rb[2]);

    std::vector<DeviceProduct> prods = {q};
    const auto V = make_V(grid);

    const auto acc_dev = run_device(grid, prods, V);
    const auto acc_host = host_integrate(grid, prods, V);
    const auto acc_cpu = cpu_collint_integrate(grid, prods, V, /*eps=*/1e-16);

    const double peak_h = max_abs(acc_host);
    ASSERT_GT(peak_h, 1e-6);
    EXPECT_LE(max_abs_diff(acc_dev, acc_host), 1e-12 * peak_h + 1e-14);

    const double peak = max_abs(acc_cpu);
    ASSERT_GT(peak, 1e-6);
    const double max_c = max_abs_diff(acc_dev, acc_cpu);
    // Slightly looser: p-type collint window vs full continuous.
    const double tol = 1e-9 * peak + 1e-13;
    EXPECT_LE(max_c, tol) << "p-type max|Δacc|=" << max_c << " peak=" << peak;
}

// ── Single-product convenience wrapper ────────────────────────────────────
TEST(CudaGpwProductIntegrate, SingleProductWrapper) {
    UniformGrid grid = cubic_grid(12, 6.0);
    const double pc[3] = {3.0, 3.0, 3.0};
    DeviceProduct q = make_ss(3.0, pc);
    const auto V = make_V(grid);
    // Shift V mid for this smaller box.
    const long ng = grid.npts();
    std::vector<double> V2(static_cast<std::size_t>(ng));
    for (long i = 0; i < ng; ++i) {
        double r[3];
        grid.point(i, r);
        V2[static_cast<std::size_t>(i)] = 0.5 + 0.03 * r[0];
    }

    auto* d_prod =
        static_cast<DeviceProduct*>(dmalloc(sizeof(DeviceProduct)));
    auto* d_V =
        static_cast<double*>(dmalloc(static_cast<size_t>(ng) * sizeof(double)));
    auto* d_acc = static_cast<double*>(dmalloc(sizeof(double)));
    check_cuda(cudaMemcpy(d_prod, &q, sizeof(DeviceProduct),
                          cudaMemcpyHostToDevice),
               "H2D");
    check_cuda(cudaMemcpy(d_V, V2.data(),
                          static_cast<size_t>(ng) * sizeof(double),
                          cudaMemcpyHostToDevice),
               "H2D V");
    integrate_product(to_desc(grid), d_prod, d_V, d_acc);
    check_cuda(cudaDeviceSynchronize(), "sync");
    double acc_dev = 0.0;
    check_cuda(cudaMemcpy(&acc_dev, d_acc, sizeof(double),
                          cudaMemcpyDeviceToHost),
               "D2H");

    std::vector<DeviceProduct> prods = {q};
    const auto acc_host = host_integrate(grid, prods, V2);
    EXPECT_NEAR(acc_dev, acc_host[0],
                1e-12 * std::abs(acc_host[0]) + 1e-14);

    cudaFree(d_prod);
    cudaFree(d_V);
    cudaFree(d_acc);
}

// ── nprod = 0 is a no-op (buffer untouched) ───────────────────────────────
TEST(CudaGpwProductIntegrate, EmptyProductListNoOp) {
    UniformGrid grid = cubic_grid(8, 6.0);
    const long ng = grid.npts();
    auto* d_V =
        static_cast<double*>(dmalloc(static_cast<size_t>(ng) * sizeof(double)));
    auto* d_acc = static_cast<double*>(dmalloc(sizeof(double)));
    check_cuda(cudaMemset(d_acc, 0xAB, sizeof(double)), "poison");
    // nprod=0 must not launch / not write acc.
    integrate_products(to_desc(grid), nullptr, 0, d_V, d_acc);
    check_cuda(cudaDeviceSynchronize(), "sync empty");
    unsigned char buf[sizeof(double)];
    check_cuda(cudaMemcpy(buf, d_acc, sizeof(double), cudaMemcpyDeviceToHost),
               "D2H");
    for (unsigned char b : buf)
        EXPECT_EQ(b, 0xAB);
    cudaFree(d_V);
    cudaFree(d_acc);
}

/// @file tests/cuda/gpw/test_weight_pack.cu
/// @brief Unit: device P→weight + acc→V scatter vs host oracles.
///
/// Single-function checks for `internal::ProductWeightPackDevice` (no SCF).

#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"
#include "atomiverse/cuda/gpw/internal/product_weight_pack.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using ATV::cuda::gpw::internal::DevicePrimContrib;
using ATV::cuda::gpw::internal::DevicePrimRef;
using ATV::cuda::gpw::internal::DeviceProduct;
using ATV::cuda::gpw::internal::DeviceProductGeom;
using ATV::cuda::gpw::internal::ProductWeightPackDevice;

namespace {

void check_cuda(cudaError_t e, const char* what) {
    ASSERT_EQ(e, cudaSuccess) << what << ": " << cudaGetErrorString(e);
}

/// Host twin of product_density_weight with column-major P (P[μ+ν·nb]).
double host_weight(const DeviceProductGeom& g,
                   const std::vector<DevicePrimRef>& prims,
                   const std::vector<DevicePrimContrib>& contribs, int nb,
                   const std::vector<double>& P) {
    const DevicePrimRef& A = prims[static_cast<std::size_t>(g.ua)];
    const DevicePrimRef& B = prims[static_cast<std::size_t>(g.ub)];
    double s = 0.0;
    for (int i = 0; i < A.cn; ++i) {
        const DevicePrimContrib& ca =
            contribs[static_cast<std::size_t>(A.cbeg + i)];
        double rs = 0.0;
        for (int j = 0; j < B.cn; ++j) {
            const DevicePrimContrib& cb =
                contribs[static_cast<std::size_t>(B.cbeg + j)];
            rs += P[static_cast<std::size_t>(ca.mu + cb.mu * nb)] * cb.coeff;
        }
        s += ca.coeff * rs;
    }
    return s * g.k;
}

/// Host twin of scatter_product_acc → column-major V.
void host_scatter(const std::vector<DeviceProductGeom>& geom,
                  const std::vector<DevicePrimRef>& prims,
                  const std::vector<DevicePrimContrib>& contribs, int nb,
                  const std::vector<double>& acc, double w_grid,
                  std::vector<double>& V) {
    V.assign(static_cast<std::size_t>(nb) * nb, 0.0);
    for (std::size_t ip = 0; ip < geom.size(); ++ip) {
        const double ig = acc[ip];
        if (ig == 0.0)
            continue;
        const DeviceProductGeom& g = geom[ip];
        const DevicePrimRef& A = prims[static_cast<std::size_t>(g.ua)];
        const DevicePrimRef& B = prims[static_cast<std::size_t>(g.ub)];
        const double sw = g.k * w_grid * ig;
        for (int i = 0; i < A.cn; ++i) {
            const DevicePrimContrib& ca =
                contribs[static_cast<std::size_t>(A.cbeg + i)];
            const double cas = ca.coeff * sw;
            for (int j = 0; j < B.cn; ++j) {
                const DevicePrimContrib& cb =
                    contribs[static_cast<std::size_t>(B.cbeg + j)];
                V[static_cast<std::size_t>(ca.mu + cb.mu * nb)] +=
                    cas * cb.coeff;
            }
        }
    }
}

double max_abs_diff(const double* a, const double* b, int n) {
    double m = 0.0;
    for (int i = 0; i < n; ++i)
        m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

} // namespace

TEST(CudaGpwWeightPack, PackWeightsMatchHost) {
    // Two prims, each contributing to one AO; one product ua=0,ub=1.
    const int nb = 2;
    std::vector<DevicePrimRef> prims = {{0, 1}, {1, 1}};
    std::vector<DevicePrimContrib> contribs = {{0, 0.5}, {1, 1.5}};
    DeviceProductGeom g{};
    g.p = 1.2;
    g.pc[0] = g.ra[0] = g.rb[0] = 0.1;
    g.pc[1] = g.ra[1] = g.rb[1] = 0.2;
    g.pc[2] = g.ra[2] = g.rb[2] = 0.3;
    g.ua = 0;
    g.ub = 1;
    g.k = 0.8;
    std::vector<DeviceProductGeom> geom = {g};

    // Column-major P: P00=1, P10=0.2, P01=0.3, P11=2
    std::vector<double> P = {1.0, 0.2, 0.3, 2.0};
    const double w_host = host_weight(g, prims, contribs, nb, P);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    ProductWeightPackDevice pack;
    pack.build(stream, nb, /*nprim=*/2, /*ncontrib=*/2);
    pack.upload_tables(prims.data(), 2, contribs.data(), 2);

    DeviceProductGeom* d_geom = nullptr;
    DeviceProduct* d_prods = nullptr;
    double* d_P = nullptr;
    check_cuda(cudaMalloc(&d_geom, sizeof(DeviceProductGeom)), "geom");
    check_cuda(cudaMalloc(&d_prods, sizeof(DeviceProduct)), "prods");
    check_cuda(cudaMalloc(&d_P, 4 * sizeof(double)), "P");
    check_cuda(cudaMemcpy(d_geom, &g, sizeof(g), cudaMemcpyHostToDevice),
               "H2D geom");
    check_cuda(cudaMemcpy(d_P, P.data(), 4 * sizeof(double),
                          cudaMemcpyHostToDevice),
               "H2D P");

    pack.pack_weights(d_P, d_geom, d_prods, 1, /*eps_drop=*/1e-20);
    DeviceProduct q;
    check_cuda(cudaMemcpy(&q, d_prods, sizeof(q), cudaMemcpyDeviceToHost),
               "D2H prod");
    check_cuda(cudaStreamSynchronize(stream), "sync");

    EXPECT_NEAR(q.weight, w_host, 1e-12) << "weight device vs host";
    EXPECT_NEAR(q.p, g.p, 0.0);
    EXPECT_NEAR(q.pc[0], g.pc[0], 0.0);

    cudaFree(d_geom);
    cudaFree(d_prods);
    cudaFree(d_P);
    cudaStreamDestroy(stream);
}

TEST(CudaGpwWeightPack, ScatterMatchesHost) {
    const int nb = 2;
    std::vector<DevicePrimRef> prims = {{0, 1}, {1, 1}};
    std::vector<DevicePrimContrib> contribs = {{0, 0.5}, {1, 1.5}};
    DeviceProductGeom g{};
    g.ua = 0;
    g.ub = 1;
    g.k = 0.8;
    std::vector<DeviceProductGeom> geom = {g};
    std::vector<double> acc = {2.5};
    const double w_grid = 0.125;

    std::vector<double> V_host;
    host_scatter(geom, prims, contribs, nb, acc, w_grid, V_host);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ProductWeightPackDevice pack;
    pack.build(stream, nb, 2, 2);
    pack.upload_tables(prims.data(), 2, contribs.data(), 2);

    DeviceProductGeom* d_geom = nullptr;
    double* d_acc = nullptr;
    check_cuda(cudaMalloc(&d_geom, sizeof(DeviceProductGeom)), "geom");
    check_cuda(cudaMalloc(&d_acc, sizeof(double)), "acc");
    check_cuda(cudaMemcpy(d_geom, &g, sizeof(g), cudaMemcpyHostToDevice),
               "H2D geom");
    check_cuda(cudaMemcpy(d_acc, acc.data(), sizeof(double),
                          cudaMemcpyHostToDevice),
               "H2D acc");

    pack.scatter_acc(d_geom, d_acc, 1, w_grid);
    std::vector<double> V_dev(4);
    pack.download_V(V_dev.data());

    const double d = max_abs_diff(V_host.data(), V_dev.data(), 4);
    EXPECT_LE(d, 1e-12) << "max|V_dev-V_host|=" << d;

    cudaFree(d_geom);
    cudaFree(d_acc);
    cudaStreamDestroy(stream);
}

TEST(CudaGpwWeightPack, FrobeniusMatchesHost) {
    const int nb = 2;
    std::vector<double> P = {1.0, 0.2, 0.3, 2.0};
    std::vector<double> M = {0.5, 0.1, 0.1, 0.4};
    double host = 0.0;
    for (int i = 0; i < 4; ++i)
        host += P[static_cast<std::size_t>(i)] * M[static_cast<std::size_t>(i)];

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    // Dummy tables (required by build).
    std::vector<DevicePrimRef> prims = {{0, 1}};
    std::vector<DevicePrimContrib> contribs = {{0, 1.0}};
    ProductWeightPackDevice pack;
    pack.build(stream, nb, 1, 1);
    pack.upload_tables(prims.data(), 1, contribs.data(), 1);

    double *d_P = nullptr, *d_M = nullptr;
    check_cuda(cudaMalloc(&d_P, 4 * sizeof(double)), "P");
    check_cuda(cudaMalloc(&d_M, 4 * sizeof(double)), "M");
    check_cuda(cudaMemcpy(d_P, P.data(), 4 * sizeof(double),
                          cudaMemcpyHostToDevice),
               "H2D P");
    check_cuda(cudaMemcpy(d_M, M.data(), 4 * sizeof(double),
                          cudaMemcpyHostToDevice),
               "H2D M");

    const double dev = pack.frobenius(d_P, d_M);
    EXPECT_NEAR(dev, host, 1e-12);

    cudaFree(d_P);
    cudaFree(d_M);
    cudaStreamDestroy(stream);
}

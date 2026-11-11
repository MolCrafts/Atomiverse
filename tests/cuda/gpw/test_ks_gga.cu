/// @file tests/cuda/gpw/test_ks_gga.cu
/// @brief CUDA GPW restricted/collinear PBE-family GGA KS chains vs CPU grid builds.
///
/// Unit only (Γ small grid): product collocate → Poisson → spectral PBE-family GGA →
/// V_eff → integrate. Not a full GPW SCF product.
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/grid_fft.h"
#include "atomiverse/cpu/gpw/internal/ks_grid_build.h"
#include "atomiverse/cuda/gpw/internal/ks_chain.cuh"
#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"
#include "atomiverse/cuda/gpw/internal/spectral_fft.cuh"
#include "atomiverse/cuda/xc/xc_pbe.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using ATV::cpu::gpw::UniformGrid;
using ATV::cpu::gpw::internal::ks_grid_build_rks;
using ATV::cpu::xc::XCType;
using ATV::cuda::gpw::internal::CollocateGridDesc;
using ATV::cuda::gpw::internal::DeviceProduct;
using ATV::cuda::gpw::internal::GpwGridDesc;
using ATV::cuda::gpw::internal::KsChainDevice;
using ATV::cuda::gpw::internal::KsChainEnergies;

using ATV::cuda::gpw::internal::SpectralFftDevice;

namespace {

class CudaGpwKsGgaFamily : public ::testing::TestWithParam<XCType> {};

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
    d.periodic = true;
    d.eps_win = 1e-14;
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

// Independent dense periodic reference: enumerate the lattice images of each
// primary-cell sample, rather than reproducing the device's unwrapped window
// walk and wrapped atomic scatter. Collocation and integration must use their
// declared support: weighted-density screening and unit bare-shape screening,
// respectively. A raw primary-cell Gaussian is a different physical input.
double periodic_product_shape(const UniformGrid& grid, const DeviceProduct& q,
                              const double r[3], bool bare) {
    const auto desc = to_desc(grid);
    const double magnitude =
        bare ? 1.0 : (q.mag > 0 ? q.mag : std::abs(q.weight));
    if (!(magnitude > 0) || !(q.p > 0))
        return 0;
    const int angular = q.L_win >= 0 ? q.L_win
                                     : q.la[0] + q.la[1] + q.la[2] + q.lb[0]
                                           + q.lb[1] + q.lb[2];
    const double log_magnitude = std::log(magnitude);
    const double log_eps = std::log(desc.eps_win);
    double radius_squared = std::max(0.0, (log_magnitude - log_eps) / q.p);
    for (int iteration = 0; iteration < 4; ++iteration) {
        const double radius = std::sqrt(std::max(radius_squared, 1e-30));
        radius_squared = std::max(
            0.0, (log_magnitude + angular * std::log(radius) - log_eps) / q.p);
    }
    const double radius = std::sqrt(radius_squared);
    int lower[3], upper[3];
    for (int d = 0; d < 3; ++d) {
        lower[d] =
            static_cast<int>(std::ceil((q.pc[d] - radius - r[d]) / grid.L[d]));
        upper[d] =
            static_cast<int>(std::floor((q.pc[d] + radius - r[d]) / grid.L[d]));
    }
    double result = 0;
    for (int i = lower[0]; i <= upper[0]; ++i)
        for (int j = lower[1]; j <= upper[1]; ++j)
            for (int k = lower[2]; k <= upper[2]; ++k) {
                const double image[3] = {r[0] + i * grid.L[0],
                                         r[1] + j * grid.L[1],
                                         r[2] + k * grid.L[2]};
                const double dx = image[0] - q.pc[0];
                const double dy = image[1] - q.pc[1];
                const double dz = image[2] - q.pc[2];
                if (dx * dx + dy * dy + dz * dz > radius_squared)
                    continue;
                result += bare ? host_product_shape_bare(q, image)
                               : host_product_shape(q, image);
            }
    return result;
}

std::vector<double> host_collocate(const UniformGrid& grid,
                                   const std::vector<DeviceProduct>& prods) {
    const long ng = grid.npts();
    std::vector<double> rho(static_cast<std::size_t>(ng), 0.0);
    for (long i = 0; i < ng; ++i) {
        double r[3];
        grid.point(i, r);
        for (const auto& q : prods)
            rho[static_cast<std::size_t>(i)] +=
                periodic_product_shape(grid, q, r, false);
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
            acc[q] += v * periodic_product_shape(grid, prods[q], r, true);
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
    std::vector<double> rho;
    std::vector<double> veff;
    std::vector<double> acc;
    KsChainEnergies e;
};

ChainHost run_device(const UniformGrid& grid,
                     const std::vector<DeviceProduct>& prods,
                     const std::vector<double>& n_ion,
                     const std::vector<double>& v_loc, bool do_integrate,
                     XCType xc) {
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

    out.rho.resize(static_cast<std::size_t>(ng));
    check_cuda(cudaMemcpy(out.rho.data(), chain.d_rho(),
                          static_cast<size_t>(ng) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "D2H rho");
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

// ── Single s–s product: V_eff + energies for each PBE-family model ───────
TEST_P(CudaGpwKsGgaFamily, SSMatchesCpuKsGrid) {
    const auto xc = GetParam();
    UniformGrid grid = cubic_grid(16, 8.0);
    const double pc[3] = {4.0, 4.0, 4.0};
    std::vector<DeviceProduct> prods = {make_ss(/*p=*/2.5, pc, /*w=*/1.3)};

    const auto rho = host_collocate(grid, prods);
    const std::size_t n = rho.size();
    std::vector<double> n_ion(n, 0.0);
    std::vector<double> v_loc;
    const auto cpu =
        ks_grid_build_rks(grid, rho, n_ion, v_loc, xc,
                          /*nonperiodic=*/false, false, false, false);

    const auto dev =
        run_device(grid, prods, n_ion, v_loc, /*integrate=*/true, xc);

    EXPECT_LE(max_abs_diff(dev.rho, rho), 1e-13 * std::max(1.0, max_abs(rho)));

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

// ── Multi-center + n_ion + V_loc for each PBE-family model ────────────────
TEST_P(CudaGpwKsGgaFamily, MultiWithIonAndVloc) {
    const auto xc = GetParam();
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
    // Gaussian values below each product's eps_win are absent from the
    // sampled density before spectral differentiation. Keeping raw tails
    // changes the GGA flux in near-vacuum regions despite tiny delta-rho.
    const long tail = (10 * grid.n[1] + 7) * grid.n[2] + 18;
    EXPECT_DOUBLE_EQ(rho[tail], 0.0);
    const double r_tail[3] = {5, 3.5, 9};
    double raw_tail = 0;
    for (const auto& q : prods)
        raw_tail += host_product_shape(q, r_tail);
    EXPECT_GT(raw_tail, 1e-14);
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

    const auto cpu = ks_grid_build_rks(grid, rho, n_ion, v_loc, xc, false,
                                       false, false, false);
    const auto dev =
        run_device(grid, prods, n_ion, v_loc, /*integrate=*/true, xc);

    EXPECT_LE(max_abs_diff(dev.rho, rho), 1e-13 * std::max(1.0, max_abs(rho)));

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
    // At (0,3,3) both +/-3-Bohr images contribute exp(-18). Omitting
    // either image shifts Padé V_xc by 1.0619e-3 Ha and E_xc by 1.3331e-10 Ha.
    const long boundary = (0 * grid.n[1] + 6) * grid.n[2] + 6;
    EXPECT_NEAR(rho[boundary], 2 * std::exp(-18.0), 1e-22);
    const double r_boundary[3] = {0, 3, 3};
    EXPECT_GT(
        std::abs(rho[boundary] - host_product_shape(prods[0], r_boundary)),
        1e-8);
    const std::size_t n = rho.size();
    std::vector<double> n_ion(n, 0.0);
    std::vector<double> v_loc;
    const auto cpu = ks_grid_build_rks(grid, rho, n_ion, v_loc, XCType::PADE,
                                       false, false, false, false);
    const auto dev = run_device(grid, prods, n_ion, v_loc, /*integrate=*/false,
                                XCType::PADE);
    EXPECT_LE(max_abs_diff(dev.rho, rho), 1e-13 * std::max(1.0, max_abs(rho)));
    EXPECT_NEAR(dev.e.e_xc, cpu.e_xc,
                1e-10 * std::max(1.0, std::abs(cpu.e_xc)));
    const double peak_v = max_abs(cpu.veff);
    EXPECT_LE(max_abs_diff(dev.veff, cpu.veff),
              1e-9 * std::max(1.0, peak_v) + 1e-12);
}

INSTANTIATE_TEST_SUITE_P(PbeFamily, CudaGpwKsGgaFamily,
                         ::testing::Values(XCType::PBE, XCType::PBEsol,
                                           XCType::RevPBE, XCType::RPBE),
                         [](const ::testing::TestParamInfo<XCType>& info)
                             -> std::string {
                             switch (info.param) {
                             case XCType::PBE:
                                 return "PBE";
                             case XCType::PBEsol:
                                 return "PBEsol";
                             case XCType::RevPBE:
                                 return "RevPBE";
                             case XCType::RPBE:
                                 return "RPBE";
                             default:
                                 return "Invalid";
                             }
                         });
namespace {
struct SpinTestBuffers {
    cudaStream_t stream;
    std::vector<void*> allocations;
    explicit SpinTestBuffers(cudaStream_t stream) : stream(stream) {}
    ~SpinTestBuffers() {
        cudaStreamSynchronize(stream);
        for (void* pointer : allocations)
            cudaFree(pointer);
    }
    template <class T>
    T* allocate(std::size_t count) {
        if (!count)
            return nullptr;
        T* pointer = nullptr;
        const auto status = cudaMalloc(&pointer, count * sizeof(T));
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
        allocations.push_back(pointer);
        return pointer;
    }
    template <class T>
    T* upload(const std::vector<T>& values) {
        auto* pointer = allocate<T>(values.size());
        if (pointer)
            check_cuda(cudaMemcpyAsync(pointer, values.data(),
                                       values.size() * sizeof(T),
                                       cudaMemcpyHostToDevice, stream),
                       "spin input upload");
        return pointer;
    }
};
struct SpinChainHost {
    KsChainEnergies e;
    std::vector<double> rho_a, rho_b, va, vb, acc_a, acc_b;
};
struct SpinTestChain {
    cudaStream_t stream = nullptr;
    std::unique_ptr<KsChainDevice> chain;
    explicit SpinTestChain(const UniformGrid& grid) {
        const auto status = cudaStreamCreate(&stream);
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
        try {
            chain = std::make_unique<KsChainDevice>(to_desc(grid), stream);
        } catch (...) {
            cudaStreamDestroy(stream);
            throw;
        }
    }
    ~SpinTestChain() {
        cudaStreamSynchronize(stream);
        chain.reset();
        cudaStreamDestroy(stream);
    }
    SpinChainHost run(const UniformGrid& grid,
                      const std::vector<DeviceProduct>& a,
                      const std::vector<DeviceProduct>& b,
                      const std::vector<double>& ions,
                      const std::vector<double>& local, XCType xc) {
        SpinTestBuffers buffers(stream);
        auto* da = buffers.upload(a);
        auto* db = buffers.upload(b);
        auto* di = buffers.upload(ions);
        auto* dl = buffers.upload(local);
        auto* aa = buffers.allocate<double>(a.size());
        auto* ab = buffers.allocate<double>(b.size());
        SpinChainHost out;
        out.e = chain->run_uks(da, int(a.size()), db, int(b.size()), di, dl, aa,
                               ab, xc);
        auto download = [&](std::vector<double>& target, const double* source,
                            std::size_t count) {
            target.resize(count);
            if (count)
                check_cuda(cudaMemcpyAsync(target.data(), source,
                                           count * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream),
                           "spin output download");
        };
        download(out.rho_a, chain->d_rho_a(), grid.npts());
        download(out.rho_b, chain->d_rho_b(), grid.npts());
        download(out.va, chain->d_veff(), grid.npts());
        download(out.vb, chain->d_veff_b(), grid.npts());
        download(out.acc_a, aa, a.size());
        download(out.acc_b, ab, b.size());
        check_cuda(cudaStreamSynchronize(stream), "spin output fence");
        return out;
    }
};
void spin_matches_cpu(const UniformGrid& grid,
                      const std::vector<DeviceProduct>& a,
                      const std::vector<DeviceProduct>& b,
                      const std::vector<double>& ions,
                      const std::vector<double>& local, XCType xc,
                      const SpinChainHost& device) {
    const auto ra = host_collocate(grid, a), rb = host_collocate(grid, b);
    const auto cpu = ATV::cpu::gpw::internal::ks_grid_build_uks(
        grid, ra, rb, ions, local, xc, false, false, false, false);
    EXPECT_LE(max_abs_diff(device.rho_a, ra),
              1e-13 * std::max(1., max_abs(ra)));
    EXPECT_LE(max_abs_diff(device.rho_b, rb),
              1e-13 * std::max(1., max_abs(rb)));
    EXPECT_NEAR(device.e.e_hartree, cpu.e_hartree,
                1e-10 * std::max(1., std::abs(cpu.e_hartree)));
    EXPECT_NEAR(device.e.e_xc, cpu.e_xc,
                1e-10 * std::max(1., std::abs(cpu.e_xc)));
    EXPECT_NEAR(device.e.e_locgrid, cpu.e_locgrid,
                1e-10 * std::max(1., std::abs(cpu.e_locgrid)));
    for (auto pair : {std::pair{&device.va, &cpu.veff_a},
                      std::pair{&device.vb, &cpu.veff_b}})
        EXPECT_LE(max_abs_diff(*pair.first, *pair.second),
                  1e-9 * std::max(1., max_abs(*pair.second)) + 1e-12);
    const auto aa = host_integrate(grid, a, cpu.veff_a),
               ab = host_integrate(grid, b, cpu.veff_b);
    EXPECT_LE(max_abs_diff(device.acc_a, aa),
              1e-9 * std::max(1., max_abs(aa)) + 1e-12);
    EXPECT_LE(max_abs_diff(device.acc_b, ab),
              1e-9 * std::max(1., max_abs(ab)) + 1e-12);
}
} // namespace

TEST_P(CudaGpwKsGgaFamily, SpinUnequalChannelsMatchCpuAndSwap) {
    const auto grid = cubic_grid(12, 6.);
    const double ca[3] = {2.2, 2.5, 2.7}, cb[3] = {3.7, 3.1, 3.4};
    const std::vector<DeviceProduct> a = {make_ss(.5, ca, .8)},
                                     b = {make_ss(.7, cb, .4)};
    const std::vector<double> ions(grid.npts(), .02), local(grid.npts(), -.03);
    SpinTestChain fixture(grid);
    const auto result = fixture.run(grid, a, b, ions, local, GetParam());
    spin_matches_cpu(grid, a, b, ions, local, GetParam(), result);
    const auto swapped = fixture.run(grid, b, a, ions, local, GetParam());
    EXPECT_NEAR(result.e.e_xc, swapped.e.e_xc, 1e-11);
    EXPECT_LE(max_abs_diff(result.va, swapped.vb),
              1e-9 * std::max(1., max_abs(result.va)));
    EXPECT_LE(max_abs_diff(result.vb, swapped.va),
              1e-9 * std::max(1., max_abs(result.vb)));
    // Different centers make the spin gradients oppose on part of the grid.
    const auto ga = ATV::cpu::gpw::gradient(grid, result.rho_a);
    const auto gb = ATV::cpu::gpw::gradient(grid, result.rho_b);
    double smallest_cross = 0;
    for (std::size_t i = 0; i < result.rho_a.size(); ++i)
        smallest_cross =
            std::min(smallest_cross, ga[0][i] * gb[0][i] + ga[1][i] * gb[1][i]
                                         + ga[2][i] * gb[2][i]);
    EXPECT_LT(smallest_cross, -1e-5);
}

TEST_P(CudaGpwKsGgaFamily, SpinEmptyChannelRetainsCompleteCpuPotential) {
    const auto grid = cubic_grid(12, 6.);
    const double center[3] = {2.2, 2.5, 2.7};
    const std::vector<DeviceProduct> a = {make_ss(.4, center, .8)}, empty;
    const std::vector<double> ions(grid.npts(), 0), local;
    SpinTestChain fixture(grid);
    const auto result = fixture.run(grid, a, empty, ions, local, GetParam());
    spin_matches_cpu(grid, a, empty, ions, local, GetParam(), result);
    for (double value : result.rho_b)
        EXPECT_DOUBLE_EQ(value, 0);
    const auto swapped = fixture.run(grid, empty, a, ions, local, GetParam());
    spin_matches_cpu(grid, empty, a, ions, local, GetParam(), swapped);
    EXPECT_LE(max_abs_diff(result.va, swapped.vb),
              1e-9 * std::max(1., max_abs(result.va)));
    EXPECT_LE(max_abs_diff(result.vb, swapped.va),
              1e-9 * std::max(1., max_abs(result.vb)));
    // Zero weighted product is distinct from the no-products admission error.
    auto zero = a;
    zero[0].weight = 0;
    const auto vacuum = fixture.run(grid, zero, empty, ions, local, GetParam());
    EXPECT_DOUBLE_EQ(vacuum.e.e_xc, 0);
    for (double value : vacuum.va)
        EXPECT_DOUBLE_EQ(value, 0);
    for (double value : vacuum.vb)
        EXPECT_DOUBLE_EQ(value, 0);
}

TEST_P(CudaGpwKsGgaFamily, SpinDiscreteEnergyDerivativeAndPadeReuse) {
    const auto grid = cubic_grid(10, 6.);
    const double ca[3] = {2.2, 2.5, 2.7}, cb[3] = {3.7, 3.1, 3.4};
    const std::vector<DeviceProduct> a = {make_ss(.3, ca, .8)},
                                     b = {make_ss(.3, cb, .4)};
    const std::vector<double> ions(grid.npts(), .02), local(grid.npts(), -.03);
    SpinTestChain fixture(grid);
    const auto baseline = fixture.run(grid, a, b, ions, local, GetParam());
    auto total = [](const SpinChainHost& state) {
        return state.e.e_hartree + state.e.e_xc + state.e.e_locgrid;
    };
    // Both spin fields stay strictly positive throughout these perturbations;
    // no density floor/endpoint extension participates in this derivative.
    EXPECT_GT(*std::min_element(baseline.rho_a.begin(), baseline.rho_a.end()),
              1e-6);
    EXPECT_GT(*std::min_element(baseline.rho_b.begin(), baseline.rho_b.end()),
              1e-6);
    for (int spin = 0; spin < 2; ++spin) {
        auto plus_a = a, minus_a = a, plus_b = b, minus_b = b;
        constexpr double h = 1e-5;
        (spin ? plus_b : plus_a)[0].weight += h;
        (spin ? minus_b : minus_a)[0].weight -= h;
        const double derivative =
            (total(fixture.run(grid, plus_a, plus_b, ions, local, GetParam()))
             - total(
                 fixture.run(grid, minus_a, minus_b, ions, local, GetParam())))
            / (2 * h);
        const double expected =
            grid.weight() * (spin ? baseline.acc_b[0] : baseline.acc_a[0]);
        EXPECT_NEAR(derivative, expected,
                    1e-7 * std::max(1., std::abs(expected)));
    }
    const auto pade = fixture.run(grid, a, b, ions, local, XCType::PADE);
    spin_matches_cpu(grid, a, b, ions, local, XCType::PADE, pade);
    const auto repeat = fixture.run(grid, a, b, ions, local, GetParam());
    EXPECT_NEAR(repeat.e.e_xc, baseline.e.e_xc, 1e-12);
    EXPECT_LE(max_abs_diff(repeat.va, baseline.va), 1e-12);
    EXPECT_LE(max_abs_diff(repeat.vb, baseline.vb), 1e-12);
}

TEST_P(CudaGpwKsGgaFamily, SpinEqualChannelsReduceToRestricted) {
    const auto grid = cubic_grid(12, 6.);
    const double center[3] = {2.2, 2.5, 2.7};
    const std::vector<DeviceProduct> a = {make_ss(.4, center, .8)};
    auto total = a;
    total[0].weight *= 2;
    const std::vector<double> ions(grid.npts(), 0), local;
    SpinTestChain fixture(grid);
    const auto spin = fixture.run(grid, a, a, ions, local, GetParam());
    const auto restricted =
        run_device(grid, total, ions, local, true, GetParam());
    EXPECT_NEAR(spin.e.e_xc, restricted.e.e_xc, 1e-10);
    EXPECT_NEAR(spin.e.e_hartree, restricted.e.e_hartree, 1e-10);
    EXPECT_LE(max_abs_diff(spin.va, restricted.veff),
              1e-9 * std::max(1., max_abs(restricted.veff)));
    EXPECT_LE(max_abs_diff(spin.vb, restricted.veff),
              1e-9 * std::max(1., max_abs(restricted.veff)));
}
namespace {
using RawSpinPoint = std::array<double, 8>;
using RawSpinValue = std::array<double, 9>; // va,vb,epsilon,flux_a[3],flux_b[3]
std::vector<RawSpinValue>
run_raw_spin_points(XCType xc, const std::vector<RawSpinPoint>& points) {
    SpinTestBuffers buffers(nullptr);
    const auto n = points.size();
    std::array<std::vector<double>, 8> input;
    std::array<double*, 8> device{};
    for (int channel = 0; channel < 8; ++channel) {
        input[channel].resize(n);
        for (std::size_t i = 0; i < n; ++i)
            input[channel][i] = points[i][channel];
        device[channel] = buffers.upload(input[channel]);
    }
    auto* va = buffers.allocate<double>(n);
    auto* vb = buffers.allocate<double>(n);
    auto* epsilon = buffers.allocate<double>(n);
    ATV::cuda::xc::xc_pbe_spin_grid(
        xc, device[0], device[1], device[2], device[3], device[4], device[5],
        device[6], device[7], va, vb, epsilon, int(n), nullptr,
        ATV::cuda::xc::SpinDensityPolicy::RawFunctional);
    std::array<const double*, 9> output = {va,        vb,        epsilon,
                                           device[2], device[3], device[4],
                                           device[5], device[6], device[7]};
    std::vector<RawSpinValue> result(n);
    for (int channel = 0; channel < 9; ++channel) {
        std::vector<double> values(n);
        check_cuda(cudaMemcpy(values.data(), output[channel],
                              n * sizeof(double), cudaMemcpyDeviceToHost),
                   "raw spin point download");
        for (std::size_t i = 0; i < n; ++i)
            result[i][channel] = values[i];
    }
    return result;
}
} // namespace

// This checks the existing GPW point API contract, including its signed-density
// extension. It does NOT establish variationality of that unphysical extension
// or the regulated empty-spin endpoint; the independent gap repro is retained
// under .molq/gpw-uks-pbe rather than presented as a passed derivative gate.
TEST_P(CudaGpwKsGgaFamily, RawSpinPointPolicyMatchesCpuContract) {
    const std::vector<RawSpinPoint> points = {
        {.2, .1, .2, -.1, .1, -.1, .05, .02},
        {.2, 0, .1, .1, 0, 0, 0, 0},
        {0, .2, 0, 0, 0, .1, .1, 0},
        {.2, -.01, .2, -.1, .1, -.1, .05, .02},
        {-.01, .2, -.1, .05, .02, .2, -.1, .1},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {-1e-4, 0, .01, 0, 0, 0, 0, 0}};
    const auto actual = run_raw_spin_points(GetParam(), points);
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto& p = points[i];
        double saa = 0, sab = 0, sbb = 0;
        for (int k = 0; k < 3; ++k) {
            saa += p[2 + k] * p[2 + k];
            sab += p[2 + k] * p[5 + k];
            sbb += p[5 + k] * p[5 + k];
        }
        const auto cpu =
            ATV::cpu::xc::evaluate_xc(GetParam(), p[0], p[1], saa, sab, sbb);
        RawSpinValue expected{};
        if (p[0] + p[1] >= 1e-12) {
            expected[0] = cpu.vrho_alpha;
            expected[1] = cpu.vrho_beta;
            expected[2] = cpu.energy_density / (p[0] + p[1]);
            for (int k = 0; k < 3; ++k) {
                expected[3 + k] =
                    2 * cpu.vsigma_aa * p[2 + k] + cpu.vsigma_ab * p[5 + k];
                expected[6 + k] =
                    2 * cpu.vsigma_bb * p[5 + k] + cpu.vsigma_ab * p[2 + k];
            }
        }
        for (int channel = 0; channel < 9; ++channel)
            EXPECT_NEAR(actual[i][channel], expected[channel],
                        1e-11 * std::max(1., std::abs(expected[channel])))
                << "point=" << i << " channel=" << channel;
        EXPECT_NEAR((p[0] + p[1]) * actual[i][2], cpu.energy_density, 1e-11);
    }
}

TEST_P(CudaGpwKsGgaFamily,
       RawSpinPhysicalEnergyDerivativesMatchPotentialAndFlux) {
    constexpr double h = 1e-6;
    const RawSpinPoint base = {.2, .07, .1, -.2, .08, -.04, .06, .03};
    std::vector<RawSpinPoint> inputs(17, base);
    for (int channel = 0; channel < 8; ++channel) {
        inputs[1 + 2 * channel][channel] += h;
        inputs[2 + 2 * channel][channel] -= h;
    }
    const auto actual = run_raw_spin_points(GetParam(), inputs);
    for (int channel = 0; channel < 8; ++channel) {
        const int plus = 1 + 2 * channel, minus = 2 + 2 * channel;
        const double eplus =
            (inputs[plus][0] + inputs[plus][1]) * actual[plus][2];
        const double eminus =
            (inputs[minus][0] + inputs[minus][1]) * actual[minus][2];
        const int output = channel < 2 ? channel : channel + 1;
        EXPECT_NEAR(actual[0][output], (eplus - eminus) / (2 * h), 5e-9)
            << "density/gradient channel=" << channel;
    }
}

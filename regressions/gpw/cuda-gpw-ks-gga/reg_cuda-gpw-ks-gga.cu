/// @file regressions/gpw/cuda-gpw-ks-gga/reg_cuda-gpw-ks-gga.cu
/// @brief END-TO-END dual-gate: composed CUDA GPW KS chain (closed-shell PBE)
///        vs CPU `ks_grid_build_rks(..., XcKind::PBE)` + continuous products.
///
/// Gates:
///   1. CPU path produces finite e_H, e_xc, V_eff, integrate acc;
///   2. CUDA path matches within stated stage tols (Hartree).
///
/// Not a full GPW method. See REFERENCE.md.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/ks_grid_build.h"
#include "atomiverse/cuda/gpw/internal/ks_chain.cuh"
#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"

using ATV::cpu::gpw::UniformGrid;
using ATV::cpu::gpw::XcKind;
using ATV::cpu::gpw::internal::ks_grid_build_rks;
using ATV::cuda::gpw::internal::CollocateGridDesc;
using ATV::cuda::gpw::internal::DeviceProduct;
using ATV::cuda::gpw::internal::KsChainDevice;
using ATV::cuda::gpw::internal::KsXcKind;

namespace {

// Stage tols (composite of spectral + PBE + Poisson foothold units; not bit-exact).
constexpr double kERel = 1.0e-10;
constexpr double kVAbs = 1.0e-12;
constexpr double kVRel = 1.0e-9;
constexpr double kAAbs = 1.0e-12;
constexpr double kARel = 1.0e-9;

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

std::vector<double> collocate(const UniformGrid& grid,
                              const std::vector<DeviceProduct>& prods) {
    const long ng = grid.npts();
    std::vector<double> rho(static_cast<std::size_t>(ng), 0.0);
    for (long i = 0; i < ng; ++i) {
        double r[3];
        grid.point(i, r);
        for (const auto& q : prods)
            rho[static_cast<std::size_t>(i)] += host_shape(q, r);
    }
    return rho;
}

std::vector<double> integrate(const UniformGrid& grid,
                              const std::vector<DeviceProduct>& prods,
                              const std::vector<double>& V) {
    const long ng = grid.npts();
    std::vector<double> acc(prods.size(), 0.0);
    for (long i = 0; i < ng; ++i) {
        double r[3];
        grid.point(i, r);
        const double v = V[static_cast<std::size_t>(i)];
        for (std::size_t q = 0; q < prods.size(); ++q)
            acc[q] += v * host_shape_bare(prods[q], r);
    }
    return acc;
}

double max_abs(const std::vector<double>& a) {
    double m = 0.0;
    for (double v : a)
        m = std::max(m, std::abs(v));
    return m;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

bool near_rel(double a, double b, double rel) {
    return std::abs(a - b) <= rel * std::max(1.0, std::abs(b));
}

std::string parse_device(int argc, char** argv) {
    std::string want = "both";
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strncmp(a, "--device=", 9) == 0)
            want = a + 9;
        else if (std::strcmp(a, "--device") == 0 && i + 1 < argc)
            want = argv[++i];
        else if (std::strcmp(a, "cpu") == 0 || std::strcmp(a, "cuda") == 0
                 || std::strcmp(a, "both") == 0)
            want = a;
    }
    return want;
}

struct Case {
    UniformGrid grid;
    std::vector<DeviceProduct> prods;
    std::vector<double> n_ion;
    std::vector<double> v_loc;
};

Case make_case() {
    Case c;
    c.grid = cubic_grid(16, 8.0);
    {
        const double c0[3] = {4.0, 4.0, 4.0};
        c.prods.push_back(make_ss(2.0, c0, 1.0));
    }
    {
        const double c1[3] = {3.5, 4.0, 4.5};
        c.prods.push_back(make_ss(2.5, c1, 0.6));
    }
    {
        const double c2[3] = {4.5, 4.0, 4.5};
        c.prods.push_back(make_ss(2.5, c2, 0.6));
    }
    const auto rho = collocate(c.grid, c.prods);
    const std::size_t n = rho.size();
    c.n_ion.assign(n, 0.0);
    c.v_loc.assign(n, -0.02);
    {
        const double pc[3] = {4.0, 4.0, 4.0};
        const double alpha = 2.5;
        const double pref = std::pow(alpha / M_PI, 1.5);
        for (std::size_t i = 0; i < n; ++i) {
            double r[3];
            c.grid.point(static_cast<long>(i), r);
            const double dx = r[0] - pc[0];
            const double dy = r[1] - pc[1];
            const double dz = r[2] - pc[2];
            c.n_ion[i] =
                pref * std::exp(-alpha * (dx * dx + dy * dy + dz * dz));
        }
    }
    return c;
}

struct Result {
    double e_hartree = 0.0;
    double e_xc = 0.0;
    double e_locgrid = 0.0;
    std::vector<double> veff;
    std::vector<double> acc;
};

Result run_cpu(const Case& c) {
    const auto rho = collocate(c.grid, c.prods);
    const auto ks =
        ks_grid_build_rks(c.grid, rho, c.n_ion, c.v_loc, XcKind::PBE, false,
                          false, false, false);
    Result r;
    r.e_hartree = ks.e_hartree;
    r.e_xc = ks.e_xc;
    r.e_locgrid = ks.e_locgrid;
    r.veff = ks.veff;
    r.acc = integrate(c.grid, c.prods, ks.veff);
    return r;
}

Result run_cuda(const Case& c) {
    const long ng = c.grid.npts();
    const int nprod = static_cast<int>(c.prods.size());
    DeviceProduct* d_prods = nullptr;
    check_cuda(cudaMalloc(&d_prods,
                          static_cast<size_t>(nprod) * sizeof(DeviceProduct)),
               "malloc prods");
    check_cuda(cudaMemcpy(d_prods, c.prods.data(),
                          static_cast<size_t>(nprod) * sizeof(DeviceProduct),
                          cudaMemcpyHostToDevice),
               "H2D prods");
    double* d_n_ion = nullptr;
    check_cuda(cudaMalloc(&d_n_ion, static_cast<size_t>(ng) * sizeof(double)),
               "malloc n_ion");
    check_cuda(cudaMemcpy(d_n_ion, c.n_ion.data(),
                          static_cast<size_t>(ng) * sizeof(double),
                          cudaMemcpyHostToDevice),
               "H2D n_ion");
    double* d_v_loc = nullptr;
    check_cuda(cudaMalloc(&d_v_loc, static_cast<size_t>(ng) * sizeof(double)),
               "malloc v_loc");
    check_cuda(cudaMemcpy(d_v_loc, c.v_loc.data(),
                          static_cast<size_t>(ng) * sizeof(double),
                          cudaMemcpyHostToDevice),
               "H2D v_loc");
    double* d_acc = nullptr;
    check_cuda(cudaMalloc(&d_acc, static_cast<size_t>(nprod) * sizeof(double)),
               "malloc acc");

    KsChainDevice chain(to_desc(c.grid));
    const auto e =
        chain.run(d_prods, nprod, d_n_ion, d_v_loc, d_acc, KsXcKind::PBE);

    Result r;
    r.e_hartree = e.e_hartree;
    r.e_xc = e.e_xc;
    r.e_locgrid = e.e_locgrid;
    r.veff.resize(static_cast<std::size_t>(ng));
    check_cuda(cudaMemcpy(r.veff.data(), chain.d_veff(),
                          static_cast<size_t>(ng) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "D2H veff");
    r.acc.resize(static_cast<std::size_t>(nprod));
    check_cuda(cudaMemcpy(r.acc.data(), d_acc,
                          static_cast<size_t>(nprod) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "D2H acc");

    cudaFree(d_prods);
    cudaFree(d_n_ion);
    cudaFree(d_v_loc);
    cudaFree(d_acc);
    return r;
}

int compare(const Result& cpu, const Result& cuda) {
    int rc = 0;
    auto check_e = [&](const char* name, double a, double b) {
        if (!near_rel(a, b, kERel)) {
            std::fprintf(stderr, "FAIL: %s cuda=%.12e cpu=%.12e |Δ|=%.3e\n",
                         name, a, b, std::abs(a - b));
            rc = 1;
        } else {
            std::printf("OK %s cuda=%.12e cpu=%.12e |Δ|=%.3e\n", name, a, b,
                        std::abs(a - b));
        }
    };
    check_e("e_hartree", cuda.e_hartree, cpu.e_hartree);
    check_e("e_xc", cuda.e_xc, cpu.e_xc);
    check_e("e_locgrid", cuda.e_locgrid, cpu.e_locgrid);

    const double peak_v = max_abs(cpu.veff);
    const double max_dv = max_abs_diff(cuda.veff, cpu.veff);
    const double tol_v = kVRel * std::max(1.0, peak_v) + kVAbs;
    if (max_dv > tol_v) {
        std::fprintf(stderr, "FAIL: max|ΔV_eff|=%.3e tol=%.3e peak=%.3e\n",
                     max_dv, tol_v, peak_v);
        rc = 1;
    } else {
        std::printf("OK V_eff max|Δ|=%.3e peak=%.3e tol=%.3e\n", max_dv, peak_v,
                    tol_v);
    }

    const double peak_a = max_abs(cpu.acc);
    const double max_da = max_abs_diff(cuda.acc, cpu.acc);
    const double tol_a = kARel * std::max(1.0, peak_a) + kAAbs;
    if (max_da > tol_a) {
        std::fprintf(stderr, "FAIL: max|Δacc|=%.3e tol=%.3e peak=%.3e\n", max_da,
                     tol_a, peak_a);
        rc = 1;
    } else {
        std::printf("OK integrate max|Δ|=%.3e peak=%.3e tol=%.3e\n", max_da,
                    peak_a, tol_a);
    }
    return rc;
}

} // namespace

int main(int argc, char** argv) {
    const std::string want = parse_device(argc, argv);
    const Case c = make_case();

    Result cpu_r, cuda_r;
    const bool run_cpu_leg = (want == "cpu" || want == "both");
    const bool run_cuda_leg = (want == "cuda" || want == "both");

    if (run_cpu_leg) {
        cpu_r = run_cpu(c);
        if (!std::isfinite(cpu_r.e_hartree) || !std::isfinite(cpu_r.e_xc)
            || !std::isfinite(cpu_r.e_locgrid))
            die("CPU energies non-finite");
        std::printf("CPU e_H=%.12e e_xc=%.12e e_loc=%.12e\n", cpu_r.e_hartree,
                    cpu_r.e_xc, cpu_r.e_locgrid);
    }
    if (run_cuda_leg) {
        cuda_r = run_cuda(c);
        if (!std::isfinite(cuda_r.e_hartree) || !std::isfinite(cuda_r.e_xc)
            || !std::isfinite(cuda_r.e_locgrid))
            die("CUDA energies non-finite");
        std::printf("CUDA e_H=%.12e e_xc=%.12e e_loc=%.12e\n", cuda_r.e_hartree,
                    cuda_r.e_xc, cuda_r.e_locgrid);
    }

    if (want == "both") {
        const int rc = compare(cpu_r, cuda_r);
        if (rc == 0)
            std::printf("PASS cuda-gpw-ks-gga dual-gate\n");
        return rc;
    }
    std::printf("PASS cuda-gpw-ks-gga (%s only)\n", want.c_str());
    return 0;
}

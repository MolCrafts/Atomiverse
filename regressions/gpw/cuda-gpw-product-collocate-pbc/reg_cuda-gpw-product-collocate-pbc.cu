/// @file regressions/gpw/cuda-gpw-product-collocate-pbc/reg_cuda-gpw-product-collocate-pbc.cu
/// @brief END-TO-END: PBC min-image product collocate vs collint + H₂ SG
///        device product KS dual-gate (drop host cbg mid-SCF).
///
/// See REFERENCE.md. Oracle: same-settings CPU collint + CPU GpwDft.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/product_collint.h"
#include "atomiverse/cuda/gpw/gpw_solver.cuh"
#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GpwAtom;
using ATV::cpu::gpw::GridProduct;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::UniformGrid;
using ATV::cpu::gpw::XcKind;
using ATV::cpu::gpw::internal::collint_collocate_product;
using ATV::cuda::gpw::internal::collocate_integrate_rho;
using ATV::cuda::gpw::internal::collocate_products;
using ATV::cuda::gpw::internal::CollocateGridDesc;
using ATV::cuda::gpw::internal::DeviceProduct;

namespace {

constexpr double kChargeRelTol = 5.0e-4;
constexpr double kETolHa = 5.0e-6;

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

// ── Edge collocate charge vs collint ──────────────────────────────────────
int check_edge_collocate() {
    UniformGrid grid;
    for (int d = 0; d < 3; ++d) {
        grid.n[d] = 24;
        grid.L[d] = 8.0;
        grid.origin[d] = 0.0;
    }
    DeviceProduct q{};
    q.p = 3.0;
    q.weight = 1.0;
    q.pc[0] = 0.15;
    q.pc[1] = 4.0;
    q.pc[2] = 4.0;
    for (int d = 0; d < 3; ++d) {
        q.ra[d] = q.pc[d];
        q.rb[d] = q.pc[d];
        q.la[d] = 0;
        q.lb[d] = 0;
    }

    const long ng = grid.npts();
    DeviceProduct* d_prods = nullptr;
    double* d_rho = nullptr;
    double* d_sum = nullptr;
    check_cuda(cudaMalloc(&d_prods, sizeof(DeviceProduct)), "malloc prods");
    check_cuda(cudaMalloc(&d_rho, static_cast<size_t>(ng) * sizeof(double)),
               "malloc rho");
    check_cuda(cudaMalloc(&d_sum, sizeof(double)), "malloc sum");
    check_cuda(
        cudaMemcpy(d_prods, &q, sizeof(DeviceProduct), cudaMemcpyHostToDevice),
        "H2D prod");
    check_cuda(cudaMemset(d_rho, 0, static_cast<size_t>(ng) * sizeof(double)),
               "zero rho");

    CollocateGridDesc desc;
    for (int i = 0; i < 3; ++i) {
        desc.n[i] = grid.n[i];
        desc.L[i] = grid.L[i];
        desc.origin[i] = grid.origin[i];
    }
    desc.periodic = true;
    collocate_products(desc, d_prods, 1, d_rho);
    collocate_integrate_rho(d_rho, ng, grid.weight(), d_sum);
    check_cuda(cudaDeviceSynchronize(), "sync collocate");
    double q_dev = 0.0;
    check_cuda(
        cudaMemcpy(&q_dev, d_sum, sizeof(double), cudaMemcpyDeviceToHost),
        "D2H charge");
    cudaFree(d_prods);
    cudaFree(d_rho);
    cudaFree(d_sum);

    GridProduct gp;
    gp.p = q.p;
    gp.k = 1.0;
    for (int d = 0; d < 3; ++d) {
        gp.pc[d] = q.pc[d];
        gp.ra[d] = q.ra[d];
        gp.rb[d] = q.rb[d];
        gp.la[d] = q.la[d];
        gp.lb[d] = q.lb[d];
    }
    std::vector<double> rho_cpu(static_cast<std::size_t>(ng), 0.0);
    collint_collocate_product(grid, gp, q.weight, std::abs(q.weight), 1e-16,
                              rho_cpu.data());
    double q_cpu = 0.0;
    for (double v : rho_cpu)
        q_cpu += v;
    q_cpu *= grid.weight();

    const double rel =
        std::abs(q_dev - q_cpu) / std::max(std::abs(q_cpu), 1e-30);
    std::printf("edge collocate: Q_dev=%.12f  Q_collint=%.12f  rel=%.3e\n",
                q_dev, q_cpu, rel);
    if (!(rel <= kChargeRelTol)) {
        std::fprintf(stderr,
                     "FAIL: edge charge rel=%.6e > tol=%.1e (dev=%.12f "
                     "collint=%.12f)\n",
                     rel, kChargeRelTol, q_dev, q_cpu);
        return 1;
    }
    // Guard: charge must be O(1), not the ~0.5 under-count of no-wrap.
    if (!(q_dev > 0.9 * q_cpu && q_cpu > 0.5)) {
        std::fprintf(stderr,
                     "FAIL: edge charge not PBC-recovered (dev=%.6f "
                     "collint=%.6f)\n",
                     q_dev, q_cpu);
        return 1;
    }
    return 0;
}

std::vector<GpwAtom> h2_atoms() {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    return {GpwAtom(rA, basis, pot, 1), GpwAtom(rB, basis, pot, 1)};
}

template <typename Method>
void set_h2(Method& m) {
    m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
    m.cutoff_ry = 80.0;
    m.n_occ = 1;
    m.max_iter = 40;
    m.tol = 1e-8;
    m.diis_depth = 6;
    m.xc = XcKind::PADE;
    m.nonperiodic = false;
    m.local_pp_on_grid = true;
    m.use_ot = false;
    m.ngrids = 1;
    m.eps_default = 1e-8;
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

} // namespace

int main(int argc, char** argv) {
    const std::string device = parse_device(argc, argv);
    const bool run_cpu = (device == "cpu" || device == "both");
    const bool run_cuda = (device == "cuda" || device == "both");
    if (!run_cpu && !run_cuda)
        die("unknown --device (use cpu|cuda|both)");

    // CUDA-only foothold: edge collocate needs a device.
    if (run_cuda) {
        if (check_edge_collocate() != 0)
            return 1;
    }

    auto atoms = h2_atoms();
    double e_cpu = 0.0, e_cuda = 0.0;

    if (run_cpu) {
        ATV::cpu::gpw::GpwDft m;
        set_h2(m);
        m.build(atoms);
        const auto& r = m.run(atoms);
        if (!r.converged) {
            std::fprintf(stderr, "FAIL: CPU H2 SCF not converged (iters=%d)\n",
                         r.iters);
            return 1;
        }
        e_cpu = r.e_total;
        std::printf("CPU H2:  E=%.12f Ha  iters=%d  ngrids=1\n", e_cpu,
                    r.iters);
    }

    if (run_cuda) {
        ATV::cuda::gpw::GpwDft m;
        set_h2(m);
        m.build(atoms);
        const auto& r = m.run(atoms);
        if (!r.converged) {
            std::fprintf(stderr,
                         "FAIL: CUDA H2 SCF not converged (iters=%d E=%.12f)\n",
                         r.iters, r.e_total);
            return 1;
        }
        e_cuda = r.e_total;
        std::printf("CUDA H2: E=%.12f Ha  iters=%d  ngrids=1 (device product "
                    "KS)\n",
                    e_cuda, r.iters);
    }

    if (run_cpu && run_cuda) {
        const double de = std::abs(e_cuda - e_cpu);
        std::printf("|E_cuda-E_cpu|=%.3e Ha  tol=%.1e\n", de, kETolHa);
        if (!(de <= kETolHa)) {
            std::fprintf(stderr, "FAIL: H2 energy dual-gate |dE|=%.6e > tol\n",
                         de);
            return 1;
        }
        std::printf(
            "PASS: cuda-gpw-product-collocate-pbc  edge charge OK + "
            "H2 |dE|=%.3e  (PBC collocate + SG device KS; perf_exempt)\n",
            de);
    } else {
        std::printf("PASS: cuda-gpw-product-collocate-pbc (--device=%s)\n",
                    device.c_str());
    }
    return 0;
}

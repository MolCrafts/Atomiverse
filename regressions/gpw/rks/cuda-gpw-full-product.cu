/// @file regressions/gpw/cuda-gpw-full-product/reg_cuda-gpw-full-product.cu
/// @brief END-TO-END dual-gate: public `cuda::gpw::GpwDft` vs CPU `GpwDft`.
///
/// First product slice — Γ closed-shell DIIS RKS Padé, H₂, energy + analytic
/// forces. Public API only for method surface; oracle is same-settings CPU.
/// See REFERENCE.md.

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../frame_fixture.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cuda/gpw/gpw_solver.cuh"
#include "harness/registry.h"

namespace atv_reg_gpw_cuda_gpw_full_product {

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::xc::XCType;
namespace fixture = ATV::regression::gpw_fixture;

namespace {

// Dual-gate tols (Hartree / Ha/Bohr). Composite of stage footholds + host
// STVnl/DIIS twin path — not bit-exact (cuFFT vs FFTW, tree reductions).
constexpr double kETolHa = 5.0e-6;
constexpr double kFTolHaBohr = 5.0e-5;

void die(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
}

std::vector<ATV::cpu::gpw::Kind> kinds() {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    return {{molrs::Element::H, basis, pot}};
}

ATV::cpu::Frame h2_frame() {
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    return fixture::frame_bohr({rA, rB}, {1, 1}, {12.0, 12.0, 12.0});
}

// Geometry and cell belong to the shared public Frame fixture.
template <typename Method>
void set_cfg(Method& m) {
    m.n_occ = 1;
    m.max_iter = 40;
    m.tol = 1e-8;
    m.diis_depth = 6;
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

int run(int argc, char** argv) {
    const std::string device = parse_device(argc, argv);
    const bool run_cpu = (device == "cpu" || device == "both");
    const bool run_cuda = (device == "cuda" || device == "both");
    if (!run_cpu && !run_cuda)
        die("unknown --device (use cpu|cuda|both)");

    const auto reference_frame = h2_frame();
    double e_cpu = 0.0, e_cuda = 0.0;
    std::vector<std::array<double, 3>> F_cpu, F_cuda;

    if (run_cpu) {
        auto frame = reference_frame;
        ATV::cpu::gpw::GpwDft m(kinds(), 80.0, XCType::PADE);
        set_cfg(m);
        m.build(frame);
        const auto& r = m.run(frame);
        if (!r.converged) {
            std::fprintf(stderr, "FAIL: CPU SCF not converged (iters=%d)\n",
                         r.iters);
            return 1;
        }
        e_cpu = r.e_total;
        m.forces(frame);
        F_cpu = fixture::forces_ha_bohr(frame);
        std::printf("CPU:  E=%.12f Ha  iters=%d  converged=%d\n", e_cpu,
                    r.iters, (int)r.converged);
    }

    if (run_cuda) {
        auto frame = reference_frame;
        ATV::cuda::gpw::GpwDft m(kinds(), 80.0, XCType::PADE);
        set_cfg(m);
        m.build(frame);
        const auto& r = m.run(frame);
        if (!r.converged) {
            std::fprintf(stderr, "FAIL: CUDA SCF not converged (iters=%d)\n",
                         r.iters);
            return 1;
        }
        e_cuda = r.e_total;
        m.forces(frame);
        F_cuda = fixture::forces_ha_bohr(frame);
        std::printf("CUDA: E=%.12f Ha  iters=%d  converged=%d\n", e_cuda,
                    r.iters, (int)r.converged);
    }

    if (run_cpu && run_cuda) {
        const double de = std::abs(e_cuda - e_cpu);
        std::printf("|E_cuda-E_cpu|=%.3e Ha  tol=%.1e\n", de, kETolHa);
        if (!(de <= kETolHa)) {
            std::fprintf(stderr, "FAIL: energy dual-gate |dE|=%.6e > tol\n",
                         de);
            return 1;
        }
        if (F_cpu.size() != F_cuda.size() || F_cpu.size() != 2) {
            die("force size mismatch");
        }
        double maxd = 0.0;
        for (std::size_t a = 0; a < F_cpu.size(); ++a)
            for (int k = 0; k < 3; ++k) {
                const double d = std::abs(F_cuda[a][k] - F_cpu[a][k]);
                maxd = std::max(maxd, d);
                std::printf("  F[%zu][%d] cpu=% .8e  cuda=% .8e  |d|=%.3e\n", a,
                            k, F_cpu[a][k], F_cuda[a][k], d);
            }
        std::printf("max|dF|=%.3e Ha/Bohr  tol=%.1e\n", maxd, kFTolHaBohr);
        if (!(maxd <= kFTolHaBohr)) {
            std::fprintf(stderr, "FAIL: force dual-gate max|dF|=%.6e > tol\n",
                         maxd);
            return 1;
        }
        std::printf("PASS: cuda-gpw-full-product  |dE|=%.3e  max|dF|=%.3e  "
                    "(H2 Γ RKS PADE; perf_exempt small cell)\n",
                    de, maxd);
    } else {
        std::printf("PASS: cuda-gpw-full-product (--device=%s only)\n",
                    device.c_str());
    }
    return 0;
}

} // namespace atv_reg_gpw_cuda_gpw_full_product

ATV_REGRESSION("gpw/rks", "cuda-gpw-full-product",
               atv_reg_gpw_cuda_gpw_full_product::run);

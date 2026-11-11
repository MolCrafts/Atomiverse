/// @file regressions/gpw/cuda-gpw-kpoints/reg_cuda-gpw-kpoints.cu
/// @brief END-TO-END dual-gate: multi-k RKS DIIS energy on public GpwDft.
///
/// H₂, k_mesh={2,1,1}, PADE, single-grid. Same-settings CPU is the oracle.
/// See REFERENCE.md. perf_exempt (small multi-k foothold).

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

namespace atv_reg_gpw_cuda_gpw_kpoints {

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::xc::XCType;
namespace fixture = ATV::regression::gpw_fixture;

namespace {

constexpr double kETolHa = 5.0e-6;

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
    m.k_mesh[0] = 2;
    m.k_mesh[1] = 1;
    m.k_mesh[2] = 1;
    // Keep both ±1/4 (nk=2); TR fold would collapse to a single off-Γ k.
    m.k_time_reversal = false;
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
    int nk_cpu = 0, nk_cuda = 0;

    if (run_cpu) {
        auto frame = reference_frame;
        ATV::cpu::gpw::GpwDft m(kinds(), 80.0, XCType::PADE);
        set_cfg(m);
        m.build(frame);
        const auto& r = m.run(frame);
        if (!r.converged) {
            std::fprintf(stderr,
                         "FAIL: CPU multi-k SCF not converged (iters=%d)\n",
                         r.iters);
            return 1;
        }
        e_cpu = r.e_total;
        nk_cpu = r.n_kpts;
        if (nk_cpu < 2) {
            std::fprintf(stderr, "FAIL: CPU n_kpts=%d expected >=2 (TR off)\n",
                         nk_cpu);
            return 1;
        }
        std::printf("CPU:  E=%.12f Ha  iters=%d  nk=%d  converged=%d\n", e_cpu,
                    r.iters, nk_cpu, (int)r.converged);
    }

    if (run_cuda) {
        auto frame = reference_frame;
        ATV::cuda::gpw::GpwDft m(kinds(), 80.0, XCType::PADE);
        set_cfg(m);
        if (!m.uses_kpath())
            die("CUDA method should report uses_kpath for k_mesh={2,1,1}");
        m.build(frame);
        const auto& r = m.run(frame);
        if (!r.converged) {
            std::fprintf(stderr,
                         "FAIL: CUDA multi-k SCF not converged (iters=%d)\n",
                         r.iters);
            return 1;
        }
        e_cuda = r.e_total;
        nk_cuda = r.n_kpts;
        if (nk_cuda < 2) {
            std::fprintf(stderr, "FAIL: CUDA n_kpts=%d expected >=2 (TR off)\n",
                         nk_cuda);
            return 1;
        }
        std::printf("CUDA: E=%.12f Ha  iters=%d  nk=%d  converged=%d\n", e_cuda,
                    r.iters, nk_cuda, (int)r.converged);
    }

    if (run_cpu && run_cuda) {
        if (nk_cpu != nk_cuda) {
            std::fprintf(stderr, "FAIL: n_kpts mismatch cpu=%d cuda=%d\n",
                         nk_cpu, nk_cuda);
            return 1;
        }
        const double de = std::abs(e_cuda - e_cpu);
        std::printf("|E_cuda-E_cpu|=%.3e Ha  tol=%.1e  nk=%d\n", de, kETolHa,
                    nk_cpu);
        if (!(de <= kETolHa)) {
            std::fprintf(stderr, "FAIL: energy dual-gate |dE|=%.6e > tol\n",
                         de);
            return 1;
        }
        std::printf(
            "PASS: cuda-gpw-kpoints  |dE|=%.3e  nk=%d  "
            "(H2 k_mesh=2x1x1 TR-off RKS PADE; perf_exempt multi-k foothold)\n",
            de, nk_cpu);
    } else {
        std::printf("PASS: cuda-gpw-kpoints (--device=%s only)\n",
                    device.c_str());
    }
    return 0;
}

} // namespace atv_reg_gpw_cuda_gpw_kpoints

ATV_REGRESSION("gpw/rks", "cuda-gpw-kpoints",
               atv_reg_gpw_cuda_gpw_kpoints::run);

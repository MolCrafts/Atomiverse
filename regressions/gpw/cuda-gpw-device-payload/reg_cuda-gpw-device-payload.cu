/// @file regressions/gpw/cuda-gpw-device-payload/reg_cuda-gpw-device-payload.cu
/// @brief END-TO-END dual-gate: device final payload eigen vs CPU energy.
///
/// H₂ Γ RKS PADE. Final C/ε come from device `eigen_from_device` (no host
/// re-diag). Public API only. See REFERENCE.md.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cuda/gpw/gpw_solver.cuh"

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GpwAtom;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::XcKind;

namespace {

constexpr double kETolHa = 5.0e-6;
constexpr double kEpsTolHa = 5.0e-5;

void die(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
}

std::vector<GpwAtom> h2_atoms() {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    return {GpwAtom(rA, basis, pot, 1), GpwAtom(rB, basis, pot, 1)};
}

template <typename Method>
void set_cfg(Method& m) {
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

    auto atoms = h2_atoms();
    double e_cpu = 0.0, e_cuda = 0.0;
    int it_cpu = 0, it_cuda = 0;
    std::vector<double> eps_cpu, eps_cuda;

    if (run_cpu) {
        ATV::cpu::gpw::GpwDft m;
        set_cfg(m);
        m.build(atoms);
        const auto& r = m.run(atoms);
        if (!r.converged) {
            std::fprintf(stderr, "FAIL: CPU SCF not converged (iters=%d)\n",
                         r.iters);
            return 1;
        }
        e_cpu = r.e_total;
        it_cpu = r.iters;
        eps_cpu = r.eps;
        std::printf("CPU:  E=%.12f Ha  iters=%d  n_eps=%zu  converged=%d\n",
                    e_cpu, it_cpu, eps_cpu.size(), (int)r.converged);
    }

    if (run_cuda) {
        ATV::cuda::gpw::GpwDft m;
        set_cfg(m);
        m.build(atoms);
        const auto& r = m.run(atoms);
        if (!r.converged) {
            std::fprintf(stderr, "FAIL: CUDA SCF not converged (iters=%d)\n",
                         r.iters);
            return 1;
        }
        if (r.C.size() == 0 || r.eps.empty()) {
            std::fprintf(stderr, "FAIL: CUDA payload missing C/eps\n");
            return 1;
        }
        e_cuda = r.e_total;
        it_cuda = r.iters;
        eps_cuda = r.eps;
        std::printf("CUDA: E=%.12f Ha  iters=%d  n_eps=%zu  converged=%d\n",
                    e_cuda, it_cuda, eps_cuda.size(), (int)r.converged);
    }

    if (run_cpu && run_cuda) {
        const double de = std::abs(e_cuda - e_cpu);
        std::printf("|E_cuda-E_cpu|=%.3e Ha  tol=%.1e\n", de, kETolHa);
        if (!(de <= kETolHa)) {
            std::fprintf(stderr, "FAIL: energy dual-gate |dE|=%.6e > tol\n",
                         de);
            return 1;
        }
        const std::size_t ncmp =
            std::min(eps_cpu.size(), eps_cuda.size());
        double deps = 0.0;
        for (std::size_t i = 0; i < ncmp; ++i)
            deps = std::max(deps, std::abs(eps_cuda[i] - eps_cpu[i]));
        std::printf("max|eps_cuda-eps_cpu|=%.3e Ha  tol=%.1e  (n=%zu)\n", deps,
                    kEpsTolHa, ncmp);
        if (!(deps <= kEpsTolHa)) {
            std::fprintf(stderr, "FAIL: eps payload |d|=%.6e > tol\n", deps);
            return 1;
        }
        std::printf(
            "PASS: cuda-gpw-device-payload  |dE|=%.3e  max|deps|=%.3e  "
            "(H2 Γ RKS PADE device final eigen payload; perf_exempt)\n",
            de, deps);
    } else {
        std::printf("PASS: cuda-gpw-device-payload single-device run\n");
    }
    return 0;
}

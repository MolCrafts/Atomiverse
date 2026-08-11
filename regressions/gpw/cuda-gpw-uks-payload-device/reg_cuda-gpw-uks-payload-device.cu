/// @file regressions/gpw/cuda-gpw-uks-payload-device/reg_cuda-gpw-uks-payload-device.cu
/// @brief END-TO-END dual-gate: device dual final payload eigen vs CPU UKS.
///
/// High-spin H₂ (n_alpha=2, n_beta=0), DZV-GTH-PADE / GTH-PADE-q1, periodic
/// 12 Bohr, CUTOFF 80 Ry, Padé LSD, DIIS depth 6. Final dual C/ε come from
/// `AoDiisEigenDevice::eigen_from_device` on both α/β channels (no host
/// dual `diagonalize_fock`). Same-settings CPU is the oracle. See REFERENCE.md.

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

// Dual-gate tols: same class as cuda-gpw-device-payload / uks-device-diis.
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
void set_uks(Method& m) {
    m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
    m.cutoff_ry = 80.0;
    m.n_occ = 0;
    m.n_alpha = 2;
    m.n_beta = 0;
    m.max_iter = 60;
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

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        m = std::max(m, std::abs(a[i] - b[i]));
    return m;
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
    double s2_cpu = 0.0, s2_cuda = 0.0;
    int it_cpu = 0, it_cuda = 0;
    std::vector<double> eps_a_cpu, eps_b_cpu, eps_a_cuda, eps_b_cuda;

    if (run_cpu) {
        ATV::cpu::gpw::GpwDft m;
        set_uks(m);
        m.build(atoms);
        const auto& r = m.run(atoms);
        if (!r.converged) {
            std::fprintf(stderr, "FAIL: CPU UKS SCF not converged (iters=%d)\n",
                         r.iters);
            return 1;
        }
        e_cpu = r.e_total;
        s2_cpu = r.s2;
        it_cpu = r.iters;
        eps_a_cpu = r.eps;
        eps_b_cpu = r.eps_beta;
        std::printf(
            "CPU:  E=%.12f Ha  <S2>=%.6f  iters=%d  n_eps_a=%zu n_eps_b=%zu\n",
            e_cpu, s2_cpu, it_cpu, eps_a_cpu.size(), eps_b_cpu.size());
    }

    if (run_cuda) {
        ATV::cuda::gpw::GpwDft m;
        set_uks(m);
        m.build(atoms);
        const auto& r = m.run(atoms);
        if (!r.converged) {
            std::fprintf(stderr,
                         "FAIL: CUDA UKS SCF not converged (iters=%d)\n",
                         r.iters);
            return 1;
        }
        if (r.C.size() == 0 || r.eps.empty()) {
            std::fprintf(stderr, "FAIL: CUDA UKS payload missing C/eps alpha\n");
            return 1;
        }
        if (r.C_beta.size() == 0 || r.eps_beta.empty()) {
            std::fprintf(stderr,
                         "FAIL: CUDA UKS payload missing C_beta/eps_beta\n");
            return 1;
        }
        if (r.veff_a.empty() || r.veff_b.empty() || r.v_es.empty())
            die("CUDA UKS result missing veff_a/veff_b/v_es after run");
        e_cuda = r.e_total;
        s2_cuda = r.s2;
        it_cuda = r.iters;
        eps_a_cuda = r.eps;
        eps_b_cuda = r.eps_beta;
        std::printf(
            "CUDA: E=%.12f Ha  <S2>=%.6f  iters=%d  n_eps_a=%zu n_eps_b=%zu\n",
            e_cuda, s2_cuda, it_cuda, eps_a_cuda.size(), eps_b_cuda.size());
    }

    if (run_cpu && run_cuda) {
        const double de = std::abs(e_cuda - e_cpu);
        std::printf("|E_cuda-E_cpu|=%.3e Ha  tol=%.1e\n", de, kETolHa);
        if (!(de <= kETolHa)) {
            std::fprintf(stderr, "FAIL: energy dual-gate |dE|=%.6e > tol\n",
                         de);
            return 1;
        }
        const double deps_a = max_abs_diff(eps_a_cuda, eps_a_cpu);
        const double deps_b = max_abs_diff(eps_b_cuda, eps_b_cpu);
        const double deps = std::max(deps_a, deps_b);
        std::printf("max|eps_a|=%.3e  max|eps_b|=%.3e  tol=%.1e\n", deps_a,
                    deps_b, kEpsTolHa);
        if (!(deps <= kEpsTolHa)) {
            std::fprintf(stderr, "FAIL: eps payload |d|=%.6e > tol\n", deps);
            return 1;
        }
        // High-spin pure triplet: <S2> ~ 2.
        if (!(s2_cuda > 1.5)) {
            std::fprintf(stderr,
                         "FAIL: CUDA <S2>=%.6f too small for high-spin\n",
                         s2_cuda);
            return 1;
        }
        std::printf(
            "PASS: cuda-gpw-uks-payload-device  |dE|=%.3e  max|deps|=%.3e  "
            "(H2 Γ UKS PADE dual device final eigen payload; perf_exempt)\n",
            de, deps);
    } else if (run_cuda) {
        std::printf(
            "PASS: cuda-gpw-uks-payload-device (cuda-only) E=%.12f Ha\n",
            e_cuda);
    } else {
        std::printf(
            "PASS: cuda-gpw-uks-payload-device (cpu-only) E=%.12f Ha\n", e_cpu);
    }
    return 0;
}

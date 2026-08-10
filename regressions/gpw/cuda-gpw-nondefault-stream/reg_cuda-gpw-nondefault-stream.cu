/// @file regressions/gpw/cuda-gpw-nondefault-stream/reg_cuda-gpw-nondefault-stream.cu
/// @brief END-TO-END: product SCF on non-default stream — dual-gate vs CPU.
///
/// Same H₂ Γ RKS PADE settings as full-product. Stream change must not move
/// energy/forces outside stated CUDA↔CPU tols. See REFERENCE.md.

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/cuda/gpw/gpw_solver.cuh"

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GpwAtom;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::XcKind;

namespace {

constexpr double kETolHa = 5.0e-6;
constexpr double kFTolHaBohr = 5.0e-5;

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
    std::vector<std::array<double, 3>> F_cpu, F_cuda;

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
        F_cpu = ATV::cpu::gpw::gpw_forces_analytic(atoms, m);
        std::printf("CPU:  E=%.12f Ha  iters=%d\n", e_cpu, r.iters);
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
        e_cuda = r.e_total;
        F_cuda = ATV::cuda::gpw::gpw_forces_analytic(atoms, m);
        std::printf("CUDA: E=%.12f Ha  iters=%d  (non-default SCF stream)\n",
                    e_cuda, r.iters);
    }

    if (run_cpu && run_cuda) {
        const double de = std::abs(e_cuda - e_cpu);
        std::printf("|E_cuda-E_cpu|=%.3e Ha  tol=%.1e\n", de, kETolHa);
        if (!(de <= kETolHa)) {
            std::fprintf(stderr, "FAIL: energy dual-gate |dE|=%.6e\n", de);
            return 1;
        }
        double max_df = 0.0;
        for (std::size_t i = 0; i < F_cpu.size(); ++i)
            for (int k = 0; k < 3; ++k)
                max_df = std::max(
                    max_df, std::abs(F_cuda[i][static_cast<std::size_t>(k)]
                                     - F_cpu[i][static_cast<std::size_t>(k)]));
        std::printf("max|F_cuda-F_cpu|=%.3e Ha/Bohr  tol=%.1e\n", max_df,
                    kFTolHaBohr);
        if (!(max_df <= kFTolHaBohr)) {
            std::fprintf(stderr, "FAIL: force dual-gate max|dF|=%.6e\n",
                         max_df);
            return 1;
        }
        std::printf(
            "PASS: cuda-gpw-nondefault-stream  |dE|=%.3e  max|dF|=%.3e\n", de,
            max_df);
    } else if (run_cuda) {
        std::printf("PASS: cuda-gpw-nondefault-stream (cuda-only)\n");
    } else {
        std::printf("PASS: cuda-gpw-nondefault-stream (cpu-only)\n");
    }
    return 0;
}

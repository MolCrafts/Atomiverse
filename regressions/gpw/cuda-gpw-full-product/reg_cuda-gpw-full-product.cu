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

// Dual-gate tols (Hartree / Ha/Bohr). Composite of stage footholds + host
// STVnl/DIIS twin path — not bit-exact (cuFFT vs FFTW, tree reductions).
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
        std::printf("CPU:  E=%.12f Ha  iters=%d  converged=%d\n", e_cpu, r.iters,
                    (int)r.converged);
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
        std::printf(
            "PASS: cuda-gpw-full-product  |dE|=%.3e  max|dF|=%.3e  "
            "(H2 Γ RKS PADE; perf_exempt small cell)\n",
            de, maxd);
    } else {
        std::printf("PASS: cuda-gpw-full-product (--device=%s only)\n",
                    device.c_str());
    }
    return 0;
}

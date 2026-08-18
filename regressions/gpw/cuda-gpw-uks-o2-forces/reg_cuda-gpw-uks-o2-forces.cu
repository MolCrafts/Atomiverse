/// @file regressions/gpw/cuda-gpw-uks-o2-forces/reg_cuda-gpw-uks-o2-forces.cu
/// @brief END-TO-END: Γ UKS O₂ analytic forces + stress dual-gate.
///
/// Same settings as `cuda-gpw-uks-o2-gate` energy. CUDA SCF payload + host
/// force/stress assembly. See REFERENCE.md.

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

constexpr double kETolHa = 5.0e-5;
constexpr double kFTol = 5.0e-4;
constexpr double kStressTol = 5.0e-5;
constexpr double kH = 2.0e-3;
constexpr double kBondBohr = 1.2075 / 0.52917721067;
constexpr double kCellBohr = 12.0;

void die(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
}

std::vector<GpwAtom> o2_atoms() {
    const auto basis = GaussianBasis(molrs::Element::O, "SZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::O, "GTH-PADE-q6");
    const double half = 0.5 * kBondBohr;
    const double c = 0.5 * kCellBohr;
    const double rA[3] = {c, c, c - half};
    const double rB[3] = {c, c, c + half};
    return {GpwAtom(rA, basis, pot, 8), GpwAtom(rB, basis, pot, 8)};
}

template <typename Method>
void set_uks(Method& m) {
    m.cell[0] = m.cell[1] = m.cell[2] = kCellBohr;
    m.cutoff_ry = 100.0;
    m.n_occ = 0;
    m.n_alpha = 7;
    m.n_beta = 5;
    m.max_iter = 80;
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

    auto atoms = o2_atoms();
    double e_cpu = 0.0, e_cuda = 0.0;
    std::vector<std::array<double, 3>> F_cpu, F_cuda;
    std::array<double, 6> s_cpu{}, s_cuda{};

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
        F_cpu = ATV::cpu::gpw::gpw_forces_analytic(atoms, m);
        s_cpu = ATV::cpu::gpw::gpw_stress(atoms, m, r, kH);
        std::printf("CPU:  E=%.12f Ha  iters=%d\n", e_cpu, r.iters);
    }

    if (run_cuda) {
        ATV::cuda::gpw::GpwDft m;
        set_uks(m);
        m.build(atoms);
        const auto& r = m.run(atoms);
        if (!r.converged) {
            std::fprintf(stderr, "FAIL: CUDA UKS SCF not converged (iters=%d)\n",
                         r.iters);
            return 1;
        }
        e_cuda = r.e_total;
        F_cuda = ATV::cuda::gpw::gpw_forces_analytic(atoms, m);
        s_cuda = ATV::cuda::gpw::gpw_stress(atoms, m, r, kH);
        std::printf("CUDA: E=%.12f Ha  iters=%d\n", e_cuda, r.iters);
    }

    if (run_cpu && run_cuda) {
        const double de = std::abs(e_cuda - e_cpu);
        std::printf("|E_cuda-E_cpu|=%.3e Ha  tol=%.1e\n", de, kETolHa);
        if (!(de <= kETolHa)) {
            std::fprintf(stderr, "FAIL: energy dual-gate |dE|=%.6e\n", de);
            return 1;
        }
        if (F_cpu.size() != F_cuda.size() || F_cpu.size() != 2)
            die("force size mismatch (want 2 atoms)");
        double max_df = 0.0;
        double max_f = 0.0;
        for (int a = 0; a < 2; ++a)
            for (int k = 0; k < 3; ++k) {
                max_df = std::max(max_df, std::abs(F_cuda[a][k] - F_cpu[a][k]));
                max_f = std::max(max_f, std::abs(F_cpu[a][k]));
            }
        std::printf("UKS F  max|Δ|=%.3e  max|F|=%.3e  tol=%.1e\n", max_df,
                    max_f, kFTol);
        if (!(max_f > 1e-4))
            die("UKS force vacuous");
        if (!(max_df <= kFTol)) {
            std::fprintf(stderr, "FAIL: UKS force parity max|dF|=%.6e\n",
                         max_df);
            return 1;
        }
        double max_ds = 0.0;
        for (int a = 0; a < 3; ++a)
            max_ds = std::max(max_ds, std::abs(s_cuda[a] - s_cpu[a]));
        std::printf("UKS σ  max|Δ|=%.3e  tol=%.1e\n", max_ds, kStressTol);
        if (!(max_ds <= kStressTol)) {
            std::fprintf(stderr, "FAIL: UKS stress parity max|dσ|=%.6e\n",
                         max_ds);
            return 1;
        }
        std::printf(
            "PASS: cuda-gpw-uks-o2-forces  |dE|=%.3e  max|dF|=%.3e  max|dσ|=%.3e\n",
            de, max_df, max_ds);
    } else {
        std::printf("PASS: cuda-gpw-uks-o2-forces (--device=%s only)\n",
                    device.c_str());
    }
    return 0;
}

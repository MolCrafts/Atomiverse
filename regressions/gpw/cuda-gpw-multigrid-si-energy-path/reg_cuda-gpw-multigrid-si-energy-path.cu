/// @file regressions/gpw/cuda-gpw-multigrid-si-energy-path/reg_cuda-gpw-multigrid-si-energy-path.cu
/// @brief END-TO-END dual-gate: Si-class Γ RKS PADE **multigrid** energy on
///        public `cuda::gpw::GpwDft` vs same-settings CPU (diamond bct 4-atom).
///
/// Production Si GTH nonlocal × multi-level after H₂ multigrid-run foothold
/// and Si single-grid product-si-gate. ngrids=4 + rel_cutoff>0 exercises
/// KsMultiGridChainDevice (PBC collocate per level + spectral transfer).
/// Oracle: CPU twin. See REFERENCE.md.

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

// Si p/d-shell + nonlocal GTH + multi-level product residual (same class as
// product-si-gate; not soft-tol).
constexpr double kETolHa = 5.0e-5;

// Å → Bohr (same constant as gpw-multigrid-kpoints / product-si-gate).
constexpr double kA2B = 1.0 / 0.52917720859;

void die(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
}

// Diamond Si body-centred-tetragonal 4-atom cell (a_t = a/√2, c = a).
std::vector<GpwAtom> si_bct(double a_bohr) {
    const double at = a_bohr / std::sqrt(2.0);
    const GaussianBasis bSi(molrs::Element::Si, "DZVP-MOLOPT-GTH");
    const GthPotential pSi(molrs::Element::Si, "GTH-PBE-q4");
    const double frac[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, 0.0, 0.25}, {0.0, 0.5, 0.75}};
    std::vector<GpwAtom> atoms;
    for (int i = 0; i < 4; ++i) {
        const double pos[3] = {frac[i][0] * at, frac[i][1] * at,
                               frac[i][2] * a_bohr};
        atoms.emplace_back(pos, bSi, pSi, 14);
    }
    return atoms;
}

template <typename Method>
void set_si_mg(Method& m, double a_bohr) {
    m.cell[0] = m.cell[1] = a_bohr / std::sqrt(2.0);
    m.cell[2] = a_bohr;
    m.cutoff_ry = 40.0;
    m.ngrids = 4;
    m.rel_cutoff = 20.0; // < cutoff → true multi-level (make_multigrid gate)
    m.n_occ = 8;         // 4 atoms × 4 valence e⁻
    m.max_iter = 60;
    m.tol = 1e-8;
    m.diis_depth = 6;
    m.xc = XcKind::PADE;
    m.nonperiodic = false;
    m.local_pp_on_grid = true;
    m.use_ot = false;
    m.eps_default = 1e-7;
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

    const double a = 5.43 * kA2B; // Si lattice constant 5.43 Å → Bohr
    auto atoms = si_bct(a);
    double e_cpu = 0.0, e_cuda = 0.0;

    if (run_cpu) {
        ATV::cpu::gpw::GpwDft m;
        set_si_mg(m, a);
        m.build(atoms);
        const auto& r = m.run(atoms);
        if (!r.converged) {
            std::fprintf(
                stderr,
                "FAIL: CPU Si MG SCF not converged (iters=%d E=%.12f)\n",
                r.iters, r.e_total);
            return 1;
        }
        e_cpu = r.e_total;
        std::printf(
            "CPU:  E=%.12f Ha  iters=%d  ngrids=%d  rel=%.1f  cutoff=%.1f\n",
            e_cpu, r.iters, m.ngrids, m.rel_cutoff, m.cutoff_ry);
    }

    if (run_cuda) {
        ATV::cuda::gpw::GpwDft m;
        set_si_mg(m, a);
        m.build(atoms);
        const auto& r = m.run(atoms);
        if (!r.converged) {
            std::fprintf(
                stderr,
                "FAIL: CUDA Si MG SCF not converged (iters=%d  E=%.12f)\n",
                r.iters, r.e_total);
            return 1;
        }
        e_cuda = r.e_total;
        std::printf(
            "CUDA: E=%.12f Ha  iters=%d  ngrids=%d  rel=%.1f  cutoff=%.1f\n",
            e_cuda, r.iters, m.ngrids, m.rel_cutoff, m.cutoff_ry);
    }

    if (run_cpu && run_cuda) {
        const double de = std::abs(e_cuda - e_cpu);
        std::printf("|E_cuda-E_cpu|=%.3e Ha  tol=%.1e\n", de, kETolHa);
        if (!(de <= kETolHa)) {
            std::fprintf(stderr, "FAIL: energy dual-gate |dE|=%.6e > tol\n",
                         de);
            return 1;
        }
        std::printf("PASS: cuda-gpw-multigrid-si-energy-path  |dE|=%.3e  "
                    "(Si diamond Γ RKS PADE ngrids=4 rel=20; perf_exempt)\n",
                    de);
    } else {
        std::printf(
            "PASS: cuda-gpw-multigrid-si-energy-path (--device=%s only)\n",
            device.c_str());
    }
    return 0;
}

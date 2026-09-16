/// @file regressions/gpw/cuda-gpw-uks-o2-gate/reg_cuda-gpw-uks-o2-gate.cu
/// @brief END-TO-END dual-gate: Γ collinear UKS O₂ energy on public GpwDft.
///
/// Production-class after H₂ UKS foothold (`cuda-gpw-uks-gamma`). O₂ triplet
/// (n_alpha=7, n_beta=5), SZV-GTH-PADE / GTH-PADE-q6, periodic 12 Bohr,
/// CUTOFF 100 Ry, Padé LSD, DIIS. Same-settings CPU is the oracle (not CP2K
/// MT twin — CUDA rejects nonperiodic). See REFERENCE.md.

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

// Dual-gate energy tol [Ha]. Composite of stage footholds + host DIIS twin +
// p-shell product collocate (cuFFT vs FFTW, tree reductions) — not bit-exact.
// H₂ pure-s UKS uses 5e-6; O₂ p-shell production gate states 5e-5.
constexpr double kETolHa = 5.0e-5;

// R(O–O) = 1.2075 Å → Bohr (a0 = 0.52917721067 Å).
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
    m.n_alpha = 7; // multiplicity 3, 12 valence e⁻
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
    double s2_cpu = 0.0, s2_cuda = 0.0;

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
        std::printf("CPU:  E=%.12f Ha  <S2>=%.6f  iters=%d  converged=%d\n",
                    e_cpu, s2_cpu, r.iters, (int)r.converged);
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
        e_cuda = r.e_total;
        s2_cuda = r.s2;
        if (r.veff_a.empty() || r.veff_b.empty() || r.v_es.empty())
            die("CUDA UKS result missing veff_a/veff_b/v_es after run");
        if (r.P_beta.size() == 0)
            die("CUDA UKS missing P_beta for n_beta>0");
        std::printf("CUDA: E=%.12f Ha  <S2>=%.6f  iters=%d  converged=%d\n",
                    e_cuda, s2_cuda, r.iters, (int)r.converged);
    }

    if (run_cpu && run_cuda) {
        const double de = std::abs(e_cuda - e_cpu);
        std::printf("|E_cuda-E_cpu|=%.3e Ha  tol=%.1e\n", de, kETolHa);
        if (!(de <= kETolHa)) {
            std::fprintf(stderr, "FAIL: energy dual-gate |dE|=%.6e > tol\n",
                         de);
            return 1;
        }
        // Multiplicity 3: ideal S(S+1)=2; Löwdin single-det near 2.
        if (!(s2_cuda > 1.5)) {
            std::fprintf(stderr,
                         "FAIL: CUDA <S2>=%.6f too small for O2 triplet\n",
                         s2_cuda);
            return 1;
        }
        std::printf(
            "PASS: cuda-gpw-uks-o2-gate  |dE|=%.3e  "
            "(O2 Γ UKS PADE; perf_exempt small cell)\n",
            de);
    } else if (run_cuda) {
        std::printf("PASS: cuda-gpw-uks-o2-gate (cuda-only) E=%.12f Ha\n",
                    e_cuda);
    } else {
        std::printf("PASS: cuda-gpw-uks-o2-gate (cpu-only) E=%.12f Ha\n",
                    e_cpu);
    }
    return 0;
}

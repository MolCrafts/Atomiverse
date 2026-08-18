/// @file regressions/gpw/cuda-gpw-product-h2o-stress/reg_cuda-gpw-product-h2o-stress.cu
/// @brief END-TO-END: water Γ RKS PADE cell stress dual-gate.
///
/// Production σ after `product-h2o-gate` (E+F) and H₂ `cuda-gpw-stress`.
/// Same settings as the water energy/force gate. See REFERENCE.md.

#include <array>
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

constexpr double kETolHa = 5.0e-5;
constexpr double kStressTol = 5.0e-5;
constexpr double kOffDiagTol = 1.0e-8;
constexpr double kH = 2.0e-3;
constexpr double kA0 = 0.52917721067;
constexpr double kCellBohr = 12.0;

void die(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
}

std::vector<GpwAtom> water_atoms() {
    const auto bas_O = GaussianBasis(molrs::Element::O, "SZV-GTH-PADE");
    const auto pot_O = GthPotential(molrs::Element::O, "GTH-PADE-q6");
    const auto bas_H = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot_H = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double dHx = 0.7586 / kA0;
    const double dHz = 0.5043 / kA0;
    const double c = 0.5 * kCellBohr;
    const double rO[3] = {c, c, c};
    const double rH1[3] = {c + dHx, c, c + dHz};
    const double rH2[3] = {c - dHx, c, c + dHz};
    return {GpwAtom(rO, bas_O, pot_O, 8), GpwAtom(rH1, bas_H, pot_H, 1),
            GpwAtom(rH2, bas_H, pot_H, 1)};
}

template <typename Method>
void set_cfg(Method& m) {
    m.cell[0] = m.cell[1] = m.cell[2] = kCellBohr;
    m.cutoff_ry = 100.0;
    m.n_occ = 4;
    m.max_iter = 80;
    m.tol = 1e-8;
    m.diis_depth = 6;
    m.mix = 0.4;
    m.level_shift = 0.0;
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

    auto atoms = water_atoms();
    double e_cpu = 0.0, e_cuda = 0.0;
    std::array<double, 6> s_cpu{}, s_cuda{};

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
        s_cpu = ATV::cpu::gpw::gpw_stress(atoms, m, r, kH);
        std::printf("CPU:  E=%.12f Ha  iters=%d\n", e_cpu, r.iters);
        std::printf("CPU σ= %.8e %.8e %.8e  (off %.3e %.3e %.3e)\n", s_cpu[0],
                    s_cpu[1], s_cpu[2], s_cpu[3], s_cpu[4], s_cpu[5]);
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
        s_cuda = ATV::cuda::gpw::gpw_stress(atoms, m, r, kH);
        std::printf("CUDA: E=%.12f Ha  iters=%d\n", e_cuda, r.iters);
        std::printf("CUDA σ= %.8e %.8e %.8e  (off %.3e %.3e %.3e)\n",
                    s_cuda[0], s_cuda[1], s_cuda[2], s_cuda[3], s_cuda[4],
                    s_cuda[5]);
        for (int k = 3; k < 6; ++k) {
            if (std::abs(s_cuda[static_cast<std::size_t>(k)]) > kOffDiagTol) {
                std::fprintf(stderr, "FAIL: CUDA off-diag σ[%d]=%.3e\n", k,
                             s_cuda[static_cast<std::size_t>(k)]);
                return 1;
            }
        }
    }

    if (run_cpu && run_cuda) {
        const double de = std::abs(e_cuda - e_cpu);
        std::printf("|E_cuda-E_cpu|=%.3e Ha  tol=%.1e\n", de, kETolHa);
        if (!(de <= kETolHa)) {
            std::fprintf(stderr, "FAIL: energy dual-gate |dE|=%.6e\n", de);
            return 1;
        }
        double max_ds = 0.0;
        for (int a = 0; a < 3; ++a) {
            const double d = std::abs(s_cuda[static_cast<std::size_t>(a)]
                                      - s_cpu[static_cast<std::size_t>(a)]);
            max_ds = std::max(max_ds, d);
            std::printf("  |Δσ_%c|=%.3e\n", "xyz"[a], d);
        }
        std::printf("max|σ_cuda-σ_cpu|=%.3e Ha/Bohr3  tol=%.1e\n", max_ds,
                    kStressTol);
        if (!(max_ds <= kStressTol)) {
            std::fprintf(stderr, "FAIL: stress dual-gate max|dσ|=%.6e\n",
                         max_ds);
            return 1;
        }
        std::printf("PASS: cuda-gpw-product-h2o-stress  |dE|=%.3e  max|dσ|=%.3e\n",
                    de, max_ds);
    } else {
        std::printf("PASS: cuda-gpw-product-h2o-stress (--device=%s only)\n",
                    device.c_str());
    }
    return 0;
}

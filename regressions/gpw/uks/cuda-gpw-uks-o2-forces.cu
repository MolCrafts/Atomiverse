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

#include "../frame_fixture.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cuda/gpw/gpw_solver.cuh"
#include "harness/registry.h"

namespace atv_reg_gpw_cuda_gpw_uks_o2_forces {

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::xc::XCType;
namespace fixture = ATV::regression::gpw_fixture;

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

std::vector<ATV::cpu::gpw::Kind> kinds() {
    const auto basis = GaussianBasis(molrs::Element::O, "SZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::O, "GTH-PADE-q6");
    return {{molrs::Element::O, basis, pot}};
}

ATV::cpu::Frame o2_frame() {
    const double half = 0.5 * kBondBohr;
    const double c = 0.5 * kCellBohr;
    const double rA[3] = {c, c, c - half};
    const double rB[3] = {c, c, c + half};
    return fixture::frame_bohr({rA, rB}, {8, 8},
                               {kCellBohr, kCellBohr, kCellBohr});
}

// Geometry and cell belong to the shared public Frame fixture.
template <typename Method>
void set_uks(Method& m) {
    m.n_occ = 0;
    m.n_alpha = 7;
    m.n_beta = 5;
    m.max_iter = 80;
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

    const auto reference_frame = o2_frame();
    double e_cpu = 0.0, e_cuda = 0.0;
    std::vector<std::array<double, 3>> F_cpu, F_cuda;
    std::array<double, 6> s_cpu{}, s_cuda{};

    if (run_cpu) {
        auto frame = reference_frame;
        ATV::cpu::gpw::GpwDft m(kinds(), 100.0, XCType::PADE);
        set_uks(m);
        m.build(frame);
        const auto& r = m.run(frame);
        if (!r.converged) {
            std::fprintf(stderr, "FAIL: CPU UKS SCF not converged (iters=%d)\n",
                         r.iters);
            return 1;
        }
        e_cpu = r.e_total;
        m.forces(frame);
        F_cpu = fixture::forces_ha_bohr(frame);
        m.stress_strain_h = kH;
        s_cpu = fixture::stress_ha_bohr3(m.stress(frame));
        std::printf("CPU:  E=%.12f Ha  iters=%d\n", e_cpu, r.iters);
    }

    if (run_cuda) {
        auto frame = reference_frame;
        ATV::cuda::gpw::GpwDft m(kinds(), 100.0, XCType::PADE);
        set_uks(m);
        m.build(frame);
        const auto& r = m.run(frame);
        if (!r.converged) {
            std::fprintf(stderr,
                         "FAIL: CUDA UKS SCF not converged (iters=%d)\n",
                         r.iters);
            return 1;
        }
        e_cuda = r.e_total;
        m.forces(frame);
        F_cuda = fixture::forces_ha_bohr(frame);
        m.stress_strain_h = kH;
        s_cuda = fixture::stress_ha_bohr3(m.stress(frame));
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
        std::printf("PASS: cuda-gpw-uks-o2-forces  |dE|=%.3e  max|dF|=%.3e  "
                    "max|dσ|=%.3e\n",
                    de, max_df, max_ds);
    } else {
        std::printf("PASS: cuda-gpw-uks-o2-forces (--device=%s only)\n",
                    device.c_str());
    }
    return 0;
}

} // namespace atv_reg_gpw_cuda_gpw_uks_o2_forces

ATV_REGRESSION("gpw/uks", "cuda-gpw-uks-o2-forces",
               atv_reg_gpw_cuda_gpw_uks_o2_forces::run);

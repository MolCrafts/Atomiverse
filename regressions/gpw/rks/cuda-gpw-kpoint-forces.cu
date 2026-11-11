/// @file regressions/gpw/cuda-gpw-kpoint-forces/reg_cuda-gpw-kpoint-forces.cu
/// @brief END-TO-END: multi-k RKS H₂ forces + stress dual-gate.
///
/// Same settings as `cuda-gpw-kpoints` energy (k_mesh={2,1,1}). CUDA SCF
/// payload + host k-path force/stress assembly. See REFERENCE.md.

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

namespace atv_reg_gpw_cuda_gpw_kpoint_forces {

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::xc::XCType;
namespace fixture = ATV::regression::gpw_fixture;

namespace {

constexpr double kETolHa = 5.0e-6;
constexpr double kFTol = 5.0e-5;
constexpr double kStressTol = 5.0e-5;
constexpr double kH = 2.0e-3;

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
    std::vector<std::array<double, 3>> F_cpu, F_cuda;
    std::array<double, 6> s_cpu{}, s_cuda{};

    if (run_cpu) {
        auto frame = reference_frame;
        ATV::cpu::gpw::GpwDft m(kinds(), 80.0, XCType::PADE);
        set_cfg(m);
        m.build(frame);
        const auto& r = m.run(frame);
        if (!r.converged) {
            std::fprintf(stderr, "FAIL: CPU SCF not converged (iters=%d)\n",
                         r.iters);
            return 1;
        }
        e_cpu = r.e_total;
        m.forces(frame);
        F_cpu = fixture::forces_ha_bohr(frame);
        m.stress_strain_h = kH;
        s_cpu = fixture::stress_ha_bohr3(m.stress(frame));
        std::printf("CPU:  E=%.12f Ha  nk=%d  iters=%d\n", e_cpu, r.n_kpts,
                    r.iters);
    }

    if (run_cuda) {
        auto frame = reference_frame;
        ATV::cuda::gpw::GpwDft m(kinds(), 80.0, XCType::PADE);
        set_cfg(m);
        m.build(frame);
        const auto& r = m.run(frame);
        if (!r.converged) {
            std::fprintf(stderr, "FAIL: CUDA SCF not converged (iters=%d)\n",
                         r.iters);
            return 1;
        }
        e_cuda = r.e_total;
        m.forces(frame);
        F_cuda = fixture::forces_ha_bohr(frame);
        m.stress_strain_h = kH;
        s_cuda = fixture::stress_ha_bohr3(m.stress(frame));
        std::printf("CUDA: E=%.12f Ha  nk=%d  iters=%d\n", e_cuda, r.n_kpts,
                    r.iters);
    }

    if (run_cpu && run_cuda) {
        const double de = std::abs(e_cuda - e_cpu);
        std::printf("|E_cuda-E_cpu|=%.3e Ha  tol=%.1e\n", de, kETolHa);
        if (!(de <= kETolHa)) {
            std::fprintf(stderr, "FAIL: energy dual-gate |dE|=%.6e\n", de);
            return 1;
        }
        if (F_cpu.size() != F_cuda.size() || F_cpu.size() != 2)
            die("force size mismatch");
        double max_df = 0.0;
        double max_f = 0.0;
        for (int a = 0; a < 2; ++a)
            for (int k = 0; k < 3; ++k) {
                max_df = std::max(max_df, std::abs(F_cuda[a][k] - F_cpu[a][k]));
                max_f = std::max(max_f, std::abs(F_cpu[a][k]));
            }
        std::printf("k-path F  max|Δ|=%.3e  max|F|=%.3e  tol=%.1e\n", max_df,
                    max_f, kFTol);
        if (!(max_f > 1e-6))
            die("multi-k force vacuous");
        if (!(max_df <= kFTol)) {
            std::fprintf(stderr, "FAIL: multi-k force parity max|dF|=%.6e\n",
                         max_df);
            return 1;
        }
        double max_ds = 0.0;
        for (int a = 0; a < 3; ++a)
            max_ds = std::max(max_ds, std::abs(s_cuda[a] - s_cpu[a]));
        std::printf("k-path σ  max|Δ|=%.3e  tol=%.1e\n", max_ds, kStressTol);
        if (!(max_ds <= kStressTol)) {
            std::fprintf(stderr, "FAIL: multi-k stress parity max|dσ|=%.6e\n",
                         max_ds);
            return 1;
        }
        std::printf("PASS: cuda-gpw-kpoint-forces  |dE|=%.3e  max|dF|=%.3e  "
                    "max|dσ|=%.3e\n",
                    de, max_df, max_ds);
    } else {
        std::printf("PASS: cuda-gpw-kpoint-forces (--device=%s only)\n",
                    device.c_str());
    }
    return 0;
}

} // namespace atv_reg_gpw_cuda_gpw_kpoint_forces

ATV_REGRESSION("gpw/rks", "cuda-gpw-kpoint-forces",
               atv_reg_gpw_cuda_gpw_kpoint_forces::run);

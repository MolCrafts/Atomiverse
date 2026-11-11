/// @file regressions/gpw/cuda-gpw-device-forces/reg_cuda-gpw-device-forces.cu
/// @brief END-TO-END dual-gate: device force product path vs CPU GpwDft.
///
/// Public `cuda::gpw::GpwDft::forces(Frame&)` (host AO Pulay + device grid HF)
/// vs same-settings CPU analytic forces. H₂ Γ RKS PADE. See REFERENCE.md.

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

namespace atv_reg_gpw_cuda_gpw_device_forces {

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::xc::XCType;
namespace fixture = ATV::regression::gpw_fixture;

namespace {

constexpr double kETolHa = 5.0e-6;
constexpr double kFTolHaBohr = 5.0e-5;

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
        std::printf("CPU:  E=%.12f Ha  iters=%d  converged=%d\n", e_cpu,
                    r.iters, r.converged ? 1 : 0);
        for (std::size_t i = 0; i < F_cpu.size(); ++i)
            std::printf("CPU F[%zu]= %.8e %.8e %.8e Ha/Bohr\n", i, F_cpu[i][0],
                        F_cpu[i][1], F_cpu[i][2]);
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
        std::printf("CUDA: E=%.12f Ha  iters=%d  converged=%d\n", e_cuda,
                    r.iters, r.converged ? 1 : 0);
        for (std::size_t i = 0; i < F_cuda.size(); ++i)
            std::printf("CUDA F[%zu]= %.8e %.8e %.8e Ha/Bohr\n", i,
                        F_cuda[i][0], F_cuda[i][1], F_cuda[i][2]);
    }

    if (run_cpu && run_cuda) {
        const double dE = std::abs(e_cuda - e_cpu);
        if (dE > kETolHa) {
            std::fprintf(stderr, "FAIL: |dE|=%.3e > tol %.3e Ha\n", dE,
                         kETolHa);
            return 1;
        }
        if (F_cpu.size() != F_cuda.size())
            die("force size mismatch");
        double max_dF = 0.0;
        for (std::size_t i = 0; i < F_cpu.size(); ++i)
            for (int k = 0; k < 3; ++k)
                max_dF = std::max(max_dF, std::abs(F_cuda[i][k] - F_cpu[i][k]));
        if (max_dF > kFTolHaBohr) {
            std::fprintf(stderr, "FAIL: max|dF|=%.3e > tol %.3e Ha/Bohr\n",
                         max_dF, kFTolHaBohr);
            return 1;
        }
        std::printf("PASS: cuda-gpw-device-forces  |dE|=%.3e  max|dF|=%.3e  "
                    "(tol E=%.1e F=%.1e Ha/Bohr)\n",
                    dE, max_dF, kETolHa, kFTolHaBohr);
    } else if (run_cuda) {
        std::printf("PASS: cuda-gpw-device-forces (cuda-only) E=%.12f Ha\n",
                    e_cuda);
    } else {
        std::printf("PASS: cuda-gpw-device-forces (cpu-only) E=%.12f Ha\n",
                    e_cpu);
    }
    return 0;
}

} // namespace atv_reg_gpw_cuda_gpw_device_forces

ATV_REGRESSION("gpw/rks", "cuda-gpw-device-forces",
               atv_reg_gpw_cuda_gpw_device_forces::run);

/// @file regressions/pw/cuda-pw-pbe-device/reg_cuda-pw-pbe-device.cu
/// @brief END-TO-END: plane-wave true PBE GGA on CUDA — dual-gate
///        (`--device=cpu|cuda|both`, default `both`).
///
/// Gates:
///   1. both legs converge to a finite total energy (Rydberg);
///   2. same-settings CPU↔CUDA parity (the CUDA-slice oracle law):
///      |E_cuda − E_cpu| < 1.0e-3 Ry.
///
/// Public API only: `cpu::pw::PW` / `cuda::pw::PW` + `Frame`. Analytic
/// GTH-PBE-q4 (no UPF file). See REFERENCE.md.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include "atomiverse/cuda/planewave/pw_solver.cuh"

using namespace ATV;

namespace {

// CPU↔CUDA parity bound. Design-time value shared with pw-cuda-lda-si:
// cuFFT vs FFTW, tree vs serial reductions, libdevice vs libm (~10
// transcendentals/point on PBE), Davidson order, 1e-4 startingwfc.
// Re-pin from the first GH200 run; never tighten at design time.
constexpr double kParityTolRy = 1.0e-3;
constexpr double kA = 5.43; // Å
constexpr double kEcutRy = 15.0;
constexpr int kMesh = 2;

enum class Device { Cpu, Cuda };

const char* device_name(Device d) {
    return d == Device::Cuda ? "cuda" : "cpu";
}

std::string parse_device_arg(int argc, char** argv) {
    std::string want = "both";
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strncmp(a, "--device=", 9) == 0) {
            want = a + 9;
        } else if (std::strcmp(a, "--device") == 0 && i + 1 < argc) {
            want = argv[++i];
        } else if (std::strcmp(a, "cpu") == 0 || std::strcmp(a, "cuda") == 0
                   || std::strcmp(a, "both") == 0) {
            want = a;
        }
    }
    return want;
}

/// @brief 2-atom fcc-primitive diamond silicon, lattice constant `a` [Å].
cpu::Frame make_si_fcc(double a) {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 14;
    f.span(kAtom_type)[1] = 14;
    f.span(kAtom_mass)[0] = 28.0855;
    f.span(kAtom_mass)[1] = 28.0855;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = a / 4.0;
    f.span(kAtom_y)[1] = a / 4.0;
    f.span(kAtom_z)[1] = a / 4.0;
    const double h = a / 2.0;
    Mat3 cell;
    cell(0, 0) = 0;
    cell(0, 1) = h;
    cell(0, 2) = h;
    cell(1, 0) = h;
    cell(1, 1) = 0;
    cell(1, 2) = h;
    cell(2, 0) = h;
    cell(2, 1) = h;
    cell(2, 2) = 0;
    f.box.set_cell(cell);
    return f;
}

/// @brief Run one backend at shared PBE settings.
/// @param dev        Which backend to run.
/// @param energy_ry  Out: total energy [Ry].
/// @return 0 on pass, 1 on failure (message already printed).
int run_on_device(Device dev, double& energy_ry) {
    cpu::Frame frame = make_si_fcc(kA);
    bool converged = false;
    int iterations = 0;
    energy_ry = 0.0;

    if (dev == Device::Cpu) {
        cpu::pw::PW method;
        method.gth_name = "GTH-PBE-q4";
        method.ecut_ry = kEcutRy;
        method.k_mesh = kMesh;
        method.tol = 1e-6;
        method.max_iter = 50;
        method.xc = cpu::scf::XCType::PBE;
        method.build(frame);
        const auto& r = method.run(frame);
        converged = r.converged;
        iterations = r.iterations;
        energy_ry = r.energy;
    } else {
        cuda::pw::PW method;
        method.gth_name = "GTH-PBE-q4";
        method.ecut_ry = kEcutRy;
        method.k_mesh = kMesh;
        method.tol = 1e-6;
        method.max_iter = 50;
        method.xc = cpu::scf::XCType::PBE; // default, set explicitly
        method.build(frame);
        const auto& r = method.run(frame);
        converged = r.converged;
        iterations = r.iterations;
        energy_ry = r.energy;
    }

    if (!converged) {
        std::fprintf(stderr,
                     "FAIL [%s]: PW PBE Si SCF did not converge (iters=%d)\n",
                     device_name(dev), iterations);
        return 1;
    }
    if (!std::isfinite(energy_ry)) {
        std::fprintf(stderr, "FAIL [%s]: non-finite energy_ry=%g\n",
                     device_name(dev), energy_ry);
        return 1;
    }

    std::printf("PASS [%s]: PW PBE Si a=%.2f A ecut=%.0f Ry k=%d  "
                "E=%.8f Ry  iters=%d  gth=GTH-PBE-q4\n",
                device_name(dev), kA, kEcutRy, kMesh, energy_ry, iterations);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::string want = parse_device_arg(argc, argv);
    if (want != "cpu" && want != "cuda" && want != "both") {
        std::fprintf(stderr, "FAIL: unknown --device=%s (want cpu|cuda|both)\n",
                     want.c_str());
        return 1;
    }

    int rc = 0;
    double e_cpu = 0.0, e_cuda = 0.0;
    const bool ran_cpu = (want == "cpu" || want == "both");
    const bool ran_cuda = (want == "cuda" || want == "both");

    if (ran_cpu)
        rc |= run_on_device(Device::Cpu, e_cpu);
    if (ran_cuda)
        rc |= run_on_device(Device::Cuda, e_cuda);

    if (rc == 0 && ran_cpu && ran_cuda) {
        const double delta = std::abs(e_cuda - e_cpu);
        if (!(delta < kParityTolRy)) {
            std::fprintf(stderr,
                         "FAIL [parity]: |E_cuda - E_cpu| = %.3e Ry  "
                         "tol=%.1e Ry  (cpu=%.8f, cuda=%.8f)\n",
                         delta, kParityTolRy, e_cpu, e_cuda);
            return 1;
        }
        std::printf("PASS [parity]: |E_cuda - E_cpu| = %.3e Ry < %.1e Ry "
                    "(cpu=%.8f, cuda=%.8f)\n",
                    delta, kParityTolRy, e_cpu, e_cuda);
    }

    return rc;
}

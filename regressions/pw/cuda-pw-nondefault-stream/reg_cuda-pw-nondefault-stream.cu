/// @file regressions/pw/cuda-pw-nondefault-stream/reg_cuda-pw-nondefault-stream.cu
/// @brief END-TO-END gate for the non-default CUDA stream on `cuda::pw::PW`.
///
/// Same Si LSDA twin as `pw-cuda-lda-si` (ABINIT-class literature window +
/// same-settings CPU↔CUDA parity). Physics is unchanged — this slug only
/// re-anchors the full public SCF path after density+band work moved off the
/// default stream (`DeviceDensity` owns the stream; `DeviceBandSolver`
/// borrows it). See REFERENCE.md and the twin regression's pins.
///
/// Public API only: `cpu::pw::PW` / `cuda::pw::PW` + `Frame`.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include "atomiverse/cuda/planewave/pw_solver.cuh"

#ifndef ATV_PW_UPF_PATH
#error "ATV_PW_UPF_PATH must be set to tests/data/pw/Si_lda_nc.upf"
#endif

using namespace ATV;

namespace {

// PseudoDojo ONCVPSP-LDA Si.djrepo deltafactor E_min (ABINIT-class), Ry/cell.
constexpr double kEminAbinitRy = -16.916316;
constexpr double kBandHeadroomRy = 1.50; // incomplete ecut/k window
// CPU↔CUDA parity bound. Design-time value: the two backends share the host
// problem assembly, the Pulay coefficient solve and the energy expression,
// but the density side is an independent device implementation of the same
// algorithm — cuFFT vs FFTW, tree vs serial reductions, libdevice vs libm —
// on top of the Davidson arithmetic order and the 1e-4 startingwfc
// perturbation. Re-pin from the first GH200 run; never tighten at design time.
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

/// @brief Run one backend at the shared settings.
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
        method.upf_path = ATV_PW_UPF_PATH;
        method.ecut_ry = kEcutRy;
        method.k_mesh = kMesh;
        method.tol = 1e-6;
        method.max_iter = 50;
        method.xc = cpu::scf::XCType::LSDA;
        method.build(frame);
        const auto& r = method.run(frame);
        converged = r.converged;
        iterations = r.iterations;
        energy_ry = r.energy;
    } else {
        cuda::pw::PW method;
        method.upf_path = ATV_PW_UPF_PATH;
        method.ecut_ry = kEcutRy;
        method.k_mesh = kMesh;
        method.tol = 1e-6;
        method.max_iter = 50;
        method.xc = cpu::scf::XCType::LSDA;
        method.build(frame);
        const auto& r = method.run(frame);
        converged = r.converged;
        iterations = r.iterations;
        energy_ry = r.energy;
    }

    if (!converged) {
        std::fprintf(stderr,
                     "FAIL [%s]: PW LSDA Si SCF did not converge (iters=%d)\n",
                     device_name(dev), iterations);
        return 1;
    }
    if (!std::isfinite(energy_ry)) {
        std::fprintf(stderr, "FAIL [%s]: non-finite energy_ry=%g\n",
                     device_name(dev), energy_ry);
        return 1;
    }

    const double lo = kEminAbinitRy;
    const double hi = kEminAbinitRy + kBandHeadroomRy;
    if (!(energy_ry >= lo && energy_ry <= hi)) {
        std::fprintf(stderr,
                     "FAIL [%s]: E_tot got=%.8f Ry  want band [%.6f, %.6f] Ry "
                     "(ABINIT-class E_min + incomplete-basis headroom; see "
                     "REFERENCE.md)\n",
                     device_name(dev), energy_ry, lo, hi);
        return 1;
    }

    std::printf("PASS [%s]: PW LSDA Si a=%.2f A ecut=%.0f Ry k=%d  "
                "E=%.8f Ry  band [%.4f, %.4f] Ry  iters=%d\n",
                device_name(dev), kA, kEcutRy, kMesh, energy_ry, lo, hi,
                iterations);
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

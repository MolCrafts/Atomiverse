/// @file regressions/pw/cuda-pw-forces-pbe/reg_cuda-pw-forces-pbe.cu
/// @brief END-TO-END: plane-wave analytic ionic forces on CUDA under **PBE**
///        — dual-gate (`--device=cpu|cuda|both`, default `both`).
///
/// Gates:
///   1. CPU forces finite and non-vacuous on stretched Si dimer (PBE SCF);
///   2. CUDA forces finite under device PBE-GGA;
///   3. same-settings max|F_cuda − F_cpu| < 5e-3 eV/Å.
///
/// Public API only. Analytic GTH-PBE-q4 (KB projectors present). See
/// REFERENCE.md. Sibling of `cuda-pw-forces` (LSDA); this slug closes the
/// PBE force dual-gate left open after `cuda-pw-pbe-device` + `cuda-pw-forces`.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include "atomiverse/cuda/planewave/pw_solver.cuh"

using namespace ATV;

namespace {

// Design-time parity bound shared with cuda-pw-forces (LSDA). Independent
// SCF + FFT/eigensolve + PBE spectral gradients are not bit-exact; re-pin
// from the first GH200 log without tightening at design time.
constexpr double kForceTolEvA = 5.0e-3;
constexpr double kA = 5.43; // Å
constexpr double kStretch = 0.08; // Å along x on atom 1
constexpr double kEcutRy = 8.0;
constexpr int kMesh = 1;

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

cpu::Frame make_si_dimer() {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_type)[i] = 14;
        f.span(kAtom_mass)[i] = 28.0855;
    }
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = kA / 4.0 + kStretch;
    f.span(kAtom_y)[1] = kA / 4.0;
    f.span(kAtom_z)[1] = kA / 4.0;
    const double h = kA / 2.0;
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

int run_on_device(Device dev, std::vector<std::array<double, 3>>& forces) {
    cpu::Frame frame = make_si_dimer();
    forces.clear();

    if (dev == Device::Cpu) {
        cpu::pw::PW method;
        method.gth_name = "GTH-PBE-q4";
        method.ecut_ry = kEcutRy;
        method.k_mesh = kMesh;
        method.tol = 1e-6;
        method.max_iter = 40;
        method.mix_beta = 0.5;
        method.xc = cpu::scf::XCType::PBE;
        method.build(frame);
        forces = cpu::pw::pw_forces_analytic(frame, method);
    } else {
        cuda::pw::PW method;
        method.gth_name = "GTH-PBE-q4";
        method.ecut_ry = kEcutRy;
        method.k_mesh = kMesh;
        method.tol = 1e-6;
        method.max_iter = 40;
        method.mix_beta = 0.5;
        method.xc = cpu::scf::XCType::PBE;
        method.build(frame);
        forces = cuda::pw::pw_forces_analytic(frame, method);
    }

    if (forces.size() != 2u) {
        std::fprintf(stderr, "FAIL [%s]: expected 2 force vectors, got %zu\n",
                     device_name(dev), forces.size());
        return 1;
    }
    double max_abs = 0.0;
    for (int ia = 0; ia < 2; ++ia) {
        for (int k = 0; k < 3; ++k) {
            const double v = forces[static_cast<std::size_t>(ia)]
                                    [static_cast<std::size_t>(k)];
            if (!std::isfinite(v)) {
                std::fprintf(stderr, "FAIL [%s]: non-finite F[%d][%d]=%g\n",
                             device_name(dev), ia, k, v);
                return 1;
            }
            max_abs = std::max(max_abs, std::abs(v));
        }
    }
    std::printf("PASS [%s]: Si dimer stretch=%.3f A ecut=%.0f Ry Γ PBE  "
                "max|F|=%.6f eV/A  F0=(%.6f,%.6f,%.6f) F1=(%.6f,%.6f,%.6f)\n",
                device_name(dev), kStretch, kEcutRy, max_abs, forces[0][0],
                forces[0][1], forces[0][2], forces[1][0], forces[1][1],
                forces[1][2]);
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

    std::vector<std::array<double, 3>> F_cpu, F_cuda;
    int rc = 0;
    if (want == "cpu" || want == "both")
        rc |= run_on_device(Device::Cpu, F_cpu);
    if (want == "cuda" || want == "both")
        rc |= run_on_device(Device::Cuda, F_cuda);
    if (rc != 0)
        return rc;

    if (want == "both") {
        double max_diff = 0.0;
        double max_cpu = 0.0;
        for (int ia = 0; ia < 2; ++ia) {
            for (int k = 0; k < 3; ++k) {
                const double fc =
                    F_cpu[static_cast<std::size_t>(ia)][static_cast<std::size_t>(k)];
                const double fg =
                    F_cuda[static_cast<std::size_t>(ia)][static_cast<std::size_t>(k)];
                max_diff = std::max(max_diff, std::abs(fc - fg));
                max_cpu = std::max(max_cpu, std::abs(fc));
            }
        }
        if (!(max_cpu > 0.05)) {
            std::fprintf(stderr,
                         "FAIL [both]: CPU force vacuous max|F|=%.6g "
                         "(need > 0.05 eV/A)\n",
                         max_cpu);
            return 1;
        }
        if (!(max_diff < kForceTolEvA)) {
            std::fprintf(stderr,
                         "FAIL [both]: max|F_cuda-F_cpu|=%.6g eV/A  tol=%g\n",
                         max_diff, kForceTolEvA);
            return 1;
        }
        std::printf("PASS [both]: max|ΔF|=%.6e eV/A  (tol=%g)  max|F_cpu|=%.6f  "
                    "xc=PBE\n",
                    max_diff, kForceTolEvA, max_cpu);
    }
    return 0;
}

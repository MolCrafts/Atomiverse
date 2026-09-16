/// @file regressions/pw/cuda-pw-stress-pbe/reg_cuda-pw-stress-pbe.cu
/// @brief END-TO-END: plane-wave Nielsen–Martin cell stress on CUDA under
///        **PBE** — dual-gate (`--device=cpu|cuda|both`, default `both`).
///
/// Gates:
///   1. CPU stress finite + cubic symmetry on diamond Si (PBE SCF);
///   2. CUDA stress finite + cubic symmetry under device PBE-GGA;
///   3. same-settings max|σ_cuda − σ_cpu| < 5e-2 eV/Å³.
///
/// Public API only. GTH-PBE-q4, PBE, Γ. See REFERENCE.md. Sibling of
/// `cuda-pw-stress` (LSDA); this slug closes the PBE stress dual-gate.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <array>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include "atomiverse/cuda/planewave/pw_solver.cuh"

using namespace ATV;

namespace {

// Design-time parity bound (eV/Å³). Same order as LSDA cuda-pw-stress and
// CPU analytic↔FD pin window on reg_pw-analytic-stress-si (0.05). Independent
// SCF + FFT + PBE spectral gradients are not bit-exact.
constexpr double kStressTolEvA3 = 5.0e-2;
constexpr double kSymTol = 2.5e-2; // eV/Å³ cubic symmetry soft bound
constexpr double kA = 5.43;       // Å
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

cpu::Frame make_si_primitive() {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_type)[i] = 14;
        f.span(kAtom_mass)[i] = 28.0855;
    }
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = kA / 4.0;
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

int check_cubic(const char* tag, const std::array<double, 6>& s) {
    for (int v = 0; v < 6; ++v) {
        if (!std::isfinite(s[static_cast<std::size_t>(v)])) {
            std::fprintf(stderr, "FAIL [%s]: non-finite stress[%d]=%g\n", tag, v,
                         s[static_cast<std::size_t>(v)]);
            return 1;
        }
    }
    if (std::abs(s[3]) > kSymTol || std::abs(s[4]) > kSymTol
        || std::abs(s[5]) > kSymTol) {
        std::fprintf(stderr,
                     "FAIL [%s]: off-diag xy=%.6g xz=%.6g yz=%.6g tol=%.3g\n",
                     tag, s[3], s[4], s[5], kSymTol);
        return 1;
    }
    if (std::abs(s[0] - s[1]) > kSymTol || std::abs(s[1] - s[2]) > kSymTol) {
        std::fprintf(stderr,
                     "FAIL [%s]: diag not equal xx=%.6g yy=%.6g zz=%.6g "
                     "tol=%.3g\n",
                     tag, s[0], s[1], s[2], kSymTol);
        return 1;
    }
    return 0;
}

int run_on_device(Device dev, std::array<double, 6>& stress, double* e_ry) {
    cpu::Frame frame = make_si_primitive();
    stress = {};
    if (e_ry)
        *e_ry = 0.0;

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
        stress = cpu::pw::pw_stress(frame, method, e_ry);
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
        stress = cuda::pw::pw_stress(frame, method, e_ry);
    }

    if (check_cubic(device_name(dev), stress) != 0)
        return 1;

    std::printf("PASS [%s]: Si a=%.2f A ecut=%.0f Ry Γ PBE  "
                "σ=%.6g %.6g %.6g | %.3g %.3g %.3g eV/A3  E=%.8f Ry\n",
                device_name(dev), kA, kEcutRy, stress[0], stress[1], stress[2],
                stress[3], stress[4], stress[5], e_ry ? *e_ry : 0.0);
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

    std::array<double, 6> s_cpu{}, s_cuda{};
    double e_cpu = 0.0, e_cuda = 0.0;
    int rc = 0;
    if (want == "cpu" || want == "both")
        rc |= run_on_device(Device::Cpu, s_cpu, &e_cpu);
    if (want == "cuda" || want == "both")
        rc |= run_on_device(Device::Cuda, s_cuda, &e_cuda);
    if (rc != 0)
        return rc;

    if (want == "both") {
        double max_diff = 0.0;
        for (int v = 0; v < 6; ++v) {
            max_diff = std::max(
                max_diff, std::abs(s_cpu[static_cast<std::size_t>(v)]
                                   - s_cuda[static_cast<std::size_t>(v)]));
        }
        // Non-vacuous: compressed diamond Si at a=5.43 has |σ_diag| ≳ 0.01.
        const double mean_diag =
            (std::abs(s_cpu[0]) + std::abs(s_cpu[1]) + std::abs(s_cpu[2]))
            / 3.0;
        if (!(mean_diag > 1e-3)) {
            std::fprintf(stderr,
                         "FAIL [both]: CPU stress vacuous mean|diag|=%.6g "
                         "(need > 1e-3 eV/A3)\n",
                         mean_diag);
            return 1;
        }
        if (!(max_diff < kStressTolEvA3)) {
            std::fprintf(stderr,
                         "FAIL [both]: max|σ_cuda-σ_cpu|=%.6g eV/A3  tol=%g\n"
                         "  cpu:  %.6g %.6g %.6g | %.3g %.3g %.3g\n"
                         "  cuda: %.6g %.6g %.6g | %.3g %.3g %.3g\n",
                         max_diff, kStressTolEvA3, s_cpu[0], s_cpu[1], s_cpu[2],
                         s_cpu[3], s_cpu[4], s_cpu[5], s_cuda[0], s_cuda[1],
                         s_cuda[2], s_cuda[3], s_cuda[4], s_cuda[5]);
            return 1;
        }
        std::printf("PASS [both]: max|Δσ|=%.6e eV/A3  (tol=%g)  "
                    "mean|diag_cpu|=%.6f  |ΔE|=%.3e Ry  xc=PBE\n",
                    max_diff, kStressTolEvA3, mean_diag,
                    std::abs(e_cpu - e_cuda));
    }
    return 0;
}

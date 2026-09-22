/// @file regressions/pw/cuda-pw-spin-stress/reg_cuda-pw-spin-stress.cu
/// @brief END-TO-END: collinear LSDA spin Nielsen–Martin cell stress on CUDA —
///        dual-gate (`--device=cpu|cuda|both`, default `both`).
///
/// Twin of landed forces `cuda-pw-spin-forces` / energy `cuda-pw-collinear-spin`
/// for the **stress** free-fn under collinear fixed-occupation UKS (nspin=2,
/// LSDA). Geometry is O₂ triplet in a cubic cell (bond stretched) so σ is
/// non-vacuous.
///
/// Gates:
///   1. CPU stress finite + non-vacuous on O₂ triplet LSDA;
///   2. CUDA stress finite;
///   3. same-settings max|σ_cuda − σ_cpu| < 5e-2 eV/Å³.
///
/// Public API only. Not spin-PBE stress / spin∧smear / multi-k spin stress.
/// See REFERENCE.md.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include "atomiverse/cuda/planewave/pw_solver.cuh"

#ifndef ATV_REG_DATA_DIR
#error "ATV_REG_DATA_DIR must be set to tests/data"
#endif

using namespace ATV;

namespace {

// Design-time parity bound (eV/Å³). Same class as reg_cuda-pw-stress closed
// shell. Independent SCF + FFT paths are not bit-exact.
constexpr double kStressTolEvA3 = 5.0e-2;
constexpr double kEcutRy = 12.0;
constexpr double kBoxAng = 8.0;
// Equilibrium O₂ ~1.2075 Å; stretch so |σ| / |F| class is non-vacuous under LSDA.
constexpr double kBondAng = 1.30;
constexpr double kTol = 5e-4;
constexpr int kMaxIter = 60;

enum class Device { Cpu, Cuda };

const char* device_name(Device d) { return d == Device::Cuda ? "cuda" : "cpu"; }

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

cpu::Frame make_o2_stretched() {
    cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 8;
        frame.span(kAtom_mass)[i] = 15.999;
        frame.span(kAtom_x)[i] = 0.0;
        frame.span(kAtom_y)[i] = 0.0;
    }
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_z)[1] = kBondAng;
    frame.box = cpu::Box::cubic(kBoxAng);
    return frame;
}

int run_on_device(Device dev, const char* upf, std::array<double, 6>& stress,
                  double* e_ry) {
    cpu::Frame frame = make_o2_stretched();
    stress = {};
    if (e_ry)
        *e_ry = 0.0;

    if (dev == Device::Cpu) {
        cpu::pw::PW method;
        method.upf_path = upf;
        method.ecut_ry = kEcutRy;
        method.k_mesh = 1;
        method.tol = kTol;
        method.max_iter = kMaxIter;
        method.mix_beta = 0.3;
        method.pulay_depth = 5;
        method.spin_polarized = true;
        method.multiplicity = 3;
        method.xc = cpu::scf::XCType::LSDA;
        method.build(frame);
        stress = cpu::pw::pw_stress(frame, method, e_ry);
    } else {
        cuda::pw::PW method;
        method.upf_path = upf;
        method.ecut_ry = kEcutRy;
        method.k_mesh = 1;
        method.tol = kTol;
        method.max_iter = kMaxIter;
        method.mix_beta = 0.3;
        method.pulay_depth = 5;
        method.spin_polarized = true;
        method.multiplicity = 3;
        method.xc = cpu::scf::XCType::LSDA;
        method.build(frame);
        stress = cuda::pw::pw_stress(frame, method, e_ry);
    }

    for (int v = 0; v < 6; ++v) {
        const double s = stress[static_cast<std::size_t>(v)];
        if (!std::isfinite(s)) {
            std::fprintf(stderr, "FAIL [%s]: non-finite stress[%d]=%g\n",
                         device_name(dev), v, s);
            return 1;
        }
    }

    const double mean_diag =
        (std::abs(stress[0]) + std::abs(stress[1]) + std::abs(stress[2])) / 3.0;
    std::printf("PASS [%s]: O2 triplet LSDA bond=%.3f A ecut=%.0f Ry Γ  "
                "σ=%.6g %.6g %.6g | %.3g %.3g %.3g eV/A3  E=%.8f Ry  "
                "mean|diag|=%.6g\n",
                device_name(dev), kBondAng, kEcutRy, stress[0], stress[1],
                stress[2], stress[3], stress[4], stress[5], e_ry ? *e_ry : 0.0,
                mean_diag);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const char* upf = ATV_REG_DATA_DIR "/pw/O_ONCV_PBE-1.2.upf";
    if (std::FILE* f = std::fopen(upf, "r")) {
        std::fclose(f);
    } else {
        std::printf("FAIL: cannot open %s\n", upf);
        return 1;
    }

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
        rc |= run_on_device(Device::Cpu, upf, s_cpu, &e_cpu);
    if (want == "cuda" || want == "both")
        rc |= run_on_device(Device::Cuda, upf, s_cuda, &e_cuda);
    if (rc != 0)
        return rc;

    if (want == "both") {
        double max_diff = 0.0;
        for (int v = 0; v < 6; ++v) {
            max_diff = std::max(
                max_diff, std::abs(s_cpu[static_cast<std::size_t>(v)]
                                   - s_cuda[static_cast<std::size_t>(v)]));
        }
        // Non-vacuous: molecular vacuum pressure on O₂ in 8 Å box.
        const double mean_diag =
            (std::abs(s_cpu[0]) + std::abs(s_cpu[1]) + std::abs(s_cpu[2]))
            / 3.0;
        if (!(mean_diag > 1e-4)) {
            std::fprintf(stderr,
                         "FAIL [both]: CPU stress vacuous mean|diag|=%.6g "
                         "(need > 1e-4 eV/A3)\n",
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
                    "mean|diag_cpu|=%.6f  |ΔE|=%.3e Ry\n",
                    max_diff, kStressTolEvA3, mean_diag,
                    std::abs(e_cpu - e_cuda));
    }
    return 0;
}

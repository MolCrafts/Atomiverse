/// @file regressions/pw/cuda-pw-smear-stress/reg_cuda-pw-smear-stress.cu
/// @brief END-TO-END: restricted Fermi–Dirac metallic-smear Nielsen–Martin
///        cell stress on CUDA — dual-gate (`--device=cpu|cuda|both`, default
///        `both`).
///
/// Twin of landed energy `cuda-pw-metallic-smear` / forces
/// `cuda-pw-smear-forces` for the **stress** free-fn under restricted FD
/// smear (Mermin free energy). Geometry is compressed FCC Al (1 atom) so
/// hydrostatic |σ| is non-vacuous and valence is odd (requires smear).
///
/// Gates:
///   1. CPU stress finite + cubic symmetry under FD smear;
///   2. CUDA stress finite + cubic symmetry;
///   3. same-settings max|σ_cuda − σ_cpu| < 5e-2 eV/Å³.
///
/// Public API only. Not spin∧smear / multi-k smear stress / smear×PBE.
/// See REFERENCE.md.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
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
constexpr double kSymTol = 5.0e-2; // eV/Å³ cubic symmetry soft bound
// Compressed vs equilibrium a≈4.05 Å so |σ_diag| is non-vacuous.
constexpr double kA = 3.90;
constexpr double kEcutRy = 16.0;
constexpr double kDegaussRy = 0.02;
constexpr double kTol = 1e-5;
constexpr int kMesh = 1;
constexpr int kMaxIter = 60;
constexpr int kExtraBands = 4;

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

/// FCC primitive cell, one Al at the origin, lattice constant a [Å].
cpu::Frame make_al_fcc_primitive(double a_ang) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 13;
    f.span(kAtom_mass)[0] = 26.9815385;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    const double h = a_ang / 2.0;
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

template <typename PwT>
void configure_smear(PwT& pw, const char* upf) {
    pw.upf_path = upf;
    pw.ecut_ry = kEcutRy;
    pw.k_mesh = kMesh;
    pw.tol = kTol;
    pw.max_iter = kMaxIter;
    pw.mix_beta = 0.4;
    pw.pulay_depth = 5;
    pw.xc = cpu::scf::XCType::LSDA;
    pw.smearing = cpu::pw::Smearing::fermi_dirac;
    pw.degauss = kDegaussRy;
    pw.extra_bands = kExtraBands;
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

int run_on_device(Device dev, const char* upf, std::array<double, 6>& stress,
                  double* e_ry) {
    cpu::Frame frame = make_al_fcc_primitive(kA);
    stress = {};
    if (e_ry)
        *e_ry = 0.0;

    if (dev == Device::Cpu) {
        cpu::pw::PW method;
        configure_smear(method, upf);
        method.build(frame);
        stress = cpu::pw::pw_stress(frame, method, e_ry);
    } else {
        cuda::pw::PW method;
        configure_smear(method, upf);
        method.build(frame);
        stress = cuda::pw::pw_stress(frame, method, e_ry);
    }

    if (check_cubic(device_name(dev), stress) != 0)
        return 1;

    std::printf("PASS [%s]: Al FCC a=%.2f A ecut=%.0f Ry Γ FD smear  "
                "σ=%.6g %.6g %.6g | %.3g %.3g %.3g eV/A3  E=%.8f Ry\n",
                device_name(dev), kA, kEcutRy, stress[0], stress[1], stress[2],
                stress[3], stress[4], stress[5], e_ry ? *e_ry : 0.0);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const char* upf = ATV_REG_DATA_DIR "/pw/Al_pz_nc.upf";
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
                    "mean|diag_cpu|=%.6f  |ΔE|=%.3e Ry\n",
                    max_diff, kStressTolEvA3, mean_diag,
                    std::abs(e_cpu - e_cuda));
    }
    return 0;
}

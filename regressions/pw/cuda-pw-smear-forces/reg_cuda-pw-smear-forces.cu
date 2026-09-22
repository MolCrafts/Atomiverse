/// @file regressions/pw/cuda-pw-smear-forces/reg_cuda-pw-smear-forces.cu
/// @brief END-TO-END: restricted Fermi–Dirac metallic-smear analytic ionic
///        forces on CUDA — dual-gate (`--device=cpu|cuda|both`, default `both`).
///
/// Twin of landed energy `cuda-pw-metallic-smear` with Al displaced off the
/// FCC origin so |F| is non-vacuous. Gates:
///   1. CPU forces finite and non-vacuous under FD smear;
///   2. CUDA forces finite;
///   3. same-settings max|F_cuda − F_cpu| < 5e-3 eV/Å.
///
/// Public API only. Not spin∧smear / stress / smear×PBE. See REFERENCE.md.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cuda/planewave/pw_solver.cuh"

#ifndef ATV_REG_DATA_DIR
#error "ATV_REG_DATA_DIR must be set to tests/data"
#endif

using namespace ATV;

namespace {

// Design-time parity bound (same class as reg_cuda-pw-forces closed shell).
constexpr double kForceTolEvA = 5.0e-3;
// Cubic cell with a stretched Al dimer: monatomic 1-atom cells have F≡0
// (origin is pure gauge); relative bond stretch yields non-vacuous |F|.
constexpr double kBoxAng = 6.5;
constexpr double kBondAng = 2.55; // slightly compressed vs FCC NN ~2.86 Å
constexpr double kEcutRy = 16.0;
constexpr double kDegaussRy = 0.02;
constexpr double kTol = 1e-5;
constexpr int kMesh = 1;
constexpr int kMaxIter = 60;
constexpr int kExtraBands = 4;

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

/// Two Al atoms along z in a cubic cell (stretched dimer; metallic occ).
cpu::Frame make_al_dimer() {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_type)[i] = 13;
        f.span(kAtom_mass)[i] = 26.9815385;
        f.span(kAtom_x)[i] = 0.0;
        f.span(kAtom_y)[i] = 0.0;
    }
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_z)[1] = kBondAng;
    f.box = cpu::Box::cubic(kBoxAng);
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
    pw.smearing = cpu::pw::Smearing::fermi_dirac;
    pw.degauss = kDegaussRy;
    pw.extra_bands = kExtraBands;
}

int run_on_device(Device dev, const char* upf,
                  std::vector<std::array<double, 3>>& forces) {
    cpu::Frame frame = make_al_dimer();
    forces.clear();

    if (dev == Device::Cpu) {
        cpu::pw::PW method;
        configure_smear(method, upf);
        method.build(frame);
        forces = cpu::pw::pw_forces_analytic(frame, method);
    } else {
        cuda::pw::PW method;
        configure_smear(method, upf);
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
    std::printf("PASS [%s]: Al2 FD smear bond=%.3f A box=%.1f A ecut=%.0f Ry "
                "k=%d  max|F|=%.6f eV/A  F0=(%.6f,%.6f,%.6f) "
                "F1=(%.6f,%.6f,%.6f)\n",
                device_name(dev), kBondAng, kBoxAng, kEcutRy, kMesh, max_abs,
                forces[0][0], forces[0][1], forces[0][2], forces[1][0],
                forces[1][1], forces[1][2]);
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

    std::vector<std::array<double, 3>> F_cpu, F_cuda;
    int rc = 0;
    if (want == "cpu" || want == "both")
        rc |= run_on_device(Device::Cpu, upf, F_cpu);
    if (want == "cuda" || want == "both")
        rc |= run_on_device(Device::Cuda, upf, F_cuda);
    if (rc != 0)
        return rc;

    if (want == "both") {
        double max_diff = 0.0;
        double max_cpu = 0.0;
        for (int ia = 0; ia < 2; ++ia) {
            for (int k = 0; k < 3; ++k) {
                const double fc = F_cpu[static_cast<std::size_t>(ia)]
                                       [static_cast<std::size_t>(k)];
                const double fg = F_cuda[static_cast<std::size_t>(ia)]
                                        [static_cast<std::size_t>(k)];
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
        std::printf(
            "PASS [both]: max|ΔF|=%.6e eV/A  (tol=%g)  max|F_cpu|=%.6f\n",
            max_diff, kForceTolEvA, max_cpu);
    }
    return 0;
}

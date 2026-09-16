/// @file regressions/pw/cuda-pw-collinear-spin/reg_cuda-pw-collinear-spin.cu
/// @brief END-TO-END: collinear fixed-occupation UKS (O2 triplet, LSDA) on
///        cuda::pw::PW — dual-gate CPU↔CUDA + magnetization identity.
///
/// Twin of `pw-spin-polarized-o2` restricted to the device-supported surface:
///   * spin_polarized = true, multiplicity = 3 (M = 2)
///   * XCType::LSDA (device collinear path is LSDA-only; no smear)
///   * same-settings CPU oracle (not a self-golden of CUDA)
///
/// Public API only. Requires ATV_BUILD_PW + ATV_BUILD_CUDA.

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

// Magnetization identity: multiplicity 3 → M = 2 (integer occupations).
constexpr double kWantMag = 2.0;
constexpr double kTolMag = 1e-9;
// Same-settings CPU↔CUDA energy parity [Ha]. Non-bit-exact (cuFFT vs FFTW,
// tree vs serial reductions, libdevice vs libm, Davidson path).
constexpr double kParityTolHa = 2.0e-3;
constexpr double kEcutRy = 12.0;
constexpr double kBoxAng = 8.0;
constexpr double kBondAng = 1.2075;

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

cpu::Frame make_o2() {
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

struct RunOut {
    double energy_ha = 0;
    double magnetization = 0;
    bool converged = false;
    int iterations = 0;
};

RunOut run_one(Device dev, const cpu::Frame& frame, const char* upf) {
    RunOut o;
    if (dev == Device::Cpu) {
        cpu::pw::PW pw;
        pw.upf_path = upf;
        pw.ecut_ry = kEcutRy;
        pw.k_mesh = 1;
        pw.tol = 5e-4;
        pw.max_iter = 60;
        pw.mix_beta = 0.3;
        pw.pulay_depth = 5;
        pw.spin_polarized = true;
        pw.multiplicity = 3;
        pw.xc = cpu::scf::XCType::LSDA;
        pw.build(frame);
        const auto& r = pw.run(frame);
        o.energy_ha = r.energy_ha;
        o.magnetization = r.magnetization;
        o.converged = r.converged;
        o.iterations = r.iterations;
    } else {
        cuda::pw::PW pw;
        pw.upf_path = upf;
        pw.ecut_ry = kEcutRy;
        pw.k_mesh = 1;
        pw.tol = 5e-4;
        pw.max_iter = 60;
        pw.mix_beta = 0.3;
        pw.pulay_depth = 5;
        pw.spin_polarized = true;
        pw.multiplicity = 3;
        pw.xc = cpu::scf::XCType::LSDA;
        pw.build(frame);
        const auto& r = pw.run(frame);
        o.energy_ha = r.energy_ha;
        o.magnetization = r.magnetization;
        o.converged = r.converged;
        o.iterations = r.iterations;
    }
    return o;
}

bool check_one(const RunOut& r, Device dev) {
    bool ok = true;
    std::printf("[%s] E=%.10f Ha  mag=%.6f  conv=%d iters=%d\n",
                device_name(dev), r.energy_ha, r.magnetization,
                (int)r.converged, r.iterations);
    if (!std::isfinite(r.energy_ha)) {
        std::printf("FAIL [%s]: non-finite energy\n", device_name(dev));
        ok = false;
    }
    if (!r.converged) {
        std::printf("FAIL [%s]: SCF not converged\n", device_name(dev));
        ok = false;
    }
    if (std::fabs(r.magnetization - kWantMag) > kTolMag) {
        std::printf("FAIL [%s]: mag got=%.12f want=%.12f\n", device_name(dev),
                    r.magnetization, kWantMag);
        ok = false;
    }
    return ok;
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
    auto frame = make_o2();

    RunOut cpu_out{}, cuda_out{};
    bool ok = true;
    if (want == "cpu" || want == "both") {
        cpu_out = run_one(Device::Cpu, frame, upf);
        ok = check_one(cpu_out, Device::Cpu) && ok;
    }
    if (want == "cuda" || want == "both") {
        cuda_out = run_one(Device::Cuda, frame, upf);
        ok = check_one(cuda_out, Device::Cuda) && ok;
    }
    if (want == "both") {
        const double de = std::fabs(cuda_out.energy_ha - cpu_out.energy_ha);
        std::printf("parity |E_cuda - E_cpu| = %.6e Ha (tol %.1e)\n", de,
                    kParityTolHa);
        if (de > kParityTolHa) {
            std::printf("FAIL: CPU↔CUDA energy parity\n");
            ok = false;
        }
    }

    if (!ok)
        return 1;
    std::printf("PASS: cuda-pw-collinear-spin (mag + LSDA dual-gate)\n");
    return 0;
}

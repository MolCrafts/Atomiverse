/// @file regressions/pw/cuda-pw-metallic-smear/reg_cuda-pw-metallic-smear.cu
/// @brief END-TO-END: restricted Fermi–Dirac metallic smear on cuda::pw::PW
///        — dual-gate CPU↔CUDA free energy + electron count (FCC Al).
///
/// Twin of `pw-smearing-al` restricted to the device-supported surface:
///   * spin_polarized = false, smearing = fermi_dirac
///   * same-settings CPU oracle (not a self-golden of CUDA)
///
/// Public API only. Requires ATV_BUILD_PW + ATV_BUILD_CUDA.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

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

// Same-settings CPU↔CUDA free-energy parity [Ry]. Non-bit-exact (cuFFT vs
// FFTW, tree vs serial reductions, libdevice vs libm, Davidson trajectory).
constexpr double kParityTolRy = 5.0e-3;
constexpr double kNeTol = 1e-5;
constexpr double kAAng = 4.05;
constexpr double kEcutRy = 20.0;
constexpr double kDegaussRy = 0.02;

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

/// FCC primitive cell with lattice constant a [Å], one Al at the origin.
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

struct RunOut {
    double energy_ry = 0;
    double e_fermi_ry = 0;
    double n_elec_occ = 0;
    double e_entropy_ry = 0;
    bool converged = false;
    int iterations = 0;
};

template <typename PwT>
RunOut run_pw(PwT& pw, const cpu::Frame& frame, const char* upf) {
    pw.upf_path = upf;
    pw.ecut_ry = kEcutRy;
    pw.k_mesh = 2;
    pw.tol = 1e-5;
    pw.max_iter = 60;
    pw.mix_beta = 0.4;
    pw.pulay_depth = 5;
    pw.smearing = cpu::pw::Smearing::fermi_dirac;
    pw.degauss = kDegaussRy;
    pw.extra_bands = 4;
    // Match the CPU Al offline pin path (default PBE members; NC LDA UPF).
    pw.build(frame);
    const auto& r = pw.run(frame);
    RunOut o;
    o.energy_ry = r.energy;
    o.e_fermi_ry = r.e_fermi;
    o.n_elec_occ = r.n_elec_occ;
    o.e_entropy_ry = r.e_entropy;
    o.converged = r.converged;
    o.iterations = r.iterations;
    return o;
}

RunOut run_one(Device dev, const cpu::Frame& frame, const char* upf) {
    if (dev == Device::Cpu) {
        cpu::pw::PW pw;
        return run_pw(pw, frame, upf);
    }
    cuda::pw::PW pw;
    return run_pw(pw, frame, upf);
}

bool check_one(const RunOut& r, Device dev) {
    bool ok = true;
    std::printf("[%s] E=%.10f Ry  e_fermi=%.6f Ry  n_occ=%.8f  e_ent=%.6f  "
                "conv=%d iters=%d\n",
                device_name(dev), r.energy_ry, r.e_fermi_ry, r.n_elec_occ,
                r.e_entropy_ry, (int)r.converged, r.iterations);
    if (!std::isfinite(r.energy_ry) || !std::isfinite(r.e_fermi_ry)) {
        std::printf("FAIL [%s]: non-finite energy or e_fermi\n",
                    device_name(dev));
        ok = false;
    }
    if (!r.converged) {
        std::printf("FAIL [%s]: SCF not converged\n", device_name(dev));
        ok = false;
    }
    if (std::fabs(r.n_elec_occ - 3.0) > kNeTol) {
        std::printf("FAIL [%s]: n_elec_occ=%.8f want 3 ± %g\n",
                    device_name(dev), r.n_elec_occ, kNeTol);
        ok = false;
    }
    if (!(std::fabs(r.e_fermi_ry) < 1e3)) {
        std::printf("FAIL [%s]: absurd e_fermi=%.6f Ry\n", device_name(dev),
                    r.e_fermi_ry);
        ok = false;
    }
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const char* upf = ATV_REG_DATA_DIR "/pw/Al_pz_nc.upf";
    if (std::FILE* fp = std::fopen(upf, "r")) {
        std::fclose(fp);
    } else {
        std::printf("FAIL: cannot open %s\n", upf);
        return 1;
    }

    const std::string want = parse_device_arg(argc, argv);
    auto frame = make_al_fcc_primitive(kAAng);

    bool ok = true;
    RunOut cpu_out{}, cuda_out{};
    bool have_cpu = false, have_cuda = false;

    if (want == "cpu" || want == "both") {
        cpu_out = run_one(Device::Cpu, frame, upf);
        ok = check_one(cpu_out, Device::Cpu) && ok;
        have_cpu = true;
    }
    if (want == "cuda" || want == "both") {
        cuda_out = run_one(Device::Cuda, frame, upf);
        ok = check_one(cuda_out, Device::Cuda) && ok;
        have_cuda = true;
    }
    if (want != "cpu" && want != "cuda" && want != "both") {
        std::printf("FAIL: unknown --device=%s (use cpu|cuda|both)\n",
                    want.c_str());
        return 1;
    }

    if (have_cpu && have_cuda) {
        const double dE = std::fabs(cuda_out.energy_ry - cpu_out.energy_ry);
        std::printf("PARITY |E_cuda-E_cpu|=%.6e Ry  (tol %.1e Ry)\n", dE,
                    kParityTolRy);
        if (dE > kParityTolRy) {
            std::printf("FAIL: free-energy parity exceeded\n");
            ok = false;
        }
        const double dn = std::fabs(cuda_out.n_elec_occ - cpu_out.n_elec_occ);
        if (dn > kNeTol) {
            std::printf("FAIL: n_occ parity |Δ|=%.3e > %g\n", dn, kNeTol);
            ok = false;
        }
    }

    if (!ok) {
        std::printf("FAIL: cuda-pw-metallic-smear\n");
        return 1;
    }
    std::printf(
        "PASS: cuda-pw-metallic-smear (FCC Al Fermi–Dirac dual-gate)\n");
    return 0;
}

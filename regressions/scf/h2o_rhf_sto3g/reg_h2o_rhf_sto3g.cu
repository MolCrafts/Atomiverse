/// @file regressions/scf/h2o_rhf_sto3g/reg_h2o_rhf_sto3g.cu
/// @brief H2O / STO-3G RHF energy — **one regression**, multi-device
///        (PyTorch-style `--device=cpu|cuda|both`, default `both`).
///
/// Same literature pin for every device. Public API only:
///   cpu::scf::RHF  or  cuda::scf::RHF  + Frame
/// Frame + concrete method only; no SCFSolver / *Config, or user Context.
/// Built only when `ATV_BUILD_CUDA=ON` (replaces the .cpp target).
/// See REFERENCE.md.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cuda/scf/rhf.cuh"

namespace {

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

cpu::Frame make_water_frame() {
    using namespace ATV;
    constexpr double kPi = 3.14159265358979323846;
    const double r_oh = 0.95;
    const double half = 0.5 * 104.5 * kPi / 180.0;
    const double hx = r_oh * std::sin(half);
    const double hz = r_oh * std::cos(half);

    cpu::Frame frame(3);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = hx;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = hz;
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_x)[2] = -hx;
    frame.span(kAtom_y)[2] = 0.0;
    frame.span(kAtom_z)[2] = hz;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

int run_on_device(Device dev) {
    using namespace ATV;
    constexpr double kEnergyRef = -74.961754;
    constexpr double kTol = 1.0e-3;

    cpu::Frame frame = make_water_frame();
    bool converged = false;
    double energy_ha = 0.0;

    if (dev == Device::Cpu) {
        cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
        rhf.run(frame);
        converged = rhf.converged();
        energy_ha = rhf.energy();
    } else {
        cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
        rhf.run(frame);
        converged = rhf.converged();
        energy_ha = rhf.energy();
    }

    const double got_meta = frame.meta.get(kEnergy) / units::kHartreeInEv;
    const double got = std::isfinite(energy_ha) ? energy_ha : got_meta;

    if (!converged) {
        std::fprintf(stderr, "FAIL [%s]: RHF/STO-3G(H2O) did not converge\n",
                     device_name(dev));
        return 1;
    }
    if (!std::isfinite(got) || std::abs(got - kEnergyRef) > kTol) {
        std::fprintf(stderr,
                     "FAIL [%s]: E_RHF/STO-3G(H2O) got=%.6f want=%.6f "
                     "tol=%.1e Ha\n",
                     device_name(dev), got, kEnergyRef, kTol);
        return 1;
    }

    std::printf("PASS [%s]: E_RHF/STO-3G(H2O) = %.6f Ha (want %.6f +/- %.1e) "
                "[LibreTexts §28.2; PySCF cross-check]\n",
                device_name(dev), got, kEnergyRef, kTol);
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
    if (want == "cpu" || want == "both") {
        rc |= run_on_device(Device::Cpu);
    }
    if (want == "cuda" || want == "both") {
        rc |= run_on_device(Device::Cuda);
    }
    return rc;
}

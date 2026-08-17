/// @file regressions/scf/h2_rhf_sto3g_aimd/reg_h2_rhf_sto3g_aimd.cu
/// @brief Short H2 / STO-3G RHF BOMD — **one case**, multi-device
///        (`--device=cpu|cuda|both`, default both). No AimdDriver.
///        See REFERENCE.md.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/scf/rhf.cuh"

namespace {

enum class Device { Cpu, Cuda };
const char* device_name(Device d) { return d == Device::Cuda ? "cuda" : "cpu"; }

std::string parse_device_arg(int argc, char** argv) {
    std::string want = "both";
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strncmp(a, "--device=", 9) == 0)
            want = a + 9;
        else if (std::strcmp(a, "--device") == 0 && i + 1 < argc)
            want = argv[++i];
        else if (std::strcmp(a, "cpu") == 0 || std::strcmp(a, "cuda") == 0
                 || std::strcmp(a, "both") == 0)
            want = a;
    }
    return want;
}

ATV::cpu::Frame make_h2() {
    using namespace ATV;
    cpu::Frame frame(2);
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_x)[1] = 0.74;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.0;
    frame.span(kAtom_mass)[0] = 1.008;
    frame.span(kAtom_mass)[1] = 1.008;
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_vx)[0] = 0.001;
    frame.span(kAtom_vx)[1] = -0.001;
    frame.box = cpu::Box::cubic(100.0);
    return frame;
}

int run_on_device(Device dev) {
    using namespace ATV;
    cpu::Frame frame = make_h2();

    if (dev == Device::Cpu) {
        cpu::md::MD sim;
        sim.set_integrator(cpu::md::VelocityVerlet{0.5});
        sim.on(md::Phase::force)
            .add(cpu::scf::RHF{cpu::scf::BasisSet::STO3G}.with_gradient(true));
        sim.build();
        sim.run(frame, 5);
    } else {
        cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, 1e-9, 100, 0.0, true};
        cuda::md::MD sim;
        sim.set_integrator(cuda::md::VelocityVerlet{0.5});
        sim.set_evaluator(std::move(rhf));
        sim.build();
        sim.run(frame, 5);
    }

    const double pe_ev = static_cast<double>(frame.meta.get(kEnergy));
    const double pe_ha = pe_ev / units::kHartreeInEv;
    if (!std::isfinite(pe_ev) || pe_ev >= 0.0) {
        std::fprintf(stderr, "FAIL [%s]: BOMD PE not finite/bound: %.6f eV\n",
                     device_name(dev), pe_ev);
        return 1;
    }
    constexpr double kPeMinEv = -35.0;
    constexpr double kPeMaxEv = -25.0;
    if (pe_ev < kPeMinEv || pe_ev > kPeMaxEv) {
        std::fprintf(stderr,
                     "FAIL [%s]: PE outside literature envelope: %.6f eV\n",
                     device_name(dev), pe_ev);
        return 1;
    }
    std::printf("PASS [%s]: H2 RHF/STO-3G short BOMD PE=%.6f eV (%.6f Ha)\n",
                device_name(dev), pe_ev, pe_ha);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::string want = parse_device_arg(argc, argv);
    if (want != "cpu" && want != "cuda" && want != "both") {
        std::fprintf(stderr, "FAIL: unknown --device=%s\n", want.c_str());
        return 1;
    }
    int rc = 0;
    if (want == "cpu" || want == "both")
        rc |= run_on_device(Device::Cpu);
    if (want == "cuda" || want == "both")
        rc |= run_on_device(Device::Cuda);
    return rc;
}

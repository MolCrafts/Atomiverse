/// @file regressions/scf/h2o_rhf_def2svp_grad/reg_h2o_rhf_def2svp_grad.cu
/// @brief H2O / def2-SVP RHF analytic gradient vs PySCF — **one case**,
///        `--device=cpu|cuda|both` (default both). See REFERENCE.md.
///
/// Public RHF + Frame only. Frame + concrete method only; no SCFSolver / *Config.
///
/// **The CUDA leg is a negative envelope gate** (REFERENCE.md §Device
/// legs): def2-SVP carries d shells, outside the device s/p STO-3G envelope,
/// so `cuda::scf::RHF::forces` must throw a how-to pointing at
/// `cpu::scf::RHF`. The day the envelope grows this must fail loudly —
/// rewrite with device tolerances. The device s/p gradient e2e is
/// `scf/h2o_rhf_sto3g_grad`.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/methods.h"
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

ATV::cpu::Frame make_frame() {
    using namespace ATV;
    cpu::Frame frame(3);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_x)[2] = 0.0;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

int run_on_device(Device dev) {
    using namespace ATV;
    cpu::Frame frame = make_frame();
    cpu::scf::GradientResult res;
    bool converged = false;

    if (dev == Device::Cpu) {
        cpu::scf::RHF rhf{cpu::scf::BasisSet::Def2SVP, 0, 1e-12, 200};
        res = rhf.forces(frame);
        converged = rhf.converged();
    } else {
        cuda::scf::RHF rhf{cpu::scf::BasisSet::Def2SVP, 0, 1e-12, 200};
        try {
            res = rhf.forces(frame);
            std::fprintf(stderr,
                         "FAIL [cuda]: def2-SVP forces returned instead of "
                         "throwing — envelope grew or host fallback returned; "
                         "update REFERENCE.md and retighten this leg\n");
            return 1;
        } catch (const std::invalid_argument& e) {
            const std::string msg(e.what());
            if (msg.find("cpu::scf::RHF") == std::string::npos
                || msg.find("STO-3G") == std::string::npos) {
                std::fprintf(stderr, "FAIL [cuda]: throw missing how-to: %s\n",
                             e.what());
                return 1;
            }
            std::printf("[cuda] path=throw (def2-SVP d shells outside the "
                        "device s/p STO-3G envelope)\n");
            std::printf("PASS [cuda]\n");
            return 0;
        }
    }

    if (!converged) {
        std::fprintf(stderr, "FAIL [%s]: RHF forces did not converge\n",
                     device_name(dev));
        return 1;
    }

    auto grad_ha_bohr = [&](int A, int d) {
        return -units::convert(res.forces(A, d), units::Force::eVPerAngstrom,
                               units::Force::HartreePerBohr);
    };

    // PySCF 2.14.0 analytic gradient [Ha/Bohr] — REFERENCE.md
    const double want[3][3] = {{0.0, 0.0, 1.804450607022e-02},
                               {0.0, 1.107031890619e-02, -9.022253035110e-03},
                               {0.0, -1.107031890619e-02, -9.022253035110e-03}};
    const double tol = 1e-6;

    double max_err = 0.0;
    double net[3] = {0.0, 0.0, 0.0};
    for (int A = 0; A < 3; ++A) {
        for (int d = 0; d < 3; ++d) {
            const double g = grad_ha_bohr(A, d);
            net[d] += g;
            max_err = std::max(max_err, std::fabs(g - want[A][d]));
        }
    }
    const double net_max =
        std::max({std::fabs(net[0]), std::fabs(net[1]), std::fabs(net[2])});

    std::printf("[%s] max_elementwise_err=%.3e tol=%.0e  |Σg|_max=%.3e\n",
                device_name(dev), max_err, tol, net_max);

    if (!(max_err < tol)) {
        std::printf("FAIL [%s]: gradient off PySCF oracle\n", device_name(dev));
        return 1;
    }
    if (!(net_max < 1e-8)) {
        std::printf("FAIL [%s]: not translationally invariant\n",
                    device_name(dev));
        return 1;
    }
    std::printf("PASS [%s]\n", device_name(dev));
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

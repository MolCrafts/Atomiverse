/// @file regressions/scf/hf-cuda-lgt1/reg_hf-cuda-lgt1.cu
/// @brief H2O RHF/6-31G* (Cartesian d) and H2O RHF/def2-TZVP (Cartesian f
///        then spherical sandwich) vs PySCF. `--device=cpu|cuda|both`.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cuda/scf/rhf.cuh"
#include "harness/registry.h"

namespace atv_reg_scf_hf_cuda_lgt1 {

namespace {

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

ATV::cpu::Frame water() {
    using namespace ATV;
    cpu::Frame frame(3);
    frame.span(kAtom_atomic_number)[0] = 8;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_atomic_number)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_atomic_number)[2] = 1;
    frame.span(kAtom_x)[2] = 0.0;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

int check_cpu(const char* label, ATV::cpu::scf::BasisSet basis, double want) {
    using namespace ATV;
    constexpr double kTol = 1e-8;
    cpu::Frame frame = water();
    cpu::scf::RHF rhf{basis, 0, 1e-12, 100};
    rhf.run(frame);
    const double got = frame.meta.get(kEnergy) / units::kHartreeInEv;
    if (!rhf.converged() || !std::isfinite(got)
        || std::abs(got - want) > kTol) {
        std::fprintf(stderr,
                     "FAIL [cpu]: %s E=%.12f want=%.12f conv=%d tol=%.0e\n",
                     label, got, want, rhf.converged() ? 1 : 0, kTol);
        return 1;
    }
    std::printf(
        "PASS [cpu]: %s E=%.12f Ha (want %.12f +/- %.0e) [PySCF 2.14.0]\n",
        label, got, want, kTol);
    return 0;
}

int check_cuda(const char* label, ATV::cpu::scf::BasisSet basis, double want) {
    using namespace ATV;
    // Device MD ERI is not bit-identical to the CPU Rys engine. 1e-6 Ha is
    // the window already used for the device H2/STO-3G energy pin.
    constexpr double kTol = 1e-6;
    cpu::Frame frame = water();
    cuda::scf::RHF rhf{basis, 0, 1e-12, 100, 0.0, false};
    rhf.run(frame);
    const double got = frame.meta.get(kEnergy) / units::kHartreeInEv;
    const double method = rhf.energy();
    if (!rhf.last_scf_on_device() || rhf.last_force_on_device()) {
        std::fprintf(stderr,
                     "FAIL [cuda]: %s is not a device SCF "
                     "(scf_on_device=%d force_on_device=%d)\n",
                     label, rhf.last_scf_on_device() ? 1 : 0,
                     rhf.last_force_on_device() ? 1 : 0);
        return 1;
    }
    if (!rhf.converged() || !std::isfinite(got) || std::abs(got - want) > kTol
        || std::abs(method - want) > kTol) {
        std::fprintf(stderr,
                     "FAIL [cuda]: %s E=%.12f method=%.12f want=%.12f conv=%d "
                     "tol=%.0e\n",
                     label, got, method, want, rhf.converged() ? 1 : 0, kTol);
        return 1;
    }
    std::printf("PASS [cuda]: %s E=%.12f Ha (want %.12f +/- %.0e) "
                "path=device [PySCF 2.14.0]\n",
                label, got, want, kTol);
    return 0;
}

} // namespace

int run(int argc, char** argv) {
    const std::string want = parse_device_arg(argc, argv);
    if (want != "cpu" && want != "cuda" && want != "both") {
        std::fprintf(stderr, "FAIL: unknown --device=%s (want cpu|cuda|both)\n",
                     want.c_str());
        return 1;
    }
    int rc = 0;
    if (want == "cpu" || want == "both") {
        rc |= check_cpu("H2O/6-31G*", ATV::cpu::scf::BasisSet::Pople631Gs,
                        -76.01050498825705);
        rc |= check_cpu("H2O/def2-TZVP", ATV::cpu::scf::BasisSet::Def2TZVP,
                        -76.05900476057711);
    }
    if (want == "cuda" || want == "both") {
        rc |= check_cuda("H2O/6-31G*", ATV::cpu::scf::BasisSet::Pople631Gs,
                         -76.01050498825705);
        rc |= check_cuda("H2O/def2-TZVP", ATV::cpu::scf::BasisSet::Def2TZVP,
                         -76.05900476057711);
    }
    return rc;
}

} // namespace atv_reg_scf_hf_cuda_lgt1

ATV_REGRESSION("scf/rhf", "hf-cuda-lgt1", atv_reg_scf_hf_cuda_lgt1::run);

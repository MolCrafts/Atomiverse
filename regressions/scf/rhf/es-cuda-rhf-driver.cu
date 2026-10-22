/// @file regressions/scf/es-cuda-rhf-driver/reg_es-cuda-rhf-driver.cu
/// @brief Staged RHF driver vs PySCF — one regression, multi-device.
///
/// `--device=cpu|cuda|both` (default both). CPU grades H2/STO-3G and
/// H2O/6-31G. CUDA grades only H2/STO-3G and requires the device SCF and
/// force flags. H2O/6-31G is not a device ERI kernel; the host-fill path
/// is not a pass. See REFERENCE.md.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cuda/scf/rhf.cuh"
#include "harness/registry.h"

namespace atv_reg_scf_es_cuda_rhf_driver {

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

ATV::cpu::Frame make_h2() {
    using namespace ATV;
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.74;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

ATV::cpu::Frame make_h2o() {
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

int check_cpu(const char* label, ATV::cpu::Frame frame,
              ATV::cpu::scf::BasisSet basis, double e_want) {
    using namespace ATV;
    constexpr double kTol = 1e-8;
    cpu::scf::RHF rhf{basis, 0, 1e-12, 200};
    rhf.run(frame);
    const double e_ha = rhf.energy();
    const double meta_ha = frame.meta.get(kEnergy) / units::kHartreeInEv;
    if (!rhf.converged() || !std::isfinite(e_ha)
        || std::abs(e_ha - e_want) > kTol || !std::isfinite(meta_ha)
        || std::abs(meta_ha - e_want) > kTol) {
        std::fprintf(stderr,
                     "FAIL [cpu]: %s E=%.12f meta=%.12f want=%.12f "
                     "conv=%d tol=%.0e\n",
                     label, e_ha, meta_ha, e_want, rhf.converged() ? 1 : 0,
                     kTol);
        return 1;
    }
    std::printf("PASS [cpu]: %s E=%.12f Ha (want %.12f +/- %.0e) "
                "[PySCF 2.14.0]\n",
                label, e_ha, e_want, kTol);
    return 0;
}

int check_cuda_h2() {
    using namespace ATV;
    // Device energy band from scf/h2o_rhf_sto3g_grad (REFERENCE.md).
    constexpr double kTol = 1e-6;
    constexpr double kWant = -1.1167593073964255;
    cpu::Frame frame = make_h2();
    cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, 1e-12, 200};
    rhf.run(frame);
    const double e_ha = rhf.energy();
    const double meta_ha = frame.meta.get(kEnergy) / units::kHartreeInEv;
    if (!rhf.last_scf_on_device() || !rhf.last_force_on_device()) {
        std::fprintf(stderr,
                     "FAIL [cuda]: H2/STO-3G host ERI path is not a device "
                     "result (scf_on_device=%d force_on_device=%d)\n",
                     rhf.last_scf_on_device() ? 1 : 0,
                     rhf.last_force_on_device() ? 1 : 0);
        return 1;
    }
    if (!rhf.converged() || !std::isfinite(e_ha)
        || std::abs(e_ha - kWant) > kTol || !std::isfinite(meta_ha)
        || std::abs(meta_ha - kWant) > kTol) {
        std::fprintf(stderr,
                     "FAIL [cuda]: H2/STO-3G E=%.12f meta=%.12f want=%.12f "
                     "conv=%d tol=%.0e\n",
                     e_ha, meta_ha, kWant, rhf.converged() ? 1 : 0, kTol);
        return 1;
    }
    std::printf("PASS [cuda]: H2/STO-3G E=%.12f Ha (want %.12f +/- %.0e) "
                "path=device [PySCF 2.14.0]\n",
                e_ha, kWant, kTol);
    return 0;
}

} // namespace

int run(int argc, char** argv) {
    static_assert(std::is_base_of_v<ATV::cuda::Driver<ATV::scf::Phase>,
                                    ATV::cuda::scf::RHF>,
                  "cuda::scf::RHF must be cuda::Driver<scf::Phase>");

    const std::string want = parse_device_arg(argc, argv);
    if (want != "cpu" && want != "cuda" && want != "both") {
        std::fprintf(stderr, "FAIL: unknown --device=%s (want cpu|cuda|both)\n",
                     want.c_str());
        return 1;
    }
    int rc = 0;
    if (want == "cpu" || want == "both") {
        rc |= check_cpu("H2/STO-3G", make_h2(), ATV::cpu::scf::BasisSet::STO3G,
                        -1.1167593073964255);
        rc |= check_cpu("H2O/6-31G", make_h2o(),
                        ATV::cpu::scf::BasisSet::Pople631G, -75.98397447272204);
    }
    if (want == "cuda" || want == "both") {
        // 6-31G stays on the CPU leg. Device kernels are STO-3G s/p only.
        std::printf("[cuda] H2O/6-31G skipped: not a device ERI kernel "
                    "(host-fill is not graded)\n");
        rc |= check_cuda_h2();
    }
    return rc;
}

} // namespace atv_reg_scf_es_cuda_rhf_driver

ATV_REGRESSION("scf/rhf", "es-cuda-rhf-driver",
               atv_reg_scf_es_cuda_rhf_driver::run);

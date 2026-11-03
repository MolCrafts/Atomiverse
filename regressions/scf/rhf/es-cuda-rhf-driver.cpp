/// @file regressions/scf/es-cuda-rhf-driver/reg_es-cuda-rhf-driver.cpp
/// @brief Staged RHF driver energies vs an offline PySCF pin — CPU path.
///
/// H2/STO-3G and H2O/6-31G. When ATV_BUILD_CUDA=ON, CMake builds
/// reg_es-cuda-rhf-driver.cu instead (`--device=cpu|cuda|both`).
/// Public API only. See REFERENCE.md.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "harness/registry.h"

namespace atv_reg_scf_es_cuda_rhf_driver {

namespace {

using ATV::cpu::Frame;

Frame make_h2() {
    using namespace ATV;
    Frame frame(2);
    frame.span(kAtom_atomic_number)[0] = 1;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_atomic_number)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.74;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

Frame make_h2o() {
    using namespace ATV;
    Frame frame(3);
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

int check_case(const char* label, Frame frame, ATV::cpu::scf::BasisSet basis,
               double e_want) {
    using namespace ATV;
    constexpr double kTol = 1e-8; // Ha — cpu band of h2o_rhf_sto3g_grad
    cpu::scf::RHF rhf{basis, /*charge=*/0, /*tol=*/1e-12, /*max_iter=*/200};
    rhf.run(frame);
    const double e_ha = rhf.energy();
    const double meta_ha = frame.meta.get(kEnergy) / units::kHartreeInEv;
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL [cpu]: %s did not converge\n", label);
        return 1;
    }
    if (!std::isfinite(e_ha) || std::abs(e_ha - e_want) > kTol) {
        std::fprintf(stderr,
                     "FAIL [cpu]: %s E got=%.12f want=%.12f tol=%.0e Ha\n",
                     label, e_ha, e_want, kTol);
        return 1;
    }
    if (!std::isfinite(meta_ha) || std::abs(meta_ha - e_want) > kTol) {
        std::fprintf(stderr, "FAIL [cpu]: %s kEnergy got=%.12f want=%.12f Ha\n",
                     label, meta_ha, e_want);
        return 1;
    }
    std::printf("PASS [cpu]: %s E=%.12f Ha (want %.12f +/- %.0e) "
                "[PySCF 2.14.0]\n",
                label, e_ha, e_want, kTol);
    return 0;
}

} // namespace

int run() {
    using namespace ATV;
    int rc = 0;
    rc |= check_case("H2/STO-3G", make_h2(), cpu::scf::BasisSet::STO3G,
                     -1.1167593073964255);
    rc |= check_case("H2O/6-31G", make_h2o(), cpu::scf::BasisSet::Pople631G,
                     -75.98397447272204);
    return rc;
}

} // namespace atv_reg_scf_es_cuda_rhf_driver

ATV_REGRESSION("scf/rhf", "es-cuda-rhf-driver",
               atv_reg_scf_es_cuda_rhf_driver::run);

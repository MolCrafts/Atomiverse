/// @file regressions/scf/hf-cuda-lgt1/reg_hf-cuda-lgt1.cpp
/// @brief H2O RHF/6-31G* and H2O RHF/def2-TZVP energies vs PySCF.
///
/// CPU-only build of the case. The CUDA build uses the .cu translation
/// unit (`--device=cpu|cuda|both`). Public Frame + RHF only.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "harness/registry.h"

namespace atv_reg_scf_hf_cuda_lgt1 {

namespace {

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

int check(const char* label, ATV::cpu::scf::BasisSet basis, double want) {
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

} // namespace

int run() {
    int rc = 0;
    rc |= check("H2O/6-31G*", ATV::cpu::scf::BasisSet::Pople631Gs,
                -76.01050498825705);
    rc |= check("H2O/def2-TZVP", ATV::cpu::scf::BasisSet::Def2TZVP,
                -76.05900476057711);
    return rc;
}

} // namespace atv_reg_scf_hf_cuda_lgt1

ATV_REGRESSION("scf/rhf", "hf-cuda-lgt1", atv_reg_scf_hf_cuda_lgt1::run);

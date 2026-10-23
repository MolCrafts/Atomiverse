/// @file regressions/scf/multireference/casscf-h2o-sto3g.cpp
/// @brief H2O CASSCF(2,2)/STO-3G total energy vs offline PySCF pin.
///
/// Unlike H2/STO-3G CAS(2,2), this active space has a frozen core and a
/// virtual orbital (ncore=4, nvirt=1, 7 STO-3G MOs), so the orbital
/// macro-iteration is not the identity CI pass. Out-of-library consumer:
/// public headers only. Product gate: e_tot to 1e-6 Ha (REFERENCE.md).

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "harness/registry.h"

namespace atv_reg_scf_casscf_h2o_sto3g {

namespace {

bool check(const char* what, double got, double want, double tol) {
    if (!std::isfinite(got) || std::abs(got - want) > tol) {
        std::fprintf(stderr, "FAIL: %s got=%.15f want=%.15f tol=%.1e\n", what,
                     got, want, tol);
        return false;
    }
    std::printf("  %-28s %18.12f (want %18.12f +/- %.1e)\n", what, got, want,
                tol);
    return true;
}

} // namespace

int run() {
    using namespace ATV;

    // Geometry (A) — pin_scf `water` preset.
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

    // Pins — REFERENCE.md (PySCF 2.14.0 mcscf.CASSCF(2,2) / RHF STO-3G).
    // ncore=4, nao=7 => nvirt=1. The orbital step is not a no-op.
    constexpr double kEHf = -74.96302313846292;       // Ha
    constexpr double kECorr = -0.0012536321488170188; // Ha
    constexpr double kETot = -74.96427677061173;      // Ha
    constexpr double kTol = 1e-6;                     // Ha

    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
    rhf.tol = 1e-12;
    rhf.max_iter = 100;

    const cpu::scf::CASSCFResult cas =
        ATV::cpu::scf::CASSCF<decltype(rhf)>{rhf, /*ncas=*/2, /*nelecas=*/2}
            .run(frame);
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF/STO-3G (H2O) did not converge\n");
        return 1;
    }
    if (!cas.converged) {
        std::fprintf(stderr, "FAIL: CASSCF did not converge (iters=%d)\n",
                     cas.iterations);
        return 1;
    }
    if (cas.ncas != 2 || cas.nelecas != 2 || cas.ncore != 4) {
        std::fprintf(
            stderr,
            "FAIL: active-space bookkeeping ncas=%d nelecas=%d ncore=%d\n",
            cas.ncas, cas.nelecas, cas.ncore);
        return 1;
    }
    // ncore=4 and ncas=2 on 7 STO-3G MOs leaves one virtual, so a converged
    // run that stopped on the first (identity) rotation did not optimize
    // orbitals. PySCF takes two macro iterations on this pin.
    if (cas.iterations < 2) {
        std::fprintf(stderr,
                     "FAIL: orbital macro-iteration did not run (iters=%d)\n",
                     cas.iterations);
        return 1;
    }

    std::printf("H2O CASSCF(2,2)/STO-3G (RHF reference, ncore=4 nvirt=1):\n");
    if (!check("E_RHF [Ha]", rhf.energy(), kEHf, kTol))
        return 1;
    if (!check("E_corr CASSCF [Ha]", cas.e_corr, kECorr, kTol))
        return 1;
    if (!check("E_tot CASSCF [Ha]", cas.e_casscf, kETot, kTol))
        return 1;
    if (!check("E_tot = E_HF+E_corr", cas.e_casscf, rhf.energy() + cas.e_corr,
               1e-12))
        return 1;
    if (!frame.meta.contains(kEnergy.key)) {
        std::fprintf(stderr, "FAIL: CASSCF did not publish kEnergy\n");
        return 1;
    }
    if (!check("kEnergy [eV]", frame.meta.get(kEnergy),
               cas.e_casscf * units::kHartreeInEv, 1e-8))
        return 1;

    std::printf("  ncas=%d  nelecas=%d  ncore=%d  iters=%d\n", cas.ncas,
                cas.nelecas, cas.ncore, cas.iterations);
    std::printf("\nPASS: casscf-h2o-sto3g\n");
    return 0;
}

} // namespace atv_reg_scf_casscf_h2o_sto3g

ATV_REGRESSION("scf/multireference", "casscf-h2o-sto3g",
               atv_reg_scf_casscf_h2o_sto3g::run);

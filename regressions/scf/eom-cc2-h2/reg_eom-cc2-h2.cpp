/// @file regressions/scf/eom-cc2-h2/reg_eom-cc2-h2.cpp
/// @brief H₂ EOM-CC2/STO-3G singlet S1 excitation energy vs offline
///        PySCF pin (REFERENCE.md).
///
/// Out-of-library consumer: public headers only
/// (`cpu/scf/methods.h`, `cpu/frame/frame.h`). Product gate: ω to 1e-4 eV.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

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

int main() {
    using namespace ATV;

    // Geometry (Å) — pin_scf `h2` preset (H–H = 0.74 Å along z).
    cpu::Frame frame(2);
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.74;
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_type)[1] = 1;
    frame.box = cpu::Box::cubic(20.0);

    // Pins — REFERENCE.md (PySCF 2.14.0 RCCSD(cc2=True)+eomee_ccsd_singlet).
    constexpr double kOmega0Ev = 26.164970973311735; // eV
    constexpr double kECc2 = -1.129897380985958;     // Ha
    constexpr double kTolEv = 1e-4;                  // eV
    constexpr double kTolHa = 1e-6;                  // Ha

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 100;

    const cpu::scf::EOMCC2Result eom = rhf.eom_cc2(frame, /*n_states=*/1);
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF/STO-3G (H2) did not converge\n");
        return 1;
    }
    if (!eom.converged) {
        std::fprintf(stderr, "FAIL: EOM-CC2 did not converge\n");
        return 1;
    }
    if (eom.energy_ev.empty()) {
        std::fprintf(stderr, "FAIL: no EOM roots returned\n");
        return 1;
    }

    std::printf("H2 EOM-CC2/STO-3G (singlet S1):\n");
    if (!check("E_CC2 [Ha]", eom.e_cc2_ha, kECc2, kTolHa))
        return 1;
    if (!check("omega_S1 [eV]", eom.energy_ev[0], kOmega0Ev, kTolEv))
        return 1;

    std::printf("  cc2_iterations             %d\n", eom.cc2_iterations);
    std::printf("\nPASS: eom-cc2-h2\n");
    return 0;
}

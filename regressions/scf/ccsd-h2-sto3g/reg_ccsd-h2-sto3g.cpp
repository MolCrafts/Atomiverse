/// @file regressions/scf/ccsd-h2-sto3g/reg_ccsd-h2-sto3g.cpp
/// @brief H₂ CCSD/STO-3G correlation + total energy vs offline PySCF pin
///        (REFERENCE.md).
///
/// Out-of-library consumer: public headers only
/// (`cpu/scf/methods.h`, `cpu/frame/frame.h`). Product gate: e_corr and
/// e_tot to 1e-6 Ha.

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

    // Pins — REFERENCE.md (PySCF 2.14.0 cc.CCSD / RHF STO-3G).
    constexpr double kEHf = -1.116759307396426;   // Ha
    constexpr double kECorr = -0.020524527092106; // Ha
    constexpr double kETot = -1.137283834488532;  // Ha
    constexpr double kTol = 1e-6;                 // Ha

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 100;

    const cpu::scf::CCSDResult ccsd = rhf.ccsd(frame);
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF/STO-3G (H2) did not converge\n");
        return 1;
    }
    if (!ccsd.converged) {
        std::fprintf(stderr, "FAIL: CCSD amplitudes did not converge\n");
        return 1;
    }

    std::printf("H2 CCSD/STO-3G (RHF reference):\n");
    if (!check("E_RHF [Ha]", rhf.energy(), kEHf, kTol))
        return 1;
    if (!check("E_corr CCSD [Ha]", ccsd.e_corr, kECorr, kTol))
        return 1;
    if (!check("E_tot CCSD [Ha]", ccsd.e_tot, kETot, kTol))
        return 1;
    if (!check("E_tot = E_HF+E_corr", ccsd.e_tot, rhf.energy() + ccsd.e_corr,
               1e-12))
        return 1;

    std::printf("  iterations                 %d\n", ccsd.iterations);
    std::printf("\nPASS: ccsd-h2-sto3g\n");
    return 0;
}

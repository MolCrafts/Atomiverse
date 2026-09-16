/// @file regressions/scf/ccsd-t-h2o-sto3g/reg_ccsd-t-h2o-sto3g.cpp
/// @brief H₂O CCSD(T)/STO-3G correlation + total energy vs offline PySCF pin
///        (REFERENCE.md).
///
/// Out-of-library consumer: public headers only
/// (`cpu/scf/methods.h`, `cpu/frame/frame.h`). Product gate: e_corr and
/// e_tot (and e_t diagnostic) to 1e-6 Ha.

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

    // Geometry (Å) — pin_scf `water` preset.
    cpu::Frame frame(3);
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_x)[2] = 0.0;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_type)[2] = 1;
    frame.box = cpu::Box::cubic(20.0);

    // Pins — REFERENCE.md (PySCF 2.14.0 cc.CCSD + ccsd_t / RHF STO-3G).
    constexpr double kEHf = -74.963023138462930;      // Ha
    constexpr double kECorrCcsd = -0.049438563031963; // Ha
    constexpr double kET = -0.000067409682944;        // Ha
    constexpr double kECorr = -0.049505972714907;     // Ha = CCSD + (T)
    constexpr double kETot = -75.012529111177841;     // Ha
    constexpr double kTol = 1e-6;                     // Ha

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 100;

    const cpu::scf::CCSDTResult ct = rhf.ccsd_t(frame);
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF/STO-3G (H2O) did not converge\n");
        return 1;
    }
    if (!ct.converged) {
        std::fprintf(stderr, "FAIL: CCSD amplitudes did not converge\n");
        return 1;
    }

    std::printf("H2O CCSD(T)/STO-3G (RHF reference):\n");
    if (!check("E_RHF [Ha]", rhf.energy(), kEHf, kTol))
        return 1;
    if (!check("E_corr CCSD [Ha]", ct.e_corr_ccsd, kECorrCcsd, kTol))
        return 1;
    if (!check("E_(T) [Ha]", ct.e_t, kET, kTol))
        return 1;
    if (!check("E_corr CCSD(T) [Ha]", ct.e_corr, kECorr, kTol))
        return 1;
    if (!check("E_tot CCSD(T) [Ha]", ct.e_tot, kETot, kTol))
        return 1;
    if (!check("E_corr = CCSD+(T)", ct.e_corr, ct.e_corr_ccsd + ct.e_t, 1e-12))
        return 1;
    if (!check("E_tot = E_HF+E_corr", ct.e_tot, rhf.energy() + ct.e_corr,
               1e-12))
        return 1;

    std::printf("  iterations                 %d\n", ct.iterations);
    std::printf("\nPASS: ccsd-t-h2o-sto3g\n");
    return 0;
}

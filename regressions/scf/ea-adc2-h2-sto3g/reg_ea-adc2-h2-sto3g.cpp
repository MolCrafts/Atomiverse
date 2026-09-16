/// @file regressions/scf/ea-adc2-h2-sto3g/reg_ea-adc2-h2-sto3g.cpp
/// @brief H2 EA-ADC(2)/STO-3G lowest attachment energy vs the offline
///        PySCF pin (REFERENCE.md).
///
/// Out-of-library consumer: public headers only (`cpu/scf/methods.h`,
/// `cpu/scf/adc.h` via rhf.h, `cpu/frame/frame.h`, `core/units.h`).
///
/// Geometry: pin_scf `h2` preset (Å). Product gate: EA0 to 1e-4 eV.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

bool check(const char* what, double got, double want, double tol) {
    if (!std::isfinite(got) || std::abs(got - want) > tol) {
        std::fprintf(stderr, "FAIL: %s got=%.12f want=%.12f tol=%.1e\n", what,
                     got, want, tol);
        return false;
    }
    std::printf("  %-36s %14.8f (want %14.8f +/- %.1e)\n", what, got, want,
                tol);
    return true;
}

} // namespace

int main() {
    using namespace ATV;

    // ── Geometry (Å) — identical to REFERENCE.md / pin_scf h2 ───────────
    cpu::Frame frame(2);
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.74;
    frame.span(kAtom_type)[1] = 1;
    frame.box = cpu::Box::cubic(20.0);

    // ── Pins (REFERENCE.md; Hartree stored, eV via engine kHartreeInEv) ─
    constexpr double kERhf = -1.116759307396; // Ha
    constexpr double kEa0Ha = 0.684281565504; // Ha
    constexpr double kEmp2 = -0.013138073590; // Ha (diagnostic)
    constexpr double kTolE = 1e-6;            // Ha
    constexpr double kTolEv = 1e-4;           // eV (product gate)

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.run(frame);
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF/STO-3G (H2) did not converge\n");
        return 1;
    }
    std::printf("H2 HF/STO-3G:\n");
    if (!check("E_RHF [Ha]", rhf.energy(), kERhf, kTolE))
        return 1;

    const cpu::scf::EAADC2Result ea = rhf.ea_adc2(/*n_states=*/1);
    if (ea.energy_ev.size() != 1) {
        std::fprintf(stderr, "FAIL: ea_adc2 returned unexpected root count\n");
        return 1;
    }

    const double ea0_ev = ea.energy_ev[0];
    std::printf("\n  EA-ADC(2) EA0 = %.6f eV\n", ea0_ev);
    std::printf("  MP2 E_corr = %.10f Ha\n\n", ea.e_corr_mp2_ha);

    if (!check("MP2 E_corr [Ha]", ea.e_corr_mp2_ha, kEmp2, kTolE))
        return 1;
    if (!check("EA-ADC(2) EA0 [eV]", ea0_ev, kEa0Ha * units::kHartreeInEv,
               kTolEv))
        return 1;

    std::printf("\nPASS: ea-adc2-h2-sto3g\n");
    return 0;
}

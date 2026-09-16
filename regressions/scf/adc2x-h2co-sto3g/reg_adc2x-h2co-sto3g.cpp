/// @file regressions/scf/adc2x-h2co-sto3g/reg_adc2x-h2co-sto3g.cpp
/// @brief Formaldehyde ADC(2)-x/STO-3G S1 excitation energy vs the offline
///        PySCF pin (REFERENCE.md).
///
/// Out-of-library consumer: public headers only (`cpu/scf/methods.h`,
/// `cpu/scf/adc.h` via rhf.h, `cpu/frame/frame.h`, `core/units.h`).
///
/// Geometry: pin_scf `h2co` preset (Å). Product gate: ADC(2)-x S1 to 1e-4 eV.

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

    // ── Geometry (Å) — identical to REFERENCE.md / pin_scf h2co ─────────
    cpu::Frame frame(4);
    const double xs[4] = {0.0, 0.0, 0.0, 0.0};
    const double ys[4] = {0.0, 0.0, 0.943, -0.943};
    const double zs[4] = {0.0, 1.210, -0.587, -0.587};
    const int types[4] = {6, 8, 1, 1};
    for (int i = 0; i < 4; ++i) {
        frame.span(kAtom_x)[i] = xs[i];
        frame.span(kAtom_y)[i] = ys[i];
        frame.span(kAtom_z)[i] = zs[i];
        frame.span(kAtom_type)[i] = types[i];
    }
    frame.box = cpu::Box::cubic(30.0);

    // ── Pins (REFERENCE.md; Hartree stored, eV via engine kHartreeInEv) ─
    constexpr double kERhf = -112.353990369522;  // Ha
    constexpr double kS1Ha = 0.121950512434326;  // Ha
    constexpr double kEmp2 = -0.117214030416689; // Ha (diagnostic)
    constexpr double kTolE = 1e-6;               // Ha
    constexpr double kTolEv = 1e-4;              // eV (product gate)

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.run(frame);
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF/STO-3G (H2CO) did not converge\n");
        return 1;
    }
    std::printf("H2CO HF/STO-3G:\n");
    if (!check("E_RHF [Ha]", rhf.energy(), kERhf, kTolE))
        return 1;

    const cpu::scf::ADC2Result adc = rhf.adc2x(/*n_states=*/1);
    if (adc.energy_ev.size() != 1) {
        std::fprintf(stderr, "FAIL: adc2x returned unexpected root count\n");
        return 1;
    }

    const double s1_ev = adc.energy_ev[0];
    std::printf("\n  ADC(2)-x S1 = %.6f eV\n", s1_ev);
    std::printf("  MP2 E_corr = %.10f Ha\n\n", adc.e_corr_mp2_ha);

    if (!check("MP2 E_corr [Ha]", adc.e_corr_mp2_ha, kEmp2, kTolE))
        return 1;
    if (!check("ADC(2)-x S1 [eV]", s1_ev, kS1Ha * units::kHartreeInEv, kTolEv))
        return 1;

    std::printf("\nPASS: adc2x-h2co-sto3g\n");
    return 0;
}

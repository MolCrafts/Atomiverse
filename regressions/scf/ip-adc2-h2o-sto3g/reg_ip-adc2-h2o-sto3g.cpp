/// @file regressions/scf/ip-adc2-h2o-sto3g/reg_ip-adc2-h2o-sto3g.cpp
/// @brief Water IP-ADC(2)/STO-3G lowest ionization potential vs the offline
///        PySCF pin (REFERENCE.md).
///
/// Out-of-library consumer: public headers only (`cpu/scf/methods.h`,
/// `cpu/frame/frame.h`, `core/units.h`).
///
/// Geometry: pin_scf `water` preset (Å). Product gate: IP1 to 1e-4 eV.

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

    // ── Geometry (Å) — identical to REFERENCE.md / pin_scf water ────────
    cpu::Frame frame(3);
    const double xs[3] = {0.0, 0.0, 0.0};
    const double ys[3] = {0.0, 0.7572, -0.7572};
    const double zs[3] = {0.1173, -0.4692, -0.4692};
    const int types[3] = {8, 1, 1};
    for (int i = 0; i < 3; ++i) {
        frame.span(kAtom_x)[i] = xs[i];
        frame.span(kAtom_y)[i] = ys[i];
        frame.span(kAtom_z)[i] = zs[i];
        frame.span(kAtom_type)[i] = types[i];
    }
    frame.box = cpu::Box::cubic(30.0);

    // ── Pins (REFERENCE.md; Hartree stored, eV via engine kHartreeInEv) ─
    constexpr double kERhf = -74.963023138463; // Ha
    constexpr double kIp1Ha = 0.304180883162;  // Ha
    constexpr double kEmp2 = -0.035545651647;  // Ha (diagnostic)
    constexpr double kTolE = 1e-6;             // Ha
    constexpr double kTolEv = 1e-4;            // eV (product gate)

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.run(frame);
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF/STO-3G (H2O) did not converge\n");
        return 1;
    }
    std::printf("H2O HF/STO-3G:\n");
    if (!check("E_RHF [Ha]", rhf.energy(), kERhf, kTolE))
        return 1;

    const cpu::scf::IPADC2Result adc = rhf.ip_adc2(/*n_states=*/1);
    if (adc.energy_ev.size() != 1) {
        std::fprintf(stderr, "FAIL: ip_adc2 returned unexpected root count\n");
        return 1;
    }

    const double ip1_ev = adc.energy_ev[0];
    std::printf("\n  IP-ADC(2) IP1 = %.6f eV\n", ip1_ev);
    std::printf("  MP2 E_corr = %.10f Ha\n\n", adc.e_corr_mp2_ha);

    if (!check("MP2 E_corr [Ha]", adc.e_corr_mp2_ha, kEmp2, kTolE))
        return 1;
    if (!check("IP-ADC(2) IP1 [eV]", ip1_ev, kIp1Ha * units::kHartreeInEv,
               kTolEv))
        return 1;

    std::printf("\nPASS: ip-adc2-h2o-sto3g\n");
    return 0;
}

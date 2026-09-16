/// @file regressions/scf/cisd-h2co-sto3g/reg_cisd-h2co-sto3g.cpp
/// @brief Formaldehyde CIS(D)/STO-3G S1 excitation energy vs the offline
///        PySCF + Head-Gordon 1994 pin (REFERENCE.md).
///
/// Out-of-library consumer: public headers only (`cpu/scf/methods.h`,
/// `cpu/scf/cisd.h` via rhf.h, `cpu/frame/frame.h`, `core/units.h`).
///
/// Geometry: pin_scf `h2co` preset (Å). Product gate: CIS(D) S1 to 1e-4 eV.

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
    constexpr double kERhf = -112.353990369522;    // Ha
    constexpr double kCisHa = 0.15722459206960743; // Ha
    constexpr double kCisdHa = 0.1670101189898645; // Ha
    constexpr double kTolE = 1e-6;                 // Ha
    constexpr double kTolEv = 1e-4;                // eV (product gate)

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

    const cpu::scf::CISDResult cisd = rhf.cisd(/*n_states=*/1);
    if (cisd.energy_ev.size() != 1 || cisd.cis_energy_ev.size() != 1) {
        std::fprintf(stderr, "FAIL: cisd returned unexpected root count\n");
        return 1;
    }

    const double cis_ev = cisd.cis_energy_ev[0];
    const double cisd_ev = cisd.energy_ev[0];
    std::printf("\n  CIS   S1 = %.6f eV\n", cis_ev);
    std::printf("  CIS(D) S1 = %.6f eV  (corr = %.6f eV)\n\n", cisd_ev,
                cisd.doubles_correction_ev[0]);

    if (!check("CIS S1 [eV]", cis_ev, kCisHa * units::kHartreeInEv, kTolEv))
        return 1;
    if (!check("CIS(D) S1 [eV]", cisd_ev, kCisdHa * units::kHartreeInEv,
               kTolEv))
        return 1;

    std::printf("\nPASS: cisd-h2co-sto3g\n");
    return 0;
}

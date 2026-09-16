/// @file regressions/scf/h2-b3lyp-rpa-tddft/reg_h2-b3lyp-rpa-tddft.cpp
/// @brief H₂ B3LYP/STO-3G hybrid full linear-response TDDFT S₁ vs PySCF pin.
///
/// Public-API consumer only. Geometry, basis, and oracle numbers are pinned
/// in REFERENCE.md (PySCF 2.14.0 offline; no live import in ctest).
///
/// Casida full RPA (A,B) on a closed-shell RKS/B3LYP hybrid reference with
/// α_hf exact-exchange on both A and B plus adiabatic semi-local f^xc —
/// not TDA and not pure-KS RPA.
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

    // Geometry [Å] — REFERENCE.md (bond on +x; equivalent to +z pin).
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.74;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.box = cpu::Box::cubic(20.0);

    // PySCF 2.14.0 pins (REFERENCE.md; cart=True, conv_tol=1e-12, B3LYP TDDFT).
    constexpr double kERksHa = -1.165418410673;  // Ha
    constexpr double kS1Ev = 25.30779734533265;  // eV (primary gate)
    constexpr double kS1Ha = 0.9300443981998157; // Ha (sanity)
    constexpr double kF =
        0.8608891208128707;           // length-gauge oscillator strength
    constexpr double kTolS1Ev = 1e-4; // eV — job gate
    constexpr double kTolEHa = 1e-4;  // Ha — ground-state sanity
    constexpr double kTolF = 0.15;    // generous f band

    cpu::scf::RKS rks(cpu::scf::XCType::B3LYP);
    rks.basis = cpu::scf::BasisSet::STO3G;
    rks.tol = 1e-10;
    rks.max_iter = 100;
    rks.run(f);
    if (!rks.converged()) {
        std::fprintf(stderr, "FAIL: RKS/B3LYP did not converge\n");
        return 1;
    }

    std::printf("H2 B3LYP/STO-3G hybrid full LR-TDDFT (Casida A,B):\n");
    if (!check("E_RKS [Ha]", rks.energy(), kERksHa, kTolEHa))
        return 1;

    const cpu::scf::ExcitedStatesResult es =
        rks.tddft(1, cpu::scf::ExcitationChannel::singlet);
    if (es.energy_ev.empty() || !std::isfinite(es.energy_ev[0])) {
        std::fprintf(stderr, "FAIL: empty/non-finite S1\n");
        return 1;
    }
    if (es.tda) {
        std::fprintf(stderr, "FAIL: tda flag true on full RPA TDDFT path\n");
        return 1;
    }
    if (es.alda) {
        std::fprintf(stderr, "FAIL: alda=true on hybrid B3LYP full LR-TDDFT\n");
        return 1;
    }

    const double s1_ev = es.energy_ev[0];
    const double s1_ha = s1_ev / units::kHartreeInEv;
    std::printf(
        "  S1 = %.8f eV (%.10f Ha)  f=%.6f  tda=%d alda=%d\n", s1_ev, s1_ha,
        es.oscillator_strength.empty() ? 0.0 : es.oscillator_strength[0],
        static_cast<int>(es.tda), static_cast<int>(es.alda));

    if (!check("S1 [eV]", s1_ev, kS1Ev, kTolS1Ev))
        return 1;
    if (!check("S1 [Ha] (consistency)", s1_ha, kS1Ha,
               kTolS1Ev / units::kHartreeInEv))
        return 1;
    if (!es.oscillator_strength.empty()) {
        if (!check("f (length gauge)", es.oscillator_strength[0], kF, kTolF))
            return 1;
    }

    // Guard: must not accidentally match hybrid TDA on this pin (25.928 eV).
    constexpr double kTdaS1Ev = 25.928364822972;
    if (std::abs(s1_ev - kTdaS1Ev) < 0.05
        && std::abs(s1_ev - kS1Ev) > kTolS1Ev) {
        std::fprintf(
            stderr,
            "FAIL: S1 matches B3LYP-TDA pin (%.6f eV) — used tda path?\n",
            kTdaS1Ev);
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}

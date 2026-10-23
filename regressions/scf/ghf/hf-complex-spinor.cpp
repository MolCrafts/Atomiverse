/// @file regressions/scf/hf-complex-spinor/reg_hf-complex-spinor.cpp
/// @brief Complex spin-orbital HF and Dirac–Hartree–Fock vs PySCF.
///
/// Public API only. Pins: REFERENCE.md.
///   - ComplexGHF x2c / real GHF on H2 and H2O STO-3G (existing pins).
///   - DHF on H2 at 1.0 Angstrom, STO-3G and cc-pVDZ (scf.DHF docstring).

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "harness/registry.h"

namespace atv_reg_scf_hf_complex_spinor {

namespace {

ATV::cpu::Frame h2(double z) {
    ATV::cpu::Frame frame(2);
    frame.span(ATV::kAtom_type)[0] = 1;
    frame.span(ATV::kAtom_type)[1] = 1;
    frame.span(ATV::kAtom_z)[1] = z;
    frame.box = ATV::cpu::Box::cubic(30.0);
    return frame;
}

ATV::cpu::Frame h2o() {
    constexpr double kPi = 3.14159265358979323846;
    const double half = 0.5 * 104.5 * kPi / 180.0;
    const double hx = 0.95 * std::sin(half);
    const double hz = 0.95 * std::cos(half);
    ATV::cpu::Frame frame(3);
    frame.span(ATV::kAtom_type)[0] = 8;
    frame.span(ATV::kAtom_type)[1] = 1;
    frame.span(ATV::kAtom_type)[2] = 1;
    frame.span(ATV::kAtom_x)[1] = hx;
    frame.span(ATV::kAtom_z)[1] = hz;
    frame.span(ATV::kAtom_x)[2] = -hx;
    frame.span(ATV::kAtom_z)[2] = hz;
    frame.box = ATV::cpu::Box::cubic(30.0);
    return frame;
}

bool check_ghf(const char* label, ATV::cpu::Frame frame, bool x2c, double pin) {
    ATV::cpu::scf::ComplexGHF ghf{ATV::cpu::scf::BasisSet::STO3G, 0, 1e-12, 80,
                                  x2c};
    ghf.run(frame);
    const double e = ghf.energy();
    const double diff = std::abs(e - pin);
    std::printf("%s: E = %.12f Ha (pin %.12f, |dE| = %.3e, iters %d)\n", label,
                e, pin, diff, ghf.iterations());
    if (!ghf.converged() || !(diff <= 1e-8)) {
        std::printf("FAIL: %s\n", label);
        return false;
    }
    const double e_ev = frame.meta.get(ATV::kEnergy) / ATV::units::kHartreeInEv;
    return std::abs(e_ev - e) <= 1e-12;
}

bool check_dhf(const char* label, ATV::cpu::scf::BasisSet basis, double pin) {
    auto frame = h2(1.0);
    ATV::cpu::scf::DHF dhf{basis, 0, 1e-12, 80};
    dhf.run(frame);
    const double e = dhf.energy();
    const double diff = std::abs(e - pin);
    std::printf("%s: E = %.12f Ha (pin %.12f, |dE| = %.3e, iters %d)\n", label,
                e, pin, diff, dhf.iterations());
    if (!dhf.converged() || !(diff <= 1e-8)) {
        std::printf("FAIL: %s\n", label);
        return false;
    }
    const double e_ev = frame.meta.get(ATV::kEnergy) / ATV::units::kHartreeInEv;
    return std::abs(e_ev - e) <= 1e-12;
}

} // namespace

int run() {
    bool ok = true;
    ok = check_ghf("H2 x2c", h2(0.74), true, -1.11677137468639) && ok;
    ok = check_ghf("H2O x2c", h2o(), true, -75.00458277099555) && ok;
    ok = check_ghf("H2O GHF", h2o(), false, -74.96175405132141) && ok;
    ok = check_dhf("H2 DHF STO-3G", ATV::cpu::scf::BasisSet::STO3G,
                   -1.0661226588590456)
         && ok;
    ok = check_dhf("H2 DHF cc-pVDZ", ATV::cpu::scf::BasisSet::CcPVDZ,
                   -1.1001626190803968)
         && ok;
    std::printf("hf-complex-spinor: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

} // namespace atv_reg_scf_hf_complex_spinor

ATV_REGRESSION("scf/ghf", "hf-complex-spinor",
               atv_reg_scf_hf_complex_spinor::run);

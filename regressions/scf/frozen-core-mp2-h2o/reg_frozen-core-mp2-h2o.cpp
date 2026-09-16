/// @file regressions/scf/frozen-core-mp2-h2o/reg_frozen-core-mp2-h2o.cpp
/// @brief H2O frozen-core RMP2/6-31G* — e_corr and e_tot vs offline PySCF pins.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Geometry and pins: REFERENCE.md (PySCF 2.14.0, cart 6-31G*, frozen=1 O 1s).
/// Never import/invoke PySCF from this binary.
///
/// Exit non-zero on failure. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

ATV::cpu::Frame h2o_frame() {
    ATV::cpu::Frame frame(3);
    frame.span(ATV::kAtom_type)[0] = 8;
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.1173;
    frame.span(ATV::kAtom_type)[1] = 1;
    frame.span(ATV::kAtom_x)[1] = 0.0;
    frame.span(ATV::kAtom_y)[1] = 0.7572;
    frame.span(ATV::kAtom_z)[1] = -0.4692;
    frame.span(ATV::kAtom_type)[2] = 1;
    frame.span(ATV::kAtom_x)[2] = 0.0;
    frame.span(ATV::kAtom_y)[2] = -0.7572;
    frame.span(ATV::kAtom_z)[2] = -0.4692;
    frame.box = ATV::cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;

    auto frame = h2o_frame();

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::Pople631Gs;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.frozen = 1; // freeze O 1s (lowest occupied spatial MO)

    const cpu::scf::MP2Result mp2 = rhf.mp2(frame);

    // PySCF 2.14.0 pins — see REFERENCE.md (never live-import PySCF here).
    constexpr double want_e_hf = -76.010504988257;
    constexpr double want_e_corr = -0.186119540118;
    constexpr double want_e_tot = -76.196624528375;
    constexpr double tol = 1e-6; // Ha, product gate for e_corr and e_tot

    const double e_hf = rhf.energy();
    std::printf("RHF  E = %.12f Ha (pin %.12f, dE = %.3e)\n", e_hf, want_e_hf,
                e_hf - want_e_hf);
    std::printf("RMP2-fc e_corr = %.12f Ha (pin %.12f, dE = %.3e)\n",
                mp2.e_corr, want_e_corr, mp2.e_corr - want_e_corr);
    std::printf("RMP2-fc e_tot  = %.12f Ha (pin %.12f, dE = %.3e)\n", mp2.e_tot,
                want_e_tot, mp2.e_tot - want_e_tot);
    std::printf("frozen = %d\n", mp2.frozen);

    bool ok = rhf.converged();
    // E_RHF is reference-only (see REFERENCE.md); product gates are e_corr/e_tot.
    ok = ok && std::fabs(e_hf - want_e_hf) < 1e-7;
    ok = ok && std::fabs(mp2.e_corr - want_e_corr) < tol;
    ok = ok && std::fabs(mp2.e_tot - want_e_tot) < tol;
    ok = ok && std::fabs(mp2.e_tot - (e_hf + mp2.e_corr)) < 1e-12;
    ok = ok && !mp2.ri;
    ok = ok && mp2.frozen == 1;
    ok = ok && mp2.e_corr < 0.0;
    // Frozen-core e_corr must be less negative than full MP2 (~ -0.1885 Ha).
    ok = ok && mp2.e_corr > -0.1880;

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

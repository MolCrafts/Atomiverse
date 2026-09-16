/// @file regressions/scf/romp2-o2-sto3g/reg_romp2-o2-sto3g.cpp
/// @brief O2 triplet ROMP2/STO-3G — e_corr and e_tot vs offline PySCF 2.14.0 pins.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Reference: ROHF (9α, 7β) then `ROHF::mp2` (PySCF convention: UMP2 on ROHF
/// spatial MOs with semi-canonical spin-Fock diagonal denominators).
/// Pins in REFERENCE.md.
///
/// Exit non-zero on failure. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

ATV::cpu::Frame o2_frame() {
    ATV::cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(ATV::kAtom_type)[i] = 8;
        frame.span(ATV::kAtom_mass)[i] = 15.999;
        frame.span(ATV::kAtom_x)[i] = 0.0;
        frame.span(ATV::kAtom_y)[i] = 0.0;
    }
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 1.2075;
    frame.box = ATV::cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;

    auto frame = o2_frame();

    cpu::scf::ROHF rohf(9, 7);
    rohf.basis = cpu::scf::BasisSet::STO3G;
    rohf.tol = 1e-12;
    rohf.max_iter = 200;

    const cpu::scf::MP2Result mp2 = rohf.mp2(frame);

    // PySCF 2.14.0 pins — see REFERENCE.md (never live-import PySCF here).
    constexpr double want_e_hf = -147.632166990682;
    constexpr double want_e_corr = -0.096718716392;
    constexpr double want_e_tot = -147.728885707074;
    constexpr double tol = 1e-6; // Ha, product gate for e_corr and e_tot

    const double e_hf = rohf.energy();
    std::printf("ROHF E = %.12f Ha (pin %.12f, dE = %.3e)\n", e_hf, want_e_hf,
                e_hf - want_e_hf);
    std::printf("ROMP2 e_corr = %.12f Ha (pin %.12f, dE = %.3e)\n", mp2.e_corr,
                want_e_corr, mp2.e_corr - want_e_corr);
    std::printf("ROMP2 e_tot  = %.12f Ha (pin %.12f, dE = %.3e)\n", mp2.e_tot,
                want_e_tot, mp2.e_tot - want_e_tot);

    bool ok = rohf.converged();
    ok = ok && std::fabs(e_hf - want_e_hf) < 1e-8;
    ok = ok && std::fabs(mp2.e_corr - want_e_corr) < tol;
    ok = ok && std::fabs(mp2.e_tot - want_e_tot) < tol;
    ok = ok && std::fabs(mp2.e_tot - (e_hf + mp2.e_corr)) < 1e-12;
    ok = ok && !mp2.ri;
    ok = ok && mp2.e_corr < 0.0;

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

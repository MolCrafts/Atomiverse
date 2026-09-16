/// @file regressions/scf/mp3-h2-sto3g/reg_mp3-h2-sto3g.cpp
/// @brief H₂ RMP3/STO-3G — e_corr (E2+E3) vs offline PySCF-oracle pins.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Geometry and pins: REFERENCE.md (Szabo/Psi4NumPy MP3 on PySCF RHF+ao2mo;
/// PySCF 2.14 has no mp.MP3). Never import/invoke PySCF from this binary.
///
/// Exit non-zero on failure. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

ATV::cpu::Frame h2_frame() {
    ATV::cpu::Frame frame(2);
    frame.span(ATV::kAtom_type)[0] = 1;
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_type)[1] = 1;
    frame.span(ATV::kAtom_x)[1] = 0.0;
    frame.span(ATV::kAtom_y)[1] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 0.74;
    frame.box = ATV::cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;

    auto frame = h2_frame();

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 100;

    const cpu::scf::MP3Result mp3 = rhf.mp3(frame);

    // Offline pins — see REFERENCE.md (never live-import PySCF here).
    constexpr double want_e_hf = -1.116759307396;
    constexpr double want_e2 = -0.013138073590;
    constexpr double want_e3 = -0.004836072637;
    constexpr double want_e_corr = -0.017974146227; // product gate
    constexpr double want_e_tot = -1.134733453623;
    constexpr double tol = 1e-6; // Ha

    const double e_hf = rhf.energy();
    std::printf("RHF      E = %.12f Ha (pin %.12f, dE = %.3e)\n", e_hf,
                want_e_hf, e_hf - want_e_hf);
    std::printf("MP2 E(2)   = %.12f Ha (pin %.12f, dE = %.3e)\n",
                mp3.e_corr_mp2, want_e2, mp3.e_corr_mp2 - want_e2);
    std::printf("MP3 E(3)   = %.12f Ha (pin %.12f, dE = %.3e)\n",
                mp3.e_corr_mp3, want_e3, mp3.e_corr_mp3 - want_e3);
    std::printf("MP3 e_corr = %.12f Ha (pin %.12f, dE = %.3e)\n", mp3.e_corr,
                want_e_corr, mp3.e_corr - want_e_corr);
    std::printf("MP3 e_tot  = %.12f Ha (pin %.12f, dE = %.3e)\n", mp3.e_tot,
                want_e_tot, mp3.e_tot - want_e_tot);

    bool ok = rhf.converged();
    ok = ok && std::fabs(e_hf - want_e_hf) < 1e-7;
    ok = ok && std::fabs(mp3.e_corr_mp2 - want_e2) < tol;
    ok = ok && std::fabs(mp3.e_corr_mp3 - want_e3) < tol;
    ok = ok && std::fabs(mp3.e_corr - want_e_corr) < tol;
    ok = ok && std::fabs(mp3.e_tot - want_e_tot) < tol;
    ok = ok && std::fabs(mp3.e_tot - (e_hf + mp3.e_corr)) < 1e-12;
    ok =
        ok && std::fabs(mp3.e_corr - (mp3.e_corr_mp2 + mp3.e_corr_mp3)) < 1e-12;
    ok = ok && mp3.frozen == 0;
    ok = ok && mp3.e_corr < 0.0;
    // MP3 more negative than MP2 alone on this system.
    ok = ok && mp3.e_corr < mp3.e_corr_mp2;
    // Between MP2 and FCI (~ -0.0205 Ha).
    ok = ok && mp3.e_corr > -0.021;

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/// @file regressions/scf/mp4-h2-sto3g/reg_mp4-h2-sto3g.cpp
/// @brief H₂ RMP4(SDTQ)/STO-3G — e_corr (E2+E3+E4) vs offline PySCF-oracle pins.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Geometry and pins: REFERENCE.md (det-space RSPT-4 on PySCF RHF+ao2mo;
/// PySCF 2.14 has no mp.MP4). Never import/invoke PySCF from this binary.
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

    const cpu::scf::MP4Result mp4 = rhf.mp4(frame);

    // Offline pins — see REFERENCE.md (never live-import PySCF here).
    constexpr double want_e_hf = -1.116759307396;
    constexpr double want_e2 = -0.013138073590;
    constexpr double want_e3 = -0.004836072637;
    constexpr double want_e4 = -0.001711078792;
    constexpr double want_e_corr = -0.019685225019; // product gate
    constexpr double want_e_tot = -1.136444532416;
    constexpr double tol = 1e-6; // Ha

    const double e_hf = rhf.energy();
    std::printf("RHF      E = %.12f Ha (pin %.12f, dE = %.3e)\n", e_hf,
                want_e_hf, e_hf - want_e_hf);
    std::printf("MP2 E(2)   = %.12f Ha (pin %.12f, dE = %.3e)\n",
                mp4.e_corr_mp2, want_e2, mp4.e_corr_mp2 - want_e2);
    std::printf("MP3 E(3)   = %.12f Ha (pin %.12f, dE = %.3e)\n",
                mp4.e_corr_mp3, want_e3, mp4.e_corr_mp3 - want_e3);
    std::printf("MP4 E(4)   = %.12f Ha (pin %.12f, dE = %.3e)\n",
                mp4.e_corr_mp4, want_e4, mp4.e_corr_mp4 - want_e4);
    std::printf("  E4(S)    = %.12f Ha\n", mp4.e_corr_s);
    std::printf("  E4(D)    = %.12f Ha\n", mp4.e_corr_d);
    std::printf("  E4(T)    = %.12f Ha\n", mp4.e_corr_t);
    std::printf("  E4(Q)    = %.12f Ha\n", mp4.e_corr_q);
    std::printf("MP4 e_corr = %.12f Ha (pin %.12f, dE = %.3e)\n", mp4.e_corr,
                want_e_corr, mp4.e_corr - want_e_corr);
    std::printf("MP4 e_tot  = %.12f Ha (pin %.12f, dE = %.3e)\n", mp4.e_tot,
                want_e_tot, mp4.e_tot - want_e_tot);

    bool ok = rhf.converged();
    ok = ok && std::fabs(e_hf - want_e_hf) < 1e-7;
    ok = ok && std::fabs(mp4.e_corr_mp2 - want_e2) < tol;
    ok = ok && std::fabs(mp4.e_corr_mp3 - want_e3) < tol;
    ok = ok && std::fabs(mp4.e_corr_mp4 - want_e4) < tol;
    ok = ok && std::fabs(mp4.e_corr - want_e_corr) < tol;
    ok = ok && std::fabs(mp4.e_tot - want_e_tot) < tol;
    ok = ok && std::fabs(mp4.e_tot - (e_hf + mp4.e_corr)) < 1e-12;
    ok = ok
         && std::fabs(mp4.e_corr
                      - (mp4.e_corr_mp2 + mp4.e_corr_mp3 + mp4.e_corr_mp4))
                < 1e-12;
    ok = ok
         && std::fabs(
                mp4.e_corr_mp4
                - (mp4.e_corr_s + mp4.e_corr_d + mp4.e_corr_t + mp4.e_corr_q))
                < 1e-12;
    ok = ok && mp4.frozen == 0;
    ok = ok && mp4.e_corr < 0.0;
    // MP4 more negative than MP3 on this system.
    ok = ok && mp4.e_corr < (mp4.e_corr_mp2 + mp4.e_corr_mp3);
    // Between MP3 (~ -0.0180) and FCI (~ -0.0205 Ha).
    ok = ok && mp4.e_corr > -0.021;
    ok = ok && mp4.e_corr < -0.018;

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

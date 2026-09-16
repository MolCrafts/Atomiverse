/// @file regressions/scf/scs-mp2-h2o-sto3g/reg_scs-mp2-h2o-sto3g.cpp
/// @brief H2O SCS-MP2/STO-3G — e_corr (and e_tot) vs offline PySCF pins.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Geometry and pins: REFERENCE.md (PySCF 2.14.0, Grimme c_OS=1.2, c_SS=1/3).
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
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    // Grimme SCS-MP2 (J. Chem. Phys. 118, 9095 (2003)): members, not a Config.
    rhf.mp2_os_scale = 1.2;
    rhf.mp2_ss_scale = 1.0 / 3.0;

    const cpu::scf::MP2Result mp2 = rhf.mp2(frame);

    // PySCF 2.14.0 pins — see REFERENCE.md (never live-import PySCF here).
    constexpr double want_e_hf = -74.963023138463;
    constexpr double want_e_os = -0.033515820085;
    constexpr double want_e_ss = -0.002029831562;
    constexpr double want_e_corr = -0.040895594623; // SCS product gate
    constexpr double want_e_tot = -75.003918733086;
    constexpr double tol = 1e-6; // Ha

    const double e_hf = rhf.energy();
    std::printf("RHF     E = %.12f Ha (pin %.12f, dE = %.3e)\n", e_hf,
                want_e_hf, e_hf - want_e_hf);
    std::printf("MP2 OS  = %.12f Ha (pin %.12f, dE = %.3e)\n", mp2.e_corr_os,
                want_e_os, mp2.e_corr_os - want_e_os);
    std::printf("MP2 SS  = %.12f Ha (pin %.12f, dE = %.3e)\n", mp2.e_corr_ss,
                want_e_ss, mp2.e_corr_ss - want_e_ss);
    std::printf("SCS-MP2 e_corr = %.12f Ha (pin %.12f, dE = %.3e)\n",
                mp2.e_corr, want_e_corr, mp2.e_corr - want_e_corr);
    std::printf("SCS-MP2 e_tot  = %.12f Ha (pin %.12f, dE = %.3e)\n", mp2.e_tot,
                want_e_tot, mp2.e_tot - want_e_tot);
    std::printf("scales: c_os=%.6f c_ss=%.6f\n", rhf.mp2_os_scale,
                rhf.mp2_ss_scale);

    bool ok = rhf.converged();
    ok = ok && std::fabs(e_hf - want_e_hf) < 1e-7;
    ok = ok && std::fabs(mp2.e_corr_os - want_e_os) < tol;
    ok = ok && std::fabs(mp2.e_corr_ss - want_e_ss) < tol;
    ok = ok && std::fabs(mp2.e_corr - want_e_corr) < tol;
    ok = ok && std::fabs(mp2.e_tot - want_e_tot) < tol;
    ok = ok && std::fabs(mp2.e_tot - (e_hf + mp2.e_corr)) < 1e-12;
    // Reconstruct SCS from unscaled channels.
    ok = ok
         && std::fabs(mp2.e_corr
                      - (1.2 * mp2.e_corr_os + (1.0 / 3.0) * mp2.e_corr_ss))
                < 1e-12;
    ok = ok && !mp2.ri;
    ok = ok && mp2.frozen == 0;
    ok = ok && mp2.e_corr < 0.0;
    // SCS should be more negative than plain RMP2 (~ -0.0355 Ha) on this system.
    ok = ok && mp2.e_corr < -0.038;

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/// @file regressions/scf/b2plyp-h2o/reg_b2plyp-h2o.cpp
/// @brief H₂O / STO-3G restricted B2PLYP double-hybrid total energy.
///
/// Out-of-library consumer: public `cpu/scf/methods.h` + `cpu/frame/frame.h`
/// only. Geometry, basis, and oracle energy are pinned in REFERENCE.md
/// (PySCF 2.14.0 offline; no live import in ctest).
///
/// B2PLYP (Grimme, J. Chem. Phys. 124, 034108 (2006)):
///   a_x = 0.53 HF exchange, (1−a_x) B88, (1−a_c) LYP with a_c = 0.27,
///   plus a_c · E_MP2 on the converged hybrid-KS orbitals.
/// Path: `RKS{XCType::B2PLYP}` + `mp2_os_scale = mp2_ss_scale = a_c` + `mp2`.

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

    // Functional metadata (no Config bag).
    constexpr auto xc = cpu::scf::XCType::B2PLYP;
    if (std::fabs(cpu::scf::hybrid_exchange_fraction(xc) - 0.53) > 1e-15) {
        std::fprintf(stderr, "FAIL: B2PLYP hybrid_exchange_fraction != 0.53\n");
        return 1;
    }
    if (std::fabs(cpu::scf::hybrid_pt2_fraction(xc) - 0.27) > 1e-15) {
        std::fprintf(stderr, "FAIL: B2PLYP hybrid_pt2_fraction != 0.27\n");
        return 1;
    }
    if (!cpu::scf::is_gga(xc)) {
        std::fprintf(stderr, "FAIL: B2PLYP is_gga == false\n");
        return 1;
    }

    auto frame = h2o_frame();

    cpu::scf::RKS rks(xc);
    rks.basis = cpu::scf::BasisSet::STO3G;
    rks.tol = 1e-10;
    rks.max_iter = 100;
    // Double-hybrid PT2 scale: E_tot = E_KS + a_c * E_MP2.
    const double a_c = cpu::scf::hybrid_pt2_fraction(xc);
    rks.mp2_os_scale = a_c;
    rks.mp2_ss_scale = a_c;

    const cpu::scf::MP2Result dh = rks.mp2(frame);

    // PySCF 2.14.0 offline pin — see REFERENCE.md (never live-import PySCF).
    constexpr double want_e_ks = -75.198832360592;  // Ha (hybrid KS only)
    constexpr double want_e_mp2 = -0.044726379786;  // Ha (unscaled full MP2)
    constexpr double want_e_corr = -0.012076122542; // Ha (a_c * E_MP2)
    constexpr double want_e_tot = -75.210908483134; // Ha (B2PLYP total)
    constexpr double tol = 1.0e-6;                  // Ha

    const double e_ks = rks.energy();
    // Reconstruct unscaled full MP2 from OS/SS channels.
    const double e_mp2_full = dh.e_corr_os + dh.e_corr_ss;

    std::printf("B2PLYP a_x=%.2f a_c=%.2f\n",
                cpu::scf::hybrid_exchange_fraction(xc), a_c);
    std::printf("KS     E = %.12f Ha (pin %.12f, dE = %.3e)\n", e_ks, want_e_ks,
                e_ks - want_e_ks);
    std::printf("MP2 OS = %.12f Ha\n", dh.e_corr_os);
    std::printf("MP2 SS = %.12f Ha\n", dh.e_corr_ss);
    std::printf("MP2 full (unscaled) = %.12f Ha (pin %.12f, dE = %.3e)\n",
                e_mp2_full, want_e_mp2, e_mp2_full - want_e_mp2);
    std::printf("PT2 a_c*E_MP2 = %.12f Ha (pin %.12f, dE = %.3e)\n", dh.e_corr,
                want_e_corr, dh.e_corr - want_e_corr);
    std::printf("B2PLYP e_tot  = %.12f Ha (pin %.12f, dE = %.3e)\n", dh.e_tot,
                want_e_tot, dh.e_tot - want_e_tot);

    bool ok = rks.converged();
    ok = ok && std::isfinite(e_ks) && std::isfinite(dh.e_tot);
    ok = ok && std::fabs(e_ks - want_e_ks) < tol;
    ok = ok && std::fabs(e_mp2_full - want_e_mp2) < tol;
    ok = ok && std::fabs(dh.e_corr - want_e_corr) < tol;
    ok = ok && std::fabs(dh.e_tot - want_e_tot) < tol;
    ok = ok && std::fabs(dh.e_tot - (e_ks + dh.e_corr)) < 1e-12;
    ok = ok && std::fabs(dh.e_corr - a_c * e_mp2_full) < 1e-12;
    ok = ok && !dh.ri;
    ok = ok && dh.frozen == 0;
    ok = ok && dh.e_corr < 0.0;

    if (!ok) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS: E_B2PLYP/STO-3G(H2O) = %.12f Ha "
                "(want %.12f +/- %.1e) [PySCF 2.14.0 offline pin]\n",
                dh.e_tot, want_e_tot, tol);
    return 0;
}

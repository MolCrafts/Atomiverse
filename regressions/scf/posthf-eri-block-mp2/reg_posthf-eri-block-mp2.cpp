/// @file regressions/scf/posthf-eri-block-mp2/reg_posthf-eri-block-mp2.cpp
/// @brief H2O RMP2/STO-3G through the block-assembled dense ERI tensor —
///        e_corr (and e_hf, e_tot) vs offline PySCF pins.
///
/// Out-of-library consumer: public headers + link atomiverse only. This case
/// specifically gates the block-assembled dense-ERI path end-to-end
/// (spec scf-review-07-posthf-perf, item 17): plain canonical RMP2 consumes the
/// full N⁴ AO ERI tensor materialized by IntegralWorkspace::eri_dense — after the
/// scalar→native::eri_block rewrite, this witness confirms the assembled tensor
/// still reproduces the PySCF correlation energy end-to-end.
///
/// Distinct from regressions/scf/h2o_mp2_sto3g (which only checks e_corr < -1e-4
/// on a different geometry): here we pin the exact PySCF RHF total energy AND
/// the plain-RMP2 correlation energy. Geometry and pins: REFERENCE.md
/// (PySCF 2.14.0, same water preset as scs-mp2-h2o-sto3g). Never import/invoke
/// PySCF from this binary.
///
/// Exit non-zero on failure. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

// PySCF `water` preset (Å): O(0,0,0.1173); H(0,±0.7572,−0.4692).
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

    // Plain canonical RMP2 (default scales, no RI) → consumes the dense ERI
    // tensor assembled by IntegralWorkspace::eri_dense (the block path under test).
    const cpu::scf::MP2Result mp2 = rhf.mp2(frame);

    // PySCF 2.14.0 pins — see REFERENCE.md (never live-import PySCF here).
    constexpr double want_e_hf = -74.963023138463;  // RHF total [Ha]
    constexpr double want_e_corr = -0.035545651647; // plain RMP2 corr [Ha]
    constexpr double want_e_tot = -74.998568790110; // = e_hf + e_corr [Ha]
    constexpr double tol_hf = 1e-7;                 // Ha (reference gate)
    constexpr double tol_corr = 1e-7;               // Ha (tighter product gate)
    constexpr double tol_tot = 1e-6;                // Ha

    const double e_hf = rhf.energy();
    std::printf("RHF      E = %.12f Ha (pin %.12f, dE = %.3e)\n", e_hf,
                want_e_hf, e_hf - want_e_hf);
    std::printf("RMP2 e_corr = %.12f Ha (pin %.12f, dE = %.3e)\n", mp2.e_corr,
                want_e_corr, mp2.e_corr - want_e_corr);
    std::printf("RMP2 e_tot  = %.12f Ha (pin %.12f, dE = %.3e)\n", mp2.e_tot,
                want_e_tot, mp2.e_tot - want_e_tot);

    bool ok = rhf.converged();
    ok = ok && std::isfinite(e_hf) && std::isfinite(mp2.e_tot);
    ok = ok && std::fabs(e_hf - want_e_hf) < tol_hf;
    ok = ok && std::fabs(mp2.e_corr - want_e_corr) < tol_corr;
    ok = ok && std::fabs(mp2.e_tot - want_e_tot) < tol_tot;
    // Internal consistency: e_tot = e_hf + e_corr to machine precision.
    ok = ok && std::fabs(mp2.e_tot - (e_hf + mp2.e_corr)) < 1e-12;
    // Plain RMP2 through the canonical (non-RI) dense-ERI path.
    ok = ok && !mp2.ri;
    ok = ok && mp2.frozen == 0;
    ok = ok && mp2.e_corr < 0.0;

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

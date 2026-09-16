/// @file regressions/scf/ccsd-spatial-so-parity/reg_ccsd-spatial-so-parity.cpp
/// @brief H2O / STO-3G closed-shell CCSD: the spatial-orbital energy path (with
///        amplitude DIIS) agrees with the spin-orbital oracle AND with a plain
///        Jacobi solve to ≤ 1e-9 Ha, and matches the offline PySCF cc.CCSD pin
///        to ≤ 1e-6 Ha.
///
/// scf-review-08-ccsd-spatial-diis — the permanent oracle cross-check that
/// catches the classic spin-summation coefficient error (2J−K weights, τ vs t2
/// in exchange/ring terms) when closed-shell CCSD is rewritten in the spatial
/// basis. See REFERENCE.md for the PySCF 2.14.0 pin (tool/version/command/date).
///
/// Out-of-library consumer: PUBLIC headers only
/// (`cpu/scf/methods.h`, `cpu/frame/frame.h`); links only `atomiverse`.
///
/// PUBLIC-API NOTE (deviation flag). regressions/ are public-API-only
/// consumers (README rule 1: "No internal/ headers … If a case cannot be
/// written against the public API, that is the defect the case just found").
/// The dual-path witness precedent (regressions/scf/ci-engine-equiv-h2-sto3g)
/// uses two PUBLIC entries that each route a different engine; this case
/// mirrors that:
///   * spatial + DIIS  → `RHF::ccsd(frame)`      (closed-shell default post-impl)
///   * spin-orbital     → `RHF::ccsd_t(frame)`   (stays spin-orbital per spec
///                        out-of-scope; `.e_corr_ccsd` is the SO CCSD-only piece)
/// The DIIS-vs-Jacobi leg (ac-003) has NO existing public toggle. It is pinned
/// against a PROPOSED minimal public member `RHF::cc_diis` (bool, default true;
/// false → plain Jacobi) — a plain public member in the house style
/// (parameters are members, no *Config bag, no factory), the single invented
/// symbol. Until the impl adds it this file is RED (compile-fail on `cc_diis`).

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

bool check(const char* what, double got, double want, double tol) {
    if (!std::isfinite(got) || std::abs(got - want) > tol) {
        std::fprintf(stderr, "FAIL: %s got=%.15f want=%.15f tol=%.1e\n", what,
                     got, want, tol);
        return false;
    }
    std::printf("  %-32s %18.12f (want %18.12f +/- %.1e)\n", what, got, want,
                tol);
    return true;
}

// H2O / STO-3G — `pin_scf` `water` preset (Å). Shared by every RHF below.
ATV::cpu::Frame make_water() {
    ATV::cpu::Frame frame(3);
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.1173;
    frame.span(ATV::kAtom_x)[1] = 0.0;
    frame.span(ATV::kAtom_y)[1] = 0.7572;
    frame.span(ATV::kAtom_z)[1] = -0.4692;
    frame.span(ATV::kAtom_x)[2] = 0.0;
    frame.span(ATV::kAtom_y)[2] = -0.7572;
    frame.span(ATV::kAtom_z)[2] = -0.4692;
    frame.span(ATV::kAtom_type)[0] = 8; // O
    frame.span(ATV::kAtom_type)[1] = 1; // H
    frame.span(ATV::kAtom_type)[2] = 1; // H
    frame.box = ATV::cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;

    // Offline PySCF 2.14.0 cc.CCSD H2O/STO-3G pin (conv_tol 1e-12) — REFERENCE.md
    // (same literal pinned by regressions/scf/ccsd-t-h2o-sto3g).
    constexpr double kECorrGolden = -0.049438563031963; // Ha
    constexpr double kGoldenTol = 1e-6;                 // Ha vs PySCF
    constexpr double kParityTol = 1e-9;                 // Ha, same fixed point

    // ── 1. spatial-orbital path (with amplitude DIIS) — closed-shell default.
    cpu::scf::RHF rhf_spatial;
    rhf_spatial.basis = cpu::scf::BasisSet::STO3G;
    rhf_spatial.tol = 1e-12;
    rhf_spatial.max_iter = 100;
    auto frame_spatial = make_water();
    const cpu::scf::CCSDResult ccsd_spatial = rhf_spatial.ccsd(frame_spatial);
    if (!rhf_spatial.converged() || !ccsd_spatial.converged) {
        std::fprintf(stderr,
                     "FAIL: spatial RHF/CCSD (H2O/STO-3G) did not converge\n");
        return 1;
    }
    const double e_corr_spatial = ccsd_spatial.e_corr;

    // ── 2. spin-orbital oracle — RHF::ccsd_t keeps the spin-orbital Gauss–Stanton
    //       kernel (spec out-of-scope); its CCSD-only correlation is the oracle.
    cpu::scf::RHF rhf_so;
    rhf_so.basis = cpu::scf::BasisSet::STO3G;
    rhf_so.tol = 1e-12;
    rhf_so.max_iter = 100;
    auto frame_so = make_water();
    const cpu::scf::CCSDTResult ccsd_so = rhf_so.ccsd_t(frame_so);
    if (!rhf_so.converged() || !ccsd_so.converged) {
        std::fprintf(
            stderr,
            "FAIL: spin-orbital RHF/CCSD (H2O/STO-3G) did not converge\n");
        return 1;
    }
    const double e_corr_so = ccsd_so.e_corr_ccsd;

    // ── 3. plain Jacobi (DIIS off) — same residual tolerance, same fixed point.
    cpu::scf::RHF rhf_jacobi;
    rhf_jacobi.basis = cpu::scf::BasisSet::STO3G;
    rhf_jacobi.tol = 1e-12;
    rhf_jacobi.max_iter = 100;
    rhf_jacobi.cc_diis = false; // PROPOSED public member (see PUBLIC-API NOTE)
    auto frame_jacobi = make_water();
    const cpu::scf::CCSDResult ccsd_jacobi = rhf_jacobi.ccsd(frame_jacobi);
    if (!rhf_jacobi.converged() || !ccsd_jacobi.converged) {
        std::fprintf(stderr,
                     "FAIL: Jacobi RHF/CCSD (H2O/STO-3G) did not converge\n");
        return 1;
    }
    const double e_corr_jacobi = ccsd_jacobi.e_corr;

    std::printf(
        "H2O / STO-3G closed-shell CCSD spatial-vs-spin-orbital parity:\n");
    // ac-006: spatial vs spin-orbital oracle (catches spin-summation errors).
    if (!check("E_corr spatial [Ha]", e_corr_spatial, kECorrGolden, kGoldenTol))
        return 1;
    if (!check("E_corr spin-orbital [Ha]", e_corr_so, kECorrGolden, kGoldenTol))
        return 1;
    if (!check("|E_corr spatial - SO| [Ha]", e_corr_spatial, e_corr_so,
               kParityTol))
        return 1;
    // ac-003: DIIS and plain Jacobi reach the same fixed point.
    if (!check("|E_corr DIIS - Jacobi| [Ha]", e_corr_spatial, e_corr_jacobi,
               kParityTol))
        return 1;
    // ac-006: spatial energy matches the offline PySCF cc.CCSD golden.
    if (!check("E_corr vs PySCF golden [Ha]", e_corr_spatial, kECorrGolden,
               kGoldenTol))
        return 1;

    std::printf("\nPASS: ccsd-spatial-so-parity\n");
    return 0;
}

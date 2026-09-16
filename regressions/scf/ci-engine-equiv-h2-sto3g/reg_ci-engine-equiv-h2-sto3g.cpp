/// @file regressions/scf/ci-engine-equiv-h2-sto3g/reg_ci-engine-equiv-h2-sto3g.cpp
/// @brief H2 / STO-3G: RHF::fci and RHF::casci(2,2) both reproduce the offline
///        PySCF full-CI pin, AND agree with each other to machine precision.
///
/// scf-review-03-posthf-dedup unifies the determinant-CI machinery (previously
/// forked between fci.cpp and casscf.cpp) into one engine. On H2/STO-3G the
/// full CI space and the CAS(2e,2o) space are identical, so FCI and CASCI must
/// give the same total energy. This case is the numeric equivalence gate: it
/// PASSES before the hoist (both methods already exist) and must keep passing
/// after, proving the merge introduced no numeric change and both entry points
/// agree. See REFERENCE.md for the PySCF 2.14.0 pin (tool/version/command/date).
///
/// Out-of-library consumer: public headers only
/// (`cpu/scf/methods.h`, `cpu/frame/frame.h`).

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
    std::printf("  %-28s %18.12f (want %18.12f +/- %.1e)\n", what, got, want,
                tol);
    return true;
}

} // namespace

int main() {
    using namespace ATV;

    // Geometry (Å) — pin_scf `h2` preset (H–H = 0.74 Å along z).
    cpu::Frame frame(2);
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.74;
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_type)[1] = 1;
    frame.box = cpu::Box::cubic(20.0);

    // Pin — REFERENCE.md (PySCF 2.14.0 full CI / CASCI(2,2), RHF STO-3G).
    constexpr double kEFullCI = -1.137283834488502; // Ha
    constexpr double kTol = 1e-6;                   // Ha vs literature
    constexpr double kEquivTol = 1e-12;             // FCI vs CASCI(full space)

    // FCI via the public entry (RHF::fci drives its own SCF).
    cpu::scf::RHF rhf_fci;
    rhf_fci.basis = cpu::scf::BasisSet::STO3G;
    rhf_fci.tol = 1e-12;
    rhf_fci.max_iter = 100;
    const cpu::scf::FCIResult fci = rhf_fci.fci(frame);
    if (!rhf_fci.converged()) {
        std::fprintf(stderr,
                     "FAIL: RHF/STO-3G (H2) FCI reference did not converge\n");
        return 1;
    }

    // CASCI(2,2) via the public entry (RHF::casci drives its own SCF).
    cpu::scf::RHF rhf_cas;
    rhf_cas.basis = cpu::scf::BasisSet::STO3G;
    rhf_cas.tol = 1e-12;
    rhf_cas.max_iter = 100;
    const cpu::scf::CASCIResult cas =
        rhf_cas.casci(frame, /*ncas=*/2, /*nelecas=*/2);
    if (!rhf_cas.converged()) {
        std::fprintf(
            stderr, "FAIL: RHF/STO-3G (H2) CASCI reference did not converge\n");
        return 1;
    }
    if (cas.ncas != 2 || cas.nelecas != 2 || cas.ncore != 0) {
        std::fprintf(
            stderr,
            "FAIL: active-space bookkeeping ncas=%d nelecas=%d ncore=%d\n",
            cas.ncas, cas.nelecas, cas.ncore);
        return 1;
    }

    std::printf("H2 / STO-3G unified CI engine equivalence:\n");
    if (!check("E_FCI [Ha]", fci.e_fci, kEFullCI, kTol))
        return 1;
    if (!check("E_CASCI(2,2) [Ha]", cas.e_casci, kEFullCI, kTol))
        return 1;
    // The unified engine drives BOTH entries: full-space CASCI == FCI exactly.
    if (!check("|E_FCI - E_CASCI| [Ha]", fci.e_fci, cas.e_casci, kEquivTol))
        return 1;

    std::printf("\nPASS: ci-engine-equiv-h2-sto3g\n");
    return 0;
}

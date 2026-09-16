/// @file regressions/scf/nevpt2-h2-sto3g/reg_nevpt2-h2-sto3g.cpp
/// @brief H₂ SC-NEVPT2/CASSCF(2,2)/STO-3G vs offline PySCF pin (REFERENCE.md).
///
/// Out-of-library consumer: public headers only
/// (`cpu/scf/methods.h`, `cpu/frame/frame.h`). Product gate: e_corr (PT2) and
/// e_tot to 1e-6 Ha.

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

    // Pins — REFERENCE.md (PySCF 2.14.0 mcscf.CASSCF + mrpt.NEVPT / STO-3G).
    constexpr double kEHf = -1.116759307396426;  // Ha
    constexpr double kECas = -1.137283834488502; // Ha
    constexpr double kECorrPt2 = 0.0;            // Ha (ncore=nvirt=0)
    constexpr double kETot = -1.137283834488502; // Ha
    constexpr double kTol = 1e-6;                // Ha

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 100;

    const cpu::scf::NEVPT2Result nev =
        rhf.nevpt2(frame, /*ncas=*/2, /*nelecas=*/2);
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF/STO-3G (H2) did not converge\n");
        return 1;
    }
    if (!nev.converged) {
        std::fprintf(stderr, "FAIL: CASSCF under NEVPT2 did not converge\n");
        return 1;
    }
    if (nev.ncas != 2 || nev.nelecas != 2 || nev.ncore != 0 || nev.nvirt != 0) {
        std::fprintf(stderr,
                     "FAIL: bookkeeping ncas=%d nelecas=%d ncore=%d nvirt=%d\n",
                     nev.ncas, nev.nelecas, nev.ncore, nev.nvirt);
        return 1;
    }

    std::printf("H2 SC-NEVPT2 / CASSCF(2,2)/STO-3G (RHF reference):\n");
    if (!check("E_RHF [Ha]", rhf.energy(), kEHf, kTol))
        return 1;
    if (!check("E_CASSCF [Ha]", nev.e_casscf, kECas, kTol))
        return 1;
    if (!check("E_corr NEVPT2 [Ha]", nev.e_corr, kECorrPt2, kTol))
        return 1;
    if (!check("E_tot NEVPT2 [Ha]", nev.e_nevpt2, kETot, kTol))
        return 1;
    if (!check("E_tot = E_CAS + E_corr", nev.e_nevpt2,
               nev.e_casscf + nev.e_corr, 1e-12))
        return 1;

    // Full-valence CAS(2,2): NEVPT2 ≡ CASSCF ≡ CASCI ≡ FCI.
    const cpu::scf::CASSCFResult cas = rhf.casscf(frame, 2, 2);
    if (!check("NEVPT2 == CASSCF", nev.e_nevpt2, cas.e_casscf, 1e-10))
        return 1;

    std::printf("  ncas=%d  nelecas=%d  ncore=%d  nvirt=%d\n", nev.ncas,
                nev.nelecas, nev.ncore, nev.nvirt);
    std::printf("\nPASS: nevpt2-h2-sto3g\n");
    return 0;
}

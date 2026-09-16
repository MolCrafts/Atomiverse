/// @file regressions/scf/direct-rpa-h2-sto3g/reg_direct-rpa-h2-sto3g.cpp
/// @brief H₂ RHF/STO-3G direct RPA correlation energy vs offline PySCF pin.
///
/// Public-API consumer only. Geometry, basis, and oracle numbers are pinned
/// in REFERENCE.md (PySCF 2.14.0 offline; no live import in ctest).
///
/// Observable: \(E_{\mathrm{c}}^{\mathrm{dRPA}}\) [Ha] from `RHF::direct_rpa()`
/// — Furche plasmon formula on the Coulomb-only Casida spectrum (exact MO
/// ERIs; non-DF twin of PySCF `gw.rpa.RPA` / `DirectRPA`).
#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

bool check(const char* what, double got, double want, double tol) {
    if (!std::isfinite(got) || std::abs(got - want) > tol) {
        std::fprintf(stderr, "FAIL: %s got=%.12f want=%.12f tol=%.1e\n", what,
                     got, want, tol);
        return false;
    }
    std::printf("  %-36s %14.8f (want %14.8f +/- %.1e)\n", what, got, want,
                tol);
    return true;
}

} // namespace

int main() {
    using namespace ATV;

    // Geometry [Å] — REFERENCE.md (bond on +z).
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.74;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.box = cpu::Box::cubic(20.0);

    // PySCF 2.14.0 exact dRPA (plasmon) pins — REFERENCE.md.
    constexpr double kERhfHa = -1.116759307396;
    constexpr double kECorrHa = -0.020633073749; // primary gate
    constexpr double kETotHa = -1.137392381145;
    constexpr double kOmegaHa = 1.570852128278;
    constexpr double kTrAHa = 1.612118275776;
    constexpr double kTolEHa = 1e-6; // job gate on e_corr

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 100;
    rhf.run(f);
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF did not converge\n");
        return 1;
    }

    std::printf("H2 RHF/STO-3G direct RPA (plasmon / non-DF):\n");
    if (!check("E_RHF [Ha]", rhf.energy(), kERhfHa, kTolEHa))
        return 1;

    const cpu::scf::DirectRPAResult rpa = rhf.direct_rpa();
    if (rpa.omega_ha.size() != 1) {
        std::fprintf(stderr, "FAIL: expected 1 dRPA pole, got %zu\n",
                     rpa.omega_ha.size());
        return 1;
    }

    std::printf("  Omega_1 = %.10f Ha\n", rpa.omega_ha[0]);
    std::printf("  Tr(A)   = %.10f Ha\n", rpa.tr_A);
    std::printf("  e_corr  = %.12f Ha\n", rpa.e_corr);
    std::printf("  e_tot   = %.12f Ha\n", rpa.e_tot);

    if (!check("e_corr [Ha]", rpa.e_corr, kECorrHa, kTolEHa))
        return 1;
    if (!check("e_tot [Ha]", rpa.e_tot, kETotHa, kTolEHa))
        return 1;
    if (!check("Omega_1 [Ha]", rpa.omega_ha[0], kOmegaHa, kTolEHa))
        return 1;
    if (!check("Tr(A) [Ha]", rpa.tr_A, kTrAHa, kTolEHa))
        return 1;

    // Plasmon identity: e_corr == 1/2 (sum Omega - Tr A).
    const double recon = 0.5 * (rpa.omega_ha[0] - rpa.tr_A);
    if (!check("plasmon recon [Ha]", recon, rpa.e_corr, 1e-12))
        return 1;

    // e_tot consistency.
    if (!check("e_tot from pieces [Ha]", rhf.energy() + rpa.e_corr, rpa.e_tot,
               1e-12))
        return 1;

    // Physical sanity: dRPA recovers more correlation than MP2 on this system
    // (ring diagrams beyond second order).
    if (!(rpa.e_corr < -0.013)) {
        std::fprintf(stderr,
                     "FAIL: dRPA e_corr %.6f not deeper than MP2 ~ -0.013\n",
                     rpa.e_corr);
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}

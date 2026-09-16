/// @file regressions/scf/scf-review-01-linalg/reg_scf-review-01-linalg.cpp
/// @brief Permanent gate for spec scf-review-01-linalg (hoist shared dense
///        linear algebra: sym_power, eig_nonsymmetric, solve_pcg, dRPA solve).
///
/// Behaviour-preserving refactor: the shared-linalg move must not change any
/// user-visible number. This out-of-library consumer (public headers only)
/// exercises the two physical paths that touch the hoisted primitives on
/// H₂/STO-3G and H₂O/STO-3G and pins them to the cited offline oracle values:
///
///   (1) direct-RPA correlation energy via `RHF::direct_rpa()`
///       — (A−B)^{1/2}(A+B)(A−B)^{1/2} Casida reduction (`sym_power` +
///       `OrbitalHessian::direct_rpa`). Same literature value as the sibling
///       case regressions/scf/direct-rpa-h2-sto3g. Gate E_c ≤ 1e-12 Ha (the
///       spec's honest bitwise-subset gate; ATV reproduces the PySCF pin to
///       all 12 printed decimals).
///   (2) CPHF static isotropic polarizability via `RHF::polarizability()`
///       — solve_pcg-backed linear response. Same literature value as the
///       sibling case regressions/scf/static-polariz-h2o. Gate α ≤ 1e-6 a.u.
///
/// All goldens are hard-coded literals with citation + date below; no live
/// third-party oracle at ctest time. ≤ 60 s, single CPU, non-zero exit on
/// mismatch.

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
    std::printf("  %-32s %16.10f (want %16.10f +/- %.1e)\n", what, got, want,
                tol);
    return true;
}

// ── Part 1: H₂/STO-3G direct-RPA correlation energy ────────────────────────
//
// Golden values copied verbatim from regressions/scf/direct-rpa-h2-sto3g
// (REFERENCE.md, "Pinned output 2026-07-23"). Oracle: PySCF 2.14.0, exact
// MO-ERI Furche plasmon formula (non-DF twin of gw.rpa.RPA), offline.
//   Furche, J. Chem. Phys. 114, 5982 (2001), DOI 10.1063/1.1353585.
//   Ren et al., New J. Phys. 14, 053020 (2012).
//   Zhu & Chan, J. Chem. Theory Comput. 17, 727 (2021).
bool run_drpa() {
    using namespace ATV;

    // Geometry [Å] — H₂ bond on +z, r(H–H) = 0.74 Å (direct-rpa-h2-sto3g).
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.74;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.box = cpu::Box::cubic(20.0);

    // PySCF 2.14.0 exact dRPA (plasmon) pins — direct-rpa-h2-sto3g/REFERENCE.md.
    constexpr double kERhfHa = -1.116759307396;
    constexpr double kECorrHa = -0.020633073749; // primary gate (spec: ≤1e-12)
    constexpr double kETotHa = -1.137392381145;
    constexpr double kOmegaHa = 1.570852128278;
    constexpr double kTrAHa = 1.612118275776;
    constexpr double kTolEc = 1e-12; // spec scf-review-01-linalg honest gate
    constexpr double kTolAux = 1e-6; // engine-vs-oracle aux quantities

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 100;
    rhf.run(f);
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF did not converge\n");
        return false;
    }

    std::printf("[1] H2 RHF/STO-3G direct RPA (sym_power + OrbitalHessian):\n");
    if (!check("E_RHF [Ha]", rhf.energy(), kERhfHa, kTolAux))
        return false;

    const cpu::scf::DirectRPAResult rpa = rhf.direct_rpa();
    if (rpa.omega_ha.size() != 1) {
        std::fprintf(stderr, "FAIL: expected 1 dRPA pole, got %zu\n",
                     rpa.omega_ha.size());
        return false;
    }

    // Primary spec gate: correlation energy to 1e-12 Ha.
    if (!check("E_c dRPA [Ha]", rpa.e_corr, kECorrHa, kTolEc))
        return false;
    if (!check("Omega_1 [Ha]", rpa.omega_ha[0], kOmegaHa, kTolAux))
        return false;
    if (!check("Tr(A) [Ha]", rpa.tr_A, kTrAHa, kTolAux))
        return false;
    if (!check("E_tot dRPA [Ha]", rpa.e_tot, kETotHa, kTolAux))
        return false;

    // Furche plasmon identity: E_c == 1/2 (sum Omega - Tr A) — internal
    // invariant, exact to numerical precision regardless of the oracle.
    const double recon = 0.5 * (rpa.omega_ha[0] - rpa.tr_A);
    if (!check("plasmon recon [Ha]", recon, rpa.e_corr, 1e-12))
        return false;
    if (!check("E_tot from pieces [Ha]", rhf.energy() + rpa.e_corr, rpa.e_tot,
               1e-12))
        return false;

    // Physical sanity: dRPA (ring diagrams to all orders) recovers more
    // correlation than MP2 (~ -0.013 Ha) on this system.
    if (!(rpa.e_corr < -0.013)) {
        std::fprintf(stderr,
                     "FAIL: dRPA e_corr %.6f not deeper than MP2 ~ -0.013\n",
                     rpa.e_corr);
        return false;
    }
    return true;
}

// ── Part 2: H₂O/STO-3G CPHF static isotropic polarizability ────────────────
//
// Golden values copied verbatim from regressions/scf/static-polariz-h2o
// (REFERENCE.md). Oracle: PySCF 2.14.0 analytic CPHF on PySCF MOs/ERIs,
// cross-checked by finite-field dμ/dF (offline). Length-gauge dipole
// response; Helgaker, Jørgensen, Olsen, "Molecular Electronic-Structure
// Theory", response chapter.
bool run_polarizability() {
    using namespace ATV;

    // Geometry [Å] — C₂v water, molecular plane yz (static-polariz-h2o).
    cpu::Frame frame(3);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_x)[2] = 0.0;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.box = cpu::Box::cubic(20.0);

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.run(frame);
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF (water) did not converge\n");
        return false;
    }

    const auto alpha = rhf.polarizability(); // 3×3 [a.u.]

    // PySCF 2.14.0 analytic CPHF diagonal α_ii [a.u.] — static-polariz-h2o.
    constexpr double kAxx = 0.04190863;
    constexpr double kAyy = 5.20973255;
    constexpr double kAzz = 2.12715820;
    // Isotropic α = Tr(α)/3 [a.u.] (spec gates the isotropic value).
    constexpr double kAisoAu = (kAxx + kAyy + kAzz) / 3.0; // 2.459599793...
    constexpr double kTolAlpha = 1e-6; // spec scf-review-01-linalg gate

    std::printf("[2] H2O RHF/STO-3G CPHF polarizability (solve_pcg):\n");
    const double want[3] = {kAxx, kAyy, kAzz};
    const char* lbl[3] = {"alpha_xx [a.u.]", "alpha_yy [a.u.]",
                          "alpha_zz [a.u.]"};
    for (int a = 0; a < 3; ++a) {
        if (!check(lbl[a], alpha(a, a), want[a], kTolAlpha))
            return false;
    }
    // Off-diagonal must vanish for C₂v water in this frame.
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < a; ++b) {
            const double off = std::fabs(alpha(a, b));
            if (off > 1e-4) {
                std::fprintf(stderr, "FAIL: alpha[%d%d]=%.3e not ~0 (C2v)\n", a,
                             b, off);
                return false;
            }
        }
    }
    const double iso = (alpha(0, 0) + alpha(1, 1) + alpha(2, 2)) / 3.0;
    if (!check("alpha_iso [a.u.]", iso, kAisoAu, kTolAlpha))
        return false;
    return true;
}

} // namespace

int main() {
    if (!run_drpa())
        return 1;
    if (!run_polarizability())
        return 1;
    std::printf("PASS\n");
    return 0;
}

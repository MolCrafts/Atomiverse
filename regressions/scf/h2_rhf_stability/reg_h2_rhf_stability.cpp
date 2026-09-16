/// @file regressions/scf/h2_rhf_stability/reg_h2_rhf_stability.cpp
/// @brief Stretched-H2 Seeger-Pople stability + broken-symmetry UHF vs the
///        PySCF-pinned oracle (Seeger & Pople 1977, DOI 10.1063/1.434318;
///        Szabo & Ostlund §3.8.7 for the UHF dissociation behavior).
///
/// Out-of-library consumer: public headers only (`cpu/scf/methods.h`,
/// `cpu/frame/frame.h`, `core/units.h`), links `atomiverse`.
///
/// H2/STO-3G (Cartesian, engine convention), RHF conv 1e-12:
///  1. R = 1.5 Å: RHF.run() then stability() — the Seeger-Pople trap:
///     internal (singlet) lowest eigenvalue ≥ 0 while the external
///     (triplet) lowest eigenvalue is NEGATIVE and matches the PySCF pin
///     −0.2092471629 Ha to 1e-6 Ha (see REFERENCE.md).
///  2. UHF seeded from StabilityResult::rotated_C (per-spin density guess
///     through the public SCFInitialGuess bridge) converges BELOW RHF:
///     E_UHF = −0.9577067934 Ha vs E_RHF = −0.9108735546 Ha
///     (ΔE = −0.0468 Ha ≫ the 1e-4 Ha gate) — the Szabo-Ostlund
///     dissociation picture.
///  3. R = 0.735 Å: both channels stable (internal +1.1351770407,
///     external +0.4114522416 Ha, PySCF pins).
///
/// Exit non-zero on mismatch, printing got / want / tol. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

ATV::cpu::Frame make_h2(double r_angstrom) {
    using namespace ATV;
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = r_angstrom;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.box = cpu::Box::cubic(20.0);
    return f;
}

bool check(const char* what, double got, double want, double tol) {
    if (!std::isfinite(got) || std::abs(got - want) > tol) {
        std::fprintf(stderr, "FAIL: %s got=%.10f want=%.10f tol=%.1e\n", what,
                     got, want, tol);
        return false;
    }
    std::printf("  %-38s %.10f (want %.10f +/- %.1e)\n", what, got, want, tol);
    return true;
}

} // namespace

int main() {
    using namespace ATV;
    using cpu::scf::SlaterDeterminant;

    // ── PySCF 2.14.0 pins (REFERENCE.md; cart=True, conv_tol=1e-12) ──────
    constexpr double kERhf15 = -0.9108735546;     // Ha
    constexpr double kInternal15 = 0.7088965814;  // Ha
    constexpr double kExternal15 = -0.2092471629; // Ha
    constexpr double kEUhf15 = -0.9577067934;     // Ha (broken symmetry)
    constexpr double kInternalEq = 1.1351770407;  // Ha, R = 0.735 Å
    constexpr double kExternalEq = 0.4114522416;  // Ha, R = 0.735 Å
    constexpr double kTolEig = 1e-6;              // Ha
    constexpr double kTolE = 1e-6;                // Ha

    // ── 1. Stretched H2: the Seeger-Pople trap ────────────────────────────
    auto frame = make_h2(1.5);
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.run(frame);
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF/STO-3G(H2, R=1.5) did not converge\n");
        return 1;
    }
    const double e_rhf = rhf.energy();
    std::printf("H2/STO-3G R=1.5 A:\n");
    if (!check("E_RHF [Ha]", e_rhf, kERhf15, kTolE))
        return 1;

    const cpu::scf::StabilityResult st = rhf.stability();
    if (!check("internal lowest eig [Ha]", st.internal_eigenvalue, kInternal15,
               kTolEig))
        return 1;
    if (!check("external lowest eig [Ha]", st.external_eigenvalue, kExternal15,
               kTolEig))
        return 1;
    if (!st.internal_stable || st.external_stable
        || st.external_eigenvalue >= 0.0) {
        std::fprintf(stderr,
                     "FAIL: Seeger-Pople trap not reproduced "
                     "(internal_stable=%d external_stable=%d)\n",
                     st.internal_stable, st.external_stable);
        return 1;
    }

    // ── 2. Broken-symmetry UHF seeded from rotated_C ──────────────────────
    // Bridge: rotated_C -> per-spin alpha density guess (P = C_occ C_occ^T),
    // beta keeps the unrotated RHF density — the asymmetric start that lets
    // UHF fall into the spin-broken minimum (scf_guess.h mechanism).
    const auto& det = rhf.determinant();
    const int n = static_cast<int>(st.rotated_C.shape()[0]);
    cpu::scf::SCFInitialGuess guess;
    guess.P_alpha = xt::zeros<double>({n, n});
    for (int mu = 0; mu < n; ++mu)
        for (int nu = 0; nu < n; ++nu)
            guess.P_alpha(mu, nu) = st.rotated_C(mu, 0) * st.rotated_C(nu, 0);
    guess.P_beta = det.density(SlaterDeterminant::Spin::beta);
    guess.has_alpha = true;
    guess.has_beta = true;

    cpu::scf::UHF uhf{1, 1};
    uhf.basis = cpu::scf::BasisSet::STO3G;
    uhf.tol = 1e-12;
    auto frame_u = make_h2(1.5);
    uhf.run(frame_u, &guess);
    if (!uhf.converged()) {
        std::fprintf(stderr, "FAIL: UHF(broken-symmetry seed) did not "
                             "converge\n");
        return 1;
    }
    const double e_uhf = uhf.energy();
    if (!check("E_UHF(broken) [Ha]", e_uhf, kEUhf15, kTolE))
        return 1;
    if (!(e_uhf < e_rhf - 1e-4)) {
        std::fprintf(stderr,
                     "FAIL: E_UHF=%.10f not below E_RHF=%.10f by 1e-4 Ha\n",
                     e_uhf, e_rhf);
        return 1;
    }
    std::printf("  E_UHF - E_RHF = %.10f Ha (< -1e-4: Szabo-Ostlund "
                "dissociation)\n",
                e_uhf - e_rhf);

    // ── 3. Equilibrium H2: both channels stable ───────────────────────────
    auto frame_eq = make_h2(0.735);
    cpu::scf::RHF rhf_eq;
    rhf_eq.basis = cpu::scf::BasisSet::STO3G;
    rhf_eq.tol = 1e-12;
    rhf_eq.run(frame_eq);
    if (!rhf_eq.converged()) {
        std::fprintf(stderr,
                     "FAIL: RHF/STO-3G(H2, R=0.735) did not converge\n");
        return 1;
    }
    const cpu::scf::StabilityResult st_eq = rhf_eq.stability();
    std::printf("H2/STO-3G R=0.735 A:\n");
    if (!check("internal lowest eig [Ha]", st_eq.internal_eigenvalue,
               kInternalEq, kTolEig))
        return 1;
    if (!check("external lowest eig [Ha]", st_eq.external_eigenvalue,
               kExternalEq, kTolEig))
        return 1;
    if (!st_eq.internal_stable || !st_eq.external_stable) {
        std::fprintf(stderr, "FAIL: equilibrium H2 flagged unstable\n");
        return 1;
    }

    std::printf("PASS: Seeger-Pople stability (internal stable / external "
                "unstable at R=1.5 A; both stable at r_e) + broken-symmetry "
                "UHF below RHF [PySCF 2.14.0 pins; Seeger-Pople 1977; "
                "Szabo-Ostlund 3.8.7]\n");
    return 0;
}

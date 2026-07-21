/// @file regressions/scf/o2_rohf_sto3g/reg_o2_rohf_sto3g.cpp
/// @brief O2 triplet ROHF/STO-3G — Roothaan effective-Fock ROHF (PySCF
///        canonical / Guest–Saunders convention) vs the externally pinned
///        PySCF 2.14.0 `scf.ROHF` value, plus the exact-spin-purity identity
///        and the same-geometry UHF spin-contamination contrast.
///
/// Out-of-library consumer: public headers + link atomiverse only.
///   - E(ROHF, O2 R = 1.2075 Å, (9,7), STO-3G) vs the PySCF pin
///     -147.632166990682 Ha to 1e-8 Ha (see REFERENCE.md for command).
///   - Spin purity: |⟨S²⟩ − S(S+1)| ≤ 1e-10 (S = 1 → ⟨S²⟩ = 2 exactly) —
///     the analytic single-MO-set identity of Roothaan, Rev. Mod. Phys. 32,
///     179 (1960), DOI 10.1103/RevModPhys.32.179.
///   - Contrast: same-geometry UHF shows contamination > 1e-3 (PySCF pin
///     ⟨S²⟩ = 2.003410857681) and a slightly LOWER energy — the classic
///     variational-freedom-vs-spin-purity trade (Tsuchimochi & Scuseria,
///     J. Chem. Phys. 133, 141102 (2010), DOI 10.1063/1.3503173).
///
/// Exit non-zero on failure. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

ATV::cpu::Frame o2_frame() {
    ATV::cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(ATV::kAtom_type)[i] = 8;
        frame.span(ATV::kAtom_mass)[i] = 15.999;
        frame.span(ATV::kAtom_x)[i] = 0.0;
        frame.span(ATV::kAtom_y)[i] = 0.0;
    }
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 1.2075;
    frame.simbox = ATV::cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;

    // ── O2 triplet, ROHF(9,7)/STO-3G ───────────────────────────────────
    auto frame = o2_frame();
    cpu::scf::ROHF rohf(9, 7);
    rohf.basis = cpu::scf::BasisSet::STO3G;
    rohf.tol = 1e-12;
    rohf.max_iter = 200;
    rohf.run(frame);

    const double e_rohf = rohf.energy();
    const double want = -147.632166990682; // PySCF 2.14.0 pin [Ha]
    std::printf("E(O2, ROHF/STO-3G) = %.12f Ha (PySCF %.12f, dE = %.3e)\n",
                e_rohf, want, e_rohf - want);
    bool ok = rohf.converged() && std::fabs(e_rohf - want) < 1e-8;

    // Spin purity: the single-MO-set identity, S = 1 → ⟨S²⟩ = 2 exactly.
    const double contamination = rohf.determinant().spin_contamination();
    std::printf("ROHF <S2> - S(S+1) = %.3e (want |.| <= 1e-10)\n",
                contamination);
    ok = ok && std::fabs(contamination) <= 1e-10;

    // ── Contrast: same-geometry UHF is contaminated and lower ──────────
    // Warm-start UHF from the converged ROHF spin densities: ROHF is
    // constrained UHF (Tsuchimochi–Scuseria), so releasing the constraint
    // relaxes downhill into the contaminated global UHF solution — the one
    // PySCF's default guess also finds (-147.633946785450 Ha,
    // ⟨S²⟩ = 2.003410857681). (The engine's cold Hcore/MINAO guesses land
    // on a higher local UHF solution for O2.)
    using Spin = cpu::scf::SlaterDeterminant::Spin;
    cpu::scf::SCFInitialGuess guess;
    guess.P_alpha = rohf.determinant().density(Spin::alpha);
    guess.P_beta = rohf.determinant().density(Spin::beta);
    guess.has_alpha = true;
    guess.has_beta = true;

    cpu::scf::UHF uhf(9, 7);
    uhf.basis = cpu::scf::BasisSet::STO3G;
    uhf.tol = 1e-12;
    uhf.max_iter = 200;
    uhf.run(frame, &guess);
    const double e_uhf = uhf.energy();
    const double want_uhf = -147.633946785450; // PySCF 2.14.0 pin [Ha]
    const double uhf_cont = uhf.determinant().spin_contamination();
    std::printf("UHF  E = %.12f Ha (PySCF %.12f, dE = %.3e), "
                "<S2> - 2 = %.6e (want > 1e-3; PySCF 3.410858e-3)\n",
                e_uhf, want_uhf, e_uhf - want_uhf, uhf_cont);
    ok = ok && uhf.converged() && std::fabs(e_uhf - want_uhf) < 1e-8;
    ok = ok && uhf_cont > 1e-3;
    // Variational ordering: UHF relaxes spin symmetry, so E_UHF < E_ROHF.
    ok = ok && e_uhf < e_rohf;

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

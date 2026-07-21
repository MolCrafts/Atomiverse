/// @file tests/cpu/scf/delta_scf_reference.h
/// @brief Shared geometries + analytic references for the ΔSCF vertical
///        ionization tests (`test_delta_scf.cpp` and the benchmark sweep
///        `qc/rhf/validation/test_delta_scf_sweep.cpp`).
///
/// ΔSCF vertical ionization: I = E(cation, N-1 e) − E(neutral, N e), both
/// evaluated at the SAME neutral geometry, basis, and convergence threshold.
///
/// ── Why no pinned external (PySCF) totals live here ──────────────────
/// The spec's original plan was to pin PySCF 2.13.1 E⁺ [Ha] / I [eV] triples.
/// PySCF is a BANNED third-party dependency in this tree: no oracle `.py`
/// script may be added and no PySCF number may be fabricated. Correctness of
/// the ΔSCF workflow is therefore established entirely from ENGINE-INTERNAL
/// PHYSICS INVARIANTS that need no external oracle:
///
///   (1) Variational bound (Koopmans is an upper bound on the relaxed
///       ionization energy): I_ΔSCF ≤ −ε_HOMO for every (system, basis).
///       Rationale: −ε_HOMO is the frozen-orbital (N-1)-electron determinant
///       energy; E_UHF(cation) variationally optimizes the same Hilbert
///       space, so E_relaxed ≤ E_frozen.
///   (2) Exact-relaxation-zero equality: for He and H₂ in a minimal basis the
///       cation is a single electron in a symmetry-fixed spatial orbital with
///       no relaxation freedom, so |I_ΔSCF − (−ε_HOMO)| = 0.
///   (3) Analytic doublet spin: a clean cation doublet has ⟨S²⟩ = S(S+1) =
///       0.75 exactly, i.e. `SlaterDeterminant::spin_contamination()` ≈ 0.
///       (Both solves use a tight `conv_tol_grad` so each reaches a reproducible
///       fixed point — the loose sqrt(tol) default leaves energies good to only
///       ~1e-5 and makes convergence thread-order dependent.)
///   (4) Two independent starts each yield a valid sub-Koopmans cation: the
///       neutral-density warm start and the Hcore cold start both converge to a
///       doublet obeying the variational bound (3). For a NON-degenerate cation
///       they reach the SAME minimum — the anchor test asserts tight warm==cold
///       agreement for He / H₂ / H₂O-STO3G. But a cation with a (near-)degenerate
///       hole — CH₄⁺ (Jahn-Teller T₂) and H₂O⁺/6-31G — has MULTIPLE UHF minima,
///       and the two guesses legitimately fall into different basins (which
///       basin is itself guess- and reduction-order-sensitive), so neither
///       warm==cold nor a systematic warm≤cold ordering holds. The sweep
///       therefore asserts only the guess-robust fact: both starts converge to
///       a sub-Koopmans doublet. A genuine excited/hole-shifted collapse would
///       break the variational bound.
///
/// An external-engine (PySCF) cross-check of the absolute E⁺ totals is an
/// ACCEPTANCE-TIME BENCHMARK, run outside this tree — it is deliberately NOT
/// pinned here. This header consequently holds only the shared geometries and
/// the analytic doublet ⟨S²⟩; the tests drive off the invariants above.
///
/// Geometries match the hf-koopmans / ch03 tutorial set: He single atom;
/// H₂ at R = 0.7414 Å; H₂O / NH₃ / CH₄ at standard experimental geometry.
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/methods.h"

namespace ATV {
namespace test {
namespace delta_scf {

/// @brief Exact ⟨S²⟩ for a spin-½ doublet cation: S(S+1) with S = ½.
///        The engine reports `spin_contamination` = ⟨S²⟩ − S(S+1), so a clean
///        doublet has contamination ≈ 0 and ⟨S²⟩ ≈ this constant.
inline constexpr double kDoubletS2 = 0.75;

/// @brief Helium atom at the origin (Z = 2, N = 2 electrons).
/// @return CPU Frame with one He atom in a 100 Å cubic box.
inline cpu::Frame make_he() {
    cpu::Frame f(1);
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_mass)[0] = 4.002602;
    f.span(ATV::kAtom_type)[0] = 2;
    f.simbox = ATV::cpu::Box::cubic(100.0);
    return f;
}

/// @brief Dihydrogen at the equilibrium bond length R = 0.7414 Å (N = 2).
/// @return CPU Frame with two H atoms in a 100 Å cubic box.
inline cpu::Frame make_h2() {
    cpu::Frame f(2);
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_x)[1] = 0.7414;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_y)[1] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_z)[1] = 0.0;
    f.span(ATV::kAtom_mass)[0] = 1.008;
    f.span(ATV::kAtom_mass)[1] = 1.008;
    f.span(ATV::kAtom_type)[0] = 1;
    f.span(ATV::kAtom_type)[1] = 1;
    f.simbox = ATV::cpu::Box::cubic(100.0);
    return f;
}

/// @brief Water at the standard experimental geometry (N = 10).
///        O at (0, 0, 0.1173) Å; H at (0, ±0.7572, −0.4692) Å.
/// @return CPU Frame with O + 2H in a 100 Å cubic box.
inline cpu::Frame make_h2o() {
    cpu::Frame f(3);
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_x)[1] = 0.0;
    f.span(ATV::kAtom_x)[2] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_y)[1] = 0.7572;
    f.span(ATV::kAtom_y)[2] = -0.7572;
    f.span(ATV::kAtom_z)[0] = 0.1173;
    f.span(ATV::kAtom_z)[1] = -0.4692;
    f.span(ATV::kAtom_z)[2] = -0.4692;
    f.span(ATV::kAtom_mass)[0] = 15.999;
    f.span(ATV::kAtom_mass)[1] = 1.008;
    f.span(ATV::kAtom_mass)[2] = 1.008;
    f.span(ATV::kAtom_type)[0] = 8;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_type)[2] = 1;
    f.simbox = ATV::cpu::Box::cubic(100.0);
    return f;
}

/// @brief Ammonia at the standard experimental geometry (r_NH = 1.012 Å,
///        HNH = 106.7°, N = 10).
/// @return CPU Frame with N + 3H in a 100 Å cubic box.
inline cpu::Frame make_nh3() {
    cpu::Frame f(4);
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_x)[1] = 0.0;
    f.span(ATV::kAtom_x)[2] = 0.812328;
    f.span(ATV::kAtom_x)[3] = -0.812328;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_y)[1] = 0.938035;
    f.span(ATV::kAtom_y)[2] = -0.469017;
    f.span(ATV::kAtom_y)[3] = -0.469017;
    f.span(ATV::kAtom_z)[0] = 0.113880;
    f.span(ATV::kAtom_z)[1] = -0.265825;
    f.span(ATV::kAtom_z)[2] = -0.265825;
    f.span(ATV::kAtom_z)[3] = -0.265825;
    f.span(ATV::kAtom_mass)[0] = 14.007;
    f.span(ATV::kAtom_mass)[1] = 1.008;
    f.span(ATV::kAtom_mass)[2] = 1.008;
    f.span(ATV::kAtom_mass)[3] = 1.008;
    f.span(ATV::kAtom_type)[0] = 7;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_type)[2] = 1;
    f.span(ATV::kAtom_type)[3] = 1;
    f.simbox = ATV::cpu::Box::cubic(100.0);
    return f;
}

/// @brief Methane at the tetrahedral geometry (r_CH = 1.087 Å, N = 10).
/// @return CPU Frame with C + 4H in a 100 Å cubic box.
inline cpu::Frame make_ch4() {
    cpu::Frame f(5);
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_x)[1] = 0.627640;
    f.span(ATV::kAtom_x)[2] = -0.627640;
    f.span(ATV::kAtom_x)[3] = 0.627640;
    f.span(ATV::kAtom_x)[4] = -0.627640;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_y)[1] = 0.627640;
    f.span(ATV::kAtom_y)[2] = -0.627640;
    f.span(ATV::kAtom_y)[3] = -0.627640;
    f.span(ATV::kAtom_y)[4] = 0.627640;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_z)[1] = 0.627640;
    f.span(ATV::kAtom_z)[2] = 0.627640;
    f.span(ATV::kAtom_z)[3] = -0.627640;
    f.span(ATV::kAtom_z)[4] = -0.627640;
    f.span(ATV::kAtom_mass)[0] = 12.011;
    f.span(ATV::kAtom_mass)[1] = 1.008;
    f.span(ATV::kAtom_mass)[2] = 1.008;
    f.span(ATV::kAtom_mass)[3] = 1.008;
    f.span(ATV::kAtom_mass)[4] = 1.008;
    f.span(ATV::kAtom_type)[0] = 6;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_type)[2] = 1;
    f.span(ATV::kAtom_type)[3] = 1;
    f.span(ATV::kAtom_type)[4] = 1;
    f.simbox = ATV::cpu::Box::cubic(100.0);
    return f;
}

/// @brief One entry of the ΔSCF system battery: a label and a Frame builder.
struct SystemSpec {
    const char* name;         ///< Human-readable system label.
    cpu::Frame (*make)();     ///< Geometry constructor.
    bool minimal_basis_exact; ///< True for He / H₂ (single-electron cation).
                              ///< In a MINIMAL (STO-3G) basis these have no
                              ///< relaxation freedom ⇒ I_ΔSCF == −ε_HOMO. In a
                              ///< larger basis (6-31G) the electron can relax,
                              ///< so only the variational bound holds.
};

/// @brief The five neutral systems, ordered as in the tutorial table.
///        `minimal_basis_exact` marks the He / H₂ single-electron cations.
inline const SystemSpec kSystems[5] = {
    {"He", &make_he, true},    {"H2", &make_h2, true},
    {"H2O", &make_h2o, false}, {"NH3", &make_nh3, false},
    {"CH4", &make_ch4, false},
};

/// @brief The two basis sets swept: minimal STO-3G and split-valence 6-31G.
inline const cpu::scf::BasisSet kBases[2] = {
    cpu::scf::BasisSet::STO3G,
    cpu::scf::BasisSet::Pople631G,
};

/// @brief Outcome of one ΔSCF vertical-ionization evaluation (all energies
///        in [Hartree]). Holds everything the invariant assertions need.
struct DeltaScfResult {
    bool neutral_converged = false;     ///< Neutral RHF converged.
    bool cation_converged = false;      ///< Warm-started cation UHF converged.
    bool cation_cold_converged = false; ///< Hcore cold-start cation converged.
    double e_neutral = 0.0;             ///< Neutral RHF total energy [Ha].
    double e_cation = 0.0;           ///< Warm-started cation UHF energy [Ha].
    double e_cation_cold = 0.0;      ///< Hcore cold-start cation energy [Ha].
    double i_delta = 0.0;            ///< I_ΔSCF = E_cation − E_neutral [Ha].
    double neg_eps_homo = 0.0;       ///< Koopmans ionization −ε_HOMO [Ha].
    double spin_contamination = 0.0; ///< ⟨S²⟩ − S(S+1) of the cation.
    int n_alpha = 0;                 ///< Cation alpha electrons.
    int n_beta = 0;                  ///< Cation beta electrons.
};

/// @brief Run the full ΔSCF workflow for one (geometry, basis) pair.
///
/// Neutral N-electron `cpu::scf::RHF` (charge 0), then the (N−1)-electron
/// doublet `cpu::scf::UHF` cation both warm-started from the neutral density
/// (`SCFInitialGuess`: P_sigma = (n_sigma/N)·P_neutral, so each channel's seed
/// trace is exactly n_sigma — the beta seed is ZERO when n_beta = 0) and
/// cold-started from the UHF Hcore guess. Parameters are method members; Frame
/// is the only I/O currency (no intermediate config or integral-pack objects).
/// Spin bookkeeping for the doublet cation: n_alpha = N/2 (= neutral n_occ),
/// n_beta = N/2 − 1. Koopmans −ε_HOMO is the neutral eigenvalue at index
/// n_occ − 1.
///
/// @param frame Neutral geometry (atom types = Z).
/// @param basis Gaussian basis set (STO-3G, 6-31G, …).
/// @param tol   SCF energy-convergence tolerance |dE| [Hartree]. Tight (1e-10)
///              so energies resolve well below the 1e-9 equality / 1e-8
///              warm-cold assertion thresholds. The He / H₂ symmetry-fixed
///              cations converge to machine precision regardless.
/// @return DeltaScfResult with all energies [Hartree] and diagnostics.
inline DeltaScfResult run_delta_scf(const cpu::Frame& frame,
                                    cpu::scf::BasisSet basis,
                                    double tol = 1e-10) {
    using namespace cpu::scf;
    DeltaScfResult out;

    // ── Neutral N-electron RHF ──────────────────────────────────────────
    // Tight gradient convergence (conv_tol_grad = 1e-9). The default
    // sqrt(tol) = 1e-5 lets the dual convergence check exit on |ΔE| alone while
    // the orbital gradient is still ~1e-5, leaving the energy / ε_HOMO good to
    // only ~1e-5 AND making that early exit fire thread-order-dependently. That
    // loose, non-reproducible convergence is what makes the warm/cold cation
    // energies wander (a single-electron cation such as He⁺/6-31G has a UNIQUE
    // minimum, yet loose warm/cold ends land ~1e-5 apart). Converging the
    // gradient tight pins the unique fixed point.
    using Spin = SlaterDeterminant::Spin;

    RHF neutral_method;
    neutral_method.basis = basis;
    neutral_method.charge = 0;
    neutral_method.tol = tol;
    neutral_method.max_iter = 200;
    neutral_method.conv_tol_grad = 1e-9;
    neutral_method.run(frame);
    out.neutral_converged = neutral_method.converged();
    out.e_neutral = neutral_method.energy();
    const SlaterDeterminant& neutral = neutral_method.determinant();
    // Closed-shell RHF: the occupied count is n_alpha (= n_electrons / 2).
    const int n_occ = neutral.n_electrons(Spin::alpha);
    out.neg_eps_homo = -neutral.orbital_energies(Spin::alpha)(n_occ - 1);

    // ── Doublet cation: N−1 electrons, M = 2 ────────────────────────────
    // Remove one electron from the highest doubly-occupied level: alpha keeps
    // the N/2 occupied orbitals, beta drops one → (N/2, N/2 − 1).
    out.n_alpha = n_occ;
    out.n_beta = n_occ - 1;

    UHF cation_method{out.n_alpha, out.n_beta};
    cation_method.basis = basis;
    cation_method.charge = +1;
    cation_method.tol = tol;
    cation_method.max_iter = 300;
    // Tight gradient convergence (same rationale as the neutral): the loose
    // sqrt(tol) default leaves the cation energy reproducible only to ~1e-5,
    // which — combined with threading — is what let warm/cold energies wander.
    cation_method.conv_tol_grad = 1e-9;

    // Warm start: partition the neutral total density BETWEEN the two spin
    // channels IN PROPORTION to each channel's electron count. The neutral P
    // integrates to N = 2·n_occ electrons, so scaling by n_sigma/N gives a
    // seed whose trace is exactly n_sigma. This matters for the single-
    // electron cations (He⁺, H₂⁺): they have n_beta = 0, so the beta seed must
    // be the ZERO matrix — seeding it with 0.5·P_neutral (trace 1) injects a
    // spurious beta electron that lets the UHF solve converge (thread-order-
    // dependently) to a contaminated 2-electron-like state instead of the
    // symmetry-fixed σg¹ doublet, breaking exact-relaxation-zero.
    const double neutral_ne = static_cast<double>(2 * n_occ);
    // The old restricted result's P_alpha was the factor-2 total density —
    // that is `density_total()` on the determinant.
    const RealMatrix P_neutral = neutral.density_total();
    SCFInitialGuess guess;
    guess.P_alpha =
        (static_cast<double>(out.n_alpha) / neutral_ne) * P_neutral;
    guess.P_beta =
        (static_cast<double>(out.n_beta) / neutral_ne) * P_neutral;
    guess.has_alpha = true;
    guess.has_beta = true;
    cation_method.run(frame, &guess);
    out.cation_converged = cation_method.converged();
    out.e_cation = cation_method.energy();
    out.spin_contamination =
        cation_method.determinant().spin_contamination();
    out.i_delta = out.e_cation - out.e_neutral;

    // Cold start: identical method, UHF Hcore default guess, no seed.
    cation_method.run(frame);
    out.cation_cold_converged = cation_method.converged();
    out.e_cation_cold = cation_method.energy();

    return out;
}

} // namespace delta_scf
} // namespace test
} // namespace ATV

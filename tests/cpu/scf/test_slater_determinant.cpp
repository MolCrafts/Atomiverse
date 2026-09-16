/// @file tests/cpu/scf/test_slater_determinant.cpp
/// @brief Unit tests for the `cpu::scf::SlaterDeterminant` value type
///        (spec scf-slater-determinant-01-type). RED-first: the header under
///        test, `atomiverse/cpu/scf/slater_determinant.h`, does not exist yet.
///
/// Contract under test (target shape — validating constructors, no factory):
///   class SlaterDeterminant {
///       enum class Spin { alpha, beta };                       // nested
///       SlaterDeterminant(RealMatrix C, RealVector epsilon,    // restricted
///                         RealVector occupation, int n_electrons_per_spin);
///       SlaterDeterminant(RealMatrix C_a, RealVector eps_a,    // unrestricted
///                         RealVector f_a, int n_alpha,
///                         RealMatrix C_b, RealVector eps_b,
///                         RealVector f_b, int n_beta);
///       bool restricted() const;
///       RealMatrix density(Spin) const;        // P^s = sum_i f_i c_i c_i^T
///       RealMatrix density_total() const;      // P^alpha + P^beta
///       double expect(const RealMatrix& O) const;  // Tr(P_total O), 1e- only
///       double orbital_energy(Spin, int i) const;  // [Ha]; out-of-range throws
///       double homo(Spin) const;               // highest-occupied eps [Ha]
///       double lumo(Spin) const;               // lowest-virtual eps [Ha];
///                                              // == homo() when no virtuals
///       SlaterDeterminant with_occupation(Spin, int from, int to) const &;
///       SlaterDeterminant with_occupation(Spin, int from, int to) &&;
///       void validate(const RealMatrix& S, double tol = 1e-8) const;
///   };
/// Constructors throw std::invalid_argument (message carries the offending
/// numeric value) on: dimension mismatch (C cols == eps size == f size,
/// C rows == n), f_i < 0, f_i > 1, |sum f - N| > 1e-12 * max(1, N).
/// with_occupation requires f_from == 1 && f_to == 0, else throws.
/// validate(S) checks Tr(P^s S) = N^s per channel and idempotency
/// P^s S P^s = P^s only when every f in the channel is integer; violation
/// throws std::runtime_error (runtime numerical invariant — the
/// CanonicalOrth precedent), never a silent fallback.
///
/// Fixtures: 3x3 orthonormal C from Pythagorean-triple Givens rotations
/// (exact short decimals), ascending eps in Hartree, toy occupations.
/// All matrices n <= 4, no SCF solve — pure value-type algebra, so the
/// handwritten references are exact to ~1e-15 and 1e-12 bounds are safe.

#ifdef ATV_HAS_HF

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <utility>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

using ATV::cpu::scf::RealMatrix;
using ATV::cpu::scf::RealVector;
using ATV::cpu::scf::SlaterDeterminant;
using Spin = ATV::cpu::scf::SlaterDeterminant::Spin;

namespace {

/// Orthonormal 3x3 alpha coefficients: 3-4-5 Givens rotation in the (0,1)
/// plane. Column i is MO i; exact decimals, C^T C = I to machine precision.
RealMatrix make_c_alpha() {
    RealMatrix c = xt::eye<double>(3);
    c(0, 0) = 0.8;
    c(1, 0) = 0.6;
    c(0, 1) = -0.6;
    c(1, 1) = 0.8;
    return c;
}

/// A distinct orthonormal 3x3 for the beta channel: 3-4-5 Givens rotation in
/// the (1,2) plane.
RealMatrix make_c_beta() {
    RealMatrix c = xt::eye<double>(3);
    c(1, 1) = 0.6;
    c(2, 1) = 0.8;
    c(1, 2) = -0.8;
    c(2, 2) = 0.6;
    return c;
}

/// Ascending orbital energies [Ha] — the spec's known-eps sequence.
RealVector make_eps() { return {-1.0, -0.5, 0.3}; }

/// Beta-channel orbital energies [Ha] (distinct from alpha on purpose).
RealVector make_eps_beta() { return {-0.9, -0.4, 0.5}; }

/// Aufbau occupation for 2 electrons in this spin channel.
RealVector make_f_aufbau() { return {1.0, 1.0, 0.0}; }

/// Handwritten reference density P = sum_i f_i c_i c_i^T (plain loops —
/// independent of both linalg::build_density_matrix and xt::linalg::dot).
RealMatrix manual_density(const RealMatrix& c, const RealVector& f) {
    const std::size_t n = c.shape(0);
    RealMatrix p = xt::zeros<double>({n, n});
    for (std::size_t i = 0; i < f.size(); ++i)
        for (std::size_t mu = 0; mu < n; ++mu)
            for (std::size_t nu = 0; nu < n; ++nu)
                p(mu, nu) += f(i) * c(mu, i) * c(nu, i);
    return p;
}

/// Elementwise |A - B| <= tol with located failure message.
void expect_matrix_near(const RealMatrix& a, const RealMatrix& b, double tol) {
    ASSERT_EQ(a.shape(0), b.shape(0));
    ASSERT_EQ(a.shape(1), b.shape(1));
    for (std::size_t i = 0; i < a.shape(0); ++i)
        for (std::size_t j = 0; j < a.shape(1); ++j)
            EXPECT_NEAR(a(i, j), b(i, j), tol)
                << "matrices differ at (" << i << "," << j << ")";
}

/// The reference restricted determinant: Aufbau, 2 electrons per spin.
SlaterDeterminant make_restricted() {
    return SlaterDeterminant(make_c_alpha(), make_eps(), make_f_aufbau(), 2);
}

/// The reference unrestricted determinant: n_alpha = 2, n_beta = 1.
SlaterDeterminant make_unrestricted() {
    return SlaterDeterminant(make_c_alpha(), make_eps(), make_f_aufbau(), 2,
                             make_c_beta(), make_eps_beta(),
                             RealVector{1.0, 0.0, 0.0}, 1);
}

} // namespace

// ── 1. Constructor happy paths (basics) ──────────────────────────────

// Basics: a valid restricted (RHF/RKS-shaped) construction succeeds and
// reports the restricted form.
TEST(SlaterDeterminantCtor, RestrictedHappyPath) {
    SlaterDeterminant sd = make_restricted();
    EXPECT_TRUE(sd.restricted());
}

// Basics: a valid unrestricted (UHF/UKS-shaped) construction succeeds and
// reports the unrestricted form.
TEST(SlaterDeterminantCtor, UnrestrictedHappyPath) {
    SlaterDeterminant sd = make_unrestricted();
    EXPECT_FALSE(sd.restricted());
}

// ── 2. Constructor validation throws (edge cases) ────────────────────

// Edge case: Pauli lower bound — any f_i < 0 is rejected. f = (1, -0.5, 0.5)
// sums to N = 1 exactly, so ONLY the negative-occupation invariant fires;
// the message must carry the offending numeric value (spec: throw with
// context, no fallback).
TEST(SlaterDeterminantCtor, NegativeOccupationThrowsWithNumericContext) {
    try {
        SlaterDeterminant sd(make_c_alpha(), make_eps(),
                             RealVector{1.0, -0.5, 0.5}, 1);
        FAIL() << "expected std::invalid_argument for f_i = -0.5";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("0.5"), std::string::npos)
            << "message must contain the offending occupation, got: "
            << e.what();
    }
}

// Edge case: Pauli upper bound — any f_i > 1 (spin-orbital occupation) is
// rejected. f = (1.5, 0.5, 0) sums to N = 2, isolating the upper bound.
TEST(SlaterDeterminantCtor, OccupationAboveOneThrows) {
    EXPECT_THROW(SlaterDeterminant(make_c_alpha(), make_eps(),
                                   RealVector{1.5, 0.5, 0.0}, 2),
                 std::invalid_argument);
}

// Edge case: electron-count consistency — |sum f - N| > 1e-12 * max(1, N)
// is rejected. f = (1,1,1) with N = 2: every f_i is in bounds, only the sum
// invariant fires.
TEST(SlaterDeterminantCtor, OccupationSumMismatchThrows) {
    EXPECT_THROW(SlaterDeterminant(make_c_alpha(), make_eps(),
                                   RealVector{1.0, 1.0, 1.0}, 2),
                 std::invalid_argument);
}

// Edge case: dimension consistency — C has 3 columns but eps/f have 2
// entries.
TEST(SlaterDeterminantCtor, DimensionMismatchThrows) {
    EXPECT_THROW(SlaterDeterminant(make_c_alpha(), RealVector{-1.0, -0.5},
                                   RealVector{1.0, 1.0}, 2),
                 std::invalid_argument);
}

// Edge case: the unrestricted constructor validates the beta channel too
// (negative beta occupation, alpha channel valid).
TEST(SlaterDeterminantCtor, UnrestrictedBadBetaChannelThrows) {
    EXPECT_THROW(SlaterDeterminant(make_c_alpha(), make_eps(), make_f_aufbau(),
                                   2, make_c_beta(), make_eps_beta(),
                                   RealVector{1.0, -0.5, 0.5}, 1),
                 std::invalid_argument);
}

// ── 3. density vs handwritten GEMM (basics / numerics) ───────────────

// Basics: Aufbau f = (1,1,0) — density(alpha) equals the handwritten
// C_occ C_occ^T via xt::linalg::dot, elementwise <= 1e-12 (the current
// build_density_matrix GEMM path).
TEST(SlaterDeterminantDensity, AlphaMatchesHandwrittenGemmAufbau) {
    SlaterDeterminant sd = make_restricted();
    RealMatrix c = make_c_alpha();
    auto c_occ = xt::view(c, xt::all(), xt::range(0, 2));
    RealMatrix ref(xt::linalg::dot(c_occ, xt::transpose(c_occ)));
    expect_matrix_near(sd.density(Spin::alpha), ref, 1e-12);
}

// ── 4. density with non-Aufbau and fractional occupations ────────────

// Non-Aufbau occupation f = (1,0,1) (the DeltaSCF shape) against the manual
// sum_i f_i c_i c_i^T reference.
TEST(SlaterDeterminantDensity, NonAufbauMatchesManualSum) {
    RealVector f = {1.0, 0.0, 1.0};
    SlaterDeterminant sd(make_c_alpha(), make_eps(), f, 2);
    expect_matrix_near(sd.density(Spin::alpha),
                       manual_density(make_c_alpha(), f), 1e-12);
}

// Fractional occupation f = (1, 0.5, 0.5) (sqrt(f)-weighted gather path)
// against the same manual reference.
TEST(SlaterDeterminantDensity, FractionalMatchesManualSum) {
    RealVector f = {1.0, 0.5, 0.5};
    SlaterDeterminant sd(make_c_alpha(), make_eps(), f, 2);
    expect_matrix_near(sd.density(Spin::alpha),
                       manual_density(make_c_alpha(), f), 1e-12);
}

// ── 5. density_total in both forms ───────────────────────────────────

// Restricted: the factor-2 closed-shell total density is DERIVED —
// density_total() == 2 * density(alpha).
TEST(SlaterDeterminantDensity, TotalRestrictedIsTwiceAlpha) {
    SlaterDeterminant sd = make_restricted();
    RealMatrix twice(2.0 * sd.density(Spin::alpha));
    expect_matrix_near(sd.density_total(), twice, 1e-12);
}

// Unrestricted (Pople–Nesbet): density_total() == P^alpha + P^beta with
// genuinely different channels.
TEST(SlaterDeterminantDensity, TotalUnrestrictedIsAlphaPlusBeta) {
    SlaterDeterminant sd = make_unrestricted();
    RealMatrix sum(sd.density(Spin::alpha) + sd.density(Spin::beta));
    expect_matrix_near(sd.density_total(), sum, 1e-12);
}

// ── 6. expect(O) — one-electron expectation value ────────────────────

// Domain validation: <Phi|O|Phi> = Tr(P_total O) for a one-electron operator
// (Szabo & Ostlund Ch. 3) against a manual trace on a symmetric toy O.
TEST(SlaterDeterminantExpect, MatchesManualTrace) {
    SlaterDeterminant sd = make_unrestricted();
    RealMatrix o = xt::zeros<double>({3, 3});
    o(0, 0) = 1.0;
    o(1, 1) = 2.0;
    o(2, 2) = 3.0;
    o(0, 1) = o(1, 0) = 0.2;
    o(1, 2) = o(2, 1) = 0.1;
    RealMatrix p = sd.density_total();
    RealMatrix po(xt::linalg::dot(p, o));
    double trace = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
        trace += po(i, i);
    EXPECT_NEAR(sd.expect(o), trace, 1e-12);
}

// ── 7. homo / lumo / orbital_energy ──────────────────────────────────

// Basics: ascending eps = (-1.0, -0.5, 0.3) [Ha], n_occ = 2 (Aufbau) —
// homo == eps(n_occ - 1) == -0.5, lumo == eps(n_occ) == 0.3 (the
// orbital_info_observer semantics being absorbed).
TEST(SlaterDeterminantOrbitals, HomoLumoAufbau) {
    SlaterDeterminant sd = make_restricted();
    EXPECT_NEAR(sd.homo(Spin::alpha), -0.5, 1e-15);
    EXPECT_NEAR(sd.lumo(Spin::alpha), 0.3, 1e-15);
}

// Edge case: no virtual orbitals (f = (1,1,1), N = 3) — lumo() returns
// homo() (observer:57-59 semantics).
TEST(SlaterDeterminantOrbitals, LumoEqualsHomoWhenAllOccupied) {
    SlaterDeterminant sd(make_c_alpha(), make_eps(), RealVector{1.0, 1.0, 1.0},
                         3);
    EXPECT_NEAR(sd.lumo(Spin::alpha), sd.homo(Spin::alpha), 1e-15);
    EXPECT_NEAR(sd.lumo(Spin::alpha), 0.3, 1e-15);
}

// Basics: orbital_energy(Spin, i) returns eps_i [Ha].
TEST(SlaterDeterminantOrbitals, OrbitalEnergyReturnsEps) {
    SlaterDeterminant sd = make_restricted();
    EXPECT_NEAR(sd.orbital_energy(Spin::alpha, 0), -1.0, 1e-15);
    EXPECT_NEAR(sd.orbital_energy(Spin::alpha, 2), 0.3, 1e-15);
}

// Edge case: out-of-range orbital index throws (no fallback).
TEST(SlaterDeterminantOrbitals, OrbitalEnergyOutOfRangeThrows) {
    SlaterDeterminant sd = make_restricted();
    EXPECT_THROW(sd.orbital_energy(Spin::alpha, 3), std::out_of_range);
    EXPECT_THROW(sd.orbital_energy(Spin::alpha, -1), std::out_of_range);
}

// ── 8. Restricted beta view ──────────────────────────────────────────

// Restricted form: the beta accessors are a view of the alpha channel —
// density(beta) == density(alpha), homo/lumo/orbital_energy agree exactly.
TEST(SlaterDeterminantRestricted, BetaViewMatchesAlpha) {
    SlaterDeterminant sd = make_restricted();
    expect_matrix_near(sd.density(Spin::beta), sd.density(Spin::alpha), 0.0);
    EXPECT_EQ(sd.homo(Spin::beta), sd.homo(Spin::alpha));
    EXPECT_EQ(sd.lumo(Spin::beta), sd.lumo(Spin::alpha));
    EXPECT_EQ(sd.orbital_energy(Spin::beta, 1),
              sd.orbital_energy(Spin::alpha, 1));
}

// ── 9. with_occupation (DeltaSCF/MOM entry point) ────────────────────

// const& path: moving the electron HOMO(1) -> LUMO(2) returns a NEW value
// with f = (1,0,1) — density and homo reflect it — while the original is
// unchanged (immutability of the copy path).
TEST(SlaterDeterminantWithOccupation, CopyPathSwapsAndPreservesOriginal) {
    SlaterDeterminant sd = make_restricted();
    SlaterDeterminant excited = sd.with_occupation(Spin::alpha, 1, 2);
    // New value carries the non-Aufbau occupation (1,0,1).
    expect_matrix_near(excited.density(Spin::alpha),
                       manual_density(make_c_alpha(), {1.0, 0.0, 1.0}), 1e-12);
    EXPECT_NEAR(excited.homo(Spin::alpha), 0.3, 1e-15);  // highest OCCUPIED
    EXPECT_NEAR(excited.lumo(Spin::alpha), -0.5, 1e-15); // lowest VIRTUAL
    // Original untouched: still Aufbau.
    expect_matrix_near(sd.density(Spin::alpha),
                       manual_density(make_c_alpha(), make_f_aufbau()), 1e-12);
    EXPECT_NEAR(sd.homo(Spin::alpha), -0.5, 1e-15);
}

// && path: chaining off a temporary (the with_gradient ref-qualified
// precedent) — construct, promote in one expression, use the result.
TEST(SlaterDeterminantWithOccupation, MovePathChainsOnTemporary) {
    SlaterDeterminant excited =
        SlaterDeterminant(make_c_alpha(), make_eps(), make_f_aufbau(), 2)
            .with_occupation(Spin::alpha, 1, 2);
    expect_matrix_near(excited.density(Spin::alpha),
                       manual_density(make_c_alpha(), {1.0, 0.0, 1.0}), 1e-12);
}

// Edge case: the transfer requires f_from == 1 and f_to == 0 — moving FROM
// an empty orbital, or TO a filled one, throws.
TEST(SlaterDeterminantWithOccupation, InvalidTransferThrows) {
    SlaterDeterminant sd = make_restricted();
    // from = 2 has f = 0 (not 1).
    EXPECT_THROW(sd.with_occupation(Spin::alpha, 2, 1), std::invalid_argument);
    // to = 0 has f = 1 (not 0).
    EXPECT_THROW(sd.with_occupation(Spin::alpha, 1, 0), std::invalid_argument);
}

// ── 10. validate(S) — the S-dependent invariants ─────────────────────

// Domain validation: orthonormal toy basis (S = I), integer occupations —
// Tr(P^s S) = N^s and idempotency P^s S P^s = P^s both hold (single
// determinant iff f in {0,1}); validate must pass for both channels.
TEST(SlaterDeterminantValidate, IdentityOverlapIntegerOccupationPasses) {
    SlaterDeterminant sd = make_unrestricted();
    RealMatrix s = xt::eye<double>(3);
    EXPECT_NO_THROW(sd.validate(s));
}

// Domain validation: fractional occupation f = (1, 0.5, 0.5) — idempotency
// does NOT hold (PSP != P for fractional f) and must be SKIPPED; the
// electron-count invariant Tr(P^s S) = sum f still holds, so validate passes.
TEST(SlaterDeterminantValidate, FractionalOccupationSkipsIdempotency) {
    SlaterDeterminant sd(make_c_alpha(), make_eps(), RealVector{1.0, 0.5, 0.5},
                         2);
    RealMatrix s = xt::eye<double>(3);
    EXPECT_NO_THROW(sd.validate(s));
}

// Edge case: a deliberately NON-orthonormal C (column 0 scaled by 2) with
// S = I — Tr(PS) = 4 + 1 = 5 != N = 2, so validate throws
// (std::runtime_error: runtime numerical invariant, CanonicalOrth
// precedent). The constructor cannot catch this (it has no S).
TEST(SlaterDeterminantValidate, NonOrthonormalCoefficientsThrow) {
    RealMatrix c = make_c_alpha();
    for (std::size_t mu = 0; mu < 3; ++mu)
        c(mu, 0) *= 2.0;
    SlaterDeterminant sd(c, make_eps(), make_f_aufbau(), 2);
    RealMatrix s = xt::eye<double>(3);
    EXPECT_THROW(sd.validate(s), std::runtime_error);
}

#endif // ATV_HAS_HF

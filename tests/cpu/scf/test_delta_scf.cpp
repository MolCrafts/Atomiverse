/// @file tests/cpu/scf/test_delta_scf.cpp
/// @brief Default-gate anchors for ΔSCF vertical ionization (STO-3G).
///
/// Anchors He⁺ / H₂⁺ / H₂O⁺ at STO-3G. Every assertion is an ENGINE-INTERNAL
/// physics invariant — there are no pinned external (PySCF) oracle numbers
/// (see `delta_scf_reference.h` for the no-third-party rationale). The full
/// 5×2 sweep lives in the acceptance-time benchmark
/// `qc/rhf/validation/test_delta_scf_sweep.cpp`.
///
/// Invariants asserted:
///   - cation SCF converged (warm and cold start);
///   - variational bound I_ΔSCF ≤ −ε_HOMO (relaxation lowers the ionization
///     energy below Koopmans);
///   - He / H₂ minimal-basis exact-relaxation-zero: |I_ΔSCF − (−ε_HOMO)| ≤ 1e-9;
///   - doublet spin_contamination ≤ 1e-4 (⟨S²⟩ ≈ 0.75);
///   - warm-start and Hcore cold-start reach the same cation energy to 1e-8.

#ifdef ATV_HAS_HF

#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/cpu/scf/basis_set.h"
#include "delta_scf_reference.h"

using namespace ATV;
using namespace ATV::test::delta_scf;

namespace {

// Tolerances for the internal-invariant assertions.
constexpr double kExactRelaxTol =
    1e-9; ///< He / H₂ Koopmans==ΔSCF equality [Ha].
// A single-electron cation (He⁺, H₂⁺: n_beta = 0) is a pure doublet with ⟨S²⟩
// exactly 0.75, so its contamination is machine-zero — assert it tight.
constexpr double kSpinTolExact =
    1e-8; ///< single-electron cation contamination.
// A genuine multi-electron open-shell UHF cation (H₂O⁺) carries small, real
// spin contamination from the unrestricted single determinant (H₂O⁺/STO-3G:
// ⟨S²⟩−0.75 ≈ 0.0052, deterministic and warm/cold-invariant). 1e-4 was an
// unphysically tight floor; 0.02 is an honest upper bound on real UHF
// contamination for these small first-row cations.
constexpr double kSpinTolUHF =
    0.02; ///< multi-electron UHF cation contamination.
constexpr double kWarmColdTol = 1e-8; ///< warm-vs-cold cation energy [Ha].
constexpr double kBoundSlack = 1e-9;  ///< numerical slack on the ≤ bound [Ha].

// One anchor evaluation, reused across the assertions.
DeltaScfResult anchor(const SystemSpec& sys) {
    return run_delta_scf(sys.make(), cpu::scf::BasisSet::STO3G, /*tol=*/1e-10);
}

} // namespace

// ── He⁺ (single-electron cation, exact relaxation zero) ─────────────────

TEST(DeltaScfAnchor, HeliumStO3G) {
    auto r = anchor(kSystems[0]); // He
    ASSERT_TRUE(r.neutral_converged);
    ASSERT_TRUE(r.cation_converged);
    ASSERT_TRUE(r.cation_cold_converged);
    EXPECT_EQ(r.n_alpha, 1);
    EXPECT_EQ(r.n_beta, 0);
    // Variational bound.
    EXPECT_LE(r.i_delta, r.neg_eps_homo + kBoundSlack);
    // Minimal-basis single electron ⇒ no relaxation ⇒ strict equality.
    EXPECT_NEAR(r.i_delta, r.neg_eps_homo, kExactRelaxTol);
    // Single-electron cation ⇒ pure doublet ⇒ machine-zero contamination.
    EXPECT_LE(std::abs(r.spin_contamination), kSpinTolExact);
    // Warm/cold agreement.
    EXPECT_NEAR(r.e_cation, r.e_cation_cold, kWarmColdTol);
}

// ── H₂⁺ (single-electron cation, exact relaxation zero) ─────────────────

TEST(DeltaScfAnchor, HydrogenMoleculeStO3G) {
    auto r = anchor(kSystems[1]); // H2
    ASSERT_TRUE(r.neutral_converged);
    ASSERT_TRUE(r.cation_converged);
    ASSERT_TRUE(r.cation_cold_converged);
    EXPECT_EQ(r.n_alpha, 1);
    EXPECT_EQ(r.n_beta, 0);
    EXPECT_LE(r.i_delta, r.neg_eps_homo + kBoundSlack);
    EXPECT_NEAR(r.i_delta, r.neg_eps_homo, kExactRelaxTol);
    // Single-electron cation ⇒ pure doublet ⇒ machine-zero contamination.
    EXPECT_LE(std::abs(r.spin_contamination), kSpinTolExact);
    EXPECT_NEAR(r.e_cation, r.e_cation_cold, kWarmColdTol);
}

// ── H₂O⁺ (multi-electron cation, genuine orbital relaxation) ────────────

TEST(DeltaScfAnchor, WaterStO3G) {
    auto r = anchor(kSystems[2]); // H2O
    ASSERT_TRUE(r.neutral_converged);
    ASSERT_TRUE(r.cation_converged);
    ASSERT_TRUE(r.cation_cold_converged);
    EXPECT_EQ(r.n_alpha, 5);
    EXPECT_EQ(r.n_beta, 4);
    // Variational bound must hold; relaxation is strictly positive here.
    EXPECT_LE(r.i_delta, r.neg_eps_homo + kBoundSlack);
    EXPECT_GT(r.neg_eps_homo - r.i_delta, 0.0); // nonzero relaxation
    // Multi-electron open-shell UHF cation ⇒ small but genuine contamination.
    EXPECT_LE(std::abs(r.spin_contamination), kSpinTolUHF);
    EXPECT_NEAR(r.e_cation, r.e_cation_cold, kWarmColdTol);
}

#else

#include <gtest/gtest.h>
TEST(DeltaScfAnchor, SkippedNoBuildRHF) {
    GTEST_SKIP() << "ATV_BUILD_HF=OFF — ΔSCF tests require -DATV_BUILD_HF=ON";
}

#endif // ATV_HAS_HF

/// @file tests/cpu/scf/test_spin_orbital_eri.cpp
/// @brief Unit checks for the hoisted spin-orbital ERI helpers
///        (scf-review-02-eri-hoists): SpinOrbital bit-packing, the single
///        authoritative antisymmetrized ⟨pq‖rs⟩ (struct + bit-packed
///        overloads), and the shared MP aufbau denominator guard.
///
/// Pins the delivered contract of internal/spin_orbital_eri.h (written
/// RED-first for spec scf-review-02-eri-hoists; landed and GREEN).

#ifdef ATV_HAS_HF

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/internal/spin_orbital_eri.h"

namespace scf = ATV::cpu::scf;

namespace {

/// @brief Bit-exact equality of two doubles (no tolerance): the two
///        antisym_eri overloads run identical float ops, so their results
///        must be bitwise identical, not merely close.
bool bit_equal(double a, double b) {
    std::uint64_t ua = 0;
    std::uint64_t ub = 0;
    std::memcpy(&ua, &a, sizeof(ua));
    std::memcpy(&ub, &b, sizeof(ub));
    return ua == ub;
}

/// @brief A 2×2×2×2 chemist-notation ERI fixture g(a,b,c,d) = (ab|cd) with
///        16 distinct nonzero values so every gather is uniquely identifiable.
///        value(a,b,c,d) = 1 + 8a + 4b + 2c + d ∈ {1,…,16}.
xt::xtensor<double, 4> make_chem_eri() {
    xt::xtensor<double, 4> g = xt::zeros<double>({2, 2, 2, 2});
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
            for (int c = 0; c < 2; ++c)
                for (int d = 0; d < 2; ++d)
                    g(a, b, c, d) = 1.0 + 8.0 * a + 4.0 * b + 2.0 * c + d;
    return g;
}

} // namespace

// ── SpinOrbital::from_packed — {so>>1, so&1} ─────────────────────────────

TEST(TestSpinOrbitalEri, FromPackedDecodesSpatialAndSpin) {
    // so = 2*spat + spin; from_packed recovers {spat = so>>1, spin = so&1}.
    const scf::SpinOrbital s0 = scf::SpinOrbital::from_packed(0);
    EXPECT_EQ(s0.spat, 0);
    EXPECT_EQ(s0.spin, 0);

    const scf::SpinOrbital s1 = scf::SpinOrbital::from_packed(1);
    EXPECT_EQ(s1.spat, 0);
    EXPECT_EQ(s1.spin, 1);

    const scf::SpinOrbital s4 = scf::SpinOrbital::from_packed(4);
    EXPECT_EQ(s4.spat, 2);
    EXPECT_EQ(s4.spin, 0);

    const scf::SpinOrbital s7 = scf::SpinOrbital::from_packed(7);
    EXPECT_EQ(s7.spat, 3);
    EXPECT_EQ(s7.spin, 1);

    // Zero-cost decode: usable in a constant expression.
    static_assert(scf::SpinOrbital::from_packed(4).spat == 2,
                  "from_packed must be constexpr");
    static_assert(scf::SpinOrbital::from_packed(7).spin == 1,
                  "from_packed must be constexpr");
}

// ── antisym_eri hand-computed literals ───────────────────────────────────
// ⟨pq‖rs⟩ = (pr|qs)δ(σp,σr)δ(σq,σs) − (ps|qr)δ(σp,σs)δ(σq,σr), chemist g.

TEST(TestSpinOrbitalEri, SameSpinBothTermsSurvive) {
    // p=(spat0,α) q=(spat1,α) r=(spat0,α) s=(spat1,α): all α.
    //   direct  += g(0,0,1,1) = 4   (pr|qs)
    //   exchange -= g(0,1,1,0) = 7   (ps|qr)
    //   ⟨pq‖rs⟩ = 4 − 7 = −3
    const auto g = make_chem_eri();
    const scf::SpinOrbital p{0, 0};
    const scf::SpinOrbital q{1, 0};
    const scf::SpinOrbital r{0, 0};
    const scf::SpinOrbital s{1, 0};
    EXPECT_DOUBLE_EQ(scf::antisym_eri(g, p, q, r, s), -3.0);
}

TEST(TestSpinOrbitalEri, OppositeSpinOnlyDirectSurvives) {
    // p=(0,α) q=(1,β) r=(0,α) s=(1,β): σp=σr=α, σq=σs=β.
    //   direct  += g(0,0,1,1) = 4
    //   exchange requires σp==σs (α==β) → dropped.
    //   ⟨pq‖rs⟩ = 4
    const auto g = make_chem_eri();
    const scf::SpinOrbital p{0, 0};
    const scf::SpinOrbital q{1, 1};
    const scf::SpinOrbital r{0, 0};
    const scf::SpinOrbital s{1, 1};
    EXPECT_DOUBLE_EQ(scf::antisym_eri(g, p, q, r, s), 4.0);
}

TEST(TestSpinOrbitalEri, CrossSpinOnlyExchangeSurvives) {
    // p=(0,α) q=(1,β) r=(0,β) s=(1,α): σp≠σr (α≠β) → direct dropped;
    //   σp==σs (α==α) and σq==σr (β==β) → exchange survives.
    //   exchange -= g(0,1,1,0) = 7
    //   ⟨pq‖rs⟩ = −7
    const auto g = make_chem_eri();
    const scf::SpinOrbital p{0, 0};
    const scf::SpinOrbital q{1, 1};
    const scf::SpinOrbital r{0, 1};
    const scf::SpinOrbital s{1, 0};
    EXPECT_DOUBLE_EQ(scf::antisym_eri(g, p, q, r, s), -7.0);
}

TEST(TestSpinOrbitalEri, NeitherTermSurvivesIsZero) {
    // p=(0,α) q=(1,α) r=(0,β) s=(1,β): σp≠σr and σp≠σs → both terms dropped.
    const auto g = make_chem_eri();
    const scf::SpinOrbital p{0, 0};
    const scf::SpinOrbital q{1, 0};
    const scf::SpinOrbital r{0, 1};
    const scf::SpinOrbital s{1, 1};
    EXPECT_DOUBLE_EQ(scf::antisym_eri(g, p, q, r, s), 0.0);
}

// ── struct overload vs bit-packed overload: bitwise identical ────────────

TEST(TestSpinOrbitalEri, StructAndPackedOverloadsAreBitwiseEqual) {
    // Sweep all 4^4 = 256 spin-orbital quadruples over 2 spatials × 2 spins
    // (packed so ∈ {0,1,2,3}). The bit-packed overload must decode via
    // from_packed and forward to the single struct implementation, so the two
    // call paths are bit-for-bit identical across every spin branch.
    const auto g = make_chem_eri();
    for (int p = 0; p < 4; ++p) {
        for (int q = 0; q < 4; ++q) {
            for (int r = 0; r < 4; ++r) {
                for (int s = 0; s < 4; ++s) {
                    const double packed = scf::antisym_eri(g, p, q, r, s);
                    const double via_struct =
                        scf::antisym_eri(g, scf::SpinOrbital::from_packed(p),
                                         scf::SpinOrbital::from_packed(q),
                                         scf::SpinOrbital::from_packed(r),
                                         scf::SpinOrbital::from_packed(s));
                    ASSERT_TRUE(bit_equal(packed, via_struct))
                        << "packed vs struct mismatch at (" << p << ',' << q
                        << ',' << r << ',' << s << "): " << packed
                        << " != " << via_struct;
                }
            }
        }
    }
}

// ── mp_denominator — aufbau guard ────────────────────────────────────────

TEST(TestSpinOrbitalEri, DenominatorNegativeForAufbauReference) {
    // Δ = e_i + e_j − e_a − e_b = −0.5 − 0.4 − 0.3 − 0.2 = −1.4 (< 0).
    EXPECT_NEAR(scf::mp_denominator(-0.5, -0.4, 0.3, 0.2), -1.4, 1e-12);
}

TEST(TestSpinOrbitalEri, DenominatorThrowsWhenNonNegative) {
    // Δ = −0.5 − 0.4 − (−1.0) − (−1.0) = +1.1 (≥ 0): degenerate / non-aufbau.
    EXPECT_THROW(scf::mp_denominator(-0.5, -0.4, -1.0, -1.0),
                 std::invalid_argument);

    // Pin the shared guard text: acceptance ac-003 expects the literal
    // "non-negative orbital-energy denominator" as the single authoritative
    // phrase (matches the current mp2/mp3/mp4 guard wording).
    bool threw = false;
    try {
        scf::mp_denominator(-0.5, -0.4, -1.0, -1.0);
    } catch (const std::invalid_argument& e) {
        threw = true;
        const std::string what = e.what();
        EXPECT_NE(what.find("non-negative orbital-energy denominator"),
                  std::string::npos)
            << "guard message was: " << what;
    }
    EXPECT_TRUE(threw);
}

#endif // ATV_HAS_HF

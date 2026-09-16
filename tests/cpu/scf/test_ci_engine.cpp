/// @file tests/cpu/scf/test_ci_engine.cpp
/// @brief Unit checks for the hoisted determinant-CI engine
///        (internal/ci_engine.h, namespace ATV::cpu::scf).
///
/// scf-review-03-posthf-dedup unified the CI machinery duplicated between
/// fci.cpp and casscf.cpp into
/// include/atomiverse/cpu/scf/internal/ci_engine.h, taking the fci.cpp
/// Slater-Condon evaluators as CANONICAL. This test pins those canonical
/// evaluators against hand-computed matrix elements (written RED-first;
/// landed and GREEN).
///
/// Convention judge (test CiElementOppositeSpinDouble): the canonical fci
/// evaluator returns the opposite-spin double element as (pa ra|qb sb) =
/// g(pa,ra,qb,sb). The old casscf fork returned g(ra,pa,sb,qb) instead, which
/// is equal ONLY when the integral tensor obeys real (pq|rs)=(qp|sr) symmetry.
/// The symmetric g used here pins the canonical value; the fork discrepancy is
/// flagged separately in the tester report and gated by the equivalence
/// regression (regressions/scf/ci-engine-equiv-h2-sto3g).

#ifdef ATV_HAS_HF

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/internal/ci_engine.h"

namespace {

namespace scf = ATV::cpu::scf;

// 2-spatial-orbital / 2-electron hand case.
//   h = [[-1.0, 0.1], [0.1, -0.5]]
xt::xtensor<double, 2> make_h2() {
    xt::xtensor<double, 2> h = xt::zeros<double>({2, 2});
    h(0, 0) = -1.0;
    h(1, 1) = -0.5;
    h(0, 1) = 0.1;
    h(1, 0) = 0.1;
    return h;
}

// Symmetric chemist ERIs (pq|rs) = g(p,q,r,s):
//   (00|00)=0.6  (11|11)=0.5
//   J01 = (00|11) = (11|00) = 0.3
//   K01 = (01|01) = (10|10) = (01|10) = (10|01) = 0.1
//   (00|01)&perms = 0.15    (11|01)&perms = 0.05
xt::xtensor<double, 4> make_g2() {
    xt::xtensor<double, 4> g = xt::zeros<double>({2, 2, 2, 2});
    g(0, 0, 0, 0) = 0.6;
    g(1, 1, 1, 1) = 0.5;
    // J01 (Coulomb)
    g(0, 0, 1, 1) = 0.3;
    g(1, 1, 0, 0) = 0.3;
    // K01 (exchange) — 4 symmetric permutations
    g(0, 1, 0, 1) = 0.1;
    g(1, 0, 1, 0) = 0.1;
    g(0, 1, 1, 0) = 0.1;
    g(1, 0, 0, 1) = 0.1;
    // (00|01) family
    g(0, 0, 0, 1) = 0.15;
    g(0, 0, 1, 0) = 0.15;
    g(0, 1, 0, 0) = 0.15;
    g(1, 0, 0, 0) = 0.15;
    // (11|01) family
    g(1, 1, 0, 1) = 0.05;
    g(1, 1, 1, 0) = 0.05;
    g(0, 1, 1, 1) = 0.05;
    g(1, 0, 1, 1) = 0.05;
    return g;
}

constexpr double kVnuc = 0.7;

// Bitstrings: bit r set == spatial orbital r occupied in that spin channel.
constexpr std::uint64_t kOrb0 = 0b01ull; // one electron in orbital 0
constexpr std::uint64_t kOrb1 = 0b10ull; // one electron in orbital 1

} // namespace

// (a) diagonal element of |0a 0b> (both electrons in orbital 0).
//     E = v_nuc + 2 h00 + (00|00) = 0.7 - 2.0 + 0.6 = -0.7
TEST(TestCiEngine, DiagonalElementGroundDeterminant) {
    const double e =
        scf::diagonal_element(kOrb0, kOrb0, make_h2(), make_g2(), kVnuc);
    EXPECT_NEAR(e, -0.7, 1e-12);

    // ci_element must dispatch the zero-excitation case to the same value.
    const double ce = scf::ci_element(kOrb0, kOrb0, kOrb0, kOrb0, make_h2(),
                                      make_g2(), kVnuc);
    EXPECT_NEAR(ce, -0.7, 1e-12);
}

// (b) single-excitation element WITH a Jordan-Wigner sign flip.
//     3-orbital source string {0,1} -> {1,2}: the 0->2 hop crosses occupied
//     orbital 1, so the JW parity is -1. Opposite channel empty.
//     v = h02 + (0 2|1 1) - (0 1|1 2) = 0.1 + 0.4 - 0.2 = 0.3
//     result = sign(-1) * 0.3 = -0.3
TEST(TestCiEngine, SingleElementJordanWignerSignFlip) {
    xt::xtensor<double, 2> h = xt::zeros<double>({3, 3});
    h(0, 2) = 0.1;
    xt::xtensor<double, 4> g = xt::zeros<double>({3, 3, 3, 3});
    g(0, 2, 1, 1) = 0.4; // same-spin spectator Coulomb (0 2|1 1)
    g(0, 1, 1, 2) = 0.2; // same-spin spectator exchange (0 1|1 2)

    const std::uint64_t same_i = 0b011ull; // orbitals 0,1 occupied (det I)
    const std::uint64_t same_j = 0b110ull; // orbitals 1,2 occupied (det J)
    const std::uint64_t other = 0ull;      // opposite spin channel empty

    const double e = scf::single_element(same_i, same_j, other, h, g);
    EXPECT_NEAR(e, -0.3, 1e-12);
}

// (c) opposite-spin double excitation via ci_element dispatch.
//     I = |0a 0b>, J = |1a 1b>: both channels excite 0->1. Only the direct
//     Coulomb term survives; canonical (fci) value is (pa ra|qb sb) =
//     (0 1|0 1) = g(0,1,0,1) = K01 = 0.1. Both JW signs are +1.
TEST(TestCiEngine, CiElementOppositeSpinDouble) {
    const std::uint64_t ai = kOrb0, bi = kOrb0; // bra I
    const std::uint64_t aj = kOrb1, bj = kOrb1; // ket J

    const double e =
        scf::ci_element(ai, bi, aj, bj, make_h2(), make_g2(), kVnuc);
    EXPECT_NEAR(e, 0.1, 1e-12);

    // Direct evaluator agrees and pins the canonical g(pa,ra,qb,sb) ordering.
    const double ed =
        scf::opposite_spin_double_element(ai, aj, bi, bj, make_g2());
    EXPECT_NEAR(ed, 0.1, 1e-12);
}

// (basics) single alpha excitation with NO sign flip, via ci_element.
//     I = |0a 0b>, J = |1a 0b>: alpha 0->1, beta spectator in orbital 0.
//     v = h01 + (0 1|0 0)_opposite = 0.1 + 0.15 = 0.25; JW sign +1.
TEST(TestCiEngine, CiElementSingleExcitation) {
    const std::uint64_t ai = kOrb0, bi = kOrb0;
    const std::uint64_t aj = kOrb1, bj = kOrb0;

    const double e =
        scf::ci_element(ai, bi, aj, bj, make_h2(), make_g2(), kVnuc);
    EXPECT_NEAR(e, 0.25, 1e-12);
}

// (d) combinatorics: choose and enumerate_strings.
TEST(TestCiEngine, ChooseAndEnumerateStrings) {
    EXPECT_EQ(scf::choose(4, 2), 6);
    EXPECT_EQ(scf::choose(2, 1), 2);
    EXPECT_EQ(scf::choose(1, 1), 1);
    EXPECT_EQ(scf::choose(4, 0), 1);
    EXPECT_EQ(scf::choose(4, 5), 0); // k > n -> 0 (range guard, no fallback)

    const std::vector<std::uint64_t> s = scf::enumerate_strings(4, 2);
    ASSERT_EQ(s.size(), static_cast<std::size_t>(6));
    for (std::size_t i = 0; i < s.size(); ++i) {
        EXPECT_EQ(__builtin_popcountll(s[i]), 2);
        if (i > 0)
            EXPECT_LT(s[i - 1], s[i]); // ascending integer order
    }
}

// (e) Jordan-Wigner single-hop parity sign.
TEST(TestCiEngine, JordanWignerSingleSign) {
    // 0 -> 1, nothing occupied strictly between -> +1.
    EXPECT_EQ(scf::jw_single_sign(0b001ull, 0, 1), 1);
    // 0 -> 2 with orbital 1 occupied in the source -> one crossing -> -1.
    EXPECT_EQ(scf::jw_single_sign(0b011ull, 0, 2), -1);
    // 0 -> 2 with orbital 1 EMPTY in the source -> +1.
    EXPECT_EQ(scf::jw_single_sign(0b101ull, 0, 2), 1);
    // 0 -> 3 with orbital 1 occupied, orbital 2 empty -> one crossing -> -1.
    EXPECT_EQ(scf::jw_single_sign(0b1011ull, 0, 3), -1);
}

#endif // ATV_HAS_HF

/// @file tests/core/test_real_harmonics.cpp
/// @brief Arbitrary-l real solid (Use A) / normalized real spherical (Use B)
///        harmonic generators vs the verified l <= 3 closed forms + orthonormality.

#include <array>
#include <cmath>
#include <map>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/real_harmonics.h"

using ATV::HarmTerm;
using ATV::real_solid_harmonics;
using ATV::real_spherical_harmonics_normalized;
using ATV::RealHarmonic;
using ATV::sphere_monomial_integral;

namespace {

using Key = std::array<int, 3>;

// Map a harmonic to {(lx,ly,lz) -> coeff} for shape comparison.
std::map<Key, double> as_map(const RealHarmonic& h) {
    std::map<Key, double> m;
    for (const auto& t : h)
        m[{t.lx, t.ly, t.lz}] += t.coeff;
    return m;
}

// L2(S^2) inner product of two r^l-scaled harmonics of the same degree.
double sphere_dot(const RealHarmonic& a, const RealHarmonic& b) {
    double s = 0.0;
    for (const auto& ta : a)
        for (const auto& tb : b)
            s += ta.coeff * tb.coeff
                 * sphere_monomial_integral(ta.lx + tb.lx, ta.ly + tb.ly,
                                            ta.lz + tb.lz);
    return s;
}

// True if `h` is proportional to the reference monomial map (Use A: scale-free).
bool proportional_to(const RealHarmonic& h, const std::map<Key, double>& ref) {
    auto hm = as_map(h);
    // drop ~0 terms
    std::map<Key, double> hc;
    for (auto& kv : hm)
        if (std::abs(kv.second) > 1e-10)
            hc[kv.first] = kv.second;
    if (hc.size() != ref.size())
        return false;
    double ratio = 0.0;
    bool have = false;
    for (const auto& kv : ref) {
        auto it = hc.find(kv.first);
        if (it == hc.end())
            return false;
        const double r = it->second / kv.second;
        if (!have) {
            ratio = r;
            have = true;
        } else if (std::abs(r - ratio) > 1e-9) {
            return false;
        }
    }
    return have && std::abs(ratio) > 1e-12;
}

// Find a harmonic in the set proportional to ref; return true if exactly one.
bool set_contains_shape(const std::vector<RealHarmonic>& set,
                        const std::map<Key, double>& ref) {
    int n = 0;
    for (const auto& h : set)
        if (proportional_to(h, ref))
            ++n;
    return n == 1;
}

} // namespace

// ---- Use A: unnormalized real solid harmonics reproduce the canonical sets ----

TEST(RealHarmonics, SolidS) {
    auto s = real_solid_harmonics(0);
    ASSERT_EQ(s.size(), 1u);
    EXPECT_TRUE(set_contains_shape(s, {{{0, 0, 0}, 1.0}}));
}

TEST(RealHarmonics, SolidP) {
    auto p = real_solid_harmonics(1);
    ASSERT_EQ(p.size(), 3u);
    EXPECT_TRUE(set_contains_shape(p, {{{1, 0, 0}, 1.0}}));
    EXPECT_TRUE(set_contains_shape(p, {{{0, 1, 0}, 1.0}}));
    EXPECT_TRUE(set_contains_shape(p, {{{0, 0, 1}, 1.0}}));
}

TEST(RealHarmonics, SolidD) {
    auto d = real_solid_harmonics(2);
    ASSERT_EQ(d.size(), 5u);
    EXPECT_TRUE(set_contains_shape(d, {{{1, 1, 0}, 1.0}})); // xy
    EXPECT_TRUE(set_contains_shape(d, {{{0, 1, 1}, 1.0}})); // yz
    EXPECT_TRUE(set_contains_shape(d, {{{1, 0, 1}, 1.0}})); // xz
    EXPECT_TRUE(set_contains_shape(
        d, {{{2, 0, 0}, 1.0}, {{0, 2, 0}, -1.0}})); // x^2-y^2
    EXPECT_TRUE(set_contains_shape(d, {{{0, 0, 2}, 2.0},
                                       {{2, 0, 0}, -1.0},
                                       {{0, 2, 0}, -1.0}})); // 2z^2-x^2-y^2
}

TEST(RealHarmonics, SolidF) {
    auto f = real_solid_harmonics(3);
    ASSERT_EQ(f.size(), 7u);
    EXPECT_TRUE(set_contains_shape(
        f, {{{3, 0, 0}, 1.0}, {{1, 2, 0}, -3.0}})); // x^3-3xy^2
    EXPECT_TRUE(set_contains_shape(
        f, {{{2, 1, 0}, 3.0}, {{0, 3, 0}, -1.0}})); // 3x^2y-y^3
    EXPECT_TRUE(set_contains_shape(
        f, {{{2, 0, 1}, 1.0}, {{0, 2, 1}, -1.0}}));         // z(x^2-y^2)
    EXPECT_TRUE(set_contains_shape(f, {{{1, 1, 1}, 1.0}})); // xyz
    // x(5z^2-r^2) = 4xz^2 - x^3 - xy^2  (since 5z^2-r^2 = 5z^2-x^2-y^2-z^2)
    EXPECT_TRUE(set_contains_shape(
        f, {{{1, 0, 2}, 4.0}, {{3, 0, 0}, -1.0}, {{1, 2, 0}, -1.0}}));
    EXPECT_TRUE(set_contains_shape(
        f, {{{0, 1, 2}, 4.0}, {{0, 3, 0}, -1.0}, {{2, 1, 0}, -1.0}}));
    // z(5z^2-3r^2) = 2z^3 - 3x^2z - 3y^2z
    EXPECT_TRUE(set_contains_shape(
        f, {{{0, 0, 3}, 2.0}, {{2, 0, 1}, -3.0}, {{0, 2, 1}, -3.0}}));
}

// ---- Use B: normalized real spherical harmonics — exact prefactors ----

// Returns the single-monomial prefactor if `h` is exactly c * (lx,ly,lz).
double mono_prefactor(const RealHarmonic& h, int lx, int ly, int lz) {
    if (h.size() != 1u)
        return 0.0 / 0.0;
    const auto& t = h[0];
    if (t.lx != lx || t.ly != ly || t.lz != lz)
        return 0.0 / 0.0;
    return t.coeff;
}

bool has_normalized_single(const std::vector<RealHarmonic>& set, int lx, int ly,
                           int lz, double expect) {
    for (const auto& h : set) {
        const double c = mono_prefactor(h, lx, ly, lz);
        if (std::isfinite(c) && std::abs(std::abs(c) - std::abs(expect)) < 1e-9)
            return true;
    }
    return false;
}

TEST(RealHarmonics, NormalizedSP) {
    auto s = real_spherical_harmonics_normalized(0);
    ASSERT_EQ(s.size(), 1u);
    EXPECT_NEAR(std::abs(s[0][0].coeff), 1.0 / std::sqrt(4.0 * M_PI),
                1e-12); // Y00
    auto p = real_spherical_harmonics_normalized(1);
    const double y1 = std::sqrt(3.0 / (4.0 * M_PI));
    EXPECT_TRUE(has_normalized_single(p, 1, 0, 0, y1)); // x
    EXPECT_TRUE(has_normalized_single(p, 0, 1, 0, y1)); // y
    EXPECT_TRUE(has_normalized_single(p, 0, 0, 1, y1)); // z
}

TEST(RealHarmonics, NormalizedD_xyPrefactor) {
    // The critical case: d_xy/d_xz/d_yz carry (1/2)sqrt(15/pi), NOT (1/4)sqrt(15/pi).
    auto d = real_spherical_harmonics_normalized(2);
    const double half = 0.5 * std::sqrt(15.0 / M_PI);
    EXPECT_TRUE(has_normalized_single(d, 1, 1, 0, half)); // xy
    EXPECT_TRUE(has_normalized_single(d, 0, 1, 1, half)); // yz
    EXPECT_TRUE(has_normalized_single(d, 1, 0, 1, half)); // xz
    // And NOT the (1/4) value.
    const double quarter = 0.25 * std::sqrt(15.0 / M_PI);
    EXPECT_FALSE(has_normalized_single(d, 1, 1, 0, quarter));
}

// ---- Use B: orthonormality on the sphere for l = 0..4 ----

TEST(RealHarmonics, OrthonormalThroughG) {
    for (int l = 0; l <= 4; ++l) {
        auto set = real_spherical_harmonics_normalized(l);
        ASSERT_EQ(set.size(), static_cast<std::size_t>(2 * l + 1));
        for (std::size_t a = 0; a < set.size(); ++a) {
            for (std::size_t b = 0; b < set.size(); ++b) {
                const double dot = sphere_dot(set[a], set[b]);
                const double expect = (a == b) ? 1.0 : 0.0;
                EXPECT_NEAR(dot, expect, 1e-9)
                    << "l=" << l << " a=" << a << " b=" << b;
            }
        }
    }
}

TEST(RealHarmonics, SphereMonomialIntegralKnownValues) {
    EXPECT_NEAR(sphere_monomial_integral(0, 0, 0), 4.0 * M_PI, 1e-12);
    EXPECT_NEAR(sphere_monomial_integral(2, 0, 0), 4.0 * M_PI / 3.0, 1e-12);
    EXPECT_NEAR(sphere_monomial_integral(1, 1, 0), 0.0, 1e-12); // odd
    EXPECT_NEAR(sphere_monomial_integral(2, 2, 0), 4.0 * M_PI / 15.0, 1e-12);
}

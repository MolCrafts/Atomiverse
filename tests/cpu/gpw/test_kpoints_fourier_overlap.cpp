/// @file test_kpoints_fourier_overlap.cpp
/// @brief Batched cross-k Gaussian Fourier operators: analytic finite image
///        sums and translation/conjugation covariance, no SCF.

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/internal/kpoints.h"

namespace {
using ATV::cpu::gpw::GridBasisFunction;
using ATV::cpu::gpw::GridPrimitive;
using ATV::cpu::gpw::internal::build_fourier_overlap_links;
using ATV::cpu::gpw::internal::FourierOverlapLink;
using Cplx = std::complex<double>;
constexpr double two_pi = 6.283185307179586476925286766559;

// Closed s/s Gaussian characteristic function for a ket shifted by T.
Cplx ss_image(const GridPrimitive& a, const GridPrimitive& b, const double T[3],
              const std::array<double, 3>& q) {
    const double p = a.alpha + b.alpha, mu = a.alpha * b.alpha / p;
    double distance2 = 0.0, q2 = 0.0, phase = 0.0;
    for (int d = 0; d < 3; ++d) {
        const double ket = b.center[d] + T[d];
        const double displacement = a.center[d] - ket;
        distance2 += displacement * displacement;
        q2 += q[d] * q[d];
        phase += q[d] * (a.alpha * a.center[d] + b.alpha * ket) / p;
    }
    return a.coeff * b.coeff * std::pow(M_PI / p, 1.5)
           * std::exp(-mu * distance2 - q2 / (4.0 * p))
           * Cplx{std::cos(phase), -std::sin(phase)};
}

TEST(GpwFourierLinks, EqualAndDistinctWavevectorsMatchAnalyticImageSums) {
    const std::vector<GridBasisFunction> basis{
        {{0.7, {0.2, -0.1, 0.3}, {0, 0, 0}, 0.8},
         {1.4, {0.2, -0.1, 0.3}, {0, 0, 0}, -0.2}},
        {{0.9, {-0.4, 0.3, 0.1}, {0, 0, 0}, 0.6},
         {1.8, {-0.4, 0.3, 0.1}, {0, 0, 0}, 0.3}}};
    const double L[3] = {2.8, 3.4, 4.1};
    const int m[3] = {1, 1, 1};
    const std::vector<double> radii{20.0, 20.0};
    const std::array<double, 3> q{0.37, -0.24, 0.19};
    const std::array<double, 3> near_q{0.370000001, -0.24, 0.19};
    const std::array<double, 3> opposite{-q[0], -q[1], -q[2]};
    // Nonadjacent equal q groups, distinct ket phases, a nearby q that must
    // never be merged by a tolerance, opposite q, and the ordinary q=0 limit.
    const std::vector<FourierOverlapLink> links{
        {{0.0, 0.0, 0.0}, q},
        {{0.25, -0.3, 0.1}, opposite},
        {{-0.25, 0.2, -0.4}, q},
        {{0.1, 0.2, 0.3}, near_q},
        {{-0.15, 0.3, -0.25}, opposite},
        {{0.125, -0.2, 0.4}, q},
        {{0.125, -0.2, 0.4}, {0, 0, 0}}};
    const auto batch =
        build_fourier_overlap_links(basis, radii, L, m, 0.0, links);
    ASSERT_EQ(batch.size(), links.size());
    for (std::size_t link = 0; link < links.size(); ++link) {
        ASSERT_EQ(batch[link].size(), 4u);
        const auto single = build_fourier_overlap_links(
            basis, radii, L, m, 0.0,
            std::span<const FourierOverlapLink>(&links[link], 1));
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                Cplx want{0, 0};
                for (int tx = -1; tx <= 1; ++tx)
                    for (int ty = -1; ty <= 1; ++ty)
                        for (int tz = -1; tz <= 1; ++tz) {
                            const double T[3] = {tx * L[0], ty * L[1],
                                                 tz * L[2]};
                            const auto& k = links[link].ket_frac;
                            const double angle =
                                two_pi * (k[0] * tx + k[1] * ty + k[2] * tz);
                            const Cplx bloch{std::cos(angle), std::sin(angle)};
                            for (const auto& a : basis[i])
                                for (const auto& b : basis[j])
                                    want +=
                                        bloch
                                        * ss_image(a, b, T, links[link].q_bohr);
                        }
                const std::size_t index = static_cast<std::size_t>(2 * i + j);
                EXPECT_NEAR(std::abs(batch[link][index] - want), 0.0, 2e-13)
                    << "link " << link << " element " << index;
                // Supplementary split-vs-batch identity; the absolute oracle
                // remains the closed characteristic-function image sum above.
                EXPECT_EQ(batch[link][index], single[0][index]);
            }
    }
}

TEST(GpwFourierLinks, MixedAngularTranslationAndConjugationCovariance) {
    std::vector<GridBasisFunction> basis{
        {{0.7, {0.2, -0.1, 0.3}, {1, 0, 0}, 0.8}},
        {{0.9, {-0.4, 0.3, 0.1}, {2, 0, 0}, 0.6},
         {0.9, {-0.4, 0.3, 0.1}, {0, 2, 0}, -0.6}}};
    const double L[3] = {2.8, 3.4, 4.1};
    const int m[3] = {1, 1, 0};
    const std::vector<double> radii{20.0, 20.0};
    const std::array<double, 3> q{0.37, -0.24, 0.19};
    const std::vector<FourierOverlapLink> links{
        {{0.25, -0.3, 0.1}, q},
        {{-0.15, 0.2, -0.4}, q},
        {{-0.25, 0.3, -0.1}, {-q[0], -q[1], -q[2]}}};
    const auto original =
        build_fourier_overlap_links(basis, radii, L, m, 0.0, links);
    const double shift[3] = {0.13, -0.27, 0.19};
    for (auto& function : basis)
        for (auto& primitive : function)
            for (int d = 0; d < 3; ++d)
                primitive.center[d] += shift[d];
    const auto translated =
        build_fourier_overlap_links(basis, radii, L, m, 0.0, links);
    for (std::size_t link = 0; link < links.size(); ++link) {
        double angle = 0;
        for (int d = 0; d < 3; ++d)
            angle += links[link].q_bohr[d] * shift[d];
        const Cplx factor{std::cos(angle), -std::sin(angle)};
        for (std::size_t entry = 0; entry < 4; ++entry)
            EXPECT_NEAR(std::abs(translated[link][entry]
                                 - factor * original[link][entry]),
                        0.0, 2e-13);
    }
    for (std::size_t entry = 0; entry < 4; ++entry)
        EXPECT_NEAR(
            std::abs(original[2][entry] - std::conj(original[0][entry])), 0.0,
            2e-13);
}

TEST(GpwFourierLinks, InvalidDomainAndWorkerExceptionsPropagate) {
    std::vector<GridBasisFunction> basis{{{0.7, {0, 0, 0}, {0, 0, 0}, 1.0}}};
    const double L[3] = {2.8, 3.4, 4.1};
    const int m[3] = {0, 0, 0};
    const std::vector<FourierOverlapLink> links{
        {{0.25, 0, 0}, {0.2, 0.1, -0.3}}};
    EXPECT_THROW(build_fourier_overlap_links(basis, {}, L, m, 0.0, links),
                 std::invalid_argument);
    auto invalid = links;
    invalid[0].q_bohr[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(build_fourier_overlap_links(basis, {20.0}, L, m, 0.0, invalid),
                 std::invalid_argument);
    // A primitive-domain error originates on an AO-row worker, and must be
    // rethrown on the caller thread rather than terminating an OpenMP region.
    basis[0][0].alpha = -0.7;
    EXPECT_THROW(build_fourier_overlap_links(basis, {20.0}, L, m, 0.0, links),
                 std::invalid_argument);
}
} // namespace

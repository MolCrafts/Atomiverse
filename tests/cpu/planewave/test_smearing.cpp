#include "atomiverse/cpu/planewave/internal/smearing.h"
#include "smearing_qe_pins.h"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using ATV::AccReal;
using namespace ATV::cpu::pw::internal;

TEST(PWSmearing, ActualQe76LocalOccupationAndEntropyPins) {
    for (const auto& p : qe_smearing_pin::points) {
        const auto kind = static_cast<SmearKind>(p.kind);
        EXPECT_NEAR(smear_f_spinless(p.x, kind, p.order), p.f, 2e-13)
            << "kind=" << p.kind << " order=" << p.order << " x=" << p.x;
        EXPECT_NEAR(smear_entropy_spinor(p.x, 0, 1, kind, p.order), p.s, 2e-13)
            << "kind=" << p.kind << " order=" << p.order << " x=" << p.x;
    }
}

TEST(PWSmearing, ActualQe76FixedMuWeightedSpectrumPins) {
    for (const auto& p : qe_smearing_pin::spectra) {
        const std::vector<AccReal> eig(p.evals.begin(), p.evals.end());
        const std::vector<AccReal> weights(eig.size(), 1);
        const auto fill =
            fill_occupations(eig, weights, p.mu, p.width,
                             static_cast<SmearKind>(p.kind), p.order, p.degen);
        EXPECT_NEAR(fill.n_occ, p.n, 2e-12);
        EXPECT_NEAR(fill.entropy, p.s, 2e-12)
            << "spectrum=" << p.id << " kind=" << p.kind
            << " order=" << p.order;
    }
}

TEST(PWSmearing, ActualQe76FermiSearchAndInsulatingPlateau) {
    for (const auto& p : qe_smearing_pin::spectra) {
        const std::vector<AccReal> eig(p.evals.begin(), p.evals.end());
        const std::vector<AccReal> weights(eig.size(), 1);
        const auto kind = static_cast<SmearKind>(p.kind);
        const auto mu = find_fermi_level(eig, weights, p.target, p.width, kind,
                                         p.order, p.degen);
        const auto fill =
            fill_occupations(eig, weights, mu, p.width, kind, p.order, p.degen);
        EXPECT_NEAR(fill.n_occ, p.target, 1.5e-10);
        if (p.id == 1) {
            // A chemical potential in a well-filled gap is not unique. The
            // true QE result and native root both have negligible entropy;
            // never drive an already-correct N plateau to a band edge.
            EXPECT_GT(mu, eig[1] + 4 * p.width);
            EXPECT_LT(mu, eig[2] - 4 * p.width);
            EXPECT_LT(std::abs(fill.entropy), 5e-8);
            EXPECT_LT(std::abs(p.s), 5e-8);
        } else {
            EXPECT_NEAR(mu, p.mu, 1e-8);
            EXPECT_NEAR(fill.entropy, p.s, 5e-8);
        }
    }
}

TEST(PWSmearing, MethfesselPaxtonSignedQuasiOccupationsAndEntropy) {
    const auto mp = SmearKind::methfessel_paxton;
    EXPECT_LT(smear_f_spinless(2, mp, 1), 0);
    EXPECT_GT(smear_f_spinless(-2, mp, 1), 1);
    EXPECT_LT(smear_entropy_spinor(2, 0, 1, mp, 1), 0);
    EXPECT_NEAR(smear_entropy_spinor(0, 0, 1, mp, 1),
                0.25 / std::sqrt(std::acos(-1.0)), 1e-15);
}

TEST(PWSmearing,
     EntropyDerivativeEqualsReducedEnergyTimesOccupationDerivative) {
    for (auto kind : {SmearKind::fermi_dirac, SmearKind::gaussian,
                      SmearKind::methfessel_paxton})
        for (int order : {1, 2, 3, 5})
            for (double x : {-3., -2., -.6, 0., .4, 1.3, 2., 3.}) {
                constexpr double h = 2e-6;
                const double df = (smear_f_spinless(x + h, kind, order)
                                   - smear_f_spinless(x - h, kind, order))
                                  / (2 * h);
                const double ds =
                    (smear_entropy_spinor(x + h, 0, 1, kind, order)
                     - smear_entropy_spinor(x - h, 0, 1, kind, order))
                    / (2 * h);
                EXPECT_NEAR(ds, x * df, 2e-8)
                    << "kind=" << static_cast<int>(kind) << " order=" << order
                    << " x=" << x;
            }
}

TEST(PWSmearing, GrandPotentialDerivativeIsSignedBandOccupation) {
    const std::vector<AccReal> eig = {-.2, 0., .2, .7};
    const std::vector<AccReal> weights = {.3, .2, .1, .4};
    constexpr double mu = .1234, width = .1, h = 2e-6;
    for (auto kind : {SmearKind::fermi_dirac, SmearKind::gaussian,
                      SmearKind::methfessel_paxton}) {
        const auto omega = [&](const std::vector<AccReal>& e) {
            const auto fill =
                fill_occupations(e, weights, mu, width, kind, 1, 2);
            double value = -width * fill.entropy - mu * fill.n_occ;
            for (std::size_t j = 0; j < e.size(); ++j)
                value += weights[j] * e[j] * fill.occ[j];
            return value;
        };
        const auto fill = fill_occupations(eig, weights, mu, width, kind, 1, 2);
        for (std::size_t i = 0; i < eig.size(); ++i) {
            auto plus = eig, minus = eig;
            plus[i] += h;
            minus[i] -= h;
            EXPECT_NEAR((omega(plus) - omega(minus)) / (2 * h),
                        weights[i] * fill.occ[i], 2e-8);
        }
    }
}

TEST(PWSmearing, FermiDiracExactEndpointsHaveZeroEntropyWithoutNan) {
    for (double x : {-1e300, -100., 100., 1e300}) {
        const double s = smear_entropy_band(x, 0, 1, SmearKind::fermi_dirac);
        EXPECT_TRUE(std::isfinite(s));
        EXPECT_DOUBLE_EQ(s, 0);
    }
    const auto fill = fill_occupations({-100., 100.}, {1., 1.}, 0, 1,
                                       SmearKind::fermi_dirac, 1, 2);
    EXPECT_DOUBLE_EQ(fill.n_occ, 2);
    EXPECT_DOUBLE_EQ(fill.entropy, 0);
}

TEST(PWSmearing, ExtremeGaussianAndMpTailsAreFiniteExactLimits) {
    for (auto kind : {SmearKind::gaussian, SmearKind::methfessel_paxton})
        for (int order : {1, 2, 10, 50})
            for (double x : {-1e300, -100., 100., 1e300}) {
                EXPECT_DOUBLE_EQ(smear_f_spinless(x, kind, order),
                                 x < 0 ? 1. : 0.);
                EXPECT_DOUBLE_EQ(smear_entropy_spinor(x, 0, 1, kind, order),
                                 0.);
            }
}

TEST(PWSmearing, ValidatesUnsupportedOrderSpectrumAndBandCapacity) {
    EXPECT_THROW(smear_f_spinless(1e300, SmearKind::methfessel_paxton, 0),
                 std::invalid_argument);
    EXPECT_THROW(smear_entropy_band(0, 0, 1, SmearKind::methfessel_paxton, 0),
                 std::invalid_argument);
    EXPECT_THROW(find_fermi_level({0, 1}, {1, 1}, 5, .1, SmearKind::gaussian),
                 std::invalid_argument);
    EXPECT_THROW(find_fermi_level({0, 1}, {1, -1}, 1, .1, SmearKind::gaussian),
                 std::invalid_argument);
    EXPECT_THROW(fill_occupations({0, 1}, {1}, 0, .1, SmearKind::gaussian),
                 std::invalid_argument);
}

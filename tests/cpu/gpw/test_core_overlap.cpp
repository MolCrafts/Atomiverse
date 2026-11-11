#include "atomiverse/cpu/gpw/internal/core_overlap.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {
using namespace ATV::gaussian_core;
using ATV::cpu::gpw::internal::add_core_overlap_forces;
using ATV::cpu::gpw::internal::core_overlap_energy;
using ATV::cpu::gpw::internal::core_overlap_virial;
constexpr int aa[6] = {0, 1, 2, 0, 0, 1};
constexpr int bb[6] = {0, 1, 2, 1, 2, 2};
const std::array<double, 3> cell{2.7, 3.1, 3.7};
const std::vector<Site> sites{{{.13, .42, .27}, 1, .41},
                              {{1.24, 1.66, 2.16}, 4, .73}};
struct Observation {
    double energy = 0;
    std::vector<std::array<double, 3>> force;
    std::array<double, 6> virial{};
};
Observation observe(const Model& model, std::size_t n) {
    Observation out;
    out.energy = core_overlap_energy(model);
    out.force.resize(n);
    add_core_overlap_forces(model, out.force);
    out.virial = core_overlap_virial(model);
    return out;
}
// Independent ordered sum: all A/B and +/- images, with an explicit 1/2.
// It neither builds the production model nor calls its scalar radial law.
Observation ordered(const std::vector<Site>& input,
                    const std::array<double, 3>& lengths, int images,
                    int strain = -1, double h = 0) {
    Observation out;
    out.force.resize(input.size());
    for (std::size_t a = 0; a < input.size(); ++a)
        for (std::size_t b = 0; b < input.size(); ++b)
            for (int x = -images; x <= images; ++x)
                for (int y = -images; y <= images; ++y)
                    for (int z = -images; z <= images; ++z) {
                        if (a == b && x == 0 && y == 0 && z == 0)
                            continue;
                        const int image[3] = {x, y, z};
                        double d[3], r2 = 0;
                        for (int k = 0; k < 3; ++k)
                            d[k] = input[a].position[k] - input[b].position[k]
                                   - image[k] * lengths[k];
                        // F=I+epsilon acts on both the positions and the
                        // lattice image. Sigma stays fixed in atomic units.
                        if (strain >= 0) {
                            const double old[3] = {d[0], d[1], d[2]};
                            d[aa[strain]] += h * old[bb[strain]];
                            if (aa[strain] != bb[strain])
                                d[bb[strain]] += h * old[aa[strain]];
                        }
                        for (double v : d)
                            r2 += v * v;
                        const double r = std::sqrt(r2);
                        const double g =
                            1
                            / std::sqrt(2
                                        * (input[a].sigma * input[a].sigma
                                           + input[b].sigma * input[b].sigma));
                        const double q = input[a].charge * input[b].charge;
                        const double energy = q * std::erfc(g * r) / r;
                        const double c = q
                                         * (std::erfc(g * r) / (r * r * r)
                                            + 2 * g / std::sqrt(std::acos(-1.0))
                                                  * std::exp(-g * g * r2) / r2);
                        out.energy += .5 * energy;
                        if (a != b)
                            for (int k = 0; k < 3; ++k) {
                                out.force[a][k] += .5 * c * d[k];
                                out.force[b][k] -= .5 * c * d[k];
                            }
                        for (int k = 0; k < 6; ++k)
                            out.virial[k] += .5 * c * d[aa[k]] * d[bb[k]];
                    }
    return out;
}
void compare(const Observation& got, const Observation& expected,
             double tolerance = 3e-14) {
    EXPECT_NEAR(got.energy, expected.energy, tolerance);
    ASSERT_EQ(got.force.size(), expected.force.size());
    for (std::size_t a = 0; a < got.force.size(); ++a)
        for (int k = 0; k < 3; ++k)
            EXPECT_NEAR(got.force[a][k], expected.force[a][k], tolerance);
    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(got.virial[k], expected.virial[k], tolerance);
}

TEST(GpwCoreOverlap, CanonicalCountingMatchesOrderedMixedWidthImageSum) {
    const auto model = make_model(sites, cell, true);
    compare(observe(model, sites.size()), ordered(sites, cell, 8));
    const auto bounds = model.tail_bounds();
    EXPECT_LE(bounds.energy, 1e-14);
    EXPECT_LE(bounds.force, 1e-14);
    EXPECT_LE(bounds.virial, 1e-14);
}

TEST(GpwCoreOverlap, IndividualLatticeTranslationsPreserveAllOutputs) {
    auto moved = sites;
    const int shifts[2][3] = {{17, -23, 11}, {-31, 9, -15}};
    for (int a = 0; a < 2; ++a)
        for (int k = 0; k < 3; ++k)
            moved[a].position[k] += shifts[a][k] * cell[k];
    compare(observe(make_model(moved, cell, true), 2),
            observe(make_model(sites, cell, true), 2));
}

TEST(GpwCoreOverlap, SelfImagesHaveEnergyAndVirialWithoutAtomicForce) {
    const std::vector<Site> one{sites[1]};
    const auto periodic = observe(make_model(one, cell, true), 1);
    EXPECT_GT(periodic.energy, 0);
    for (int k = 0; k < 3; ++k) {
        EXPECT_GT(periodic.virial[k], 0);
        EXPECT_DOUBLE_EQ(periodic.force[0][k], 0);
    }
    const auto isolated = observe(make_model(one, cell, false), 1);
    EXPECT_DOUBLE_EQ(isolated.energy, 0);
    for (double w : isolated.virial)
        EXPECT_DOUBLE_EQ(w, 0);
}

TEST(GpwCoreOverlap, IsolatedUsesActualSeparationWithoutMinimumImage) {
    auto moved = sites;
    const int shifts[2][3] = {{17, -23, 11}, {-31, 9, -15}};
    for (int a = 0; a < 2; ++a)
        for (int k = 0; k < 3; ++k)
            moved[a].position[k] += shifts[a][k] * cell[k];
    const auto original = observe(make_model(sites, cell, false), 2);
    const auto translated = observe(make_model(moved, cell, false), 2);
    EXPECT_GT(original.energy, .001);
    EXPECT_LT(translated.energy, 1e-100);
    compare(original, ordered(sites, cell, 0));
    compare(translated, ordered(moved, cell, 0));
}

TEST(GpwCoreOverlap, AllForceAndVirialLanesMatchIndependentDifferenceSweep) {
    // Five-point derivatives remove the coarse h^2 truncation without relaxing
    // tolerances. At each h the independent ordered law supplies all energies.
    for (bool periodic : {false, true}) {
        const int images = periodic ? 8 : 0;
        const auto base = observe(make_model(sites, cell, periodic), 2);
        for (double h : {1e-3, 3e-4, 1e-4, 3e-5}) {
            SCOPED_TRACE(::testing::Message()
                         << "periodic=" << periodic << " h=" << h);
            for (int a = 0; a < 2; ++a)
                for (int k = 0; k < 3; ++k) {
                    auto plus = sites, minus = sites;
                    plus[a].position[k] += h;
                    minus[a].position[k] -= h;
                    const double fd = -(ordered(plus, cell, images).energy
                                        - ordered(minus, cell, images).energy)
                                      / (2 * h);
                    plus[a].position[k] += h;
                    minus[a].position[k] -= h;
                    const double fd2 = -(ordered(plus, cell, images).energy
                                         - ordered(minus, cell, images).energy)
                                       / (4 * h);
                    EXPECT_NEAR((4 * fd - fd2) / 3, base.force[a][k], 1e-9);
                }
            for (int k = 0; k < 6; ++k) {
                const double shear = k < 3 ? 1 : 2;
                const double fd =
                    -(ordered(sites, cell, images, k, h).energy
                      - ordered(sites, cell, images, k, -h).energy)
                    / (2 * h * shear);
                const double fd2 =
                    -(ordered(sites, cell, images, k, 2 * h).energy
                      - ordered(sites, cell, images, k, -2 * h).energy)
                    / (4 * h * shear);
                EXPECT_NEAR((4 * fd - fd2) / 3, base.virial[k], 1e-9);
                if (k < 3) {
                    // Rebuild production periodic image domains at strained
                    // diagonal cells too, including fixed-fractional atoms.
                    auto plus = sites, minus = sites;
                    auto lp = cell, lm = cell;
                    lp[k] *= 1 + h;
                    lm[k] *= 1 - h;
                    for (int a = 0; a < 2; ++a) {
                        plus[a].position[k] *= 1 + h;
                        minus[a].position[k] *= 1 - h;
                    }
                    const double rebuilt =
                        -(core_overlap_energy(make_model(plus, lp, periodic))
                          - core_overlap_energy(
                              make_model(minus, lm, periodic)))
                        / (2 * h);
                    EXPECT_NEAR(rebuilt, fd, 2e-11);
                }
            }
        }
    }
}

TEST(GpwCoreOverlap, SmallLengthScalePreservesTailBudgetsAndPhysicalScaling) {
    auto tiny = sites;
    auto small_cell = cell;
    for (auto& site : tiny) {
        site.sigma *= 1e-3;
        for (double& p : site.position)
            p *= 1e-3;
    }
    for (double& length : small_cell)
        length *= 1e-3;
    const auto model = make_model(tiny, small_cell, true);
    const auto bounds = model.tail_bounds();
    EXPECT_LE(bounds.energy, 1e-14);
    EXPECT_LE(bounds.force, 1e-14);
    EXPECT_LE(bounds.virial, 1e-14);
    auto scaled = observe(model, 2);
    scaled.energy *= 1e-3;
    for (auto& f : scaled.force)
        for (double& component : f)
            component *= 1e-6;
    for (double& w : scaled.virial)
        w *= 1e-3;
    compare(scaled, observe(make_model(sites, cell, true), 2));
}

TEST(GpwCoreOverlap, ShortRangeTailBoundsDominateAnIndependentLargerImageSum) {
    // A deliberately short one-atom cube leaves observable tails. Verify the
    // complete tensor norm, especially in Bohr scales below one.
    const std::array<double, 3> lengths{.0027, .0031, .0037};
    const std::vector<Site> one{{{0, 0, 0}, 4, .00073}};
    const auto large = ordered(one, lengths, 8);
    const auto short_sum = ordered(one, lengths, 0);
    const double gamma = 1 / (2 * one[0].sigma);
    const auto bound = detail::image_tail(gamma, 16, lengths, {0, 0, 0});
    EXPECT_LE(std::abs(large.energy - short_sum.energy), bound.energy);
    double norm2 = 0;
    for (int k = 0; k < 6; ++k) {
        const double d = large.virial[k] - short_sum.virial[k];
        norm2 += (k < 3 ? 1 : 2) * d * d;
    }
    EXPECT_LE(std::sqrt(norm2), bound.virial);
    // Self-image force is zero by translation invariance, independently of the
    // energy/virial tail; mixed-site force tails are bounded below.
    auto mixed = sites;
    mixed[1].position[2] = 1.8; // Each base separation is <= L_k/2.
    for (auto& site : mixed) {
        site.sigma *= 1e-3;
        for (double& p : site.position)
            p *= 1e-3;
    }
    const auto mixed_large = ordered(mixed, lengths, 8);
    const auto mixed_short = ordered(mixed, lengths, 0);
    const double mixed_gamma =
        1
        / std::sqrt(2
                    * (mixed[0].sigma * mixed[0].sigma
                       + mixed[1].sigma * mixed[1].sigma));
    const auto mixed_bound =
        detail::image_tail(mixed_gamma, 4, lengths, {0, 0, 0});
    for (int a = 0; a < 2; ++a) {
        double f2 = 0;
        for (int k = 0; k < 3; ++k) {
            const double d = mixed_large.force[a][k] - mixed_short.force[a][k];
            f2 += d * d;
        }
        EXPECT_LE(std::sqrt(f2), mixed_bound.force);
    }
}

TEST(GpwCoreOverlap, RejectsInvalidAndSingularInputsBeforeModelPublication) {
    auto bad = sites;
    bad[1].sigma = 0;
    EXPECT_THROW(make_model(bad, cell, true), std::invalid_argument);
    bad = sites;
    bad[1].charge = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(make_model(bad, cell, true), std::invalid_argument);
    bad = sites;
    for (int k = 0; k < 3; ++k)
        bad[1].position[k] = bad[0].position[k];
    EXPECT_THROW(make_model(bad, cell, true), std::invalid_argument);
    EXPECT_THROW(make_model(sites, {0, 1, 1}, true), std::invalid_argument);
    EXPECT_TRUE(make_model({}, cell, true).pairs().empty());
}
} // namespace

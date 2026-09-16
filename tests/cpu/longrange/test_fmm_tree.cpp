/// @file tests/longrange/test_fmm_tree.cpp
/// @brief Unit tests for adaptive-octree FmmTree vs direct O(N²).

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/longrange/fmm_tree.h"

using namespace ATV;
using ATV::longrange::FmmTree;

namespace {

AccReal direct_energy(const std::vector<Vec3>& pos,
                      const std::vector<AccReal>& q) {
    AccReal e = 0.0;
    const std::size_t N = pos.size();
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i + 1; j < N; ++j) {
            const AccReal r = (pos[i] - pos[j]).norm();
            if (r > 0.0)
                e += q[i] * q[j] / r;
        }
    return e;
}

void direct_potentials(const std::vector<Vec3>& pos,
                       const std::vector<AccReal>& q,
                       std::vector<AccReal>& phi) {
    const std::size_t N = pos.size();
    phi.assign(N, 0.0);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            if (i == j)
                continue;
            const AccReal r = (pos[i] - pos[j]).norm();
            if (r > 0.0)
                phi[i] += q[j] / r;
        }
}

} // namespace

TEST(FmmTree, RejectsBadCtorArgs) {
    EXPECT_THROW(FmmTree(0, 4), std::invalid_argument);
    EXPECT_THROW(FmmTree(-1, 4), std::invalid_argument);
    EXPECT_THROW(FmmTree(4, 0), std::invalid_argument);
    EXPECT_THROW(FmmTree(4, -2), std::invalid_argument);
}

TEST(FmmTree, SingleParticleZeroEnergy) {
    FmmTree tree(4, 4);
    std::vector<Vec3> pos = {Vec3{1, 2, 3}};
    std::vector<AccReal> q = {1.0};
    EXPECT_NEAR(tree.energy(pos, q), 0.0, 1e-15);
    std::vector<AccReal> phi;
    tree.potentials(pos, q, phi);
    ASSERT_EQ(phi.size(), 1u);
    EXPECT_NEAR(phi[0], 0.0, 1e-15);
}

TEST(FmmTree, TwoParticlesNearField) {
    // Two particles always fall into near field of a coarse tree.
    FmmTree tree(6, 8);
    std::vector<Vec3> pos = {Vec3{0, 0, 0}, Vec3{1, 0, 0}};
    std::vector<AccReal> q = {1.0, -1.0};
    const AccReal e_ref = direct_energy(pos, q);
    EXPECT_NEAR(tree.energy(pos, q), e_ref, 1e-12);
}

TEST(FmmTree, SingleLeafDegenerate) {
    // max_per_leaf large enough that root never splits → pure direct sum.
    FmmTree tree(4, 100);
    std::vector<Vec3> pos;
    std::vector<AccReal> q;
    for (int i = 0; i < 10; ++i) {
        pos.push_back(Vec3{0.1 * i, 0.05 * i, -0.02 * i});
        q.push_back((i % 2 == 0) ? 1.0 : -1.0);
    }
    EXPECT_NEAR(tree.energy(pos, q), direct_energy(pos, q), 1e-12);
    EXPECT_TRUE(tree.node_is_leaf(0));
}

TEST(FmmTree, RandomVsDirect) {
    const int N = 128;
    const int p = 8;
    const int leaf = 8;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<Vec3> pos(static_cast<std::size_t>(N));
    std::vector<AccReal> q(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        pos[static_cast<std::size_t>(i)] = Vec3{U(rng), U(rng), U(rng)};
        q[static_cast<std::size_t>(i)] = U(rng);
    }

    FmmTree tree(p, leaf);
    const AccReal e_fmm = tree.energy(pos, q);
    const AccReal e_ref = direct_energy(pos, q);
    const AccReal rel =
        std::abs(e_fmm - e_ref) / std::max(std::abs(e_ref), 1e-30);
    EXPECT_LT(rel, 1e-6) << "e_fmm=" << e_fmm << " e_ref=" << e_ref;

    std::vector<AccReal> phi_fmm, phi_ref;
    tree.potentials(pos, q, phi_fmm);
    direct_potentials(pos, q, phi_ref);
    AccReal max_abs = 0.0;
    AccReal max_rel = 0.0;
    for (int i = 0; i < N; ++i) {
        const AccReal dphi = std::abs(phi_fmm[static_cast<std::size_t>(i)]
                                      - phi_ref[static_cast<std::size_t>(i)]);
        max_abs = std::max(max_abs, dphi);
        const AccReal denom =
            std::max(std::abs(phi_ref[static_cast<std::size_t>(i)]), 1e-8);
        max_rel = std::max(max_rel, dphi / denom);
    }
    // Energy is the tight pin; per-particle φ can be slightly looser for
    // near-cancellations. Absolute 1e-5 covers the MAC truncation.
    EXPECT_LT(max_abs, 1e-5) << "max_rel=" << max_rel;
}

TEST(FmmTree, InteractionListWellSeparated) {
    const int N = 64;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<Vec3> pos(static_cast<std::size_t>(N));
    std::vector<AccReal> q(static_cast<std::size_t>(N), 1.0);
    for (int i = 0; i < N; ++i)
        pos[static_cast<std::size_t>(i)] = Vec3{U(rng), U(rng), U(rng)};

    FmmTree tree(6, 4);
    (void)tree.energy(pos, q);

    for (const auto& pair : tree.interaction_list()) {
        const int a = pair.first;
        const int b = pair.second;
        EXPECT_NE(a, b);
        // MAC: |cA−cB| / (rA+rB) ≥ 2 with r = half·√3.
        const Vec3 ca = tree.node_center(a);
        const Vec3 cb = tree.node_center(b);
        const AccReal ha = tree.node_half_width(a);
        const AccReal hb = tree.node_half_width(b);
        const AccReal ra = ha * std::sqrt(3.0);
        const AccReal rb = hb * std::sqrt(3.0);
        const AccReal dist = (ca - cb).norm();
        const AccReal c_ratio = dist / (ra + rb);
        EXPECT_GE(c_ratio, 2.0 * (1.0 - 1e-12))
            << "pair (" << a << "," << b << ") c=" << c_ratio;
    }
}

TEST(FmmTree, LeafCapacityRespected) {
    const int N = 50;
    const int leaf = 3;
    std::vector<Vec3> pos(static_cast<std::size_t>(N));
    std::vector<AccReal> q(static_cast<std::size_t>(N), 1.0);
    for (int i = 0; i < N; ++i)
        pos[static_cast<std::size_t>(i)] =
            Vec3{0.1 * (i % 5), 0.1 * ((i / 5) % 5), 0.1 * (i / 25)};

    FmmTree tree(4, leaf);
    (void)tree.energy(pos, q);
    for (int i = 0; i < tree.num_nodes(); ++i) {
        if (tree.node_is_leaf(i))
            EXPECT_LE(tree.node_particle_count(i), leaf);
    }
}

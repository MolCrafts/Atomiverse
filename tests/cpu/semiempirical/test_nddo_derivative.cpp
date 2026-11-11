/// @file
/// @brief Analytic pair integral derivatives, without any SCF iteration.
#include "atomiverse/cpu/semiempirical/internal/nddo_derivative.h"
#include "atomiverse/semiempirical/rm1_parameters.h"
#include <gtest/gtest.h>

namespace {
using namespace ATV::cpu::semiempirical;

TEST(NddoDerivative, ElementaryChainRule) {
    internal::NDDODerivative x{1.3, {1.0, 0.0, 0.0}};
    internal::NDDODerivative y{0.7, {0.0, 1.0, 0.0}};
    const auto f =
        internal::exp(x * y) / internal::sqrt(x) + internal::pow(y, 3);
    EXPECT_NEAR(f.value,
                std::exp(1.3 * 0.7) / std::sqrt(1.3) + std::pow(0.7, 3), 1e-14);
    EXPECT_NEAR(f.gradient[0],
                std::exp(1.3 * 0.7) / std::sqrt(1.3) * (0.7 - 0.5 / 1.3),
                1e-14);
    EXPECT_NEAR(f.gradient[1],
                std::exp(1.3 * 0.7) * std::sqrt(1.3) + 3 * 0.7 * 0.7, 1e-14);
    EXPECT_EQ(f.gradient[2], 0.0);
}

TEST(NddoDerivative, PairIntegralsAllSupportedSpElementsAndPolarAxes) {
    // Check derivatives against independent displacements of the value kernel.
    // The external whole-method energy/force pins remain regression gates.
    constexpr double h = 1e-5;
    for (const auto method : {ATV::semiempirical::SemiEmpiricalMethod::AM1,
                              ATV::semiempirical::SemiEmpiricalMethod::RM1}) {
        for (int z = 1; z <= 100; ++z) {
            const auto* pi = ATV::semiempirical::nddo_parameter(method, z);
            if (!pi || pi->n_orbitals > 4 || pi->n_orbitals == 0)
                continue;
            for (int zj : {1, 6}) {
                const auto* pj = ATV::semiempirical::nddo_parameter(method, zj);
                NDDOSystem pair;
                pair.method = method;
                pair.atoms = {{zj, 0, pj->n_orbitals - 1, pj->n_orbitals,
                               pj->core_charge, pj},
                              {z, pj->n_orbitals,
                               pj->n_orbitals + pi->n_orbitals - 1,
                               pi->n_orbitals, pi->core_charge, pi}};
                pair.n_orbitals = pi->n_orbitals + pj->n_orbitals;
                for (const auto pos :
                     {std::array<double, 3>{0.73, -0.38, 1.17},
                      std::array<double, 3>{0, 0, 1.43},
                      std::array<double, 3>{0, 0, -1.43},
                      std::array<double, 3>{1.43, 0, 0},
                      std::array<double, 3>{0, 1.43, 0},
                      std::array<double, 3>{0.6, 0, 0.8 - 1e-8},
                      std::array<double, 3>{0.6, 0, 0.8 + 1e-8}}) {
                    SCOPED_TRACE(::testing::Message()
                                 << "method=" << int(method) << " Z=" << z
                                 << "," << zj << " pos=" << pos[0] << ","
                                 << pos[1] << "," << pos[2]);
                    std::array<std::array<ATV::StorageReal, 2>, 3> xyz{};
                    for (int d = 0; d < 3; ++d)
                        xyz[d][1] = pos[d];
                    const auto dh =
                        internal::hcore_pair_derivative(pair, pos, {0, 0, 0});
                    const auto dw = internal::two_center_integral_derivative(
                        pair, xyz[0].data(), xyz[1].data(), xyz[2].data(), 1,
                        0);
                    // Bounded outputs must initialize unused slots even when
                    // successive calls change from p-containing to s-only pairs.
                    for (int d = 0; d < 3; ++d) {
                        const int packed =
                            pair.n_orbitals * (pair.n_orbitals + 1) / 2;
                        const int pair_count_i =
                            pi->n_orbitals * (pi->n_orbitals + 1) / 2;
                        const int pair_count_j =
                            pj->n_orbitals * (pj->n_orbitals + 1) / 2;
                        for (int k = packed; k < 36; ++k)
                            EXPECT_EQ(dh[d].packed_hcore[k], 0.0);
                        for (int k = pair_count_i * pair_count_j; k < 100; ++k)
                            EXPECT_EQ(dw[d][k], 0.0);
                    }
                    for (int d = 0; d < 3; ++d) {
                        xyz[d][1] = pos[d] + h;
                        const auto hp = NDDOHCoreBuilder{}.build(
                            pair, xyz[0].data(), xyz[1].data(), xyz[2].data());
                        NDDOTwoCenterIntegralBlock wp;
                        build_two_center_integral_block(
                            pair, xyz[0].data(), xyz[1].data(), xyz[2].data(),
                            1, 0, wp);
                        xyz[d][1] = pos[d] - h;
                        const auto hm = NDDOHCoreBuilder{}.build(
                            pair, xyz[0].data(), xyz[1].data(), xyz[2].data());
                        NDDOTwoCenterIntegralBlock wm;
                        build_two_center_integral_block(
                            pair, xyz[0].data(), xyz[1].data(), xyz[2].data(),
                            1, 0, wm);
                        xyz[d][1] = pos[d];
                        EXPECT_NEAR(
                            dh[d].nuclear_repulsion_ev,
                            (hp.nuclear_repulsion_ev - hm.nuclear_repulsion_ev)
                                / (2 * h),
                            2e-6);
                        for (std::size_t k = 0; k < hp.packed_hcore.size(); ++k)
                            EXPECT_NEAR(
                                dh[d].packed_hcore[k],
                                (hp.packed_hcore[k] - hm.packed_hcore[k])
                                    / (2 * h),
                                2e-6)
                                << "axis=" << d << " hcore=" << k;
                        for (int k = 0; k < wp.n_pairs_i * wp.n_pairs_j; ++k)
                            EXPECT_NEAR(dw[d][k],
                                        (wp.values[k] - wm.values[k]) / (2 * h),
                                        2e-6)
                                << "axis=" << d << " eri=" << k;
                    }
                }
            }
        }
    }
}
TEST(NddoDerivative, ValueBlockReusesActiveRegionAcrossPairSizes) {
    for (const auto method : {ATV::semiempirical::SemiEmpiricalMethod::AM1,
                              ATV::semiempirical::SemiEmpiricalMethod::RM1}) {
        NDDOTwoCenterIntegralBlock reused;
        reused.values.fill(731.0);
        // Dense p/p, s/s, and both rectangular orientations change both the
        // active count and row stride. Include polar axes with exact zeros.
        int step = 0;
        for (const auto elements : {std::array<int, 2>{6, 6},
                                    {1, 1},
                                    {6, 1},
                                    {1, 6},
                                    {6, 6},
                                    {1, 1},
                                    {6, 6}}) {
            SCOPED_TRACE(::testing::Message()
                         << "method=" << int(method) << " step=" << step);
            const auto* pi =
                ATV::semiempirical::nddo_parameter(method, elements[0]);
            const auto* pj =
                ATV::semiempirical::nddo_parameter(method, elements[1]);
            NDDOSystem pair;
            pair.method = method;
            pair.atoms = {{elements[0], 0, pi->n_orbitals - 1, pi->n_orbitals,
                           pi->core_charge, pi},
                          {elements[1], pi->n_orbitals,
                           pi->n_orbitals + pj->n_orbitals - 1, pj->n_orbitals,
                           pj->core_charge, pj}};
            pair.n_orbitals = pi->n_orbitals + pj->n_orbitals;
            const bool polar = (++step % 2 == 1);
            std::array<ATV::StorageReal, 2> x{0, polar ? 0.0 : 0.73};
            std::array<ATV::StorageReal, 2> y{0, polar ? 0.0 : -0.38};
            std::array<ATV::StorageReal, 2> z{0, polar ? 1.43 : 1.17};
            const auto before = reused.values;
            NDDOTwoCenterIntegralBlock fresh;
            build_two_center_integral_block(pair, x.data(), y.data(), z.data(),
                                            0, 1, fresh);
            build_two_center_integral_block(pair, x.data(), y.data(), z.data(),
                                            0, 1, reused);
            ASSERT_EQ(reused.atom_i, 0);
            ASSERT_EQ(reused.atom_j, 1);
            ASSERT_EQ(reused.n_pairs_i,
                      pi->n_orbitals * (pi->n_orbitals + 1) / 2);
            ASSERT_EQ(reused.n_pairs_j,
                      pj->n_orbitals * (pj->n_orbitals + 1) / 2);
            const int active = reused.n_pairs_i * reused.n_pairs_j;
            for (int k = 0; k < active; ++k)
                EXPECT_DOUBLE_EQ(reused.values[k], fresh.values[k])
                    << "entry=" << k;
            for (int k = active; k < 100; ++k)
                EXPECT_DOUBLE_EQ(reused.values[k], before[k])
                    << "inactive=" << k;
            EXPECT_THROW(reused.at(reused.n_pairs_i, 0), std::runtime_error);
            EXPECT_THROW(reused.at(0, reused.n_pairs_j), std::runtime_error);
        }
    }
}

TEST(NddoDerivative, RejectsPairOutsideFixedSpCapacity) {
    NDDOSystem pair;
    const auto* p = ATV::semiempirical::nddo_parameter(
        ATV::semiempirical::SemiEmpiricalMethod::AM1, 6);
    pair.atoms = {{6, 0, 8, 9, p->core_charge, p},
                  {6, 9, 17, 9, p->core_charge, p}};
    pair.n_orbitals = 18;
    std::array<ATV::StorageReal, 2> x{0, 1}, y{}, z{};
    EXPECT_THROW(internal::hcore_pair_derivative(pair, {1, 0, 0}, {0, 0, 0}),
                 std::runtime_error);
    EXPECT_THROW(internal::two_center_integral_derivative(
                     pair, x.data(), y.data(), z.data(), 1, 0),
                 std::runtime_error);
    NDDOTwoCenterIntegralBlock value;
    value.values.fill(731.0);
    EXPECT_THROW(build_two_center_integral_block(pair, x.data(), y.data(),
                                                 z.data(), 1, 0, value),
                 std::runtime_error);
    pair.atoms[1].n_orbitals = 1;
    for (const int count : {0, 2, 3}) {
        pair.atoms[0].n_orbitals = count;
        EXPECT_THROW(build_two_center_integral_block(pair, x.data(), y.data(),
                                                     z.data(), 1, 0, value),
                     std::runtime_error);
    }
    for (double entry : value.values)
        EXPECT_EQ(entry, 731.0);
    pair.atoms.clear();
    EXPECT_THROW(internal::hcore_pair_derivative(pair, {1, 0, 0}, {0, 0, 0}),
                 std::runtime_error);
}
} // namespace

/// @file cpu/semiempirical/test_nddo_solver.cpp
/// @brief NDDO preparation boundaries; no self-consistency loop.
#include <algorithm>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_solver.h"
#include "atomiverse/cpu/semiempirical/nddo_density.h"
#include "atomiverse/cpu/semiempirical/nddo_fock.h"

namespace {
using namespace ATV;
using namespace ATV::cpu::semiempirical;

TEST(NddoDensity, ReusesStorageAcrossSizesAndOccupationBranches) {
    NDDODensityBuilder builder;
    std::vector<double> density(10, std::numeric_limits<double>::quiet_NaN());
    const auto* storage = density.data();
    const auto capacity = density.capacity();
    // Orthonormal columns: occupied (3/5, 4/5), virtual (-4/5, 3/5).
    const std::vector<double> rotated{0.6, -0.8, 0.8, 0.6};
    const NDDODensityInput small{2, 2, 1, 2.0, 1, 0.0, 1};
    std::vector<double> identity(16, 0.0);
    for (int i = 0; i < 4; ++i)
        identity[5 * i] = 1.0;
    for (const int mode : {1, 2, 1}) {
        builder.build_packed(rotated, small, density);
        ASSERT_EQ(density.size(), 3u);
        EXPECT_NEAR(density[0], 0.72, 1e-15);
        EXPECT_NEAR(density[1], 0.96, 1e-15);
        EXPECT_NEAR(density[2], 1.28, 1e-15);
        EXPECT_EQ(density.data(), storage);
        // Fractional occupations (2,.5,.5,0). Modes 1 and 2 exercise the
        // complementary-orbital and occupied-orbital expressions respectively.
        const NDDODensityInput large{4, 4, 1, 2.0, 3, 0.5, mode};
        builder.build_packed(identity, large, density);
        ASSERT_EQ(density.size(), 10u);
        const double expected[4] = {2.0, 0.5, 0.5, 0.0};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j <= i; ++j)
                EXPECT_DOUBLE_EQ(density[packed_index(i, j)],
                                 i == j ? expected[i] : 0.0);
        EXPECT_EQ(density.data(), storage);
        EXPECT_EQ(density.capacity(), capacity);
    }
}

TEST(NddoDensity, InvalidInputPreservesOutputAndAllowsRetry) {
    NDDODensityBuilder builder;
    const std::vector<double> coefficients{1.0, 0.0, 0.0, 1.0};
    const NDDODensityInput valid{2, 2, 1, 2.0, 1, 0.0, 1};
    std::vector<double> density{731.0, 732.0, 733.0};
    const auto original = density;
    for (const NDDODensityInput invalid :
         {NDDODensityInput{0, 2, 1, 2.0, 1, 0.0, 1},
          NDDODensityInput{1, 2, 1, 2.0, 1, 0.0, 1},
          NDDODensityInput{2, 2, 3, 2.0, 3, 0.0, 1}}) {
        EXPECT_THROW(builder.build_packed(coefficients, invalid, density),
                     std::runtime_error);
        EXPECT_EQ(density, original);
    }
    EXPECT_THROW(builder.build_packed({1.0}, valid, density),
                 std::runtime_error);
    EXPECT_EQ(density, original);
    auto alias = coefficients;
    EXPECT_THROW(builder.build_packed(alias, valid, alias),
                 std::invalid_argument);
    EXPECT_EQ(alias, coefficients);
    builder.build_packed(coefficients, valid, density);
    EXPECT_EQ(density, (std::vector<double>{2.0, 0.0, 0.0}));
}

TEST(NddoPreparation, MndoPm3RejectUnvalidatedIdentityBeforeScf) {
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = .8;
    for (const auto method : {ATV::semiempirical::SemiEmpiricalMethod::MNDO,
                              ATV::semiempirical::SemiEmpiricalMethod::PM3}) {
        NDDOConfig config;
        config.method = method;
        NDDOSolver supported{config};
        EXPECT_NO_THROW(supported.build(frame));
        EXPECT_EQ(supported.result().method, method);
        EXPECT_FALSE(supported.result().converged);
        config.charge = 1;
        NDDOSolver charged{config};
        EXPECT_THROW(charged.build(frame), std::invalid_argument);
        config.charge = 0;
        config.multiplicity = 3;
        NDDOSolver triplet{config};
        EXPECT_THROW(triplet.build(frame), std::invalid_argument);
        frame.span(kAtom_type)[0] = 16;
        EXPECT_THROW(supported.scf_prepare(frame), std::runtime_error);
        frame.span(kAtom_type)[0] = 1;
        EXPECT_NO_THROW(supported.scf_prepare(frame));
    }
}

TEST(NddoPreparation, RebuildsElementIdentityAtUnchangedSize) {
    cpu::Frame frame(1);
    frame.span(kAtom_type)[0] = 6;
    frame.span(kAtom_x)[0] = frame.span(kAtom_y)[0] = frame.span(kAtom_z)[0] =
        0;
    NDDOSolver solver{NDDOConfig{}};
    solver.build(frame);
    frame.span(kAtom_type)[0] = 8;
    solver.scf_prepare(frame);
    EXPECT_EQ(solver.system().atoms[0].atomic_number, 8);
    EXPECT_EQ(solver.system().n_electrons, 6);
    EXPECT_EQ(solver.system().n_orbitals, 4);
    EXPECT_FALSE(solver.has_density());
}

TEST(NddoPreparation, RebuildsAtomCount) {
    cpu::Frame frame(1);
    frame.span(kAtom_type)[0] = 6;
    frame.span(kAtom_x)[0] = frame.span(kAtom_y)[0] = frame.span(kAtom_z)[0] =
        0;
    NDDOSolver solver{NDDOConfig{}};
    solver.build(frame);
    cpu::Frame hydrogen(2);
    for (int a = 0; a < 2; ++a) {
        hydrogen.span(kAtom_type)[a] = 1;
        hydrogen.span(kAtom_x)[a] = 0.74 * a;
        hydrogen.span(kAtom_y)[a] = hydrogen.span(kAtom_z)[a] = 0;
    }
    solver.scf_prepare(hydrogen);
    EXPECT_EQ(solver.system().atoms.size(), 2);
    EXPECT_EQ(solver.system().n_orbitals, 2);
    EXPECT_EQ(solver.system().n_electrons, 2);
}

TEST(NddoPreparation, InvalidGeometryFailsBeforeScfAndCanRetry) {
    cpu::Frame frame(2);
    for (int a = 0; a < 2; ++a) {
        frame.span(kAtom_type)[a] = 1;
        frame.span(kAtom_x)[a] = 0.74 * a;
        frame.span(kAtom_y)[a] = frame.span(kAtom_z)[a] = 0;
    }
    NDDOSolver solver{NDDOConfig{}};
    solver.build(frame);
    solver.result().converged = true;
    frame.span(kAtom_x)[1] = std::numeric_limits<StorageReal>::quiet_NaN();
    EXPECT_THROW(solver.scf_prepare(frame), std::invalid_argument);
    EXPECT_FALSE(solver.result().converged);
    frame.span(kAtom_x)[1] = 0;
    EXPECT_THROW(solver.scf_prepare(frame), std::invalid_argument);
    frame.span(kAtom_x)[1] = 0.74;
    EXPECT_NO_THROW(solver.scf_prepare(frame));
    cpu::Frame empty;
    EXPECT_THROW(solver.scf_prepare(empty), std::invalid_argument);
}
TEST(NddoPreparation, FailedForceRequestClearsSuccessAndOldForces) {
    cpu::Frame frame(1);
    frame.span(kAtom_type)[0] = 6;
    frame.span(kAtom_x)[0] = frame.span(kAtom_y)[0] = frame.span(kAtom_z)[0] =
        0;
    NDDOSolver solver{NDDOConfig{}};
    solver.build(frame);
    // A stale result flag must never bypass the converged-density precondition.
    solver.result().converged = true;
    solver.result().forces_ev_a = {1, 2, 3};
    EXPECT_THROW(solver.compute_forces(frame), std::runtime_error);
    EXPECT_FALSE(solver.result().converged);
    EXPECT_TRUE(solver.result().forces_ev_a.empty());
    EXPECT_NO_THROW(solver.scf_prepare(frame));
}
TEST(NddoPreparation, ScfControlsUpdateWithoutRebuildingWorkspace) {
    cpu::Frame frame(1);
    frame.span(kAtom_type)[0] = 6;
    frame.span(kAtom_x)[0] = frame.span(kAtom_y)[0] = frame.span(kAtom_z)[0] =
        0;
    NDDOSolver solver{NDDOConfig{}};
    solver.build(frame);
    const auto* const matrix = solver.state().packed_hcore.data();
    const auto hcore = solver.state().packed_hcore;
    solver.set_scf_controls(700, 1e-11, 1e-13);
    EXPECT_EQ(solver.config().max_iter, 700);
    EXPECT_EQ(solver.config().density_tol, 1e-11);
    EXPECT_EQ(solver.config().energy_tol, 1e-13);
    EXPECT_EQ(solver.state().packed_hcore.data(), matrix);
    EXPECT_EQ(solver.state().packed_hcore, hcore);
    EXPECT_THROW(solver.set_scf_controls(0, 1e-8, 1e-10),
                 std::invalid_argument);
    EXPECT_THROW(solver.set_scf_controls(5, 0, 1e-10), std::invalid_argument);
    EXPECT_THROW(solver.set_scf_controls(
                     5, 1e-8, std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
    EXPECT_THROW(solver.set_scf_controls(
                     5, std::numeric_limits<double>::quiet_NaN(), 1e-10),
                 std::invalid_argument);
    EXPECT_EQ(solver.config().max_iter, 700);
    EXPECT_EQ(solver.config().density_tol, 1e-11);
    EXPECT_EQ(solver.config().energy_tol, 1e-13);
    NDDOConfig invalid;
    invalid.max_iter = 0;
    EXPECT_THROW((NDDOSolver{invalid}), std::invalid_argument);
}
TEST(NddoPreparation, ForceStorageIsReservedByBuildAndPreservedOnPreparation) {
    cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_x)[i] = 0.74 * i;
        frame.span(kAtom_y)[i] = frame.span(kAtom_z)[i] = 0;
    }
    NDDOSolver solver{NDDOConfig{}};
    solver.build(frame);
    EXPECT_TRUE(solver.result().forces_ev_a.empty());
    EXPECT_GE(solver.result().forces_ev_a.capacity(), 6u);
    const auto capacity = solver.result().forces_ev_a.capacity();
    solver.scf_prepare(frame);
    EXPECT_EQ(solver.result().forces_ev_a.capacity(), capacity);
    EXPECT_THROW(solver.commit_density(), std::runtime_error);
    EXPECT_FALSE(solver.has_density());

    cpu::Frame larger(4);
    for (int i = 0; i < 4; ++i) {
        larger.span(kAtom_type)[i] = 1;
        larger.span(kAtom_x)[i] = 0.74 * i;
        larger.span(kAtom_y)[i] = larger.span(kAtom_z)[i] = 0;
    }
    solver.scf_prepare(larger);
    EXPECT_GE(solver.result().forces_ev_a.capacity(), 12u);
    EXPECT_TRUE(solver.result().forces_ev_a.empty());
    EXPECT_FALSE(solver.has_density());
}
TEST(NddoDiagonalizer, ReusesOutputForSuccessiveSymmetricMatrices) {
    NDDODiagonalizer solver;
    solver.build(2);
    std::vector<double> coefficients(4);
    const auto* storage = coefficients.data();
    for (const double coupling : {1.0, -0.25, 1.0}) {
        const std::vector<double> packed{2.0, coupling, 2.0};
        solver.diagonalize_packed(packed, coefficients);
        ASSERT_EQ(coefficients.size(), 4u);
        EXPECT_EQ(coefficients.data(), storage);
        const double eigenvalues[2] = {2.0 - std::abs(coupling),
                                       2.0 + std::abs(coupling)};
        for (int col = 0; col < 2; ++col) {
            // Analytic eigenvalues and A*C=C*e fix the column/row layout.
            for (int row = 0; row < 2; ++row)
                EXPECT_NEAR(2.0 * coefficients[2 * row + col]
                                + coupling * coefficients[2 * (1 - row) + col],
                            eigenvalues[col] * coefficients[2 * row + col],
                            1.e-13);
            for (int other = 0; other < 2; ++other)
                EXPECT_NEAR(coefficients[col] * coefficients[other]
                                + coefficients[2 + col]
                                      * coefficients[2 + other],
                            col == other ? 1.0 : 0.0, 1.e-13);
        }
    }
}

TEST(NddoDiagonalizer, RebuildsDimensionsAndKeepsSortedColumns) {
    NDDODiagonalizer solver;
    std::vector<double> coefficients;
    for (const int n : {3, 2, 3}) {
        solver.build(n);
        std::vector<double> packed(static_cast<std::size_t>(n) * (n + 1) / 2,
                                   0.0);
        for (int i = 0; i < n; ++i)
            packed[packed_index(i, i)] = 2.0 + i;
        solver.diagonalize_packed(packed, coefficients);
        ASSERT_EQ(coefficients.size(), static_cast<std::size_t>(n) * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                EXPECT_NEAR(std::abs(coefficients[i * n + j]),
                            i == j ? 1.0 : 0.0, 1.e-14);
    }
}

TEST(NddoDiagonalizer, RejectsInvalidInputAndCanRetry) {
    ATV::cpu::linalg::RealMatrix dense;
    EXPECT_THROW(NDDOMatrixBuilder{}.packed_to_dense(
                     {}, std::numeric_limits<int>::max(), dense),
                 std::runtime_error);
    NDDODiagonalizer solver;
    const std::vector<double> packed{2.0, 1.0, 2.0};
    std::vector<double> coefficients(4, 731.0);
    EXPECT_THROW(solver.diagonalize_packed(packed, coefficients),
                 std::runtime_error);
    EXPECT_THROW(solver.build(0), std::invalid_argument);
    EXPECT_THROW(solver.build(std::numeric_limits<int>::max()),
                 std::invalid_argument);
    solver.build(2);
    const std::vector<double> wrong_size{1.0};
    EXPECT_THROW(solver.diagonalize_packed(wrong_size, coefficients),
                 std::runtime_error);
    auto invalid = packed;
    invalid[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(solver.diagonalize_packed(invalid, coefficients),
                 std::runtime_error);
    for (double value : coefficients)
        EXPECT_EQ(value, 731.0);
    EXPECT_NO_THROW(solver.diagonalize_packed(packed, coefficients));
    for (double value : coefficients)
        EXPECT_TRUE(std::isfinite(value));
    // An invalid new dimension must not invalidate the already-built size.
    EXPECT_THROW(solver.build(-1), std::invalid_argument);
    EXPECT_NO_THROW(solver.diagonalize_packed(packed, coefficients));
}

TEST(NddoFock, CachedTwoCenterBlocksReproduceTheDirectBuild) {
    // Formaldehyde plus a distant water: s/p and s-only blocks of every size.
    cpu::Frame frame(7);
    const int z[7] = {6, 8, 1, 1, 8, 1, 1};
    const double xyz[7][3] = {{0, 0, 0},         {0, 0, 1.21}, {0.94, 0, -0.54},
                              {-0.94, 0, -0.54}, {3, 0, 0},    {3.96, 0, 0},
                              {2.76, 0.93, 0}};
    for (int a = 0; a < 7; ++a) {
        frame.span(kAtom_type)[a] = z[a];
        frame.span(kAtom_x)[a] = xyz[a][0];
        frame.span(kAtom_y)[a] = xyz[a][1];
        frame.span(kAtom_z)[a] = xyz[a][2];
    }
    const NDDOSystem system{frame, 0, 1};
    const auto packed = static_cast<std::size_t>(system.n_orbitals)
                        * (system.n_orbitals + 1) / 2;
    std::vector<double> hcore(packed), density(packed), spin(packed);
    for (std::size_t k = 0; k < packed; ++k) {
        hcore[k] = -1.0 + 0.01 * static_cast<double>(k % 17);
        density[k] = 0.3 * std::sin(0.7 * static_cast<double>(k));
        spin[k] = 0.5 * density[k];
    }
    NDDOFockBuilder direct, cached;
    const auto expected = direct.build(system, frame, hcore, density, spin);
    EXPECT_FALSE(cached.has_two_center_cache(system));
    EXPECT_THROW(cached.build_cached(system, hcore, density, spin),
                 std::logic_error);
    ASSERT_TRUE(cached.cache_two_center(
        system, frame.span(kAtom_x), frame.span(kAtom_y), frame.span(kAtom_z)));
    ASSERT_TRUE(cached.has_two_center_cache(system));
    const auto fock = cached.build_cached(system, hcore, density, spin);
    ASSERT_EQ(fock.size(), expected.size());
    for (std::size_t k = 0; k < fock.size(); ++k)
        EXPECT_NEAR(fock[k], expected[k],
                    1e-12 * (1.0 + std::abs(expected[k])));
    cached.release_two_center();
    EXPECT_FALSE(cached.has_two_center_cache(system));
}

TEST(NddoDiagonalizer, PseudoStepKeepsEigenvectorsAndOrthonormality) {
    // 4x4, two occupied orbitals with a 2 eV gap.
    const int n = 4;
    const std::vector<double> fock{-3.0, 0.1,  -2.5, 0.0, 0.2,
                                   1.0,  0.05, 0.0,  0.1, 2.0};
    NDDODiagonalizer solver;
    solver.build(n);
    std::vector<double> c(16, 731.0);
    EXPECT_FALSE(solver.pseudo_diagonalize_packed(fock, 2, 0.5, c));
    EXPECT_EQ(c, std::vector<double>(16, 731.0));
    std::vector<double> exact;
    solver.diagonalize_packed(fock, exact);
    std::vector<double> rotated;
    // At self-consistency (F_ov = 0) the step is the identity.
    ASSERT_TRUE(solver.pseudo_diagonalize_packed(fock, 2, 0.5, rotated));
    for (std::size_t k = 0; k < 16; ++k)
        EXPECT_NEAR(rotated[k], exact[k], 1e-12);
    EXPECT_FALSE(solver.pseudo_diagonalize_packed(fock, 2, 1e3, rotated));
    EXPECT_FALSE(solver.pseudo_diagonalize_packed(fock, 0, 0.5, rotated));
    // A perturbed operator: the rotated set stays orthonormal and its
    // occupied-virtual coupling shrinks.
    auto perturbed = fock;
    perturbed[packed_index(2, 0)] += 0.2;
    perturbed[packed_index(3, 1)] -= 0.15;
    auto coupling = [&](const std::vector<double>& cr) {
        double largest = 0.0;
        for (int i = 0; i < 2; ++i)
            for (int a = 2; a < 4; ++a) {
                double f = 0.0;
                for (int mu = 0; mu < n; ++mu)
                    for (int nu = 0; nu < n; ++nu)
                        f += cr[mu * n + i] * perturbed[packed_index(mu, nu)]
                             * cr[nu * n + a];
                largest = std::max(largest, std::abs(f));
            }
        return largest;
    };
    const double before = coupling(exact);
    ASSERT_TRUE(solver.pseudo_diagonalize_packed(perturbed, 2, 0.5, rotated));
    EXPECT_LT(coupling(rotated), 0.2 * before);
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q) {
            double s = 0.0;
            for (int mu = 0; mu < n; ++mu)
                s += rotated[mu * n + p] * rotated[mu * n + q];
            EXPECT_NEAR(s, p == q ? 1.0 : 0.0, 1e-12);
        }
}
} // namespace

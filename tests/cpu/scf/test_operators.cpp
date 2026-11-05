/// @file tests/cpu/scf/test_operators.cpp
/// @brief Thread-invariant stored-ERI G unit test. Layout mirrors
///        `cpu/scf/operators.cpp`
///        (`internal::IntegralWorkspace::coulomb_exchange_fused`); the exact
///        J/K engine itself is covered by test_exact_jk.cpp.

#ifdef ATV_HAS_HF

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/integrals.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ATV;
using namespace ATV::cpu::scf;

namespace {

/// @brief Water (H2O) frame, C2v.
cpu::Frame make_h2o() {
    cpu::Frame f(3);
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_x)[1] = 0.757;
    f.span(ATV::kAtom_y)[1] = 0.587;
    f.span(ATV::kAtom_z)[1] = 0.0;
    f.span(ATV::kAtom_x)[2] = -0.757;
    f.span(ATV::kAtom_y)[2] = 0.587;
    f.span(ATV::kAtom_z)[2] = 0.0;
    f.span(ATV::kAtom_atomic_number)[0] = 8;
    f.span(ATV::kAtom_atomic_number)[1] = 1;
    f.span(ATV::kAtom_atomic_number)[2] = 1;
    f.span(ATV::kAtom_mass)[0] = 15.999;
    f.span(ATV::kAtom_mass)[1] = 1.008;
    f.span(ATV::kAtom_mass)[2] = 1.008;
    f.box = ATV::cpu::Box::cubic(100.0);
    return f;
}

/// @brief Dense symmetric test density with off-diagonals (exercises all
///        Coulomb + exchange contributions).
RealMatrix make_density(int n) {
    RealMatrix P = xt::zeros<double>(
        {static_cast<std::size_t>(n), static_cast<std::size_t>(n)});
    for (int i = 0; i < n; ++i)
        P(i, i) = 0.5;
    for (int i = 0; i < n - 1; ++i) {
        P(i, i + 1) = 0.1;
        P(i + 1, i) = 0.1;
    }
    return P;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════
// TestIntegralWorkspace — coulomb_exchange_fused thread-count invariance (ac-004)
// ═══════════════════════════════════════════════════════════════════════

// Parallel s8 contraction reduces thread-local matrices. Floating-point sum
// order may differ; validate to 1e-12, below the HF energy/force gates.
TEST(TestIntegralWorkspace, CoulombExchangeFusedThreadInvariant) {
    auto frame = make_h2o();
    auto pack = internal::IntegralWorkspace{frame, BasisSet::STO3G, 0,
                                            /*compute_eri=*/true};
    const int n = pack.n_basis;
    RealMatrix P = make_density(n);

#ifdef _OPENMP
    const int saved = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    RealMatrix G1;
    pack.coulomb_exchange_fused(P, G1);

#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    RealMatrix G4;
    pack.coulomb_exchange_fused(P, G4);

#ifdef _OPENMP
    omp_set_num_threads(saved);
#endif

    ASSERT_EQ(G1.size(), G4.size());
    for (std::size_t i = 0; i < G1.size(); ++i)
        EXPECT_NEAR(G1.data()[i], G4.data()[i], 1e-12)
            << "coulomb_exchange_fused element " << i
            << " differs between 1 and 4 threads";
}

#else

#include <gtest/gtest.h>
TEST(TestIntegralWorkspace, SkippedNoBuildHF) {
    GTEST_SKIP()
        << "ATV_BUILD_HF=OFF — operator tests require -DATV_BUILD_HF=ON";
}

#endif // ATV_HAS_HF

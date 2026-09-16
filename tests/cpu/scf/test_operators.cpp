/// @file tests/cpu/scf/test_operators.cpp
/// @brief Fused one-sweep JK and thread-invariant stored-ERI K unit tests
///        (spec scf-review-06-scf-perf, Item 16 / Item 20 — ac-003, ac-004).
///
/// Layout mirrors `cpu/scf/operators.cpp` (the two-electron Fock operators:
/// `ShellBlockPairList::{g_rhf,k,jk_rhf}` and
/// `internal::IntegralWorkspace::coulomb_exchange_fused`).
///
/// RED until the planned `ShellBlockPairList::jk_rhf(P_total, J, K)` member
/// (one shell-block sweep producing bare J and bare K) exists.
///
/// The binding scientist invariant (ac-003): fusing J and K into one sweep is
/// algebraically exact vs the two-pass g_rhf + k. The ≤1e-12 per-element gate
/// exists specifically to catch the classic one-sweep-JK bug where the 8-fold
/// ERI permutation multiplicities unpack differently for J than for K. K is
/// emitted unscaled (RSH-separable) so a later erf-attenuated K can be added.

#ifdef ATV_HAS_HF

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"

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
    f.span(ATV::kAtom_type)[0] = 8;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_type)[2] = 1;
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

/// @brief Per-element max |A - B| over two [n x n] matrices.
double max_abs_diff(const RealMatrix& a, const RealMatrix& b) {
    double worst = 0.0;
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i)
        worst = std::max(worst, std::abs(a.data()[i] - b.data()[i]));
    return worst;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════
// TestShellBlockPairList — fused jk_rhf vs two-pass g_rhf + k (ac-003)
// ═══════════════════════════════════════════════════════════════════════

namespace {

/// @brief One-sweep jk_rhf must reproduce the two-pass J/K and the assembled
///        hybrid Fock to ≤1e-12 Ha for the given basis (includes a d-shell
///        fixture for the classic multiplicity-unpack bug).
void check_fused_vs_twopass(const cpu::Frame& frame, BasisSet basis,
                            const char* label) {
    auto pack = internal::IntegralWorkspace{frame, basis};
    auto sp = ShellBlockPairList{pack};
    const int n = sp.n_basis;
    RealMatrix P = make_density(n);

    // Fused one-sweep: bare J and bare K (planned API — RED until it exists).
    RealMatrix J_fused, K_fused;
    sp.jk_rhf(P, J_fused, K_fused);

    // Two-pass reference: g_rhf gives G = J - 0.5*K, k gives bare K, hence
    // J_ref = G + 0.5*K_ref.
    RealMatrix G, K_ref;
    sp.g_rhf(P, G);
    sp.k(P, K_ref);
    RealMatrix J_ref = G + 0.5 * K_ref;

    // K must be emitted unscaled — same normalization as ShellBlockPairList::k.
    EXPECT_LE(max_abs_diff(K_fused, K_ref), 1e-12)
        << label << ": fused K must match k() (bare, RSH-separable)";
    EXPECT_LE(max_abs_diff(J_fused, J_ref), 1e-12)
        << label << ": fused J must match g_rhf-recovered J";

    // Binding gate: assemble the two-electron hybrid Fock two ways and compare
    // per element for both B3LYP (a_x=0.20) and PBE0 (a_x=0.25). Hcore + Vxc
    // are common to both assemblies, so comparing the J/K part is sufficient
    // and isolates the one-sweep-JK multiplicity-unpack invariant.
    for (double a_x : {0.20, 0.25}) {
        RealMatrix F_fused = J_fused - 0.5 * a_x * K_fused;
        RealMatrix F_twopass = J_ref - 0.5 * a_x * K_ref;
        EXPECT_LE(max_abs_diff(F_fused, F_twopass), 1e-12)
            << label << ": F_fused vs F_twopass at a_x=" << a_x
            << " exceeds 1e-12 Ha gate (one-sweep-JK multiplicity bug)";
    }
}

} // namespace

TEST(TestShellBlockPairList, FusedJkMatchesTwoPass_H2O_STO3G) {
    check_fused_vs_twopass(make_h2o(), BasisSet::STO3G, "H2O/STO-3G");
}

TEST(TestShellBlockPairList, FusedJkMatchesTwoPass_H2O_631Gs_DShell) {
    // 6-31G* puts a Cartesian d-shell on oxygen → d block quartets appear,
    // exercising the 8-fold permutation unpack that differs for J vs K.
    check_fused_vs_twopass(make_h2o(), BasisSet::Pople631Gs, "H2O/6-31G*");
}

// ═══════════════════════════════════════════════════════════════════════
// TestIntegralWorkspace — coulomb_exchange_fused thread-count invariance (ac-004)
// ═══════════════════════════════════════════════════════════════════════

// NOTE: today the K contraction in internal::IntegralWorkspace::coulomb_exchange_fused is
// serial, so this test passes trivially. It becomes the guarding regression
// AFTER the planned `#pragma omp parallel for` reduction is added to the K
// contraction: each output element (i, j>=j') and its mirror (j, i) is owned
// by a single i-iteration and reduced in ascending k,l order, so the result
// must stay bit-for-bit identical for any thread count.
TEST(TestIntegralWorkspace, CoulombExchangeFusedThreadInvariant) {
    auto frame = make_h2o();
    auto pack = internal::IntegralWorkspace{frame, BasisSet::STO3G, 0, /*compute_eri=*/true};
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
        EXPECT_EQ(G1.data()[i], G4.data()[i])
            << "coulomb_exchange_fused element " << i
            << " differs between 1 and 4 threads (must be bit-invariant)";
}

#else

#include <gtest/gtest.h>
TEST(TestShellBlockPairList, SkippedNoBuildHF) {
    GTEST_SKIP()
        << "ATV_BUILD_HF=OFF — operator tests require -DATV_BUILD_HF=ON";
}

#endif // ATV_HAS_HF

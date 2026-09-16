/// @file tests/cpu/scf/test_xc_grid_engine.cpp
/// @brief Equivalence unit tests for the geometry-fixed AO-on-grid cache
///        (spec scf-review-06-scf-perf, Item 15 / ac-001, ac-002).
///
/// Layout mirrors `cpu/scf/xc_grid_engine.h`. RED until the planned
/// `XCGridBasis` workspace type and the precomputed-basis
/// `evaluate_xc_batched` overload exist.
///
/// The refactor splits `evaluate_xc_batched` into
///   1. `XCGridBasis::build(grid, shells, n_basis, is_gga)` — build Phi (and
///      dPhi_x/y/z for GGA) once per geometry, and
///   2. `evaluate_xc_batched(const XCGridBasis&, grid, P_alpha, P_beta,
///      xc_type)` — the density GEMM → XC batch → Vxc GEMM steps on the
///      precomputed basis.
/// It must be a pure performance change: the cached path is bit-for-bit
/// identical to the original rebuild-every-iteration path. This pins that
/// invariant on a small stored H2/STO-3G grid fixture (runs in milliseconds)
/// for both an LDA and a PBE (GGA) functional.

#ifdef ATV_HAS_HF

#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include "atomiverse/cpu/scf/xc_grid_engine.h"

using namespace ATV;
using namespace ATV::cpu::scf;

namespace {

/// @brief Build an H2 frame at 0.74 Å (smallest KS-capable system).
cpu::Frame make_h2() {
    cpu::Frame f(2);
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_x)[1] = 0.74;
    f.span(ATV::kAtom_y)[1] = 0.0;
    f.span(ATV::kAtom_z)[1] = 0.0;
    f.span(ATV::kAtom_type)[0] = 1;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_mass)[0] = 1.008;
    f.span(ATV::kAtom_mass)[1] = 1.008;
    f.box = ATV::cpu::Box::cubic(100.0);
    return f;
}

/// @brief A small symmetric closed-shell-shaped density (spin halves equal).
///
/// Any non-degenerate symmetric P exercises the density / Vxc GEMMs; the two
/// code paths must agree on identical input, so a physical density is not
/// required.
xt::xtensor<double, 2> make_density(std::size_t N) {
    xt::xtensor<double, 2> P = xt::zeros<double>({N, N});
    for (std::size_t i = 0; i < N; ++i)
        P(i, i) = 0.5;
    for (std::size_t i = 0; i + 1 < N; ++i) {
        P(i, i + 1) = 0.1;
        P(i + 1, i) = 0.1;
    }
    return P;
}

/// @brief Assert two [N x N] tensors are bit-for-bit identical.
void expect_bitwise_2d(const xt::xtensor<double, 2>& cached,
                       const xt::xtensor<double, 2>& rebuild,
                       const char* what) {
    ASSERT_EQ(cached.shape(0), rebuild.shape(0)) << what << " rows";
    ASSERT_EQ(cached.shape(1), rebuild.shape(1)) << what << " cols";
    const std::size_t n = cached.size();
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_EQ(cached.data()[i], rebuild.data()[i])
            << what << " element " << i << " not bit-identical";
}

/// @brief Compare the cached-basis overload to the rebuild-every-call overload
///        for one functional; both must be bitwise-equal on the same inputs.
void check_cache_equivalence(XCType xc, bool is_gga, const char* label) {
    auto frame = make_h2();
    auto pack = internal::IntegralWorkspace{frame, BasisSet::STO3G};
    const int N = pack.n_ao_cart();
    MolecularGrid grid(pack.atoms.data(), static_cast<int>(pack.atoms.size()),
                       GridLevel::SG0, /*n_radial=*/30, /*n_angular=*/50);

    auto Pa = make_density(static_cast<std::size_t>(N));
    auto Pb = Pa;

    // Original path: rebuilds Phi/dPhi inside every call.
    XCGridResult rebuild =
        evaluate_xc_batched(grid, pack.shells, N, Pa, Pb, xc);

    // Cached path: build the AO-on-grid basis once, then the precomputed
    // overload runs steps 2–5 (planned API — RED until it exists).
    XCGridBasis aog;
    aog.build(grid, pack.shells, N, is_gga);
    XCGridResult cached = evaluate_xc_batched(aog, grid, Pa, Pb, xc);

    expect_bitwise_2d(cached.Vxc_alpha, rebuild.Vxc_alpha, label);
    expect_bitwise_2d(cached.Vxc_beta, rebuild.Vxc_beta, label);
    EXPECT_EQ(cached.E_xc, rebuild.E_xc) << label << " E_xc not bit-identical";
    EXPECT_EQ(cached.n_alpha_integrated, rebuild.n_alpha_integrated)
        << label << " n_alpha_integrated not bit-identical";
    EXPECT_EQ(cached.n_beta_integrated, rebuild.n_beta_integrated)
        << label << " n_beta_integrated not bit-identical";
}

} // namespace

// ── ac-001: cached vs rebuild bitwise equality ─────────────────────────

TEST(TestXcGridEngine, CachedBasisMatchesRebuild_LDA) {
    check_cache_equivalence(XCType::LSDA, /*is_gga=*/false,
                            "LDA cache vs rebuild");
}

TEST(TestXcGridEngine, CachedBasisMatchesRebuild_PBE) {
    // GGA path also exercises dPhi_x/y/z in the cached basis.
    check_cache_equivalence(XCType::PBE, /*is_gga=*/true,
                            "PBE cache vs rebuild");
}

// ── build() latches the built + gga flags ──────────────────────────────

TEST(TestXcGridEngine, BuildSetsBuiltAndGgaFlags) {
    auto frame = make_h2();
    auto pack = internal::IntegralWorkspace{frame, BasisSet::STO3G};
    const int N = pack.n_ao_cart();
    MolecularGrid grid(pack.atoms.data(), static_cast<int>(pack.atoms.size()),
                       GridLevel::SG0, /*n_radial=*/30, /*n_angular=*/50);

    XCGridBasis lda;
    lda.build(grid, pack.shells, N, /*is_gga=*/false);
    EXPECT_TRUE(lda.built) << "build() must latch built=true";
    EXPECT_FALSE(lda.gga) << "LDA build must leave gga=false";

    XCGridBasis gga;
    gga.build(grid, pack.shells, N, /*is_gga=*/true);
    EXPECT_TRUE(gga.built) << "build() must latch built=true";
    EXPECT_TRUE(gga.gga) << "GGA build must set gga=true";
}

#else

#include <gtest/gtest.h>
TEST(TestXcGridEngine, SkippedNoBuildHF) {
    GTEST_SKIP()
        << "ATV_BUILD_HF=OFF — XC grid engine tests require -DATV_BUILD_HF=ON";
}

#endif // ATV_HAS_HF

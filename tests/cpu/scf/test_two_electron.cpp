/// @file tests/cpu/scf/test_two_electron.cpp
/// @brief Equivalence guard for internal::IntegralWorkspace::eri_dense (src/cpu/scf/two_electron.cpp).
///
/// Spec scf-review-07-posthf-perf, item 17: the dense N⁴ AO ERI tensor is
/// re-implemented from a per-AO scalar `native::eri` double shell-pair loop to a
/// `native::eri_block` shell-block assembly (amortized Boys/R-table + OpenMP,
/// 8-fold-symmetry scatter into disjoint cells). Both paths are independently
/// benchmarked element-wise against libcint (McMurchie–Davidson scalar vs
/// Rys/xsimd block), so on s+p bases they must agree to ≤1e-12 per element —
/// that is the equivalence judge here.
///
/// These tests pass TODAY on the scalar implementation (the reference tensor is
/// built from the same `native::eri` the current body calls, so the delta is
/// pure FP round-off ~1e-16) and must still pass AFTER the block/GEMM rewrite:
/// they are equivalence guards, not compile-RED. The anchor literals were
/// captured from the current scalar implementation on 2026-07-24 and guard
/// same-direction drift of BOTH implementations.

#ifdef ATV_HAS_HF

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/native_integrals.h"

using namespace ATV;
using cpu::scf::internal::IntegralWorkspace;

namespace {

/// @brief H2 along x at a given bond length [Angstrom], STO-3G (pure-s: 2 AO).
cpu::Frame make_h2(double bond_angstrom) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = bond_angstrom;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.box = cpu::Box::cubic(20.0);
    return f;
}

/// @brief Water, STO-3G (7 AO: O 1s/2s/2px/2py/2pz + 2×H 1s) — contains p shells.
cpu::Frame make_h2o() {
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = 0.757;
    f.span(kAtom_y)[1] = 0.586;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_x)[2] = -0.757;
    f.span(kAtom_y)[2] = 0.586;
    f.span(kAtom_z)[2] = 0.0;
    f.box = cpu::Box::cubic(20.0);
    return f;
}

/// @brief Reference dense ERI tensor, assembled DIRECTLY from per-AO
///        `native::eri` calls over the pack's shell list — the still-present,
///        libcint-benchmarked scalar kernel. This is deliberately NOT the
///        (about-to-be-rewritten) `eri_dense`; it is the independent judge.
/// @param pack internal::IntegralWorkspace built with compute_eri=true on a Cartesian basis.
/// @return (n × n × n × n) chemist-notation tensor [Hartree].
xt::xtensor<double, 4> reference_eri(const cpu::scf::internal::IntegralWorkspace& pack) {
    const int n = pack.n_basis;
    xt::xtensor<double, 4> ref = xt::zeros<double>(
        {static_cast<std::size_t>(n), static_cast<std::size_t>(n),
         static_cast<std::size_t>(n), static_cast<std::size_t>(n)});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    ref(i, j, k, l) =
                        cpu::scf::native::eri(pack.shells[i], pack.shells[j],
                                              pack.shells[k], pack.shells[l]);
    return ref;
}

/// @brief Largest element-wise |block − reference| over the whole tensor.
double max_abs_diff(const xt::xtensor<double, 4>& a,
                    const xt::xtensor<double, 4>& b) {
    const int n = static_cast<int>(a.shape(0));
    double m = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l) {
                    const double d = std::abs(a(i, j, k, l) - b(i, j, k, l));
                    if (d > m)
                        m = d;
                }
    return m;
}

// Equivalence judge (block vs scalar reference): FP reassociation only.
constexpr double kEriEquivTol = 1e-12;

// Anchor guard (same-direction drift of both implementations). Comfortably
// tighter than the block-vs-scalar 1e-12 reassociation bound would allow to
// slip; energy-exact FP tier.
constexpr double kEriAnchorTol = 1e-10;

// Hard-coded ERI anchors captured from the CURRENT scalar internal::IntegralWorkspace::eri_dense
// (H2/STO-3G @ 0.74 Å), 2026-07-24, GCC 14.3, double precision.
constexpr double kH2_ERI_0000 = 7.7460592980626808e-01;
constexpr double kH2_ERI_0011 = 5.6999487185769371e-01;
constexpr double kH2_ERI_0101 = 2.9759054552702741e-01;
constexpr double kH2_ERI_1111 = 7.7460592980626808e-01;

// H2O/STO-3G (pp|pp) anchors (O 2px index 2, 2py index 3), same capture.
constexpr double kH2O_ERI_2222 = 8.8015912773873739e-01;
constexpr double kH2O_ERI_2323 = 4.7444446970738241e-02;

} // namespace

// (a) Block-assembled dense ERI == per-AO scalar reference on a p-shell fixture.
TEST(TestEriDense, EriDenseMatchesScalarReference) {
    const cpu::Frame frame = make_h2o();
    const cpu::scf::internal::IntegralWorkspace pack{frame, cpu::scf::BasisSet::STO3G, 0,
                            /*compute_eri=*/true};

    ASSERT_EQ(pack.n_basis, 7);
    ASSERT_FALSE(pack.spherical())
        << "STO-3G must be Cartesian so pack.ERI is the raw dense tensor "
           "the per-AO reference can compare against directly";
    ASSERT_GT(pack.ERI.size(), 0u) << "compute_eri=true must materialize ERI";

    const xt::xtensor<double, 4> ref = reference_eri(pack);
    EXPECT_LE(max_abs_diff(pack.ERI, ref), kEriEquivTol)
        << "block-assembled eri_dense diverged from the scalar native::eri "
           "reference on H2O/STO-3G";
}

// (b) Pure-s fixture (H2/STO-3G) — exercises the eri_blk_ss_ss path.
TEST(TestEriDense, EriDensePureSFixture) {
    const cpu::Frame frame = make_h2(0.74);
    const cpu::scf::internal::IntegralWorkspace pack{frame, cpu::scf::BasisSet::STO3G, 0,
                            /*compute_eri=*/true};

    ASSERT_EQ(pack.n_basis, 2) << "H2/STO-3G is a pure-s 2-AO fixture";
    ASSERT_FALSE(pack.spherical());
    ASSERT_GT(pack.ERI.size(), 0u);

    const xt::xtensor<double, 4> ref = reference_eri(pack);
    EXPECT_LE(max_abs_diff(pack.ERI, ref), kEriEquivTol)
        << "block-assembled eri_dense diverged from the scalar reference on "
           "the pure-s H2/STO-3G fixture";
}

// (c) 8-fold permutation symmetry of the assembled tensor: the disjoint-cell
//     scatter must fill every mirror of a canonical block identically.
TEST(TestEriDense, EriDenseSymmetrySpotChecks) {
    const cpu::Frame frame = make_h2o();
    const cpu::scf::internal::IntegralWorkspace pack{frame, cpu::scf::BasisSet::STO3G, 0,
                            /*compute_eri=*/true};
    const auto& E = pack.ERI;

    // A handful of (i,j,k,l) tuples spanning s and p AOs on water.
    const int tuples[][4] = {
        {0, 2, 3, 4}, {2, 3, 4, 5}, {0, 1, 2, 6}, {2, 4, 3, 5}, {1, 3, 5, 6},
    };
    for (const auto& t : tuples) {
        const int i = t[0], j = t[1], k = t[2], l = t[3];
        const double v = E(i, j, k, l);
        // (ij|kl) == (kl|ij)
        EXPECT_NEAR(E(k, l, i, j), v, kEriEquivTol)
            << "(kl|ij) != (ij|kl) at " << i << j << k << l;
        // (ij|kl) == (ji|lk)
        EXPECT_NEAR(E(j, i, l, k), v, kEriEquivTol)
            << "(ji|lk) != (ij|kl) at " << i << j << k << l;
        // (ij|kl) == (ji|kl)
        EXPECT_NEAR(E(j, i, k, l), v, kEriEquivTol)
            << "(ji|kl) != (ij|kl) at " << i << j << k << l;
        // (ij|kl) == (ij|lk)
        EXPECT_NEAR(E(i, j, l, k), v, kEriEquivTol)
            << "(ij|lk) != (ij|kl) at " << i << j << k << l;
        // (ij|kl) == (lk|ji)
        EXPECT_NEAR(E(l, k, j, i), v, kEriEquivTol)
            << "(lk|ji) != (ij|kl) at " << i << j << k << l;
    }
}

// (d) Hard-coded ERI anchors (H2/STO-3G + one H2O (pp|pp)) — guard both
//     implementations against same-direction drift the equivalence judge
//     would miss.
TEST(TestEriDense, EriDenseAnchorValues) {
    const cpu::scf::internal::IntegralWorkspace h2{make_h2(0.74), cpu::scf::BasisSet::STO3G, 0,
                          /*compute_eri=*/true};
    EXPECT_NEAR(h2.ERI(0, 0, 0, 0), kH2_ERI_0000, kEriAnchorTol);
    EXPECT_NEAR(h2.ERI(0, 0, 1, 1), kH2_ERI_0011, kEriAnchorTol);
    EXPECT_NEAR(h2.ERI(0, 1, 0, 1), kH2_ERI_0101, kEriAnchorTol);
    EXPECT_NEAR(h2.ERI(1, 1, 1, 1), kH2_ERI_1111, kEriAnchorTol);

    const cpu::scf::internal::IntegralWorkspace h2o{make_h2o(), cpu::scf::BasisSet::STO3G, 0,
                           /*compute_eri=*/true};
    EXPECT_NEAR(h2o.ERI(2, 2, 2, 2), kH2O_ERI_2222, kEriAnchorTol);
    EXPECT_NEAR(h2o.ERI(2, 3, 2, 3), kH2O_ERI_2323, kEriAnchorTol);
}

#else // !ATV_HAS_HF

#include <gtest/gtest.h>
TEST(TestEriDense, Disabled) {
    GTEST_SKIP() << "eri_dense equivalence tests require -DATV_BUILD_HF=ON";
}

#endif

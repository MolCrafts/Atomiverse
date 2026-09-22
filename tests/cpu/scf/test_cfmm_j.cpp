/// @file tests/cpu/scf/test_cfmm_j.cpp
/// @brief Unit tests for internal::CfmmJ (link 21): s|s monopole = overlap,
///        complementary near/far tables, H2O/STO-3G single j(P) vs dense
///        direct J, and CFMM × RI mutual exclusion. No converged SCF.

#if defined(ATV_HAS_HF)

#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/internal/cfmm_j.h"
#include "atomiverse/cpu/scf/internal/hamiltonian_builder.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/scf_enums.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::scf;

namespace {

cpu::Frame make_h2o() {
    cpu::Frame frame(3);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_x)[2] = 0.0;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.box = cpu::Box::cubic(100.0);
    return frame;
}

/// Two H atoms far apart so pure s|s far-field pairs exist
/// (need ≳ 50 Bohr ≈ 26 Å under the CFMM absolute far-field floor).
cpu::Frame make_h2_far() {
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 40.0; // Å ≳ 75 Bohr — exercises multipole far
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.0;
    frame.box = cpu::Box::cubic(100.0);
    return frame;
}

double max_abs_diff(const RealMatrix& A, const RealMatrix& B) {
    double m = 0.0;
    const int n = static_cast<int>(A.shape(0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            m = std::max(m, std::abs(A(i, j) - B(i, j)));
    return m;
}

} // namespace

TEST(CfmmJ, SSMonopoleEqualsOverlap) {
    // H2 STO-3G: all AOs are s — unit monopole must match overlap.
    auto frame = make_h2_far();
    internal::IntegralWorkspace pack(frame, BasisSet::STO3G, 0,
                                     /*compute_eri=*/false);
    internal::CfmmJ cfmm(pack.shells, /*order=*/6, /*threshold=*/1e-9);
    cfmm.prepare();
    ASSERT_GT(cfmm.n_distributions(), 0);
    for (int d = 0; d < cfmm.n_distributions(); ++d) {
        if (!cfmm.distribution_spherical(d))
            continue;
        auto [i, j] = cfmm.distribution_aos(d);
        const double S =
            native::overlap(pack.shells[static_cast<std::size_t>(i)],
                            pack.shells[static_cast<std::size_t>(j)]);
        EXPECT_NEAR(cfmm.unit_monopole(d), S, 1e-12)
            << "dist " << d << " ao (" << i << "," << j << ")";
    }
}

TEST(CfmmJ, NearFarComplementaryNoDoubleCount) {
    auto frame = make_h2_far();
    internal::IntegralWorkspace pack(frame, BasisSet::STO3G, 0,
                                     /*compute_eri=*/false);
    // Loose threshold → smaller extents → more far pairs.
    internal::CfmmJ cfmm(pack.shells, /*order=*/6, /*threshold=*/1e-3);
    cfmm.prepare();
    const int nd = cfmm.n_distributions();
    // Expected canonical pair count after Schwarz (upper bound).
    std::set<std::pair<int, int>> near_set, far_set;
    for (const auto& p : cfmm.near_pairs()) {
        EXPECT_GE(p.first, p.second);
        EXPECT_TRUE(near_set.insert(p).second) << "duplicate near";
    }
    for (const auto& p : cfmm.far_pairs()) {
        EXPECT_GE(p.first, p.second);
        EXPECT_TRUE(far_set.insert(p).second) << "duplicate far";
        EXPECT_EQ(near_set.count(p), 0u) << "pair in both near and far";
    }
    // Far path should be exercised on this stretched H2.
    EXPECT_GT(cfmm.far_pairs().size(), 0u);
    EXPECT_GT(cfmm.near_pairs().size(), 0u);
    // Union size ≤ nd*(nd+1)/2 (Schwarz may drop some).
    const std::size_t union_n = near_set.size() + far_set.size();
    EXPECT_LE(union_n, static_cast<std::size_t>(nd) * (nd + 1) / 2);
}

TEST(CfmmJ, H2OSto3GMatchesDirectJ) {
    auto frame = make_h2o();
    internal::IntegralWorkspace pack(frame, BasisSet::STO3G, 0,
                                     /*compute_eri=*/false);
    const int n = pack.n_ao_cart();

    // A simple positive-semidefinite density (not from SCF).
    RealMatrix P = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i)
        P(i, i) = 0.5;
    // Off-diagonal coupling so the full matrix is exercised.
    if (n > 1) {
        P(0, 1) = P(1, 0) = 0.1;
    }

    ShellPairList sp(pack);
    RealMatrix J_direct;
    sp.j(P, J_direct);

    internal::CfmmJ cfmm(pack.shells, /*order=*/10, /*threshold=*/1e-9);
    cfmm.prepare();
    RealMatrix J_cfmm = cfmm.j(P);

    const double max_diff = max_abs_diff(J_cfmm, J_direct);
    EXPECT_LE(max_diff, 1e-8) << "max |J_CFMM - J_direct| = " << max_diff;
}

TEST(CfmmJ, CfmmTimesRiThrows) {
    auto frame = make_h2o();
    internal::IntegralWorkspace pack(frame, BasisSet::STO3G, 0,
                                     /*compute_eri=*/false);
    internal::HamiltonianBuilder hb;
    EXPECT_THROW(hb.prepare(pack, /*is_hf=*/true, GridLevel::SG1, 0, 0, nullptr,
                            RI::J, AuxBasis::Def2JKFIT, Coulomb::CFMM, 10,
                            1e-9),
                 std::runtime_error);
}

TEST(CfmmJ, DefaultDirectUnaffected) {
    // prepare with defaults must not build a CFMM workspace path that
    // changes the Direct branch (smoke: no throw, step returns finite F).
    auto frame = make_h2o();
    internal::IntegralWorkspace pack(frame, BasisSet::STO3G, 0,
                                     /*compute_eri=*/false);
    const int n = pack.n_basis;
    RealMatrix P = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i)
        P(i, i) = 1.0;

    internal::HamiltonianBuilder hb;
    hb.prepare(pack, /*is_hf=*/true, GridLevel::SG1, 0, 0, nullptr);
    auto fr = hb.step(pack, /*restricted=*/true, /*is_hf=*/true, XCType::LSDA,
                      /*hybrid_alpha_override=*/-1.0, P, P);
    EXPECT_TRUE(std::isfinite(fr.F_alpha(0, 0)));
}

#else

#include <gtest/gtest.h>
TEST(CfmmJ, SkippedWithoutHF) { GTEST_SKIP() << "ATV_HAS_HF off"; }

#endif

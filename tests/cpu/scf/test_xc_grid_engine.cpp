/// @file tests/cpu/scf/test_xc_grid_engine.cpp
/// @brief Unit tests of the blocked molecular XC engine. Layout mirrors
///        `cpu/scf/xc_grid_engine.h`.
///
/// - The geometry-fixed cache (`XCGridBasis::build` once, then
///   `evaluate_xc_batched(const XCGridBasis&, ...)`) is bit-for-bit
///   identical to the build-per-call overload (H2/STO-3G, LDA and PBE).
/// - The blocked, AO-screened GEMM engine matches an independent
///   point-by-point quadrature of the scalar `evaluate_xc` (water/def2-SVP
///   Cartesian d shells, unequal spin densities, LDA and GGA).
/// - The restricted single-channel path equals the collinear two-channel
///   evaluation of the same closed-shell density.

#ifdef ATV_HAS_HF

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_eval.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/xc_grid_engine.h"
#include "atomiverse/cpu/xc/xc_functional.h"

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
    f.span(ATV::kAtom_atomic_number)[0] = 1;
    f.span(ATV::kAtom_atomic_number)[1] = 1;
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
void check_cache_equivalence(ATV::cpu::xc::XCType xc, bool is_gga,
                             const char* label) {
    auto frame = make_h2();
    auto pack = internal::IntegralWorkspace{frame, BasisSet::STO3G};
    const int N = pack.n_ao_cart();
    MolecularGrid grid(pack.atoms.data(), static_cast<int>(pack.atoms.size()),
                       GridLevel::SG0, /*n_radial=*/30, /*n_angular=*/50);

    auto Pa = make_density(static_cast<std::size_t>(N));
    auto Pb = Pa;

    const ATV::cpu::xc::BuiltinXc functional(xc);
    // Rebuild Phi/dPhi inside this call.
    XCGridResult rebuild =
        evaluate_xc_batched(grid, pack.shells, N, Pa, Pb, functional);

    // Cache AO-on-grid samples once; both paths consume the same XC provider.
    XCGridBasis aog;
    aog.build(grid, pack.shells, N, is_gga);
    XCGridResult cached = evaluate_xc_batched(aog, grid, Pa, Pb, functional);

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
    check_cache_equivalence(ATV::cpu::xc::XCType::LSDA, /*is_gga=*/false,
                            "LDA cache vs rebuild");
}

TEST(TestXcGridEngine, CachedBasisMatchesRebuild_PBE) {
    // GGA path also exercises the cached AO gradient blocks.
    check_cache_equivalence(ATV::cpu::xc::XCType::PBE, /*is_gga=*/true,
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

namespace {

cpu::Frame make_water() {
    cpu::Frame f(3);
    const int z[] = {8, 1, 1};
    const double xyz[][3] = {
        {0, 0, 0.1173}, {0, 0.7572, -0.4692}, {0, -0.7572, -0.4692}};
    for (int i = 0; i < 3; ++i) {
        f.span(ATV::kAtom_atomic_number)[i] = z[i];
        f.span(ATV::kAtom_x)[i] = xyz[i][0];
        f.span(ATV::kAtom_y)[i] = xyz[i][1];
        f.span(ATV::kAtom_z)[i] = xyz[i][2];
    }
    f.box = ATV::cpu::Box::cubic(30.0);
    return f;
}

xt::xtensor<double, 2> spin_density(std::size_t n, double phase) {
    xt::xtensor<double, 2> P = xt::zeros<double>({n, n});
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            P(i, j) = 0.04 * std::cos(0.3 * (double(i) - double(j)) + phase)
                      + (i == j ? 0.15 : 0.0);
    return xt::xtensor<double, 2>(0.5 * (P + xt::transpose(P)));
}

/// Literal quadrature: every AO at every point, scalar functional, no
/// blocks, no screening, no BLAS.
XCGridResult pointwise(const MolecularGrid& grid,
                       const std::vector<native::Shell>& shells,
                       const xt::xtensor<double, 2>& Pa,
                       const xt::xtensor<double, 2>& Pb,
                       ATV::cpu::xc::XCType type, bool gga) {
    const std::size_t n = shells.size();
    XCGridResult r;
    r.Vxc_alpha = xt::zeros<double>({n, n});
    r.Vxc_beta = xt::zeros<double>({n, n});
    r.E_xc = r.n_alpha_integrated = r.n_beta_integrated = 0;
    std::vector<double> v(n),
        g[3] = {std::vector<double>(n), std::vector<double>(n),
                std::vector<double>(n)};
    for (const auto& pt : grid.points) {
        for (std::size_t mu = 0; mu < n; ++mu) {
            if (gga)
                eval_basis_val_grad(shells[mu], pt.x, pt.y, pt.z, v[mu],
                                    g[0][mu], g[1][mu], g[2][mu]);
            else
                v[mu] = eval_basis(shells[mu], pt.x, pt.y, pt.z);
        }
        double rho[2] = {0, 0}, grad[2][3] = {};
        const xt::xtensor<double, 2>* P[2] = {&Pa, &Pb};
        for (int s = 0; s < 2; ++s)
            for (std::size_t mu = 0; mu < n; ++mu)
                for (std::size_t nu = 0; nu < n; ++nu) {
                    const double p = (*P[s])(mu, nu);
                    rho[s] += p * v[mu] * v[nu];
                    if (gga)
                        for (int d = 0; d < 3; ++d)
                            grad[s][d] += 2.0 * p * v[nu] * g[d][mu];
                }
        rho[0] = std::max(rho[0], 0.0);
        rho[1] = std::max(rho[1], 0.0);
        double saa = 0, sab = 0, sbb = 0;
        for (int d = 0; d < 3; ++d) {
            saa += grad[0][d] * grad[0][d];
            sab += grad[0][d] * grad[1][d];
            sbb += grad[1][d] * grad[1][d];
        }
        const auto x =
            ATV::cpu::xc::evaluate_xc(type, rho[0], rho[1], saa, sab, sbb);
        const double w = pt.weight;
        r.E_xc += w * x.energy_density;
        r.n_alpha_integrated += w * rho[0];
        r.n_beta_integrated += w * rho[1];
        double ca[3] = {}, cb[3] = {};
        for (int d = 0; d < 3; ++d) {
            ca[d] = 2 * x.vsigma_aa * grad[0][d] + x.vsigma_ab * grad[1][d];
            cb[d] = 2 * x.vsigma_bb * grad[1][d] + x.vsigma_ab * grad[0][d];
        }
        for (std::size_t mu = 0; mu < n; ++mu)
            for (std::size_t nu = 0; nu < n; ++nu) {
                double a = x.vrho_alpha * v[mu] * v[nu];
                double b = x.vrho_beta * v[mu] * v[nu];
                if (gga)
                    for (int d = 0; d < 3; ++d) {
                        const double pq = v[mu] * g[d][nu] + g[d][mu] * v[nu];
                        a += ca[d] * pq;
                        b += cb[d] * pq;
                    }
                r.Vxc_alpha(mu, nu) += w * a;
                r.Vxc_beta(mu, nu) += w * b;
            }
    }
    return r;
}

void expect_near_2d(const xt::xtensor<double, 2>& a,
                    const xt::xtensor<double, 2>& b, double tol,
                    const char* what) {
    ASSERT_EQ(a.shape(), b.shape()) << what;
    for (std::size_t i = 0; i < a.size(); ++i)
        EXPECT_NEAR(a.data()[i], b.data()[i], tol) << what << " element " << i;
}

} // namespace

TEST(TestXcGridEngine, BlockedEngineMatchesPointwiseQuadrature) {
    auto frame = make_water();
    auto pack = internal::IntegralWorkspace{frame, BasisSet::Pople631Gs};
    const std::size_t N = pack.n_ao_cart();
    MolecularGrid grid(pack.atoms.data(), static_cast<int>(pack.atoms.size()),
                       GridLevel::SG0, 30, 50);
    ASSERT_GT(grid.points.size(),
              std::size_t(2 * XCGridBasis::kBlockSize)); // several blocks
    const auto Pa = spin_density(N, 0.0), Pb = spin_density(N, 0.8);
    for (auto [type, gga] : {std::pair{ATV::cpu::xc::XCType::LSDA, false},
                             std::pair{ATV::cpu::xc::XCType::PBE, true}}) {
        const ATV::cpu::xc::BuiltinXc functional(type);
        const XCGridResult got =
            evaluate_xc_batched(grid, pack.shells, N, Pa, Pb, functional);
        const XCGridResult ref =
            pointwise(grid, pack.shells, Pa, Pb, type, gga);
        expect_near_2d(got.Vxc_alpha, ref.Vxc_alpha, 1e-10, "Vxc_alpha");
        expect_near_2d(got.Vxc_beta, ref.Vxc_beta, 1e-10, "Vxc_beta");
        EXPECT_NEAR(got.E_xc, ref.E_xc, 1e-10) << (gga ? "GGA" : "LDA");
        EXPECT_NEAR(got.n_alpha_integrated, ref.n_alpha_integrated, 1e-10);
        EXPECT_NEAR(got.n_beta_integrated, ref.n_beta_integrated, 1e-10);
    }
}

TEST(TestXcGridEngine, RestrictedSingleChannelEqualsCollinear) {
    auto frame = make_water();
    auto pack = internal::IntegralWorkspace{frame, BasisSet::Pople631Gs};
    const std::size_t N = pack.n_ao_cart();
    MolecularGrid grid(pack.atoms.data(), static_cast<int>(pack.atoms.size()),
                       GridLevel::SG0, 30, 50);
    const auto P = spin_density(N, 0.3);
    for (auto type :
         {ATV::cpu::xc::XCType::LSDA, ATV::cpu::xc::XCType::B3LYP}) {
        const ATV::cpu::xc::BuiltinXc functional(type);
        XCGridBasis aog;
        aog.build(grid, pack.shells, static_cast<int>(N),
                  type != ATV::cpu::xc::XCType::LSDA);
        const auto r = evaluate_xc_batched(aog, grid, P, P, functional,
                                           ATV::electronic::XcSpin::Restricted);
        const auto c = evaluate_xc_batched(aog, grid, P, P, functional,
                                           ATV::electronic::XcSpin::Collinear);
        expect_near_2d(r.Vxc_alpha, c.Vxc_alpha, 1e-12, "restricted Va");
        expect_near_2d(r.Vxc_beta, c.Vxc_beta, 1e-12, "restricted Vb");
        EXPECT_NEAR(r.E_xc, c.E_xc, 1e-12);
        EXPECT_EQ(r.n_alpha_integrated, r.n_beta_integrated);
    }
}

#else

#include <gtest/gtest.h>
TEST(TestXcGridEngine, SkippedNoBuildHF) {
    GTEST_SKIP()
        << "ATV_BUILD_HF=OFF — XC grid engine tests require -DATV_BUILD_HF=ON";
}

#endif // ATV_HAS_HF

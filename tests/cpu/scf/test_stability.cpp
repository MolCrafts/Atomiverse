/// @file tests/cpu/scf/test_stability.cpp
/// @brief Unit checks for cpu::scf::internal::StabilityAnalysis and the
///        RHF::stability() guards: Seeger-Pople channel split on H2/STO-3G,
///        Davidson-vs-dense lowest-root parity, no-virtuals / before-run
///        throws.
///
/// NO SCF loop runs here (the full RHF.run()+stability() path lives in
/// regressions/scf/h2_rhf_stability). The converged H2/STO-3G determinant
/// is symmetry-exact in closed form (see test_orbital_hessian.cpp for the
/// derivation): φ_g = (χ₁+χ₂)/√(2(1+S₁₂)), φ_u = (χ₁−χ₂)/√(2(1−S₁₂)),
/// ε_p = c_pᵀ F c_p from ONE Fock build over the symmetry-fixed density.
///
/// PySCF 2.14.0 pins (H2/STO-3G, cart=True, RHF conv_tol=1e-12; (A±B)
/// built from ao2mo MO integrals with the same closed-shell formulas —
/// commands in regressions/scf/h2_rhf_stability/REFERENCE.md):
///   R = 0.735 Å: λ_min[¹(A+B)] = +1.1351770407 Ha,
///                λ_min[³(A+B)] = +0.4114522416 Ha  (both stable)
///   R = 1.5 Å:   λ_min[¹(A+B)] = +0.7088965814 Ha,
///                λ_min[³(A+B)] = −0.2092471629 Ha  (the Seeger-Pople trap:
///                internal stable, external unstable)

#ifdef ATV_HAS_HF

#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/orbital_hessian.h"
#include "atomiverse/cpu/scf/internal/stability.h"
#include "atomiverse/cpu/scf/rhf.h"

using namespace ATV;
using cpu::scf::internal::IntegralWorkspace;
using cpu::scf::RealMatrix;
using cpu::scf::RealVector;
using cpu::scf::SlaterDeterminant;
using cpu::scf::internal::Channel;
using cpu::scf::internal::OrbitalHessian;
using cpu::scf::internal::StabilityAnalysis;

namespace {

cpu::Frame make_h2(double r_angstrom) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = r_angstrom;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.box = cpu::Box::cubic(20.0);
    return f;
}

struct H2Reference {
    SlaterDeterminant det;
    xt::xtensor<double, 4> eri;
};

H2Reference make_h2_reference(double r_angstrom) {
    auto frame = make_h2(r_angstrom);
    cpu::scf::internal::IntegralWorkspace pack{frame, cpu::scf::BasisSet::STO3G, 0,
                      /*compute_eri=*/true};
    const double s12 = pack.S(0, 1);
    const double cg = 1.0 / std::sqrt(2.0 * (1.0 + s12));
    const double cu = 1.0 / std::sqrt(2.0 * (1.0 - s12));
    RealMatrix C = xt::zeros<double>({2, 2});
    C(0, 0) = cg;
    C(1, 0) = cg;
    C(0, 1) = cu;
    C(1, 1) = -cu;
    RealMatrix P = xt::zeros<double>({2, 2});
    for (int mu = 0; mu < 2; ++mu)
        for (int nu = 0; nu < 2; ++nu)
            P(mu, nu) = 2.0 * C(mu, 0) * C(nu, 0);
    RealMatrix G = xt::zeros<double>({2, 2});
    pack.coulomb_exchange_fused(P, G);
    const RealMatrix F = pack.Hcore + G;
    RealVector eps = xt::zeros<double>({2});
    for (int p = 0; p < 2; ++p) {
        double e = 0.0;
        for (int mu = 0; mu < 2; ++mu)
            for (int nu = 0; nu < 2; ++nu)
                e += C(mu, p) * F(mu, nu) * C(nu, p);
        eps(p) = e;
    }
    RealVector occ = xt::zeros<double>({2});
    occ(0) = 1.0;
    return {SlaterDeterminant{C, eps, occ, 1}, pack.ERI};
}

/// Deterministic LCG in [-0.5, 0.5].
struct Lcg {
    std::uint64_t s;
    explicit Lcg(std::uint64_t seed) : s(seed) {}
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>((s >> 11) & 0xFFFFFFFFFFFFFULL)
                   / static_cast<double>(0x10000000000000ULL)
               - 0.5;
    }
};

struct Synthetic {
    SlaterDeterminant det;
    xt::xtensor<double, 4> eri;
};

Synthetic make_synthetic(int n, int no, std::uint64_t seed) {
    Lcg rng(seed);
    RealMatrix C = xt::zeros<double>({n, n});
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            C(r, c) = rng.next() + (r == c ? 1.0 : 0.0);
    RealVector eps = xt::zeros<double>({static_cast<std::size_t>(n)});
    for (int p = 0; p < n; ++p)
        eps(p) = -1.0 + 0.3 * p;
    RealVector occ = xt::zeros<double>({static_cast<std::size_t>(n)});
    for (int i = 0; i < no; ++i)
        occ(i) = 1.0;
    xt::xtensor<double, 4> raw = xt::zeros<double>({n, n, n, n});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    raw(i, j, k, l) = 0.1 * rng.next();
    xt::xtensor<double, 4> eri = xt::zeros<double>({n, n, n, n});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    eri(i, j, k, l) =
                        (raw(i, j, k, l) + raw(j, i, k, l) + raw(i, j, l, k)
                         + raw(j, i, l, k) + raw(k, l, i, j) + raw(l, k, i, j)
                         + raw(k, l, j, i) + raw(l, k, j, i))
                        / 8.0;
    return {SlaterDeterminant{C, eps, occ, no}, eri};
}

} // namespace

TEST(StabilityAnalysis, EquilibriumH2BothChannelsStable) {
    // ac-003 anchor: R = 0.735 Å — both channels non-negative, and the
    // eigenvalues match the PySCF pins to 1e-6 Ha.
    auto ref = make_h2_reference(0.735);
    OrbitalHessian h{ref.det, ref.eri};
    StabilityAnalysis stab{h};
    const double internal = stab.lowest(Channel::singlet);
    const double external = stab.lowest(Channel::triplet);
    EXPECT_GE(internal, 0.0);
    EXPECT_GE(external, 0.0);
    EXPECT_NEAR(internal, 1.1351770407, 1e-6);
    EXPECT_NEAR(external, 0.4114522416, 1e-6);
}

TEST(StabilityAnalysis, SeegerPopleTrapStretchedH2) {
    // The Seeger-Pople classification content itself: at R = 1.5 Å the RHF
    // solution is INTERNALLY stable yet EXTERNALLY (triplet) unstable.
    auto ref = make_h2_reference(1.5);
    OrbitalHessian h{ref.det, ref.eri};
    StabilityAnalysis stab{h};
    const double internal = stab.lowest(Channel::singlet);
    const double external = stab.lowest(Channel::triplet);
    EXPECT_GE(internal, 0.0);
    EXPECT_LT(external, 0.0);
    EXPECT_NEAR(internal, 0.7088965814, 1e-6);
    EXPECT_NEAR(external, -0.2092471629, 1e-6);
}

TEST(StabilityAnalysis, DavidsonMatchesDense) {
    // n_ov = 4 × 8 = 32 synthetic system: the matrix-free Davidson lowest
    // root must agree with the dense eigensolve to 1e-10, both channels.
    auto syn = make_synthetic(12, 4, 17);
    OrbitalHessian h{syn.det, syn.eri};
    StabilityAnalysis stab{h};
    ASSERT_EQ(h.n_ov(), 32);
    for (Channel ch : {Channel::singlet, Channel::triplet})
        EXPECT_NEAR(stab.lowest_davidson(ch), stab.lowest_dense(ch), 1e-10);
}

TEST(StabilityAnalysis, DispatchHonorsDenseThreshold) {
    auto syn = make_synthetic(8, 3, 19);
    OrbitalHessian h{syn.det, syn.eri};
    StabilityAnalysis stab{h};
    const double dense = stab.lowest(Channel::triplet); // n_ov = 15 ≤ 512
    stab.dense_threshold = 0;                           // force Davidson
    const double davidson = stab.lowest(Channel::triplet);
    EXPECT_NEAR(dense, davidson, 1e-10);
}

TEST(StabilityAnalysis, LowestVectorIsNormalizedEigenvector) {
    auto ref = make_h2_reference(1.5);
    OrbitalHessian h{ref.det, ref.eri};
    StabilityAnalysis stab{h};
    const RealVector x = stab.lowest_vector(Channel::triplet);
    ASSERT_EQ(static_cast<int>(x.size()), h.n_ov());
    double norm = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        norm += x(i) * x(i);
    EXPECT_NEAR(norm, 1.0, 1e-12);
    // A·x = λ·x for the dense build.
    const RealMatrix M = h.dense_plus(Channel::triplet);
    const double lambda = stab.lowest(Channel::triplet);
    for (int r = 0; r < h.n_ov(); ++r) {
        double acc = 0.0;
        for (int c = 0; c < h.n_ov(); ++c)
            acc += M(r, c) * x(c);
        EXPECT_NEAR(acc, lambda * x(r), 1e-10);
    }
}

TEST(RhfStability, ThrowsBeforeRun) {
    cpu::scf::RHF rhf;
    EXPECT_THROW(rhf.stability(), std::runtime_error);
}

TEST(OrbitalHessian, NoVirtualsThrowsWithClearMessage) {
    // He/STO-3G shape: one basis function, one doubly occupied orbital —
    // no occ→virt rotation exists; the kernel must refuse (no fallback).
    RealMatrix C = xt::zeros<double>({1, 1});
    C(0, 0) = 1.0;
    RealVector eps = xt::zeros<double>({1});
    eps(0) = -0.876;
    RealVector occ = xt::zeros<double>({1});
    occ(0) = 1.0;
    SlaterDeterminant det{C, eps, occ, 1};
    xt::xtensor<double, 4> eri = xt::zeros<double>({1, 1, 1, 1});
    eri(0, 0, 0, 0) = 0.77;
    try {
        OrbitalHessian h{det, eri};
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("no virtual orbitals"),
                  std::string::npos);
    }
}

TEST(OrbitalHessian, UnrestrictedDeterminantThrows) {
    RealMatrix C = xt::zeros<double>({2, 2});
    C(0, 0) = 1.0;
    C(1, 1) = 1.0;
    RealVector eps = xt::zeros<double>({2});
    RealVector fa = xt::zeros<double>({2});
    fa(0) = 1.0;
    RealVector fb = xt::zeros<double>({2});
    fb(0) = 1.0;
    SlaterDeterminant det{C, eps, fa, 1, C, eps, fb, 1};
    xt::xtensor<double, 4> eri = xt::zeros<double>({2, 2, 2, 2});
    EXPECT_THROW(OrbitalHessian(det, eri), std::invalid_argument);
}

#endif // ATV_HAS_HF

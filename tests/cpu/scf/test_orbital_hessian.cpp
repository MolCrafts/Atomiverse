/// @file tests/cpu/scf/test_orbital_hessian.cpp
/// @brief Unit checks for the MO orbital-Hessian A/B kernel
///        (cpu::scf::internal::OrbitalHessian): dense-vs-sigma parity,
///        element-wise formula identity on H2/STO-3G, channel distinctness.
///
/// NO SCF loop runs here. The H2/STO-3G converged determinant is built in
/// closed form from symmetry: with two identical s-type AOs the occupied MO
/// is fixed by D∞h symmetry as φ_g = (χ₁ + χ₂)/√(2(1+S₁₂)) and the virtual
/// as φ_u = (χ₁ − χ₂)/√(2(1−S₁₂)), independent of any iteration. The
/// orbital energies follow from ONE Fock build over the symmetry-determined
/// density (ε_p = c_pᵀ F c_p; the density is already self-consistent, so
/// this single build IS the converged Fock). Integrals come from
/// IntegralPack (internal workspace — allowed in unit tests).

#ifdef ATV_HAS_HF

#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/orbital_hessian.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

using namespace ATV;
using cpu::scf::IntegralPack;
using cpu::scf::RealMatrix;
using cpu::scf::RealVector;
using cpu::scf::SlaterDeterminant;
using cpu::scf::internal::Channel;
using cpu::scf::internal::OrbitalHessian;

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
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

/// The symmetry-exact converged H2/STO-3G determinant + its dense AO ERI.
struct H2Reference {
    SlaterDeterminant det;
    xt::xtensor<double, 4> eri;
};

H2Reference make_h2_reference(double r_angstrom) {
    auto frame = make_h2(r_angstrom);
    IntegralPack pack{frame, cpu::scf::BasisSet::STO3G, 0,
                      /*compute_eri=*/true};
    const double s12 = pack.S(0, 1);
    const double cg = 1.0 / std::sqrt(2.0 * (1.0 + s12));
    const double cu = 1.0 / std::sqrt(2.0 * (1.0 - s12));
    RealMatrix C = xt::zeros<double>({2, 2});
    C(0, 0) = cg;
    C(1, 0) = cg;
    C(0, 1) = cu;
    C(1, 1) = -cu;

    // One Fock build over the symmetry-determined (self-consistent) density.
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
        eps(p) = e; // c_p is unit-normalized in the S metric by construction
    }

    RealVector occ = xt::zeros<double>({2});
    occ(0) = 1.0;
    return {SlaterDeterminant{C, eps, occ, 1}, pack.ERI};
}

/// Deterministic LCG in [-0.5, 0.5] — no <random>, fully reproducible.
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

/// Synthetic kernel input: n AOs, no occupied, an 8-fold-symmetric random
/// "ERI" tensor and a random determinant — exercises the index plumbing on a
/// non-trivial n_ov without any physics.
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
    // Impose the real-orbital 8-fold permutational symmetry of (ij|kl).
    xt::xtensor<double, 4> eri = xt::zeros<double>({n, n, n, n});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    eri(i, j, k, l) =
                        (raw(i, j, k, l) + raw(j, i, k, l) + raw(i, j, l, k)
                         + raw(j, i, l, k) + raw(k, l, i, j)
                         + raw(l, k, i, j) + raw(k, l, j, i)
                         + raw(l, k, j, i))
                        / 8.0;
    return {SlaterDeterminant{C, eps, occ, no}, eri};
}

RealVector random_vector(int n, std::uint64_t seed) {
    Lcg rng(seed);
    RealVector v = xt::zeros<double>({static_cast<std::size_t>(n)});
    for (int i = 0; i < n; ++i)
        v(i) = rng.next();
    return v;
}

double max_abs_diff(const RealVector& a, const RealVector& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::abs(a(i) - b(i)));
    return m;
}

RealVector dense_times(const RealMatrix& M, const RealVector& v) {
    RealVector y = xt::zeros<double>({v.size()});
    for (std::size_t r = 0; r < M.shape()[0]; ++r) {
        double acc = 0.0;
        for (std::size_t c = 0; c < M.shape()[1]; ++c)
            acc += M(r, c) * v(c);
        y(r) = acc;
    }
    return y;
}

} // namespace

TEST(OrbitalHessian, Accessors) {
    auto ref = make_h2_reference(0.735);
    OrbitalHessian h{ref.det, ref.eri};
    EXPECT_EQ(h.n_occ(), 1);
    EXPECT_EQ(h.n_virt(), 1);
    EXPECT_EQ(h.n_ov(), 1);
}

TEST(OrbitalHessian, SingletAFormulaElementwiseH2) {
    // ¹A_ai,bj = δδ(ε_a−ε_i) + 2(ai|jb) − (ij|ab), checked element-wise
    // against MO integrals transformed independently in this test.
    auto ref = make_h2_reference(0.735);
    OrbitalHessian h{ref.det, ref.eri};
    const RealMatrix A =
        0.5 * (h.dense_plus(Channel::singlet) + h.dense_minus(Channel::singlet));

    const RealMatrix& C = ref.det.coefficients(SlaterDeterminant::Spin::alpha);
    // Independent AO→MO transform by explicit loops (n = 2, occ = {0},
    // virt = {1}): (pq|rs) = Σ C_up C_vq C_lr C_ss' eri(u,v,l,s').
    auto mo = [&](int p, int q, int r, int s) {
        double acc = 0.0;
        for (int u = 0; u < 2; ++u)
            for (int v = 0; v < 2; ++v)
                for (int l = 0; l < 2; ++l)
                    for (int m = 0; m < 2; ++m)
                        acc += C(u, p) * C(v, q) * C(l, r) * C(m, s)
                               * ref.eri(u, v, l, m);
        return acc;
    };
    const double v_aijb = mo(1, 0, 0, 1); // (ai|jb), a=MO1, i=j=MO0, b=MO1
    const double v_ijab = mo(0, 0, 1, 1); // (ij|ab)
    const double de = ref.det.orbital_energy(SlaterDeterminant::Spin::alpha, 1)
                      - ref.det.orbital_energy(SlaterDeterminant::Spin::alpha, 0);
    EXPECT_NEAR(A(0, 0), de + 2.0 * v_aijb - v_ijab, 1e-12);
    // ac-002: the δδ(ε_a − ε_i) diagonal term, coupling terms subtracted.
    EXPECT_NEAR(A(0, 0) - (2.0 * v_aijb - v_ijab), de, 1e-12);
}

TEST(OrbitalHessian, TripletFormulaElementwiseH2) {
    // ³A = δδ(ε_a−ε_i) − (ij|ab); ³B = −(aj|bi).
    auto ref = make_h2_reference(0.735);
    OrbitalHessian h{ref.det, ref.eri};
    const RealMatrix A3 =
        0.5 * (h.dense_plus(Channel::triplet) + h.dense_minus(Channel::triplet));
    const RealMatrix B3 =
        0.5 * (h.dense_plus(Channel::triplet) - h.dense_minus(Channel::triplet));

    const RealMatrix& C = ref.det.coefficients(SlaterDeterminant::Spin::alpha);
    auto mo = [&](int p, int q, int r, int s) {
        double acc = 0.0;
        for (int u = 0; u < 2; ++u)
            for (int v = 0; v < 2; ++v)
                for (int l = 0; l < 2; ++l)
                    for (int m = 0; m < 2; ++m)
                        acc += C(u, p) * C(v, q) * C(l, r) * C(m, s)
                               * ref.eri(u, v, l, m);
        return acc;
    };
    const double v_ijab = mo(0, 0, 1, 1);
    const double v_ajbi = mo(1, 0, 1, 0); // (aj|bi)
    const double de = ref.det.orbital_energy(SlaterDeterminant::Spin::alpha, 1)
                      - ref.det.orbital_energy(SlaterDeterminant::Spin::alpha, 0);
    EXPECT_NEAR(A3(0, 0), de - v_ijab, 1e-12);
    EXPECT_NEAR(B3(0, 0), -v_ajbi, 1e-12);
}

TEST(OrbitalHessian, DenseVsSigmaH2BothChannels) {
    auto ref = make_h2_reference(0.735);
    OrbitalHessian h{ref.det, ref.eri};
    const RealVector v = random_vector(h.n_ov(), 42);
    for (Channel ch : {Channel::singlet, Channel::triplet}) {
        EXPECT_LE(max_abs_diff(dense_times(h.dense_plus(ch), v),
                               h.sigma_plus(v, ch)),
                  1e-12);
        EXPECT_LE(max_abs_diff(dense_times(h.dense_minus(ch), v),
                               h.sigma_minus(v, ch)),
                  1e-12);
    }
}

TEST(OrbitalHessian, DenseVsSigmaSyntheticBothChannels) {
    // n_ov = 3 × 5 = 15: non-trivial index plumbing, random trial vectors.
    auto syn = make_synthetic(8, 3, 7);
    OrbitalHessian h{syn.det, syn.eri};
    ASSERT_EQ(h.n_ov(), 15);
    for (std::uint64_t seed : {1ULL, 2ULL, 3ULL}) {
        const RealVector v = random_vector(h.n_ov(), seed);
        for (Channel ch : {Channel::singlet, Channel::triplet}) {
            EXPECT_LE(max_abs_diff(dense_times(h.dense_plus(ch), v),
                                   h.sigma_plus(v, ch)),
                      1e-12);
            EXPECT_LE(max_abs_diff(dense_times(h.dense_minus(ch), v),
                                   h.sigma_minus(v, ch)),
                      1e-12);
        }
    }
}

TEST(OrbitalHessian, DensePlusIsSymmetric) {
    auto syn = make_synthetic(8, 3, 11);
    OrbitalHessian h{syn.det, syn.eri};
    for (Channel ch : {Channel::singlet, Channel::triplet}) {
        const RealMatrix M = h.dense_plus(ch);
        for (int p = 0; p < h.n_ov(); ++p)
            for (int q = 0; q < p; ++q)
                EXPECT_NEAR(M(p, q), M(q, p), 1e-13);
    }
}

TEST(OrbitalHessian, SingletAndTripletDiffer) {
    auto ref = make_h2_reference(0.735);
    OrbitalHessian h{ref.det, ref.eri};
    const RealMatrix s = h.dense_plus(Channel::singlet);
    const RealMatrix t = h.dense_plus(Channel::triplet);
    // The channels differ by 4(ai|bj) — order 1 Ha for H2/STO-3G.
    EXPECT_GT(std::abs(s(0, 0) - t(0, 0)), 1e-3);
}

TEST(OrbitalHessian, DiagonalPlusMatchesDense) {
    auto syn = make_synthetic(8, 3, 13);
    OrbitalHessian h{syn.det, syn.eri};
    for (Channel ch : {Channel::singlet, Channel::triplet}) {
        const RealMatrix M = h.dense_plus(ch);
        const RealVector d = h.diagonal_plus(ch);
        for (int p = 0; p < h.n_ov(); ++p)
            EXPECT_NEAR(d(p), M(p, p), 1e-13);
    }
}

TEST(OrbitalHessian, RejectsBadInputs) {
    auto ref = make_h2_reference(0.735);
    // Wrong ERI dimension.
    xt::xtensor<double, 4> bad = xt::zeros<double>({3, 3, 3, 3});
    EXPECT_THROW(OrbitalHessian(ref.det, bad), std::invalid_argument);
    // Wrong sigma vector length.
    OrbitalHessian h{ref.det, ref.eri};
    RealVector v2 = xt::zeros<double>({2});
    EXPECT_THROW(h.sigma_plus(v2, Channel::singlet), std::invalid_argument);
    EXPECT_THROW(h.sigma_minus(v2, Channel::triplet), std::invalid_argument);
}

#endif // ATV_HAS_HF

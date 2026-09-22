/// @file tests/cpu/scf/test_tdhf_pack.cpp
/// @brief Unit checks for the TDA + TDHF/RPA lowest-roots solvers
///        (cpu::scf::internal::TDHFPack) over the link-08 OrbitalHessian
///        kernel: dense TDA == A eigenvalues, reduced-form RPA == full
///        nonsymmetric [[A,B],[−B,−A]] spectrum, ω_RPA ≤ ω_TDA, Davidson ==
///        dense, and the triplet-instability throw.
///
/// NO SCF loop runs here (test_orbital_hessian precedent): the H2/STO-3G
/// converged determinant is symmetry-exact in closed form, and the larger
/// kernel is a deterministic synthetic (fixed LCG seed, no <random>).

#ifdef ATV_HAS_HF

#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/orbital_hessian.h"
#include "atomiverse/cpu/scf/internal/tdhf_pack.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

using namespace ATV;
using cpu::scf::RealMatrix;
using cpu::scf::RealVector;
using cpu::scf::SlaterDeterminant;
using cpu::scf::internal::Channel;
using cpu::scf::internal::IntegralWorkspace;
using cpu::scf::internal::OrbitalHessian;
using cpu::scf::internal::TDHFPack;

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

/// The symmetry-exact converged H2/STO-3G determinant + its dense AO ERI.
struct H2Reference {
    SlaterDeterminant det;
    xt::xtensor<double, 4> eri;
};

H2Reference make_h2_reference(double r_angstrom) {
    auto frame = make_h2(r_angstrom);
    cpu::scf::internal::IntegralWorkspace pack{frame, cpu::scf::BasisSet::STO3G,
                                               0,
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

/// Synthetic kernel input (test_orbital_hessian precedent): n AOs, no
/// occupied, an 8-fold-symmetric small random "ERI" and a well-gapped ε
/// ladder — diagonally dominant, so both channels stay stable and (A−B)
/// positive definite.
struct Synthetic {
    SlaterDeterminant det;
    xt::xtensor<double, 4> eri;
};

Synthetic make_synthetic(int n, int no, std::uint64_t seed, double scale) {
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
                    raw(i, j, k, l) = scale * rng.next();
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

/// Positive-real eigenvalues of the full nonsymmetric RPA supermatrix
/// [[A, B], [−B, −A]], ascending — the parity anchor for the reduced form.
std::vector<double> full_rpa_spectrum(const OrbitalHessian& h, Channel ch) {
    const RealMatrix Mp = h.dense_plus(ch);
    const RealMatrix Mm = h.dense_minus(ch);
    const std::size_t n = Mp.shape()[0];
    const RealMatrix A = 0.5 * (Mp + Mm);
    const RealMatrix B = 0.5 * (Mp - Mm);
    xt::xtensor<double, 2> S = xt::zeros<double>({2 * n, 2 * n});
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = 0; c < n; ++c) {
            S(r, c) = A(r, c);
            S(r, n + c) = B(r, c);
            S(n + r, c) = -B(r, c);
            S(n + r, n + c) = -A(r, c);
        }
    auto [w, V] = xt::linalg::eig(S); // complex spectrum ±ω
    std::vector<double> pos;
    for (std::size_t i = 0; i < 2 * n; ++i) {
        EXPECT_NEAR(w(i).imag(), 0.0, 1e-10)
            << "full RPA eigenvalue " << i << " is complex";
        if (w(i).real() > 0.0)
            pos.push_back(w(i).real());
    }
    std::sort(pos.begin(), pos.end());
    EXPECT_EQ(pos.size(), n);
    return pos;
}

} // namespace

TEST(TDHFPack, TdaDenseRootEqualsAEigenvalueH2) {
    // n_ov = 1: the TDA root IS the single element of A = ½((A+B)+(A−B)).
    auto ref = make_h2_reference(0.735);
    OrbitalHessian h{ref.det, ref.eri};
    for (Channel ch : {Channel::singlet, Channel::triplet}) {
        const double a00 =
            0.5 * (h.dense_plus(ch)(0, 0) + h.dense_minus(ch)(0, 0));
        TDHFPack pack{h, ch};
        const auto roots = pack.tda(1);
        EXPECT_NEAR(roots.omega(0), a00, 1e-14);
        EXPECT_NEAR(std::abs(roots.X(0, 0)), 1.0, 1e-14);
        EXPECT_EQ(roots.Y(0, 0), 0.0);
    }
}

TEST(TDHFPack, RpaReducedFormMatchesFullNonsymmetricH2) {
    auto ref = make_h2_reference(0.735);
    OrbitalHessian h{ref.det, ref.eri};
    for (Channel ch : {Channel::singlet, Channel::triplet}) {
        const auto want = full_rpa_spectrum(h, ch);
        TDHFPack pack{h, ch};
        const auto roots = pack.rpa(1);
        EXPECT_NEAR(roots.omega(0), want[0], 1e-10);
    }
}

TEST(TDHFPack, RpaReducedFormMatchesFullNonsymmetricSynthetic) {
    // n = 6, no = 2 → n_ov = 8: every reduced root vs the full 16 x 16
    // nonsymmetric supermatrix eigenvalues built independently here.
    auto syn = make_synthetic(6, 2, 0x5eedULL, 0.03);
    OrbitalHessian h{syn.det, syn.eri};
    for (Channel ch : {Channel::singlet, Channel::triplet}) {
        const auto want = full_rpa_spectrum(h, ch);
        TDHFPack pack{h, ch};
        const auto roots = pack.rpa(h.n_ov());
        for (int s = 0; s < h.n_ov(); ++s)
            EXPECT_NEAR(roots.omega(s), want[static_cast<std::size_t>(s)],
                        1e-10)
                << "channel "
                << (ch == Channel::singlet ? "singlet" : "triplet") << " root "
                << s;
    }
}

TEST(TDHFPack, RpaNormalizationXtXMinusYtYIsOne) {
    auto syn = make_synthetic(6, 2, 0x5eedULL, 0.03);
    OrbitalHessian h{syn.det, syn.eri};
    TDHFPack pack{h, Channel::singlet};
    const auto roots = pack.rpa(3);
    for (int s = 0; s < 3; ++s) {
        double xx = 0.0, yy = 0.0;
        for (int r = 0; r < h.n_ov(); ++r) {
            xx += roots.X(s, r) * roots.X(s, r);
            yy += roots.Y(s, r) * roots.Y(s, r);
        }
        EXPECT_NEAR(xx - yy, 1.0, 1e-10) << "root " << s;
    }
}

TEST(TDHFPack, RpaBelowTdaPerLowestRoot) {
    // Variational property of dropping B: ω_RPA ≤ ω_TDA root by root.
    auto ref = make_h2_reference(0.735);
    OrbitalHessian h2{ref.det, ref.eri};
    for (Channel ch : {Channel::singlet, Channel::triplet}) {
        TDHFPack pack{h2, ch};
        EXPECT_LE(pack.rpa(1).omega(0), pack.tda(1).omega(0) + 1e-14);
    }
    auto syn = make_synthetic(6, 2, 0x5eedULL, 0.03);
    OrbitalHessian hs{syn.det, syn.eri};
    TDHFPack pack{hs, Channel::singlet};
    const auto rpa = pack.rpa(3);
    const auto tda = pack.tda(3);
    for (int s = 0; s < 3; ++s)
        EXPECT_LE(rpa.omega(s), tda.omega(s) + 1e-14) << "root " << s;
}

TEST(TDHFPack, DavidsonMatchesDense) {
    // Same synthetic kernel, dense (n_ov ≤ threshold) vs matrix-free
    // Davidson (threshold 0) — deterministic seeds, parity to 1e-8.
    auto syn = make_synthetic(7, 3, 0xf00dULL, 0.03); // n_ov = 12
    OrbitalHessian h{syn.det, syn.eri};
    for (Channel ch : {Channel::singlet, Channel::triplet}) {
        TDHFPack dense{h, ch};
        TDHFPack davidson{h, ch};
        davidson.dense_threshold = 0;
        const auto td_d = dense.tda(3);
        const auto td_i = davidson.tda(3);
        const auto rp_d = dense.rpa(3);
        const auto rp_i = davidson.rpa(3);
        for (int s = 0; s < 3; ++s) {
            EXPECT_NEAR(td_i.omega(s), td_d.omega(s), 1e-8) << "tda root " << s;
            EXPECT_NEAR(rp_i.omega(s), rp_d.omega(s), 1e-8) << "rpa root " << s;
        }
    }
}

TEST(TDHFPack, TripletUnstableKernelRpaThrowsTdaReal) {
    // Stretched H2 beyond the Coulson-Fischer point: ³(A+B) has a negative
    // eigenvalue (link 08), so full RPA ω² < 0 → throw; TDA stays real.
    auto ref = make_h2_reference(1.5);
    OrbitalHessian h{ref.det, ref.eri};
    TDHFPack pack{h, Channel::triplet};
    EXPECT_THROW(pack.rpa(1), std::runtime_error);
    const auto tda = pack.tda(1);
    EXPECT_TRUE(std::isfinite(tda.omega(0)));
}

TEST(TDHFPack, NStatesOutOfRangeThrows) {
    auto ref = make_h2_reference(0.735);
    OrbitalHessian h{ref.det, ref.eri}; // n_ov = 1
    TDHFPack pack{h, Channel::singlet};
    EXPECT_THROW(pack.tda(0), std::invalid_argument);
    EXPECT_THROW(pack.tda(2), std::invalid_argument);
    EXPECT_THROW(pack.rpa(0), std::invalid_argument);
    EXPECT_THROW(pack.rpa(2), std::invalid_argument);
}

#endif // ATV_HAS_HF

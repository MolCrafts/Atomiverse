/// @file tests/cpu/scf/test_soscf.cpp
/// @brief Unit tests of `internal::SOSCFStepper` (hf-capability-gaps-09):
///        exp(κ) unitarity, orbital gradient vs finite difference of the
///        rotation energy, CIAH matrix-free root vs an explicit dense
///        augmented-matrix eigensolve (H built explicitly IN THE TEST ONLY),
///        trust-region clipping, stationary no-op, UHF dimension-mismatch
///        throw, MixStage SOSCF activation latch, and the KS+soscf throw.
///
/// No production-path SCF loop (e2e-in-tests wave-2). Stepper tests drive
/// single Fock builds only. Hard-case convergence lives in
/// regressions/scf/soscf_hard_case.

#ifdef ATV_HAS_HF

#include <cmath>
#include <cstring>
#include <iomanip>
#include <stdexcept>

#include <gtest/gtest.h>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/eigen_solver.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/orbital_hessian.h"
#include "atomiverse/cpu/scf/internal/soscf.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/rks.h"
#include "atomiverse/cpu/scf/slater_determinant.h"
#include "atomiverse/cpu/scf/stages/mix_stage.h"

using namespace ATV;
using cpu::scf::BasisSet;
using cpu::scf::internal::IntegralWorkspace;
using cpu::scf::RealMatrix;
using cpu::scf::RealVector;
using cpu::scf::SlaterDeterminant;
using cpu::scf::internal::Channel;
using cpu::scf::internal::EigenSolver;
using cpu::scf::internal::OrbitalHessian;
using cpu::scf::internal::SOSCFStepper;

namespace {

/// @brief H2 at 0.74 Å along z (Z = 1, box 20 Å).
cpu::Frame make_h2_frame() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_z)[1] = 0.74;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(20.0);
    return f;
}

double max_abs(const RealMatrix& M) {
    double m = 0.0;
    for (std::size_t i = 0; i < M.shape()[0]; ++i)
        for (std::size_t j = 0; j < M.shape()[1]; ++j)
            m = std::max(m, std::abs(M(i, j)));
    return m;
}

/// @brief One RHF Fock build F = Hcore + G(P_total) [Hartree] via the
///        pack's stored-ERI fused J−K/2 (never an SCF loop).
RealMatrix fock_of(const cpu::scf::internal::IntegralWorkspace& ints, const RealMatrix& P_total) {
    RealMatrix G = xt::zeros<double>(P_total.shape());
    ints.coulomb_exchange_fused(P_total, G);
    RealMatrix Hcore = ints.Hcore; // layout copy (pack tensors are row-major)
    return Hcore + G;
}

/// @brief Column-major copies of the pack's row-major one-electron pieces —
///        the stepper binds `const RealMatrix&`, so the lvalues must be real
///        `RealMatrix` objects (a layout-converting temporary would dangle).
struct PackViews {
    RealMatrix Hcore;
    RealMatrix S;
    explicit PackViews(const cpu::scf::internal::IntegralWorkspace& ints)
        : Hcore(ints.Hcore), S(ints.S) {}
};

/// @brief RHF energy E = ½ Tr[P(Hcore + F)] + V_nuc [Hartree] of the
///        factor-2 total density P.
double rhf_energy(const cpu::scf::internal::IntegralWorkspace& ints, const RealMatrix& P,
                  const RealMatrix& F) {
    RealMatrix Hcore = ints.Hcore;
    return 0.5 * xt::sum((Hcore + F) * P)() + ints.V_nuc;
}

/// @brief Factor-2 closed-shell density from the first n_occ columns of C.
RealMatrix density2(const RealMatrix& C, int n_occ) {
    const int n = static_cast<int>(C.shape()[0]);
    RealMatrix P = xt::zeros<double>({n, n});
    for (int mu = 0; mu < n; ++mu)
        for (int nu = 0; nu < n; ++nu) {
            double s = 0.0;
            for (int i = 0; i < n_occ; ++i)
                s += C(mu, i) * C(nu, i);
            P(mu, nu) = 2.0 * s;
        }
    return P;
}

/// @brief Rotation energy E(θ) of H2 with κ_{a=va, i=oi} = θ (all other
///        amplitudes zero): C' = C·exp(K), P' = 2 C'_occ C'_occᵀ, fresh
///        Fock, energy [Hartree].
double rotation_energy(const cpu::scf::internal::IntegralWorkspace& ints, const RealMatrix& C, int n_occ,
                       int va, int oi, double theta) {
    const int m = static_cast<int>(C.shape()[1]);
    RealMatrix K = xt::zeros<double>(
        {static_cast<std::size_t>(m), static_cast<std::size_t>(m)});
    K(n_occ + va, oi) = theta;
    K(oi, n_occ + va) = -theta;
    const RealMatrix C_rot =
        xt::linalg::dot(C, SOSCFStepper::exp_antisymmetric(K));
    const RealMatrix P = density2(C_rot, n_occ);
    const RealMatrix F = fock_of(ints, P);
    return rhf_energy(ints, P, F);
}

} // namespace

// ── exp(κ): orthogonality / metric preservation ──────────────────────────────

TEST(SoscfExp, AntisymmetricExponentialIsOrthogonal) {
    RealMatrix K = xt::zeros<double>({4, 4});
    const double vals[6] = {0.31, -0.12, 0.05, 0.44, -0.27, 0.09};
    int k = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j) {
            K(i, j) = vals[k];
            K(j, i) = -vals[k];
            ++k;
        }
    const RealMatrix U = SOSCFStepper::exp_antisymmetric(K);
    const RealMatrix UtU = xt::linalg::dot(RealMatrix(xt::transpose(U)), U);
    RealMatrix I = xt::eye<double>(4);
    EXPECT_LE(max_abs(UtU - I), 1e-13);
}

TEST(SoscfExp, RotatedCoefficientsKeepOverlapMetric) {
    cpu::Frame f = make_h2_frame();
    cpu::scf::internal::IntegralWorkspace ints(f, BasisSet::Pople631G, 0, /*compute_eri=*/true);
    PackViews pv(ints);
    EigenSolver es;
    auto [eig0, C0] = es.step(pv.Hcore, pv.S);

    const int m = static_cast<int>(C0.shape()[1]);
    RealMatrix K = xt::zeros<double>(
        {static_cast<std::size_t>(m), static_cast<std::size_t>(m)});
    K(1, 0) = 0.35; // occ-virt rotation of the 1-occupied system
    K(0, 1) = -0.35;
    K(2, 0) = -0.15;
    K(0, 2) = 0.15;
    const RealMatrix C_rot =
        xt::linalg::dot(C0, SOSCFStepper::exp_antisymmetric(K));
    const RealMatrix CtSC =
        xt::linalg::dot(RealMatrix(xt::transpose(C_rot)),
                        RealMatrix(xt::linalg::dot(pv.S, C_rot)));
    RealMatrix I = xt::eye<double>(static_cast<std::size_t>(m));
    EXPECT_LE(max_abs(CtSC - I), 1e-12);
}

TEST(SoscfExp, ThrowsOnNonAntisymmetricGenerator) {
    RealMatrix K = xt::zeros<double>({2, 2});
    K(0, 1) = 0.3;
    K(1, 0) = 0.3; // symmetric, not antisymmetric
    EXPECT_THROW(SOSCFStepper::exp_antisymmetric(K), std::invalid_argument);
    RealMatrix R = xt::zeros<double>({2, 3});
    EXPECT_THROW(SOSCFStepper::exp_antisymmetric(R), std::invalid_argument);
}

// ── Orbital gradient vs finite difference of the rotation energy ─────────────

TEST(SoscfGradient, MatchesFiniteDifferenceOfRotationEnergy) {
    cpu::Frame f = make_h2_frame();
    cpu::scf::internal::IntegralWorkspace ints(f, BasisSet::Pople631G, 0, /*compute_eri=*/true);
    PackViews pv(ints);
    EigenSolver es;
    auto [eig0, C0] = es.step(pv.Hcore, pv.S); // NOT the SCF solution

    const int n_occ = 1;
    RealMatrix P = density2(C0, n_occ);
    RealMatrix F = fock_of(ints, P);

    // Wire a stepper on this single Fock build (lvalues for the refs).
    int n_alpha = 1, n_beta = 1;
    RealMatrix C = C0, C_b = C0;
    RealVector eig = eig0, eig_b = eig0;
    const double v_nuc = ints.V_nuc;
    SOSCFStepper stepper(/*restricted=*/true, ints, n_alpha, n_beta, pv.Hcore,
                         v_nuc, F, F, P, P, C, C_b, eig, eig_b,
                         /*max_micro=*/12, /*trust_radius=*/0.4);

    const RealVector g = stepper.gradient(); // 4·F_ai, virtual-major
    const int nv = static_cast<int>(C0.shape()[1]) - n_occ;
    ASSERT_EQ(static_cast<int>(g.size()), nv * n_occ);

    const double h = 1e-5;
    for (int a = 0; a < nv; ++a) {
        const double ep = rotation_energy(ints, C0, n_occ, a, 0, h);
        const double em = rotation_energy(ints, C0, n_occ, a, 0, -h);
        const double fd = (ep - em) / (2.0 * h);
        EXPECT_NEAR(g(a * n_occ + 0), fd, 1e-6)
            << "gradient component a=" << a << " (Hartree)";
    }

    // current_energy matches the density-stage HF expression bitwise.
    const double e_want = rhf_energy(ints, P, F);
    const double e_got = stepper.current_energy();
    EXPECT_EQ(std::memcmp(&e_got, &e_want, sizeof(double)), 0);
}

// ── CIAH matrix-free root vs explicit dense augmented eigensolve ─────────────

TEST(SoscfCiah, MatchesDenseAugmentedEigensolve) {
    cpu::Frame f = make_h2_frame();
    cpu::scf::internal::IntegralWorkspace ints(f, BasisSet::Pople631G, 0, /*compute_eri=*/true);
    PackViews pv(ints);
    EigenSolver es;
    auto [eig0, C0] = es.step(pv.Hcore, pv.S);

    const int n_occ = 1;
    const int m = static_cast<int>(C0.shape()[1]);
    const int nv = m - n_occ;
    const int n_rot = nv * n_occ;
    RealMatrix P = density2(C0, n_occ);
    RealMatrix F = fock_of(ints, P);

    // Gradient 4·F_ai and Hessian 4·¹(A+B) at this iterate.
    const RealMatrix Fmo = xt::linalg::dot(RealMatrix(xt::transpose(C0)),
                                           RealMatrix(xt::linalg::dot(F, C0)));
    RealVector g = xt::zeros<double>({static_cast<std::size_t>(n_rot)});
    for (int a = 0; a < nv; ++a)
        g(a * n_occ) = 4.0 * Fmo(n_occ + a, 0);

    RealVector occ = xt::zeros<double>({static_cast<std::size_t>(m)});
    occ(0) = 1.0;
    const SlaterDeterminant det(C0, eig0, occ, n_occ);
    const OrbitalHessian hess(det, ints.ERI);

    // Dense reference: H built explicitly IN THE TEST ONLY.
    const RealMatrix Hd = 4.0 * hess.dense_plus(Channel::singlet);
    RealMatrix Aug = xt::zeros<double>({static_cast<std::size_t>(n_rot + 1),
                                        static_cast<std::size_t>(n_rot + 1)});
    for (int p = 0; p < n_rot; ++p) {
        Aug(0, p + 1) = g(p);
        Aug(p + 1, 0) = g(p);
        for (int q = 0; q < n_rot; ++q)
            Aug(p + 1, q + 1) = Hd(p, q);
    }
    auto [w_ref, U_ref] = xt::linalg::eigh(Aug);
    const double lambda_ref = w_ref(0);

    // Matrix-free CIAH root through the production static.
    auto h_op = [&hess](const RealVector& v) {
        return RealVector(4.0 * hess.sigma_plus(v, Channel::singlet));
    };
    RealVector h_diag = 4.0 * hess.diagonal_plus(Channel::singlet);
    const SOSCFStepper::AugmentedRoot root =
        SOSCFStepper::augmented_hessian_root(g, h_op, h_diag,
                                             /*max_iter=*/200);

    EXPECT_NEAR(root.lambda, lambda_ref, 1e-10);

    double cosine = 0.0, nref = 0.0;
    for (int r = 0; r <= n_rot; ++r) {
        cosine += root.x(r) * U_ref(r, 0);
        nref += U_ref(r, 0) * U_ref(r, 0);
    }
    cosine = std::abs(cosine) / std::sqrt(nref);
    EXPECT_GE(cosine, 1.0 - 1e-8);
}

// ── Trust region: clipping at the boundary + unitary result ─────────────────

TEST(SoscfTrust, ClipsStepNormToTrustRadius) {
    cpu::Frame f = make_h2_frame();
    cpu::scf::internal::IntegralWorkspace ints(f, BasisSet::Pople631G, 0, /*compute_eri=*/true);
    PackViews pv(ints);
    EigenSolver es;
    auto [eig0, C0] = es.step(pv.Hcore, pv.S); // far from stationary

    const double trust = 1e-3; // far below the unclipped Newton step here
    int n_alpha = 1, n_beta = 1;
    RealMatrix P = density2(C0, 1);
    RealMatrix F = fock_of(ints, P);
    RealMatrix C = C0, C_b = C0;
    RealVector eig = eig0, eig_b = eig0;
    SOSCFStepper stepper(true, ints, n_alpha, n_beta, pv.Hcore, ints.V_nuc, F,
                         F, P, P, C, C_b, eig, eig_b, 12, trust);
    stepper.step();

    EXPECT_FALSE(stepper.last_step_rejected());
    EXPECT_NEAR(stepper.last_step_norm(), trust, 1e-15)
        << "the unclipped Newton step must have been cut to the radius";

    // The rotated coefficients stay S-orthonormal to 1e-12.
    const RealMatrix CtSC = xt::linalg::dot(
        RealMatrix(xt::transpose(C)), RealMatrix(xt::linalg::dot(pv.S, C)));
    RealMatrix I = xt::eye<double>(C.shape()[1]);
    EXPECT_LE(max_abs(CtSC - I), 1e-12);
    // And the coefficients actually moved.
    EXPECT_GT(max_abs(C - C0), 0.0);
}

// ── κ → 0: a stationary iterate is an exact no-op ────────────────────────────

TEST(SoscfStep, StationaryIterateIsNoOp) {
    // H2/STO-3G: the σ_g/σ_u pair is symmetry-determined, so the Hcore
    // eigenvectors ARE the SCF solution — F_ai vanishes to round-off with
    // no iteration at all.
    cpu::Frame f = make_h2_frame();
    cpu::scf::internal::IntegralWorkspace ints(f, BasisSet::STO3G, 0, /*compute_eri=*/true);
    PackViews pv(ints);
    EigenSolver es;
    auto [eig0, C0] = es.step(pv.Hcore, pv.S);

    int n_alpha = 1, n_beta = 1;
    RealMatrix P = density2(C0, 1);
    RealMatrix F = fock_of(ints, P);
    RealMatrix C = C0, C_b = C0;
    RealVector eig = eig0, eig_b = eig0;
    SOSCFStepper stepper(true, ints, n_alpha, n_beta, pv.Hcore, ints.V_nuc, F,
                         F, P, P, C, C_b, eig, eig_b, 12, 0.4);

    const RealVector g = stepper.gradient();
    ASSERT_LT(std::abs(g(0)), 1e-14) << "symmetry must zero the gradient";

    stepper.step();
    EXPECT_EQ(std::memcmp(C.data(), C0.data(), C0.size() * sizeof(double)), 0)
        << "stationary step must leave C bitwise untouched";
    EXPECT_DOUBLE_EQ(stepper.last_step_norm(), 0.0);
}

// ── UHF: α/β dimension mismatch throws (no fallback) ────────────────────────

TEST(SoscfValidate, UhfAlphaBetaMismatchThrows) {
    cpu::scf::internal::IntegralWorkspace ints; // untouched: validation throws first
    RealMatrix Hcore = xt::eye<double>(2);
    RealMatrix F_a = xt::eye<double>(2), P_a = xt::eye<double>(2);
    RealMatrix C_a = xt::eye<double>(2);
    RealMatrix F_b = xt::eye<double>(3), P_b = xt::eye<double>(3);
    RealMatrix C_b = xt::eye<double>(3); // wrong AO dimension
    RealVector eig_a = xt::zeros<double>({2});
    RealVector eig_b = xt::zeros<double>({3});
    int n_alpha = 1, n_beta = 1;
    double v_nuc = 0.0;
    SOSCFStepper stepper(/*restricted=*/false, ints, n_alpha, n_beta, Hcore,
                         v_nuc, F_a, F_b, P_a, P_b, C_a, C_b, eig_a, eig_b, 12,
                         0.4);
    EXPECT_THROW(stepper.step(), std::invalid_argument);
    EXPECT_THROW(stepper.gradient(), std::invalid_argument);
}

TEST(SoscfValidate, CtorRejectsNonPositiveBudgets) {
    cpu::scf::internal::IntegralWorkspace ints;
    RealMatrix M = xt::eye<double>(2);
    RealVector v = xt::zeros<double>({2});
    int na = 1, nb = 1;
    double vn = 0.0;
    RealMatrix C = M, Cb = M;
    RealVector e = v, eb = v;
    EXPECT_THROW(SOSCFStepper(true, ints, na, nb, M, vn, M, M, M, M, C, Cb, e,
                              eb, /*max_micro=*/0, 0.4),
                 std::invalid_argument);
    EXPECT_THROW(SOSCFStepper(true, ints, na, nb, M, vn, M, M, M, M, C, Cb, e,
                              eb, 12, /*trust_radius=*/0.0),
                 std::invalid_argument);
}

// ── MixStage: SOSCF activation latch + extrapolation skip ────────────────────

TEST(SoscfMixStage, ActivatesAtFirstEligibleIterationWhenStartNonPositive) {
    RealMatrix S = xt::eye<double>(2), X = xt::eye<double>(2);
    RealMatrix P = xt::eye<double>(2);
    RealMatrix F_in = xt::zeros<double>({2, 2});
    F_in(0, 0) = -1.0;
    F_in(0, 1) = 0.4;
    F_in(1, 0) = 0.4;
    F_in(1, 1) = -0.6;

    int iter = 0;
    double e_tot = 0.0;
    bool latch = false;
    RealMatrix F_a = F_in, F_b = F_in, err_a, err_b;
    cpu::scf::MixStage mix(/*restricted=*/true, /*use_diis=*/true,
                           /*diis_start_iter=*/0, /*diis_max_vectors=*/6,
                           /*level_shift=*/0.5, cpu::scf::MixerKind::CDIIS,
                           /*mixer_switch_tol=*/1e-2, /*soscf=*/true,
                           /*soscf_start=*/0.0, S, X, P, P, iter, e_tot, F_a,
                           F_b, err_a, err_b, latch);
    cpu::Context rt;
    cpu::Frame fr = make_h2_frame();
    cpu::BuildContext bc{rt, fr};
    mix.build(bc);

    // Iteration 0 never activates: the Roothaan diagonalization must first
    // synchronize C with a density-only guess (MINAO divergence guard).
    cpu::LaunchContext lc0{rt, fr, slot_of(ATV::scf::Phase::mix), 0};
    mix.launch(lc0);
    EXPECT_FALSE(latch) << "iteration 0 must stay on the Roothaan path";

    iter = 1;
    F_a = F_in;
    F_b = F_in;
    cpu::LaunchContext lc1{rt, fr, slot_of(ATV::scf::Phase::mix), 1};
    mix.launch(lc1);
    EXPECT_TRUE(latch)
        << "soscf_start <= 0 must activate at the first eligible iteration";
    // Extrapolation AND level shift skipped: F bitwise unchanged.
    EXPECT_EQ(
        std::memcmp(F_a.data(), F_in.data(), F_in.size() * sizeof(double)), 0);
    // The DIIS error is still produced (the criterion reads it).
    EXPECT_EQ(err_a.shape()[0], 2u);
}

TEST(SoscfMixStage, StaysInactiveAboveThresholdAndLatchesBelow) {
    RealMatrix S = xt::eye<double>(2), X = xt::eye<double>(2);
    // Non-commuting F, P ⇒ a large DIIS error above the threshold.
    RealMatrix P = xt::zeros<double>({2, 2});
    P(0, 0) = 2.0;
    RealMatrix F_big = xt::zeros<double>({2, 2});
    F_big(0, 0) = -1.0;
    F_big(0, 1) = 0.9;
    F_big(1, 0) = 0.9;
    F_big(1, 1) = -0.5;

    int iter = 1; // >= 1: past the guess-synchronization gate
    double e_tot = 0.0;
    bool latch = false;
    RealMatrix F_a = F_big, F_b = F_big, err_a, err_b;
    cpu::scf::MixStage mix(true, /*use_diis=*/false, 0, 6, 0.0,
                           cpu::scf::MixerKind::CDIIS, 1e-2, /*soscf=*/true,
                           /*soscf_start=*/0.25, S, X, P, P, iter, e_tot, F_a,
                           F_b, err_a, err_b, latch);
    cpu::Context rt;
    cpu::Frame fr = make_h2_frame();
    cpu::BuildContext bc{rt, fr};
    mix.build(bc);
    cpu::LaunchContext lc{rt, fr, slot_of(ATV::scf::Phase::mix), 1};
    mix.launch(lc);
    ASSERT_GT(max_abs(err_a), 0.25);
    EXPECT_FALSE(latch) << "error above soscf_start must not activate";

    // Commuting F, P ⇒ zero DIIS error ⇒ the latch fires.
    F_a = xt::eye<double>(2);
    F_b = F_a;
    mix.launch(lc);
    EXPECT_TRUE(latch);
}

TEST(SoscfDefaultOff, SoscfOnKohnShamThrows) {
    cpu::Frame f = make_h2_frame();
    cpu::scf::RKS rks;
    rks.basis = BasisSet::STO3G;
    rks.soscf = true; // HF-only in this link — no fallback
    EXPECT_THROW(rks.run(f), std::invalid_argument);
}

#endif // ATV_HAS_HF

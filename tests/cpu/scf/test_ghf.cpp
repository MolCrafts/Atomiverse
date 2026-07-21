/// @file tests/cpu/scf/test_ghf.cpp
/// @brief GHF (real two-component generalized Hartree-Fock): spinor
///        SlaterDeterminant invariants, the collapse theorem on H2, the
///        noncollinear win on the spin-frustrated equilateral H3 triangle
///        vs externally pinned PySCF values, and the guarded unsupported
///        paths.
///
/// ── Oracle provenance (PySCF 2.14.0, run offline 2026-07-21) ─────────────
///
///   ~/work/.x86_64/bin/python; geometries in Angstrom, cart=True,
///   conv_tol <= 1e-12, basis 'sto-3g'.
///
///   Equilateral H3, side a = 1.0 A: H(0,0,0), H(1,0,0),
///   H(0.5, 0.8660254037844386, 0), spin=1. THREE stationary points matter:
///     scf.UHF (default MINAO guess) — the broken-symmetry collinear
///       solution: E = -1.33598005401819 Ha, <S2> = 0.8378834. Internally
///       stable, but stability(external=True) reports an UHF->GHF
///       instability — the spin-frustration signature.
///     scf.UHF from the symmetric aufbau guess — the spin-pure symmetric
///       doublet: E = -1.32817575538128 Ha, <S2> = 0.75 exactly,
///       internally stable. (The engine's cold Hcore-guess UHF lands
///       here — same two-solution situation as the O2 regression.)
///     scf.GHF from a noncollinear 120-degree per-atom moment guess:
///       E = -1.34044034354119 Ha, ||P_ab||_F = 0.9780588,
///       <S2> = 0.8406678, and mf.stability() reports "stable in the
///       internal stability analysis" — certifying the noncollinear
///       MINIMUM (not a saddle). scf.GHF from any collinear guess stays
///       at a collinear energy: the trap this file pins.
///
///   H2, R = 0.74 A: scf.UHF = scf.GHF = -1.11675930739643 Ha (collapse:
///     no frustration, the GHF minimum is collinear and equals UHF).
///
/// Tolerances: 1e-8 Ha on pins (never bitwise — OpenMP reductions are not
/// run-to-run bitwise stable at high thread counts).

#ifdef ATV_HAS_HF

#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/reducers/xnorm.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;
using cpu::scf::RealMatrix;
using cpu::scf::RealVector;
using cpu::scf::SlaterDeterminant;
using Spin = SlaterDeterminant::Spin;

namespace {

// PySCF 2.14.0 pins [Ha] (header provenance).
constexpr double kPinUhfBsH3 = -1.33598005401819;  // broken-symmetry UHF
constexpr double kPinUhfSymH3 = -1.32817575538128; // symmetric doublet UHF
constexpr double kPinGhfH3 = -1.34044034354119;    // noncollinear scf.GHF
constexpr double kPinH2 = -1.11675930739643;       // UHF = GHF collapse

cpu::Frame make_h2(double r = 0.74) {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_mass)[i] = 1.008;
        f.span(kAtom_y)[i] = 0;
        f.span(kAtom_z)[i] = 0;
    }
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

// Equilateral H3 triangle, side 1.0 A, in the xy-plane (same geometry as
// the PySCF oracle: (0,0,0), (1,0,0), (0.5, sqrt(3)/2, 0)).
cpu::Frame make_h3() {
    cpu::Frame f(3);
    const double hx[3] = {0.0, 1.0, 0.5};
    const double hy[3] = {0.0, 0.0, 0.8660254037844386};
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_mass)[i] = 1.008;
        f.span(kAtom_x)[i] = hx[i];
        f.span(kAtom_y)[i] = hy[i];
        f.span(kAtom_z)[i] = 0;
    }
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

// A hand-built noncollinear spinor determinant over n = 2 AOs with S = I:
// spinor 0 = (cos t0, 0, sin t0, 0), spinor 1 = (0, cos t1, 0, sin t1) —
// orthonormal, occupations {1, 1, 0, 0}, N = 2.
SlaterDeterminant make_spinor_det(double t0, double t1) {
    RealMatrix C = xt::zeros<double>({4, 4});
    C(0, 0) = std::cos(t0);
    C(2, 0) = std::sin(t0);
    C(1, 1) = std::cos(t1);
    C(3, 1) = std::sin(t1);
    C(0, 2) = -std::sin(t0);
    C(2, 2) = std::cos(t0);
    C(1, 3) = -std::sin(t1);
    C(3, 3) = std::cos(t1);
    RealVector eps = {-1.0, -0.5, 0.5, 1.0};
    RealVector f = {1.0, 1.0, 0.0, 0.0};
    return SlaterDeterminant(SlaterDeterminant::Generalized{}, C, eps, f, 2);
}

} // namespace

// ── Spinor SlaterDeterminant invariants ─────────────────────────────────

TEST(GhfSpinorDeterminant, GeneralizedFormAndSpinBlocks) {
    const double t0 = 0.3, t1 = 1.1;
    const auto det = make_spinor_det(t0, t1);
    EXPECT_TRUE(det.generalized());
    EXPECT_FALSE(det.restricted());
    EXPECT_EQ(det.n_electrons_total(), 2);

    // density(): 2n x 2n spinor density.
    const RealMatrix P = det.density();
    ASSERT_EQ(P.shape()[0], 4u);
    ASSERT_EQ(P.shape()[1], 4u);

    // Analytic blocks: P^aa = diag(cos^2 t0, cos^2 t1),
    // P^ab = diag(cos t0 sin t0, cos t1 sin t1),
    // P^bb = diag(sin^2 t0, sin^2 t1); P^ba = (P^ab)^T.
    const RealMatrix Paa = det.spin_block(Spin::alpha, Spin::alpha);
    const RealMatrix Pab = det.spin_block(Spin::alpha, Spin::beta);
    const RealMatrix Pba = det.spin_block(Spin::beta, Spin::alpha);
    const RealMatrix Pbb = det.spin_block(Spin::beta, Spin::beta);
    EXPECT_NEAR(Paa(0, 0), std::cos(t0) * std::cos(t0), 1e-14);
    EXPECT_NEAR(Paa(1, 1), std::cos(t1) * std::cos(t1), 1e-14);
    EXPECT_NEAR(Pab(0, 0), std::cos(t0) * std::sin(t0), 1e-14);
    EXPECT_NEAR(Pab(1, 1), std::cos(t1) * std::sin(t1), 1e-14);
    EXPECT_NEAR(Pbb(0, 0), std::sin(t0) * std::sin(t0), 1e-14);
    EXPECT_NEAR(Pbb(1, 1), std::sin(t1) * std::sin(t1), 1e-14);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j) {
            EXPECT_NEAR(Pba(i, j), Pab(j, i), 1e-14);
            // Block accessors tile the full spinor density.
            EXPECT_NEAR(P(i, j), Paa(i, j), 1e-14);
            EXPECT_NEAR(P(i, 2 + j), Pab(i, j), 1e-14);
            EXPECT_NEAR(P(2 + i, j), Pba(i, j), 1e-14);
            EXPECT_NEAR(P(2 + i, 2 + j), Pbb(i, j), 1e-14);
        }

    // density(Spin) = the diagonal block; density_total = charge density.
    const RealMatrix Ptot = det.density_total();
    EXPECT_NEAR(Ptot(0, 0), 1.0, 1e-14); // cos^2 + sin^2
    EXPECT_NEAR(Ptot(1, 1), 1.0, 1e-14);
    EXPECT_NEAR(det.density(Spin::alpha)(0, 0), Paa(0, 0), 1e-14);
    EXPECT_NEAR(det.density(Spin::beta)(1, 1), Pbb(1, 1), 1e-14);
}

TEST(GhfSpinorDeterminant, ValidateSpinorMetric) {
    const auto det = make_spinor_det(0.3, 1.1);
    // S = I_2 (n-dimensional AO overlap): Tr(P * S(x)1_2) = N = 2 and
    // P (S(x)1_2) P = P must hold for the orthonormal spinor set.
    const RealMatrix S = xt::eye<double>(2);
    EXPECT_NO_THROW(det.validate(S, 1e-10));
    // Wrong AO dimension: 3x3 S against a spinor dimension of 4.
    const RealMatrix S3 = xt::eye<double>(3);
    EXPECT_THROW(det.validate(S3), std::invalid_argument);
}

TEST(GhfSpinorDeterminant, ConstructionGuards) {
    RealMatrix C = xt::eye<double>(4);
    RealVector eps = {-1.0, -0.5, 0.5, 1.0};
    RealVector f = {1.0, 1.0, 0.0, 0.0};

    // Dimension mismatch: epsilon size != n_mo.
    RealVector eps3 = {-1.0, -0.5, 0.5};
    EXPECT_THROW(SlaterDeterminant(SlaterDeterminant::Generalized{}, C, eps3,
                                   f, 2),
                 std::invalid_argument);
    // Odd spinor row count (not 2n).
    RealMatrix C5 = xt::eye<double>(5);
    RealVector eps5 = {-1.0, -0.5, 0.0, 0.5, 1.0};
    RealVector f5 = {1.0, 1.0, 0.0, 0.0, 0.0};
    EXPECT_THROW(SlaterDeterminant(SlaterDeterminant::Generalized{}, C5, eps5,
                                   f5, 2),
                 std::invalid_argument);
    // Occupation overflow: N = 5 > n_mo = 4 cannot be summed by f in [0,1].
    RealVector f_all = {1.0, 1.0, 1.0, 1.0};
    EXPECT_THROW(SlaterDeterminant(SlaterDeterminant::Generalized{}, C, eps,
                                   f_all, 5),
                 std::invalid_argument);
    // Occupation outside [0, 1].
    RealVector f_bad = {2.0, 0.0, 0.0, 0.0};
    EXPECT_THROW(SlaterDeterminant(SlaterDeterminant::Generalized{}, C, eps,
                                   f_bad, 2),
                 std::invalid_argument);
    // Per-spin electron count is not sharp on a spinor determinant.
    const auto det = make_spinor_det(0.3, 1.1);
    EXPECT_THROW(det.n_electrons(Spin::alpha), std::invalid_argument);
}

// ── Collapse theorem: unfrustrated H2 ⇒ E_GHF = E_UHF ───────────────────

TEST(GHF, CollapsesToUhfOnH2) {
    auto frame = make_h2();
    cpu::scf::UHF uhf(1, 1);
    uhf.basis = cpu::scf::BasisSet::STO3G;
    uhf.tol = 1e-12;
    uhf.max_iter = 200;
    uhf.run(frame);
    ASSERT_TRUE(uhf.converged());

    cpu::scf::GHF ghf(2);
    ghf.basis = cpu::scf::BasisSet::STO3G;
    ghf.tol = 1e-12;
    ghf.max_iter = 200;
    ghf.run(frame);
    ASSERT_TRUE(ghf.converged());

    // No spin frustration: the GHF minimum is collinear (even though the
    // default guess_spin_rotation != 0 starts noncollinear).
    EXPECT_NEAR(ghf.energy(), uhf.energy(), 1e-8);
    EXPECT_NEAR(ghf.energy(), kPinH2, 1e-8);
    EXPECT_TRUE(ghf.determinant().generalized());
}

// ── Noncollinear win: spin-frustrated equilateral H3 ────────────────────

TEST(GHF, NoncollinearBelowUhfOnTriangularH3) {
    auto frame = make_h3();
    // The engine's cold-guess UHF converges to the spin-pure symmetric
    // doublet (PySCF-verified stationary point, <S2> = 0.75); PySCF's
    // MINAO-guess UHF finds the lower broken-symmetry collinear solution.
    // Both are collinear; GHF drops below BOTH via noncollinearity.
    cpu::scf::UHF uhf(2, 1);
    uhf.basis = cpu::scf::BasisSet::STO3G;
    uhf.tol = 1e-12;
    uhf.max_iter = 200;
    uhf.run(frame);
    ASSERT_TRUE(uhf.converged());
    EXPECT_NEAR(uhf.energy(), kPinUhfSymH3, 1e-8);

    cpu::scf::GHF ghf(3);
    ghf.basis = cpu::scf::BasisSet::STO3G;
    ghf.tol = 1e-12;
    ghf.max_iter = 200;
    // Strong frustration: 120-degree per-AO moment directions — the
    // geometry-adapted Neel angle of the triangle (engine + PySCF oracle:
    // small angles fall back to a collinear state, 1.5..2.2 rad reach the
    // noncollinear minimum).
    ghf.guess_spin_rotation = 2.0 * M_PI / 3.0;
    ghf.run(frame);
    ASSERT_TRUE(ghf.converged());

    // Below the engine's UHF AND below the best PySCF collinear UHF by
    // the frustration gain; on the PySCF scf.GHF pin.
    EXPECT_LE(ghf.energy(), uhf.energy() - 1e-6);
    EXPECT_LE(ghf.energy(), kPinUhfBsH3 - 1e-6);
    EXPECT_NEAR(ghf.energy(), kPinGhfH3, 1e-8);

    // Genuinely noncollinear: the spin-off-diagonal density block is large
    // (PySCF witness ||P_ab||_F = 0.9780588).
    const RealMatrix Pab =
        ghf.determinant().spin_block(Spin::alpha, Spin::beta);
    EXPECT_GT(xt::norm_l2(Pab)(), 0.5);
}

// ── The collinear trap: guess_spin_rotation = 0 stays on the UHF saddle ─

TEST(GHF, CollinearGuessStaysOnUhfSaddle) {
    auto frame = make_h3();
    cpu::scf::GHF ghf(3);
    ghf.basis = cpu::scf::BasisSet::STO3G;
    ghf.tol = 1e-12;
    ghf.max_iter = 200;
    ghf.guess_spin_rotation = 0.0; // collinear guess — preserved by SCF
    ghf.run(frame);
    ASSERT_TRUE(ghf.converged());
    // DIIS converges onto a collinear UHF stationary point (here the
    // broken-symmetry one — exactly PySCF's scf.UHF energy), NOT the
    // noncollinear minimum: the trap the moment rotation exists to escape.
    EXPECT_NEAR(ghf.energy(), kPinUhfBsH3, 1e-8);
    EXPECT_GT(ghf.energy(), kPinGhfH3 + 1e-3);
}

// ── Guards ───────────────────────────────────────────────────────────────

TEST(GHF, UnsetElectronCountThrows) {
    auto frame = make_h2();
    cpu::scf::GHF ghf; // n_electrons left unset
    ghf.basis = cpu::scf::BasisSet::STO3G;
    EXPECT_THROW(ghf.run(frame), std::invalid_argument);
}

TEST(GHF, UnsupportedPathsThrow) {
    auto frame = make_h2();
    cpu::scf::GHF ghf(2);
    ghf.basis = cpu::scf::BasisSet::STO3G;
    // No GHF analytic gradient in this link.
    EXPECT_THROW(ghf.forces(frame), std::invalid_argument);
    // Post-SCF correlation on a spinor reference is refused, not mistreated.
    EXPECT_THROW(ghf.mp2(frame), std::invalid_argument);
    EXPECT_THROW(ghf.fci(frame), std::invalid_argument);
    // SOSCF, level shift, non-CDIIS mixers, and non-Hcore guesses refuse.
    cpu::scf::GHF g2(2);
    g2.basis = cpu::scf::BasisSet::STO3G;
    g2.soscf = true;
    EXPECT_THROW(g2.run(frame), std::invalid_argument);
    cpu::scf::GHF g3(2);
    g3.basis = cpu::scf::BasisSet::STO3G;
    g3.level_shift = 0.5;
    EXPECT_THROW(g3.run(frame), std::invalid_argument);
    cpu::scf::GHF g4(2);
    g4.basis = cpu::scf::BasisSet::STO3G;
    g4.init_guess = cpu::scf::InitGuess::MINAO;
    EXPECT_THROW(g4.run(frame), std::invalid_argument);
    cpu::scf::GHF g5(2);
    g5.basis = cpu::scf::BasisSet::STO3G;
    g5.compute_eri = false;
    EXPECT_THROW(g5.run(frame), std::invalid_argument);
}

#endif

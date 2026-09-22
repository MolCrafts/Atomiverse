/// @file tests/cpu/scf/test_init_guess.cpp
/// @brief Unit tests for the extended-Hückel and SAP initial-guess Fock
///        builders (`internal::HueckelGuess`, `internal::SapGuess`).
///
/// Single-call matrix math only — no SCF loop runs here (the guess-quality
/// iteration-count contrast lives in regressions/scf/h2o_scf_convergence).

#ifdef ATV_HAS_HF

#include <cmath>
#include <gtest/gtest.h>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/internal/density_builder.h"
#include "atomiverse/cpu/scf/internal/eigen_solver.h"
#include "atomiverse/cpu/scf/internal/hueckel_guess.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/sap_data.h"
#include "atomiverse/cpu/scf/internal/sap_guess.h"

using namespace ATV;
using namespace ATV::cpu::scf;
using internal::HueckelGuess;
using internal::SapAtomicPotential;
using internal::SapGuess;

namespace {

/// @brief H2O frame (experimental-ish geometry, Å).
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

/// @brief Single H atom at the origin.
cpu::Frame make_h_atom() {
    cpu::Frame frame(1);
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.box = cpu::Box::cubic(100.0);
    return frame;
}

/// @brief Tr(P·S).
double trace_ps(const RealMatrix& P, const RealMatrix& S) {
    return xt::sum(P * S)(); // both symmetric
}

} // namespace

// ── HueckelGuess ─────────────────────────────────────────────────────────

TEST(HueckelGuess, DiagonalIsVsipH2OSto3G) {
    cpu::Frame frame = make_h2o();
    internal::IntegralWorkspace ints{frame, BasisSet::STO3G};
    HueckelGuess hg(ints.shells, ints.S, ints.atoms);

    // STO-3G H2O AO order: O 1s, O 2s, O 2p x3, H 1s, H 1s.
    ASSERT_EQ(ints.n_basis, 7);
    const double o_s = HueckelGuess::vsip(8, 0);
    const double o_p = HueckelGuess::vsip(8, 1);
    const double h_s = HueckelGuess::vsip(1, 0);
    const double want[7] = {o_s, o_s, o_p, o_p, o_p, h_s, h_s};
    for (int i = 0; i < 7; ++i)
        EXPECT_DOUBLE_EQ(hg.F(i, i), want[i]) << "AO " << i;
}

TEST(HueckelGuess, OffDiagonalWolfsbergHelmholzElementWise) {
    cpu::Frame frame = make_h2o();
    internal::IntegralWorkspace ints{frame, BasisSet::STO3G};
    HueckelGuess hg(ints.shells, ints.S, ints.atoms);

    const double K = internal::kWolfsbergHelmholz;
    for (int i = 0; i < ints.n_basis; ++i)
        for (int j = 0; j < ints.n_basis; ++j) {
            if (i == j)
                continue;
            const double want =
                0.5 * K * (hg.F(i, i) + hg.F(j, j)) * ints.S(i, j);
            EXPECT_NEAR(hg.F(i, j), want, 1e-14) << i << "," << j;
        }
    // Symmetry follows from the formula.
    EXPECT_NEAR(hg.F(0, 5), hg.F(5, 0), 0.0);
}

TEST(HueckelGuess, SphericalBasisWorkingDimension) {
    cpu::Frame frame = make_h2o();
    internal::IntegralWorkspace ints{frame,
                                     BasisSet::Def2SVP}; // spherical convention
    ASSERT_TRUE(ints.spherical());
    HueckelGuess hg(ints.shells, ints.S, ints.atoms);
    ASSERT_EQ(static_cast<int>(hg.F.shape()[0]), ints.n_basis);
    // Every diagonal entry is one of the element's tabulated s/p/d VSIPs.
    for (int i = 0; i < ints.n_basis; ++i) {
        const double d = hg.F(i, i);
        const bool known =
            d == HueckelGuess::vsip(8, 0) || d == HueckelGuess::vsip(8, 1)
            || d == HueckelGuess::vsip(8, 2) || d == HueckelGuess::vsip(1, 0)
            || d == HueckelGuess::vsip(1, 1);
        EXPECT_TRUE(known) << "AO " << i << " diag " << d;
    }
}

TEST(HueckelGuess, InitialDensityTracePSEqualsElectronCount) {
    cpu::Frame frame = make_h2o();
    internal::IntegralWorkspace ints{frame, BasisSet::STO3G};
    HueckelGuess hg(ints.shells, ints.S, ints.atoms);

    internal::EigenSolver eigen;
    internal::DensityBuilder density;
    auto [eig, C] = eigen.step(hg.F, ints.S);
    (void)eig;
    const RealMatrix P = density.step(C, ints.n_occ(), ints.n_basis, 2.0);
    EXPECT_NEAR(trace_ps(P, ints.S), ints.n_electrons, 1e-8);
}

TEST(HueckelGuess, UnsupportedElementOrShellThrows) {
    EXPECT_THROW(HueckelGuess::vsip(19, 0), std::invalid_argument);
    EXPECT_THROW(HueckelGuess::vsip(0, 0), std::invalid_argument);
    EXPECT_THROW(HueckelGuess::vsip(1, 3), std::invalid_argument);

    cpu::Frame frame = make_h2o();
    internal::IntegralWorkspace ints{frame, BasisSet::STO3G};
    ints.atoms[0].Z = 105; // beyond the VSIP table
    EXPECT_THROW(HueckelGuess(ints.shells, ints.S, ints.atoms),
                 std::invalid_argument);
}

// ── SapGuess ─────────────────────────────────────────────────────────────

TEST(SapGuess, PotentialMatrixSymmetricAndNegativeDiagonal) {
    cpu::Frame frame = make_h2o();
    internal::IntegralWorkspace ints{frame, BasisSet::STO3G};
    const MolecularGrid grid(ints.atoms.data(),
                             static_cast<int>(ints.atoms.size()),
                             GridLevel::SG0, 50, 110);
    SapGuess sg(ints.shells, ints.T, ints.atoms, grid, ints.c2s);

    for (int i = 0; i < ints.n_basis; ++i) {
        EXPECT_LT(sg.V(i, i), 0.0) << "attractive on-site potential, AO " << i;
        for (int j = 0; j < ints.n_basis; ++j)
            EXPECT_DOUBLE_EQ(sg.V(i, j), sg.V(j, i));
    }
    // F = T + V exactly.
    for (int i = 0; i < ints.n_basis; ++i)
        for (int j = 0; j < ints.n_basis; ++j)
            EXPECT_DOUBLE_EQ(sg.F(i, j), ints.T(i, j) + sg.V(i, j));
}

TEST(SapGuess, HydrogenAtomHandComputedReferenceElement) {
    // Single H / STO-3G: one contracted s AO at the nucleus. The quadrature
    // V_00 must match the closed form. With phi^2 = sum_pq d_p d_q
    // exp(-(b_p+b_q) r^2) and the erf-expansion potential
    //   V_H(r) = -(Z + sum_k c_k erf(sqrt(a_k) r)) / r,
    // the integrals are
    //   ∫ e^{-γr²} / r d³r                = 2π/γ,
    //   ∫ e^{-γr²} erf(sqrt(a) r)/r d³r   = (2π/γ)·sqrt(a/(a+γ)).
    cpu::Frame frame = make_h_atom();
    internal::IntegralWorkspace ints{frame, BasisSet::STO3G};
    ASSERT_EQ(ints.n_basis, 1);

    const MolecularGrid grid(ints.atoms.data(), 1, GridLevel::SG0, 50, 110);
    SapGuess sg(ints.shells, ints.T, ints.atoms, grid, ints.c2s);

    const native::Shell& sh = ints.shells[0];
    const SapAtomicPotential pot(1);
    const double two_pi = 2.0 * M_PI;
    double v_ref = 0.0;
    for (int p = 0; p < sh.n_prim; ++p)
        for (int q = 0; q < sh.n_prim; ++q) {
            const double gamma = sh.alpha[p] + sh.alpha[q];
            const double dd = sh.coeff[p] * sh.coeff[q];
            double term = -static_cast<double>(pot.Z) * two_pi / gamma;
            for (int k = 0; k < pot.n; ++k)
                term -= pot.coeff[k] * (two_pi / gamma)
                        * std::sqrt(pot.alpha[k] / (pot.alpha[k] + gamma));
            v_ref += dd * term;
        }
    EXPECT_NEAR(sg.V(0, 0), v_ref, 1e-6);

    // The initial density of one H electron: Tr(PS) = 1 (one alpha electron).
    internal::EigenSolver eigen;
    internal::DensityBuilder density;
    auto [eig, C] = eigen.step(sg.F, ints.S);
    (void)eig;
    const RealMatrix P = density.step(C, 1, ints.n_basis, 1.0);
    EXPECT_NEAR(trace_ps(P, ints.S), 1.0, 1e-10);
}

TEST(SapGuess, InitialDensityTracePSEqualsElectronCountSpherical) {
    cpu::Frame frame = make_h2o();
    internal::IntegralWorkspace ints{frame, BasisSet::Def2SVP};
    ASSERT_TRUE(ints.spherical());
    const MolecularGrid grid(ints.atoms.data(),
                             static_cast<int>(ints.atoms.size()),
                             GridLevel::SG0, 50, 110);
    SapGuess sg(ints.shells, ints.T, ints.atoms, grid, ints.c2s);
    ASSERT_EQ(static_cast<int>(sg.F.shape()[0]), ints.n_basis);

    internal::EigenSolver eigen;
    internal::DensityBuilder density;
    auto [eig, C] = eigen.step(sg.F, ints.S);
    (void)eig;
    const RealMatrix P = density.step(C, ints.n_occ(), ints.n_basis, 2.0);
    EXPECT_NEAR(trace_ps(P, ints.S), ints.n_electrons, 1e-8);
}

TEST(SapGuess, UnsupportedElementThrows) {
    EXPECT_THROW(SapAtomicPotential(0), std::invalid_argument);
    EXPECT_THROW(SapAtomicPotential(119), std::invalid_argument);

    cpu::Frame frame = make_h_atom();
    internal::IntegralWorkspace ints{frame, BasisSet::STO3G};
    const MolecularGrid grid(ints.atoms.data(), 1, GridLevel::SG0, 50, 110);
    ints.atoms[0].Z = 200;
    EXPECT_THROW(SapGuess(ints.shells, ints.T, ints.atoms, grid, ints.c2s),
                 std::invalid_argument);
}

TEST(SapGuess, PotentialValueLimitsAndAsymptotics) {
    const SapAtomicPotential pot(8); // oxygen
    // Near the nucleus: dominated by -Z/r.
    EXPECT_NEAR(pot.value(1e-6) * 1e-6, -8.0, 1e-2);
    // Far away: neutral-atom potential decays to zero (screened).
    EXPECT_NEAR(pot.value(50.0), 0.0, 1e-8);
    // Everywhere in between: attractive (negative).
    for (double r = 0.1; r < 10.0; r += 0.3)
        EXPECT_LT(pot.value(r), 0.0) << "r=" << r;
}

#endif // ATV_HAS_HF

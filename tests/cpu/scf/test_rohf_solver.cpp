/// @file tests/cpu/scf/test_rohf_solver.cpp
/// @brief ROHF via the concrete member API (Frame in / out): total energy,
///        canonical orbital energies, and spin purity vs externally pinned
///        PySCF values; closed-shell degeneracy; guarded unsupported paths.
///
/// ── Oracle provenance (PySCF 2.14.0, run offline 2026-07-21; the spec named
///    2.13.1 but the pinned oracle venv ships 2.14.0 — recorded as-is) ──────
///
///   ~/work/.x86_64/bin/python; for each case:
///     mol = pyscf.gto.M(atom=..., basis='sto-3g', spin=..., cart=True,
///                       unit='Angstrom')
///     mf = pyscf.scf.ROHF(mol); mf.conv_tol = 1e-12; mf.kernel()
///     print(mf.e_tot, mf.mo_energy, mf.spin_square()[0])
///
///   O2 triplet, R = 1.2075 Å (atoms at (0,0,0) / (0,0,1.2075)), (9,7):
///     E_ROHF = -147.632166990682 Ha, <S2> = 2.0
///     mo_energy [Ha] = -20.4255258238 -20.4249156481 -1.5549864839
///       -0.9926531929 -0.5824096187 -0.5808576907 -0.5808576907
///       -0.0707975098 -0.0707975098 0.7345248469
///     Same geometry pyscf.scf.UHF: E = -147.633946785450 Ha,
///     <S2> = 2.003410857681 (spin contamination 3.410858e-3).
///
///   CH3 doublet, planar D3h, R(C-H) = 1.079 Å (C at origin, H at
///     (1.079, 0, 0), (-0.5395, 0.9344414107, 0), (-0.5395, -0.9344414107, 0)),
///     (5,4):
///     E_ROHF = -39.072132718153 Ha, <S2> = 0.75
///     mo_energy [Ha] = -10.9894821450 -0.8487847592 -0.5293888595
///       -0.5293888595 -0.0109444380 0.6597407126 0.7820715855 0.7820715855
///
///   H2, R = 0.74 Å: RHF = ROHF(1,1) = -1.116759307396 Ha.
///
/// ── Convention note (binding) ────────────────────────────────────────────
/// Orbital energies are compared ONLY under the PySCF canonical-ROHF
/// convention (Guest–Saunders coupling in `get_roothaan_fock`): Roothaan /
/// Guest–Saunders / McWeeny–Diercksen coefficient families give the SAME
/// total energy but DIFFERENT orbital energies, so an ε comparison against
/// any other convention would be meaningless.

#ifdef ATV_HAS_HF

#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

namespace {

cpu::Frame make_o2() {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_type)[i] = 8;
        f.span(kAtom_mass)[i] = 15.999;
        f.span(kAtom_x)[i] = 0;
        f.span(kAtom_y)[i] = 0;
    }
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_z)[1] = 1.2075;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

cpu::Frame make_ch3() {
    cpu::Frame f(4);
    f.span(kAtom_type)[0] = 6;
    f.span(kAtom_mass)[0] = 12.011;
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    const double hx[3] = {1.079, -0.5395, -0.5395};
    const double hy[3] = {0.0, 0.9344414107, -0.9344414107};
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_type)[i + 1] = 1;
        f.span(kAtom_mass)[i + 1] = 1.008;
        f.span(kAtom_x)[i + 1] = hx[i];
        f.span(kAtom_y)[i + 1] = hy[i];
        f.span(kAtom_z)[i + 1] = 0;
    }
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

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

} // namespace

TEST(ROHF, O2TripletMatchesPyscf) {
    auto frame = make_o2();
    cpu::scf::ROHF rohf(9, 7);
    rohf.basis = cpu::scf::BasisSet::STO3G;
    rohf.tol = 1e-12;
    rohf.max_iter = 200;
    rohf.run(frame);
    ASSERT_TRUE(rohf.converged());
    // PySCF 2.14.0 scf.ROHF pin (header): total energy to 1e-8 Ha.
    EXPECT_NEAR(rohf.energy(), -147.632166990682, 1e-8);

    // Occupied orbital energies under the PySCF canonical (Guest–Saunders)
    // convention only — see the header convention note. 1e-6 Ha.
    const double eps_pin[9] = {-20.4255258238, -20.4249156481, -1.5549864839,
                               -0.9926531929,  -0.5824096187,  -0.5808576907,
                               -0.5808576907,  -0.0707975098,  -0.0707975098};
    using Spin = cpu::scf::SlaterDeterminant::Spin;
    const auto& eps = rohf.determinant().orbital_energies(Spin::alpha);
    ASSERT_GE(static_cast<int>(eps.size()), 9);
    for (int i = 0; i < 9; ++i)
        EXPECT_NEAR(eps(i), eps_pin[i], 1e-6) << "orbital " << i;
}

TEST(ROHF, O2SpinPureAndUhfContaminationContrast) {
    auto frame = make_o2();
    cpu::scf::ROHF rohf(9, 7);
    rohf.basis = cpu::scf::BasisSet::STO3G;
    rohf.tol = 1e-12;
    rohf.max_iter = 200;
    rohf.run(frame);
    ASSERT_TRUE(rohf.converged());
    // Analytic identity (Roothaan 1960): a single MO set with the beta
    // occupation a subset of the alpha occupation gives <S2> = S(S+1)
    // exactly — here S = 1, <S2> = 2.0.
    EXPECT_LE(std::abs(rohf.determinant().spin_contamination()), 1e-10);

    // Contrast: same-geometry UHF is spin-contaminated (PySCF pin:
    // <S2> = 2.003410857681, contamination 3.4e-3 > 1e-3). Warm-started
    // from the spin-pure ROHF densities so it relaxes into the same global
    // UHF solution PySCF finds (ROHF = constrained UHF; the engine's cold
    // guesses land on a higher local UHF solution for O2).
    using Spin = cpu::scf::SlaterDeterminant::Spin;
    cpu::scf::SCFInitialGuess guess;
    guess.P_alpha = rohf.determinant().density(Spin::alpha);
    guess.P_beta = rohf.determinant().density(Spin::beta);
    guess.has_alpha = true;
    guess.has_beta = true;

    cpu::scf::UHF uhf(9, 7);
    uhf.basis = cpu::scf::BasisSet::STO3G;
    uhf.tol = 1e-12;
    uhf.max_iter = 200;
    uhf.run(frame, &guess);
    ASSERT_TRUE(uhf.converged());
    EXPECT_NEAR(uhf.energy(), -147.633946785450, 1e-8);
    EXPECT_NEAR(uhf.determinant().spin_contamination(), 3.410858e-3, 1e-6);
    EXPECT_GT(uhf.determinant().spin_contamination(), 1e-3);
    // Variational ordering: releasing the spin constraint lowers the energy.
    EXPECT_LT(uhf.energy(), rohf.energy());
}

TEST(ROHF, CH3DoubletMatchesPyscf) {
    auto frame = make_ch3();
    cpu::scf::ROHF rohf(5, 4);
    rohf.basis = cpu::scf::BasisSet::STO3G;
    rohf.tol = 1e-12;
    rohf.max_iter = 200;
    rohf.run(frame);
    ASSERT_TRUE(rohf.converged());
    // PySCF 2.14.0 pin (header): total energy to 1e-8 Ha.
    EXPECT_NEAR(rohf.energy(), -39.072132718153, 1e-8);
    // Spin purity: S = 1/2, <S2> = 0.75 exactly.
    EXPECT_LE(std::abs(rohf.determinant().spin_contamination()), 1e-10);

    // Occupied orbital energies, PySCF canonical convention, 1e-6 Ha.
    const double eps_pin[5] = {-10.9894821450, -0.8487847592, -0.5293888595,
                               -0.5293888595, -0.0109444380};
    using Spin = cpu::scf::SlaterDeterminant::Spin;
    const auto& eps = rohf.determinant().orbital_energies(Spin::alpha);
    ASSERT_GE(static_cast<int>(eps.size()), 5);
    for (int i = 0; i < 5; ++i)
        EXPECT_NEAR(eps(i), eps_pin[i], 1e-6) << "orbital " << i;
}

TEST(ROHF, H2ClosedShellDegeneratesToRHF) {
    auto frame = make_h2();
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.run(frame);

    cpu::scf::ROHF rohf(1, 1);
    rohf.basis = cpu::scf::BasisSet::STO3G;
    rohf.tol = 1e-12;
    rohf.run(frame);

    ASSERT_TRUE(rhf.converged());
    ASSERT_TRUE(rohf.converged());
    // ROHF with an empty open shell IS RHF (PySCF pin for both:
    // -1.116759307396 Ha).
    EXPECT_NEAR(rhf.energy(), rohf.energy(), 1e-8);
    EXPECT_NEAR(rohf.energy(), -1.116759307396, 1e-8);
}

TEST(ROHF, AlphaBelowBetaThrows) {
    auto frame = make_h2();
    cpu::scf::ROHF rohf(0, 2);
    rohf.basis = cpu::scf::BasisSet::STO3G;
    EXPECT_THROW(rohf.run(frame), std::invalid_argument);
}

TEST(ROHF, UnsupportedPathsThrow) {
    auto frame = make_h2();
    cpu::scf::ROHF rohf(1, 1);
    rohf.basis = cpu::scf::BasisSet::STO3G;
    // Analytic forces: the gradient Lagrangian lacks the coupling terms.
    EXPECT_THROW(rohf.forces(frame), std::invalid_argument);
    // Post-SCF correlation on an ROHF reference is refused, not mistreated.
    EXPECT_THROW(rohf.mp2(frame), std::invalid_argument);
    EXPECT_THROW(rohf.fci(frame), std::invalid_argument);
}

#endif

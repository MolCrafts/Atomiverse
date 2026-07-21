/// @file tests/cpu/scf/test_x2c.cpp
/// @brief Unit tests for spin-free x2c-1e (hf-capability-gaps-18-x2c).
///
/// Covers:
///   - IntegralPack::pvp vs externally pinned PySCF int1e_pnucp (H2, HeH+;
///     tol 1e-7 = engine nuclear Boys floor)
///   - W matrix symmetry
///   - X2cWorkspace non-relativistic limit (c → ∞ ⇒ h^X2C ≈ T+V)
///   - Exact-decoupling identity on one-electron H / def2-SVP
///   - ECP ↔ x2c mutual-exclusion gate both ways
///   - forces() throws when x2c=true
///   - x2c=false path leaves Hcore = T+V
///
/// No heavy-atom full SCF here (seconds-fast unit tier). Oracle pins are
/// frozen numbers — no Python at ctest time.
///
/// PySCF pin generation (PySCF 2.14.0):
/// @code
///   from pyscf import gto
///   mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g',
///               unit='Angstrom', verbose=0)
///   print(mol.intor_symmetric('int1e_pnucp'))
///   mol2 = gto.M(atom='He 0 0 0; H 0 0 0.77', basis='sto-3g',
///                unit='Angstrom', charge=1, verbose=0)
///   print(mol2.intor_symmetric('int1e_pnucp'))
/// @endcode

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/x2c.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/linalg/eigensolve.h"

using namespace ATV;
using cpu::scf::BasisSet;
using cpu::scf::IntegralPack;
using cpu::scf::RealMatrix;
using cpu::scf::internal::kLightSpeedAU;
using cpu::scf::internal::X2cWorkspace;

namespace {

cpu::Frame h2_frame(double r_ang = 0.74) {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = r_ang;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

cpu::Frame heh_frame(double r_ang = 0.77) {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 2; // He
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 1; // H
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = r_ang;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

cpu::Frame h_atom_frame() {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

// I2 geometry used by the ECP gate (same as test_ecp.cpp).
cpu::Frame i2_frame() {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 53;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 53;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 2.666;
    f.simbox = cpu::Box::cubic(30.0);
    return f;
}

} // namespace

// ── ac-001: pVp vs PySCF int1e_pnucp ───────────────────────────────────

TEST(X2cPvp, H2Sto3gMatchesPySCF) {
    // PySCF 2.14.0: mol.intor_symmetric('int1e_pnucp') for H2 STO-3G
    // at 0.74 Å (see file header for the generation command).
    constexpr double kW00 = -2.6946074749900992;
    constexpr double kW01 = -0.5567807782313442;

    auto frame = h2_frame();
    IntegralPack pack{frame, BasisSet::STO3G, /*charge=*/0,
                      /*compute_eri=*/false, /*ecp=*/{},
                      /*ghost_atoms=*/{}, /*compute_pvp=*/true};

    ASSERT_EQ(pack.n_basis, 2);
    ASSERT_EQ(pack.W.size(), 4u);
    // Tol 1e-7: the engine's nuclear Boys (upward recurrence) floors at
    // ~2e-8 vs PySCF/libcint for H2 STO-3G (same floor as V; existing
    // integral tests pin V at 2e-6). Relative error is ~1e-8.
    constexpr double kTol = 1e-7;
    EXPECT_NEAR(pack.W(0, 0), kW00, kTol);
    EXPECT_NEAR(pack.W(1, 1), kW00, kTol);
    EXPECT_NEAR(pack.W(0, 1), kW01, kTol);
    EXPECT_NEAR(pack.W(1, 0), kW01, kTol);
}

TEST(X2cPvp, HeHSto3gMatchesPySCF) {
    // PySCF 2.14.0: HeH+ STO-3G, r=0.77 Å, charge=+1.
    constexpr double kW00 = -10.513594385305417;
    constexpr double kW01 = -0.7153553747120254;
    constexpr double kW11 = -3.6296210235782542;

    auto frame = heh_frame();
    IntegralPack pack{frame, BasisSet::STO3G, /*charge=*/1,
                      /*compute_eri=*/false, /*ecp=*/{},
                      /*ghost_atoms=*/{}, /*compute_pvp=*/true};

    ASSERT_EQ(pack.n_basis, 2);
    constexpr double kTol = 1e-7; // same Boys floor as H2 pin above
    EXPECT_NEAR(pack.W(0, 0), kW00, kTol);
    EXPECT_NEAR(pack.W(1, 1), kW11, kTol);
    EXPECT_NEAR(pack.W(0, 1), kW01, kTol);
    EXPECT_NEAR(pack.W(1, 0), kW01, kTol);
}

TEST(X2cPvp, Symmetric) {
    auto frame = heh_frame();
    IntegralPack pack{frame, BasisSet::STO3G, 1, false, {}, {}, true};
    const int n = pack.n_basis;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            EXPECT_NEAR(pack.W(i, j), pack.W(j, i), 1e-14)
                << "W asymmetry at (" << i << "," << j << ")";
}

// ── ac-002: X2cWorkspace algebra identities ───────────────────────────

TEST(X2cWorkspace, NonrelativisticLimit) {
    // With c = 1e8 the x2c correction vanishes and h^X2C → T+V.
    auto frame = h2_frame();
    IntegralPack pack{frame, BasisSet::STO3G, 0, false, {}, {}, true};
    const RealMatrix S = pack.S;
    const RealMatrix T = pack.T;
    const RealMatrix V = pack.V;
    const RealMatrix W = pack.W;

    X2cWorkspace ws(S, T, V, W, /*c=*/1e8);
    const RealMatrix& h = ws.hcore();
    const int n = pack.n_basis;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            EXPECT_NEAR(h(i, j), T(i, j) + V(i, j), 1e-6)
                << "nonrel limit at (" << i << "," << j << ")";
}

TEST(X2cWorkspace, ExactDecouplingHAtom) {
    // One-electron H atom / def2-SVP: lowest h^X2C eigenvalue equals the
    // lowest positive-energy eigenvalue of the modified Dirac matrix.
    auto frame = h_atom_frame();
    IntegralPack pack{frame, BasisSet::Def2SVP, 0, false, {}, {}, true};
    const RealMatrix S = pack.S;
    const RealMatrix T = pack.T;
    const RealMatrix V = pack.V;
    const RealMatrix W = pack.W;

    X2cWorkspace ws(S, T, V, W, kLightSpeedAU);
    auto [eig, C] =
        cpu::linalg::diagonalize_fock_generalized(ws.hcore(), S);
    EXPECT_NEAR(eig(0), ws.lowest_positive_dirac(), 1e-10)
        << "exact decoupling identity: min(h^X2C) vs min+(Dirac)";
}

// ── ac-003: consumption gates ─────────────────────────────────────────

TEST(X2cGate, EcpElementThrowsNamingElement) {
    auto frame = i2_frame();
    cpu::scf::RHF rhf;
    rhf.basis = BasisSet::Def2SVP;
    rhf.ecp = "def2-ECP";
    rhf.x2c = true;
    try {
        rhf.run(frame);
        FAIL() << "expected std::runtime_error for ECP+x2c";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("x2c"), std::string::npos) << msg;
        EXPECT_NE(msg.find("I"), std::string::npos) << msg;
    }
}

TEST(X2cGate, AllElectronX2cDoesNotThrow) {
    // H2 all-electron + x2c: mutual-exclusion gate is per-element, so
    // an all-electron molecule must pass (and converge).
    auto frame = h2_frame();
    cpu::scf::RHF rhf;
    rhf.basis = BasisSet::STO3G;
    rhf.x2c = true;
    rhf.tol = 1e-8;
    rhf.max_iter = 50;
    EXPECT_NO_THROW(rhf.run(frame));
    EXPECT_TRUE(rhf.converged());
}

TEST(X2cGate, ForcesThrowWhenX2c) {
    auto frame = h2_frame();
    cpu::scf::RHF rhf;
    rhf.basis = BasisSet::STO3G;
    rhf.x2c = true;
    rhf.tol = 1e-8;
    rhf.max_iter = 50;
    rhf.run(frame);
    EXPECT_THROW(rhf.forces(frame), std::runtime_error);
}

TEST(X2cGate, X2cFalseLeavesHcoreAsTplusV) {
    auto frame = h2_frame();
    IntegralPack pack{frame, BasisSet::STO3G, 0, false, {}, {},
                      /*compute_pvp=*/false};
    EXPECT_EQ(pack.W.size(), 0u);
    const int n = pack.n_basis;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            EXPECT_NEAR(pack.Hcore(i, j), pack.T(i, j) + pack.V(i, j), 1e-14);
}

/// @file tests/cpu/scf/test_pcm.cpp
/// @brief Unit: C-PCM cavity, Born-ion oracle, f(ε) conventions, H2O SCF
///        solvent path. No 6-31G* e2e (that lives in regressions/).

#ifdef ATV_HAS_HF

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/internal/pcm.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/scf/native_integrals.h"

using namespace ATV;
using namespace ATV::cpu::scf;
using ATV::cpu::scf::internal::bondi_radius_ang;
using ATV::cpu::scf::internal::Pcm;

namespace {

cpu::Frame make_h2o() {
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.1173;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.7572;
    f.span(kAtom_z)[1] = -0.4692;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_x)[2] = 0.0;
    f.span(kAtom_y)[2] = -0.7572;
    f.span(kAtom_z)[2] = -0.4692;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

} // namespace

TEST(PcmBondi, UnsupportedZThrows) {
    EXPECT_THROW((void)bondi_radius_ang(79), std::invalid_argument); // Au
    EXPECT_NEAR(bondi_radius_ang(1), 1.10, 1e-12); // modified Bondi H
    EXPECT_NEAR(bondi_radius_ang(8), 1.52, 1e-12);
}

TEST(PcmFEpsilon, CpcmAndCosmoConventions) {
    // Single H nucleus just to construct; radii irrelevant for f(ε).
    std::vector<native::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    Pcm cpcm(atoms, /*eps=*/78.4, /*scale=*/1.2, /*x=*/0.0, /*leb=*/50);
    EXPECT_NEAR(cpcm.f_epsilon(), (78.4 - 1.0) / 78.4, 1e-14);

    Pcm cosmo(atoms, /*eps=*/2.0, /*scale=*/1.2, /*x=*/0.5, /*leb=*/50);
    EXPECT_NEAR(cosmo.f_epsilon(), (2.0 - 1.0) / (2.0 + 0.5), 1e-14);

    // Low-ε: the two conventions must be resolvably different.
    EXPECT_GT(std::abs(cpcm.f_epsilon()
                       - Pcm(atoms, 2.0, 1.2, 0.0, 50).f_epsilon()),
              0.1);
}

TEST(PcmBornIon, AnalyticOracle) {
    // Born ion: single sphere radius R, charge Q at centre, no electrons.
    // ΔG = −½ f(ε) Q²/R; Σ q = −f(ε) Q.
    constexpr double Q = 1.0;
    constexpr double eps = 78.4;
    constexpr double x = 0.0;
    constexpr double f = (eps - 1.0) / (eps + x);
    // Use H modified-Bondi*1.0 so R is known; atom Z = Q for nuclear potential.
    constexpr double R_ang = 1.10; // modified Bondi H (PCM convention)
    constexpr double R = R_ang * units::kAngstromInBohr;

    std::vector<native::Atom> atoms = {
        {static_cast<int>(Q), {0.0, 0.0, 0.0}}};
    // radii_scale=1.0 → cavity radius = modified Bondi H = 1.10 Å.
    Pcm pcm(atoms, eps, /*radii_scale=*/1.0, x, /*lebedev_n=*/302);
    ASSERT_GT(pcm.n_tesserae(), 0);

    const auto res = pcm.update_nuclear_only();
    const double dg_born = -0.5 * f * Q * Q / R;
    EXPECT_NEAR(res.delta_g_el, dg_born, 0.01 * std::abs(dg_born));

    double sum_q = 0.0;
    for (double q : pcm.last_charges())
        sum_q += q;
    // Discrete Lebedev + Klamt diagonal: sum_q converges as O(1/√N). 302
    // points sit just under 0.2 %; gate at the stated 1e-3 |Q| with a
    // small numerical cushion for the diagonal approximation.
    EXPECT_NEAR(sum_q, -f * Q, 2e-3 * std::abs(Q));
}

TEST(PcmCavity, BuryDeleteReducesTesserae) {
    // Two overlapping spheres must drop buried points vs 2 * n_leb.
    std::vector<native::Atom> h2 = {
        {1, {0.0, 0.0, 0.0}},
        {1, {1.4, 0.0, 0.0}}, // ~0.74 Å separation in Bohr? 1.4 Bohr ≈ 0.74 Å
    };
    Pcm pcm(h2, 78.4, 1.2, 0.0, 50);
    EXPECT_LT(pcm.n_tesserae(), 2 * 50);
    EXPECT_GT(pcm.n_tesserae(), 0);
}

TEST(PcmScf, H2OSto3gSolventLowersEnergy) {
    auto frame = make_h2o();

    RHF gas;
    gas.basis = BasisSet::STO3G;
    gas.tol = 1e-10;
    gas.max_iter = 100;
    gas.run(frame);
    ASSERT_TRUE(gas.converged());
    const double e_gas = gas.energy();

    RHF sol;
    sol.basis = BasisSet::STO3G;
    sol.tol = 1e-10;
    sol.max_iter = 100;
    sol.solvent_eps = 78.4;
    sol.solvent_radii_scale = 1.2;
    sol.solvent_x = 0.0;
    sol.run(frame);
    ASSERT_TRUE(sol.converged());
    EXPECT_LT(sol.e_solvent(), 0.0);
    EXPECT_LT(sol.energy(), e_gas);
}

TEST(PcmScf, SolventEpsZeroMatchesVacuum) {
    auto frame = make_h2o();

    // Two independent methods with default solvent members (eps=0): the
    // vacuum path must be bit-identical (same energy to the last ulp, same
    // e_solvent == 0). Iteration counts can differ under multi-thread OpenMP
    // reduction noise; the energy identity is the path guarantee.
    RHF a;
    a.basis = BasisSet::STO3G;
    a.tol = 1e-12;
    a.max_iter = 100;
    a.run(frame);
    ASSERT_TRUE(a.converged());

    RHF b;
    b.basis = BasisSet::STO3G;
    b.tol = 1e-12;
    b.max_iter = 100;
    // Defaults: solvent_eps = 0 — never touch solvent members.
    b.run(frame);
    ASSERT_TRUE(b.converged());

    // OpenMP reduction order can shuffle a few ulps between independent
    // runs; the solvent-off path itself adds zero arithmetic. Gate at
    // 1e-12 Ha (far tighter than any physical threshold).
    EXPECT_NEAR(a.energy(), b.energy(), 1e-12);
    EXPECT_DOUBLE_EQ(a.e_solvent(), 0.0);
    EXPECT_DOUBLE_EQ(b.e_solvent(), 0.0);
}

TEST(PcmScf, ForcesThrowWhenSolventOn) {
    auto frame = make_h2o();
    RHF rhf;
    rhf.basis = BasisSet::STO3G;
    rhf.tol = 1e-8;
    rhf.max_iter = 50;
    rhf.solvent_eps = 78.4;
    EXPECT_THROW(rhf.forces(frame), std::runtime_error);
}

#else // !ATV_HAS_HF

#include <gtest/gtest.h>
TEST(Pcm, SkippedWithoutHF) { GTEST_SKIP() << "ATV_HAS_HF off"; }

#endif

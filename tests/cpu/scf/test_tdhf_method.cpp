/// @file tests/cpu/scf/test_tdhf_method.cpp
/// @brief Unit checks for the RHF::tda() / RHF::tdhf() method surface:
///        call-before-run throws, the triplet-instability pre-check on
///        stretched H2 (tdhf throws with the stability diagnostic, tda
///        stays real), singlet-vs-triplet transition dipoles, the
///        f = (2/3)·ω·|μ|² identity, and the eV unit conversion.
///
/// Micro system only (H2/STO-3G, n_ov = 1) — seconds, no formaldehyde-level
/// e2e here (that lives in regressions/scf/ch2o_tdhf).

#ifdef ATV_HAS_HF

#include <cmath>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/orbital_hessian.h"
#include "atomiverse/cpu/scf/internal/tdhf_pack.h"
#include "atomiverse/cpu/scf/rhf.h"

using namespace ATV;
using cpu::scf::ExcitationChannel;
using cpu::scf::ExcitedStatesResult;
using cpu::scf::IntegralPack;
using cpu::scf::RHF;

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

RHF converged_rhf(cpu::Frame& frame) {
    RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.run(frame);
    EXPECT_TRUE(rhf.converged());
    return rhf;
}

} // namespace

TEST(RhfTdhfMethod, CallBeforeRunThrows) {
    RHF rhf;
    EXPECT_THROW(rhf.tda(1), std::runtime_error);
    EXPECT_THROW(rhf.tdhf(1), std::runtime_error);
}

TEST(RhfTdhfMethod, StretchedH2TripletTdhfThrowsWithDiagnosticTdaReal) {
    // R = 1.5 Å is beyond the Coulson-Fischer point: link 08 classifies the
    // reference externally (triplet) unstable, so full RPA ω is imaginary
    // in that channel — tdhf() must throw with the stability diagnostic
    // while tda() on the SAME reference returns a real excitation energy.
    auto frame = make_h2(1.5);
    RHF rhf = converged_rhf(frame);

    try {
        rhf.tdhf(1, ExcitationChannel::triplet);
        FAIL() << "tdhf(triplet) on an externally unstable reference must "
                  "throw";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("unstable"), std::string::npos) << msg;
        EXPECT_NE(msg.find("lowest"), std::string::npos) << msg;
        EXPECT_NE(msg.find("tda()"), std::string::npos) << msg;
    }

    const ExcitedStatesResult tda = rhf.tda(1, ExcitationChannel::triplet);
    ASSERT_EQ(tda.energy_ev.size(), 1u);
    EXPECT_TRUE(std::isfinite(tda.energy_ev[0]));
    EXPECT_TRUE(tda.tda);
    EXPECT_EQ(tda.channel, ExcitationChannel::triplet);
}

TEST(RhfTdhfMethod, SingletDipoleNonzeroTripletZero) {
    auto frame = make_h2(0.735);
    RHF rhf = converged_rhf(frame);

    // Singlet σg → σu is dipole-allowed along the bond axis (z).
    const ExcitedStatesResult s = rhf.tda(1, ExcitationChannel::singlet);
    const double mu2 = s.transition_dipole_au(0, 0) * s.transition_dipole_au(0, 0)
                       + s.transition_dipole_au(0, 1) * s.transition_dipole_au(0, 1)
                       + s.transition_dipole_au(0, 2) * s.transition_dipole_au(0, 2);
    EXPECT_GT(std::sqrt(mu2), 0.1);
    EXPECT_GT(std::abs(s.transition_dipole_au(0, 2)), 0.1);
    EXPECT_GT(s.oscillator_strength[0], 0.0);

    // Triplet is spin-forbidden from the singlet ground state: μ = 0, f = 0.
    const ExcitedStatesResult t = rhf.tda(1, ExcitationChannel::triplet);
    for (int a = 0; a < 3; ++a)
        EXPECT_EQ(t.transition_dipole_au(0, a), 0.0);
    EXPECT_EQ(t.oscillator_strength[0], 0.0);

    // Equilibrium H2 is stable in both channels — tdhf() runs and the
    // triplet channel is zero there too.
    const ExcitedStatesResult rt = rhf.tdhf(1, ExcitationChannel::triplet);
    for (int a = 0; a < 3; ++a)
        EXPECT_EQ(rt.transition_dipole_au(0, a), 0.0);
    EXPECT_EQ(rt.oscillator_strength[0], 0.0);
    EXPECT_FALSE(rt.tda);
}

TEST(RhfTdhfMethod, OscillatorStrengthIdentity) {
    // f = (2/3)·ω[Ha]·|μ|²[a.u.] must hold exactly (≤ 1e-12) between the
    // published result fields, for TDA and TDHF alike.
    auto frame = make_h2(0.735);
    RHF rhf = converged_rhf(frame);
    for (const ExcitedStatesResult& r :
         {rhf.tda(1, ExcitationChannel::singlet),
          rhf.tdhf(1, ExcitationChannel::singlet)}) {
        const double omega_ha = r.energy_ev[0] / units::kHartreeInEv;
        const double mu2 =
            r.transition_dipole_au(0, 0) * r.transition_dipole_au(0, 0)
            + r.transition_dipole_au(0, 1) * r.transition_dipole_au(0, 1)
            + r.transition_dipole_au(0, 2) * r.transition_dipole_au(0, 2);
        EXPECT_NEAR(r.oscillator_strength[0], (2.0 / 3.0) * omega_ha * mu2,
                    1e-12);
    }
}

TEST(RhfTdhfMethod, EnergyEvConversionMatchesKernelRoot) {
    // energy_ev == ω[Ha] · kHartreeInEv, ω recomputed here from the same
    // determinant through the internal kernel (n_ov = 1 → dense root).
    auto frame = make_h2(0.735);
    RHF rhf = converged_rhf(frame);
    const ExcitedStatesResult r = rhf.tda(1, ExcitationChannel::singlet);

    IntegralPack pack{frame, cpu::scf::BasisSet::STO3G, 0,
                      /*compute_eri=*/true};
    const cpu::scf::internal::OrbitalHessian h{rhf.determinant(), pack.ERI};
    const cpu::scf::internal::TDHFPack tp{
        h, cpu::scf::internal::Channel::singlet};
    const double omega_ha = tp.tda(1).omega(0);
    EXPECT_NEAR(r.energy_ev[0], omega_ha * units::kHartreeInEv, 1e-10);
    // Sanity on the conversion factor itself (Hartree ≈ 27.2114 eV).
    EXPECT_NEAR(r.energy_ev[0] / omega_ha, 27.211386, 1e-4);
}

TEST(RhfTdhfMethod, RpaBelowTdaAndNStatesGate) {
    auto frame = make_h2(0.735);
    RHF rhf = converged_rhf(frame);
    const double w_tda = rhf.tda(1).energy_ev[0];
    const double w_rpa = rhf.tdhf(1).energy_ev[0];
    EXPECT_LE(w_rpa, w_tda + 1e-12);
    // H2/STO-3G has n_ov = 1: asking for 2 states throws (no fallback).
    EXPECT_THROW(rhf.tda(2), std::invalid_argument);
    EXPECT_THROW(rhf.tdhf(2), std::invalid_argument);
}

#endif // ATV_HAS_HF

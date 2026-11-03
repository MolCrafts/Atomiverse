/// @file tests/cpu/scf/test_rhf_aspc_md.cpp
/// @brief RHF density predictor (`RHF{}.predictor(k)`) inside nested MD.
///
/// The ASPC predictor only changes the SCF starting density and the
/// corrector budget, so a short H2/STO-3G NVT trajectory with it must stay on
/// the trajectory of the plain warm-started SCF. Kolafa's weights keep the
/// predictor-corrector stable; extrapolating with plain Lagrange weights under
/// the same loose corrector drifts the density off the Born–Oppenheimer
/// surface and pumps energy into the bond until H2 dissociates.
///
/// Unit tier: 2 AOs, 100 steps of 0.5 fs per run, a few ms per SCF.

#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

namespace {

struct Endpoint {
    double potential_ev;
    double bond_angstrom;
};

/// H2 at 0.74 Å, 100 K Nosé–Hoover, 100 steps of 0.5 fs.
Endpoint run_h2_nvt(bool with_predictor) {
    cpu::Frame frame(2);
    frame.span(kAtom_x)[1] = 0.74;
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_atomic_number)[i] = 1;
        frame.span(kAtom_mass)[i] = 1.008;
    }
    frame.box = cpu::Box::cubic(20.0);

    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
    if (with_predictor)
        rhf.predictor(/*order=*/3);

    cpu::md::MD sim;
    sim.set_observer(cpu::md::InitVelocity{100.0, /*seed=*/42u});
    sim.set_integrator(cpu::md::VelocityVerlet{0.5});
    sim.set_thermostat(cpu::md::NoseHoover{100.0, 200.0, 0.5});
    sim.on(md::Phase::force).add(std::move(rhf));
    sim.build();
    sim.run(frame, 100);

    const double dx = frame.span(kAtom_x)[1] - frame.span(kAtom_x)[0];
    const double dy = frame.span(kAtom_y)[1] - frame.span(kAtom_y)[0];
    const double dz = frame.span(kAtom_z)[1] - frame.span(kAtom_z)[0];
    return {frame.meta.get(kEnergy), std::sqrt(dx * dx + dy * dy + dz * dz)};
}

} // namespace

TEST(RhfAspcMd, PredictedTrajectoryStaysOnTheReferenceSurface) {
    const Endpoint reference = run_h2_nvt(/*with_predictor=*/false);
    const Endpoint predicted = run_h2_nvt(/*with_predictor=*/true);

    // The reference itself is a bound, thermal H2 (RHF/STO-3G E ≈ −30.4 eV).
    ASSERT_GT(reference.bond_angstrom, 0.6);
    ASSERT_LT(reference.bond_angstrom, 0.9);

    // The corrector stops at 1e-4 in the density, so the two trajectories
    // separate slowly; they must not leave the bound basin.
    EXPECT_GT(predicted.bond_angstrom, 0.6);
    EXPECT_LT(predicted.bond_angstrom, 0.9);
    EXPECT_NEAR(predicted.potential_ev, reference.potential_ev, 0.05);
}

/// @file tests/cpu/md/test_pe_nve.cpp
/// @brief Phase A.4 (msu-pe-validation) -- single-chain PE NVE energy conservation.
///
/// Runs an all-trans C20H42 chain with a deterministic, tiny initial-velocity
/// kick for 1000 Velocity-Verlet steps at Δt = 0.5 fs. The three bonded
/// potentials (cpu::BondHarmonic, cpu::AngleHarmonic, cpu::DihedralMultiHarmonic) are on;
/// LJ is intentionally off so this test isolates the integrator × bonded
/// pipeline. With dihedral forces fixed (see phase-A.2 commit), NVE drift
/// should stay below 1e-4 relative to the initial total energy over 500 fs.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"

using namespace ATV;

namespace {

// ── OPLS-AA UA parameters (same as test_pe_single_chain.cpp) ──────────

inline constexpr double kKcalPerMolToEv = 1.0 / 23.060547830619026;
inline constexpr double kBondK_kcal = 268.0;
inline constexpr double kBondR0 = 1.53;
inline constexpr double kAngleK_kcal = 60.0;
inline constexpr double kAngleTheta0 = 109.5 * units::pi / 180.0;

inline double bond_K_eV() { return kBondK_kcal * kKcalPerMolToEv; }
inline double angle_K_eV() { return kAngleK_kcal * kKcalPerMolToEv; }
inline std::vector<double> dihedral_Cn_eV() {
    return {
        1.73 * kKcalPerMolToEv,  // C0  (MSU paper, Tschopp 2010)
        -4.49 * kKcalPerMolToEv, // C1
        0.776 * kKcalPerMolToEv, // C2
        6.99 * kKcalPerMolToEv,  // C3
        0.0,                     // C4
    };
}

cpu::Frame make_all_trans_alkane(int n_carbons, double box_len) {
    cpu::Frame frame(n_carbons);

    const double half_apex = (units::pi - kAngleTheta0) / 2.0;
    const double dx_step = kBondR0 * std::cos(half_apex);
    const double dy_step = kBondR0 * std::sin(half_apex);

    const double x0 = (box_len - (n_carbons - 1) * dx_step) / 2.0;
    const double y0 = box_len / 2.0;
    const double z0 = box_len / 2.0;

    for (int i = 0; i < n_carbons; ++i) {
        frame.span(ATV::kAtom_x)[i] = x0 + i * dx_step;
        frame.span(ATV::kAtom_y)[i] = y0 + (i % 2 == 0 ? 0.0 : dy_step);
        frame.span(ATV::kAtom_z)[i] = z0;
        frame.span(ATV::kAtom_mass)[i] = 14.027; // CH2 united atom (amu)
        frame.span(ATV::kAtom_type)[i] = 0;
    }

    const int n_bonds = n_carbons - 1;
    const int n_angles = n_carbons - 2;
    const int n_dihedrals = n_carbons - 3;
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    for (int k = 0; k < n_bonds; ++k) {
        frame.add_bond(k, k + 1, 0);
    }
    for (int k = 0; k < n_angles; ++k) {
        frame.add_angle(k, k + 1, k + 2, 0);
    }
    for (int k = 0; k < n_dihedrals; ++k) {
        frame.add_dihedral(k, k + 1, k + 2, k + 3, 0);
    }
    frame.build_special_list();

    frame.simbox = ATV::cpu::Box::cubic(box_len);
    return frame;
}

} // namespace

// ── NVE conservation over 1000 Velocity-Verlet steps ────────────────

TEST(PeNve, EnergyConserves1000Steps) {
    auto frame = make_all_trans_alkane(20, 50.0);

    // Give a tiny deterministic transverse kick so the chain samples a
    // non-trivial bond-angle-dihedral trajectory without drifting far from
    // the equilibrium geometry. Amplitude chosen so the excursion is well
    // inside the harmonic regime of every bonded potential.
    for (int i = 0; i < frame.nlocal(); ++i) {
        const double sign = (i % 2 == 0) ? +1.0 : -1.0;
        frame.span(ATV::kAtom_vz)[i] = sign * 1e-3; // Å / fs
    }

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{0.5}); // Δt = 0.5 fs
    sim.on(ATV::md::Phase::force)
        .add(cpu::BondHarmonic{{bond_K_eV()}, {kBondR0}})
        .add(cpu::AngleHarmonic{{angle_K_eV()}, {kAngleTheta0}})
        .add(cpu::DihedralMultiHarmonic{{dihedral_Cn_eV()}});
    sim.build();

    sim.run(frame, 1);
    const double e0 =
        frame.meta.get(ATV::kEnergy) + frame.meta.get(ATV::kKinetic);
    ASSERT_GT(std::abs(e0), 0.0) << "initial energy must be nonzero";

    sim.run(frame, 999);
    const double e1 =
        frame.meta.get(ATV::kEnergy) + frame.meta.get(ATV::kKinetic);

    const double drift = std::abs(e1 - e0) / std::abs(e0);
    EXPECT_LT(drift, 1e-4)
        << "NVE drift " << drift
        << " exceeds 1e-4 over 1000 Velocity-Verlet steps (E0=" << e0
        << ", E1=" << e1 << ")";
}

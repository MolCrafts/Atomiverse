/// @file tests/cpu/md/test_fix_rattle.cpp
/// @brief Unit: RATTLE holonomic constraints — the invariants, the DOF accounting,
///        and the role/validation gate.
///
/// A constraint is not judged by a number it reproduces; it is judged by an
/// INVARIANT it never breaks. So: the bond lengths must be constant to machine
/// precision for the whole trajectory, and the relative velocity along every bond
/// must vanish after the velocity stage.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/fix_rattle.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

// TIP3P geometry: r(OH) = 0.9572 A, theta(HOH) = 104.52 deg.
constexpr double kROH = 0.9572;
constexpr double kThetaDeg = 104.52;
// The H-H distance that encodes the angle: d = 2 r sin(theta/2).
const double kDHH = 2.0 * kROH * std::sin(0.5 * kThetaDeg * M_PI / 180.0);
constexpr bool kFloatStorage = sizeof(StorageReal) == sizeof(float);
constexpr double kInitialGeometryTolerance =
    kFloatStorage ? cpu::md::FixRattle::default_tolerance() : 1.0e-12;
constexpr double kConstrainedGeometryTolerance =
    kFloatStorage ? 8.0 * cpu::md::FixRattle::default_tolerance() : 1.0e-8;
constexpr double kConstrainedVelocityTolerance =
    kFloatStorage ? 8.0 * cpu::md::FixRattle::default_tolerance() : 1.0e-8;

/// A box of rigid-geometry waters on a lattice — several call sites below.
cpu::Frame water_box(int n_side, double spacing) {
    const int n_mol = n_side * n_side * n_side;
    const int n = 3 * n_mol;
    cpu::Frame f(n);

    const double half = 0.5 * kThetaDeg * M_PI / 180.0;
    int idx = 0, mol = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                const double ox = (ix + 0.5) * spacing;
                const double oy = (iy + 0.5) * spacing;
                const double oz = (iz + 0.5) * spacing;
                // Rotate each molecule a little so they are not all aligned.
                const double phi = 0.7 * double(mol);

                const int o = idx, h1 = idx + 1, h2 = idx + 2;
                f.span(ATV::kAtom_x)[o] = ox;
                f.span(ATV::kAtom_y)[o] = oy;
                f.span(ATV::kAtom_z)[o] = oz;
                f.span(ATV::kAtom_mass)[o] = 15.9994;
                f.span(ATV::kAtom_type)[o] = 8;

                f.span(ATV::kAtom_x)[h1] =
                    ox + kROH * std::sin(half) * std::cos(phi);
                f.span(ATV::kAtom_y)[h1] =
                    oy + kROH * std::sin(half) * std::sin(phi);
                f.span(ATV::kAtom_z)[h1] = oz + kROH * std::cos(half);
                f.span(ATV::kAtom_mass)[h1] = 1.008;
                f.span(ATV::kAtom_type)[h1] = 1;

                f.span(ATV::kAtom_x)[h2] =
                    ox - kROH * std::sin(half) * std::cos(phi);
                f.span(ATV::kAtom_y)[h2] =
                    oy - kROH * std::sin(half) * std::sin(phi);
                f.span(ATV::kAtom_z)[h2] = oz + kROH * std::cos(half);
                f.span(ATV::kAtom_mass)[h2] = 1.008;
                f.span(ATV::kAtom_type)[h2] = 1;

                idx += 3;
                ++mol;
            }
    f.simbox = ATV::cpu::Box::cubic(n_side * spacing);
    f.rebuild_tag_index(0, n);

    // Topology through the loader API (it owns the LAMMPS ownership rules: the
    // smaller-tag atom owns a bond; the centre atom owns an angle).
    for (int m = 0; m < n_mol; ++m) {
        const int o = 3 * m, h1 = o + 1, h2 = o + 2;
        f.add_bond(f.span(ATV::kAtom_tag)[o], f.span(ATV::kAtom_tag)[h1], 0);
        f.add_bond(f.span(ATV::kAtom_tag)[o], f.span(ATV::kAtom_tag)[h2], 0);
        f.add_angle(f.span(ATV::kAtom_tag)[h1], f.span(ATV::kAtom_tag)[o],
                    f.span(ATV::kAtom_tag)[h2], 0);
    }
    // Intramolecular pairs MUST be excluded from the pair style. The O and its H's
    // sit 0.96 A apart — well inside sigma — so an unexcluded LJ would blast the
    // molecule apart and no constraint solver could hold it. (This is exactly what
    // my first draft of this test did: the position stage failed to converge, and
    // the bug was in the test, not in RATTLE.)
    f.build_special_list();
    return f;
}

/// Worst relative deviation of any constrained distance from its target.
double worst_geometry_error(const cpu::Frame& f, int n_mol) {
    double worst = 0.0;
    for (int m = 0; m < n_mol; ++m) {
        const int o = 3 * m, h1 = o + 1, h2 = o + 2;
        const std::pair<std::pair<int, int>, double> checks[3] = {
            {{o, h1}, kROH}, {{o, h2}, kROH}, {{h1, h2}, kDHH}};
        for (const auto& c : checks) {
            double dx = f.span(ATV::kAtom_x)[c.first.first]
                        - f.span(ATV::kAtom_x)[c.first.second];
            double dy = f.span(ATV::kAtom_y)[c.first.first]
                        - f.span(ATV::kAtom_y)[c.first.second];
            double dz = f.span(ATV::kAtom_z)[c.first.first]
                        - f.span(ATV::kAtom_z)[c.first.second];
            f.simbox.min_image(dx, dy, dz);
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            worst = std::max(worst, std::fabs(r - c.second) / c.second);
        }
    }
    return worst;
}

} // namespace

/// The DOF accounting: 3 constraints per rigid water, so 9 - 3 = 6 DOF per molecule.
TEST(FixRattle, ReportsThreeConstraintsPerWater) {
    cpu::Frame f = water_box(2, 6.0); // 8 molecules
    cpu::md::FixRattle r{1.0};
    r.constrain_bond(0, kROH);
    r.constrain_angle(0, kDHH);
    const cpu::Context rt{}; // single-rank
    EXPECT_EQ(r.n_constraints(rt, f), 3 * 8);
}

TEST(FixRattle, RejectsAnUnrepresentableTolerance) {
    EXPECT_THROW(cpu::md::FixRattle(
                     1.0, 0.5 * cpu::md::FixRattle::storage_tolerance_floor()),
                 std::invalid_argument);
}

/// The gate the ComponentRole::Constraint entry exists to keep shut: a Plan with a
/// constraint but NO integrator must not run. Before Constraint existed, tagging the
/// component as an Integrator would have satisfied cpu::md::MD's validation and
/// the system would have "run" with nothing propagating the equations of motion.
TEST(FixRattle, ConstraintDoesNotSatisfyTheIntegratorRequirement) {
    cpu::Frame f = water_box(2, 6.0);
    cpu::md::MD sim;
    sim.add_potential(cpu::PairLJCut{0.1, 3.0, 8.0});
    sim.set_neighbor(cpu::md::NeighborList<>{8.0, 0.3});
    cpu::md::FixRattle rattle{1.0};
    rattle.constrain_bond(0, kROH);
    rattle.constrain_angle(0, kDHH);
    sim.add_ref(rattle);
    // No integrator was registered.
    EXPECT_THROW(sim.build(), std::runtime_error);
}

/// An unresolvable partner THROWS. A bonded force style silently drops such a tuple
/// (a small missing energy term); a dropped CONSTRAINT is a molecule that is simply
/// not rigid — a physics corruption, not an approximation.
TEST(FixRattle, ThrowsOnUnresolvableConstraint) {
    cpu::Frame f = water_box(1, 6.0);
    f.span(ATV::kAtom_bond_atom)(0, 0) =
        999999; // a tag that is not in the frame
    f.rebuild_tag_index(0, f.nlocal());

    cpu::md::MD sim;
    sim.add_potential(
        cpu::PairLJCut{0.0067, 3.15, 6.0}.set_special_bonds(0.0, 0.0, 0.0));
    sim.set_neighbor(cpu::md::NeighborList<>{6.0, 0.5});
    sim.set_integrator(cpu::md::VelocityVerlet{1.0});
    cpu::md::FixRattle rattle{1.0};
    rattle.constrain_bond(0, kROH);
    sim.add_ref(rattle);
    sim.set_threads(2); // 24 atoms: OpenMP fork/join would dwarf the physics
    sim.enable_timing(false);
    sim.build();
    // The tag has no mass in the (global) table: no rank owns such an atom.
    EXPECT_THROW(sim.run(f, 1), std::runtime_error);
}

/// THE INVARIANT: the geometry stays rigid for the whole trajectory.
TEST(FixRattle, GeometryStaysRigidOverATrajectory) {
    const int n_side = 2, n_mol = n_side * n_side * n_side; // 8 waters — unit
    cpu::Frame f = water_box(n_side, 5.0); // tests stay seconds-fast
    const double dt = 2.0; // fs — larger than a flexible water could survive

    cpu::md::MD sim;
    sim.add_potential(
        cpu::PairLJCut{0.0067, 3.15, 6.0}.set_special_bonds(0.0, 0.0, 0.0));
    sim.set_neighbor(cpu::md::NeighborList<>{6.0, 0.5});
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    cpu::md::FixRattle rattle{dt};
    rattle.constrain_bond(0, kROH);
    rattle.constrain_angle(0, kDHH);
    sim.add_ref(rattle);
    sim.set_threads(2); // 24 atoms: OpenMP fork/join would dwarf the physics
    sim.enable_timing(false);
    sim.build();

    // Kick the molecules so the constraints have something to fight.
    for (int i = 0; i < f.nlocal(); ++i) {
        f.span(ATV::kAtom_vx)[i] = 0.004 * std::sin(double(3 * i + 1));
        f.span(ATV::kAtom_vy)[i] = 0.004 * std::cos(double(5 * i + 2));
        f.span(ATV::kAtom_vz)[i] = 0.004 * std::sin(double(7 * i + 3));
    }

    EXPECT_LT(worst_geometry_error(f, n_mol), kInitialGeometryTolerance)
        << "start geometry is not representably rigid";

    for (int chunk = 0; chunk < 5; ++chunk) {
        sim.run(f, 20);
        EXPECT_LT(worst_geometry_error(f, n_mol), kConstrainedGeometryTolerance)
            << "constrained geometry drifted after " << 20 * (chunk + 1)
            << " steps";
    }
}

/// THE OTHER INVARIANT: after the velocity stage, the relative velocity along every
/// constrained bond vanishes — d/dt |r_ij|^2 = 0. Skip this stage and the lengths
/// stay right while the molecule silently absorbs energy along its own bonds.
TEST(FixRattle, RelativeVelocityAlongBondsVanishes) {
    const int n_side = 2, n_mol = 8;
    cpu::Frame f = water_box(n_side, 6.0);
    const double dt = 2.0;

    cpu::md::MD sim;
    sim.add_potential(
        cpu::PairLJCut{0.0067, 3.15, 6.0}.set_special_bonds(0.0, 0.0, 0.0));
    sim.set_neighbor(cpu::md::NeighborList<>{6.0, 0.5});
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    cpu::md::FixRattle rattle{dt};
    rattle.constrain_bond(0, kROH);
    rattle.constrain_angle(0, kDHH);
    sim.add_ref(rattle);
    sim.set_threads(2); // 24 atoms: OpenMP fork/join would dwarf the physics
    sim.enable_timing(false);
    sim.build();

    for (int i = 0; i < f.nlocal(); ++i) {
        f.span(ATV::kAtom_vx)[i] = 0.003 * std::sin(double(2 * i + 1));
        f.span(ATV::kAtom_vy)[i] = 0.003 * std::cos(double(4 * i));
        f.span(ATV::kAtom_vz)[i] = 0.003 * std::sin(double(6 * i + 5));
    }
    sim.run(f, 25);

    for (int m = 0; m < n_mol; ++m) {
        const int o = 3 * m, h1 = o + 1, h2 = o + 2;
        const std::pair<int, int> pairs[3] = {{o, h1}, {o, h2}, {h1, h2}};
        for (const auto& p : pairs) {
            double dx =
                f.span(ATV::kAtom_x)[p.first] - f.span(ATV::kAtom_x)[p.second];
            double dy =
                f.span(ATV::kAtom_y)[p.first] - f.span(ATV::kAtom_y)[p.second];
            double dz =
                f.span(ATV::kAtom_z)[p.first] - f.span(ATV::kAtom_z)[p.second];
            f.simbox.min_image(dx, dy, dz);
            const double dvx = f.span(ATV::kAtom_vx)[p.first]
                               - f.span(ATV::kAtom_vx)[p.second];
            const double dvy = f.span(ATV::kAtom_vy)[p.first]
                               - f.span(ATV::kAtom_vy)[p.second];
            const double dvz = f.span(ATV::kAtom_vz)[p.first]
                               - f.span(ATV::kAtom_vz)[p.second];
            const double rv = dx * dvx + dy * dvy + dz * dvz;
            EXPECT_LT(std::fabs(rv), kConstrainedVelocityTolerance)
                << "molecule " << m
                << ": relative velocity along a constrained bond";
        }
    }
}

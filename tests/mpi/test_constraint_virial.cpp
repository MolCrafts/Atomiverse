#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/fix_rattle.h"
#include "atomiverse/cpu/dynamics/fix_shake.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include <gtest/gtest.h>
#include <mpi.h>
using namespace ATV;
namespace {
template <class Constraint>
void molecular_pressure() {
    cpu::Frame f(12);
    f.box = cpu::Box::cubic(24);
    const ColKey<StorageReal> pos[3] = {kAtom_x, kAtom_y, kAtom_z};
    const ColKey<StorageReal> vel[3] = {kAtom_vx, kAtom_vy, kAtom_vz};
    for (int c = 0; c < 6; ++c) {
        const int axis = c % 3;
        for (int a = 0; a < 2; ++a) {
            int i = 2 * c + a;
            f.span(kAtom_mass)[i] = 1;
            f.span(kAtom_tag)[i] = i + 1;
            for (int d = 0; d < 3; ++d)
                f.span(pos[d])[i] =
                    d == axis ? 12 + (a ? 0.5 : -0.5) : (c < 3 ? 6 : 18);
            f.span(vel[(axis + 1) % 3])[i] = a ? 0.1 : -0.1;
        }
    }
    f.clear_tag_index();
    f.rebuild_tag_index(0, 12);
    for (int c = 0; c < 6; ++c)
        f.add_bond(2 * c + 1, 2 * c + 2, 0);
    cpu::md::MD sim;
    sim.set_units("lj");
    sim.set_neighbor(cpu::md::NeighborList<>{2.0, 0.5});
    sim.set_evaluator(cpu::BondHarmonic{{0.0}, {1.0}});
    sim.set_integrator(cpu::md::VelocityVerlet{0.002});
    Constraint fix{0.002, 1e-13};
    fix.constrain_bond(0, 1);
    sim.set_observer(std::move(fix));
    sim.build();
    sim.run(f, 0);
    auto trace = [&]() {
        const auto& w = f.meta.get(kStress);
        return w[0] + w[1] + w[2];
    };
    EXPECT_NEAR(trace(), -0.12, 1e-8);
    sim.run(f, 1);
    EXPECT_NEAR(trace() + 2 * f.meta.get(kKinetic), 0, 1e-6);
}
} // namespace
TEST(MpiConstraintVirial, ShakeCountsEachCrossRankPairOnce) {
    molecular_pressure<cpu::md::FixShake>();
}
TEST(MpiConstraintVirial, RattleCountsEachCrossRankPairOnce) {
    molecular_pressure<cpu::md::FixRattle>();
}

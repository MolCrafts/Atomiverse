#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/fix_rattle.h"
#include "atomiverse/cpu/dynamics/fix_shake.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/task/task_materializer.h"
#include <gtest/gtest.h>
using namespace ATV;
namespace {
template <class Params, class Native>
void replay_constraint() {
    cpu::Frame f(2);
    f.box = cpu::Box::cubic(10);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_x)[i] = 4.5 + i;
        f.span(kAtom_y)[i] = 5;
        f.span(kAtom_z)[i] = 5;
        f.span(kAtom_mass)[i] = 1;
        f.span(kAtom_vy)[i] = i ? 0.1 : -0.1;
    }
    f.clear_tag_index();
    f.rebuild_tag_index(0, 2);
    f.add_bond(0, 1, 0);
    auto direct = f;
    Params p{0.002, 1e-10, 500};
    p.constrain_bond(0, 1);
    using namespace cpu::task;
    TaskMaterializer task{
        TaskSpec(MDTask{{{VelocityVerletParams{0.002}, std::nullopt},
                         {BondHarmonicParams{{0.0}, {1.0}}, std::nullopt},
                         {p, std::nullopt}},
                        1})};
    task.run(f);
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{0.002});
    sim.set_evaluator(cpu::BondHarmonic{{0.0}, {1.0}});
    sim.set_observer(p.template component<Native>());
    sim.build();
    sim.run(direct, 1);
    for (auto key : {kAtom_x, kAtom_y, kAtom_z, kAtom_vx, kAtom_vy, kAtom_vz})
        for (int i = 0; i < 2; ++i)
            EXPECT_DOUBLE_EQ(f.span(key)[i], direct.span(key)[i]);
    for (int d = 0; d < 6; ++d)
        EXPECT_DOUBLE_EQ(f.meta.get(kStress)[d], direct.meta.get(kStress)[d]);
}
} // namespace
TEST(ConstraintTask, ShakeCategoricalReplayMatchesNative) {
    replay_constraint<cpu::task::FixShakeParams, cpu::md::FixShake>();
}
TEST(ConstraintTask, RattleCategoricalReplayMatchesNative) {
    replay_constraint<cpu::task::FixRattleParams, cpu::md::FixRattle>();
}

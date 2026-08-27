/// @file tests/cpu/dynamics/test_fix_rattle_packer.cpp
/// @brief FixRattle XSHAKE / VP / V pack goldens. No MD closed loop.
///
/// L = 10 Å, x = 9 Å, pbc = {+1,0,0} → XSHAKE packed 19.0 Å.
/// VP / V vx = 0.4 Å/fs packs 0.4 (no image shift).
/// Cite LAMMPS `fix_rattle.cpp:652-678` (checkout `78a7ddf`).

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/component/constraint.h"
#include "atomiverse/cpu/component/constraint_protocol.h"
#include "atomiverse/cpu/dynamics/fix_rattle.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

cpu::Frame one_atom_box() {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 9.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_vx)[0] = 0.4;
    f.span(kAtom_vy)[0] = 0.0;
    f.span(kAtom_vz)[0] = 0.0;
    f.span(kAtom_mass)[0] = 1.0;
    f.span(kAtom_type)[0] = 0;
    f.box = cpu::Box::cubic(10.0);
    return f;
}

void bind_frame(cpu::md::FixRattle& rattle, cpu::Frame& frame,
                ATV::md::Phase phase) {
    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, frame, ATV::slot_of(phase), 0};
    rattle.launch(lc);
}

class StubConstraint : public cpu::Constraint, public cpu::ConstraintProvider {
  public:
    std::string name() const override { return "stub_constraint"; }
    const cpu::ConstraintProvider* as_constraint() const override {
        return this;
    }
    int n_constraints(const cpu::Context&, const cpu::Frame&) const override {
        return 0;
    }
    void launch(const cpu::LaunchContext&) override {}
};

class StubCharge : public cpu::Component {
  public:
    std::string name() const override { return "stub_charge"; }
    bool updates_owned_charges() const override { return true; }
    void launch(const cpu::LaunchContext&) override {}
};

bool plan_has_name(const cpu::md::MD& sim, const char* needle) {
    std::unordered_set<std::string> names;
    for (const auto& slot : sim.plan().slots) {
        for (const auto& op : slot.ops) {
            if (op.component)
                names.insert(op.component->name());
        }
    }
    return names.count(needle) != 0;
}

} // namespace

TEST(TestFixRattlePacker, XshakeAppliesBoxPbc) {
    cpu::md::FixRattle rattle{1.0};
    cpu::Frame frame = one_atom_box();
    bind_frame(rattle, frame, ATV::md::Phase::post_integrate);
    rattle.set_comm_mode(cpu::md::FixRattle::CommMode::XSHAKE);
    int list[] = {0};
    int pbc[] = {1, 0, 0};
    double buf[3] = {};
    EXPECT_EQ(rattle.comm_forward(), 3);
    EXPECT_EQ(rattle.pack_forward_comm(1, list, buf, 1, pbc), 3);
    EXPECT_DOUBLE_EQ(buf[0], 19.0);
    EXPECT_DOUBLE_EQ(buf[1], 0.0);
    EXPECT_DOUBLE_EQ(buf[2], 0.0);
}

TEST(TestFixRattlePacker, VpDoesNotApplyPbc) {
    cpu::md::FixRattle rattle{1.0};
    cpu::Frame frame = one_atom_box();
    bind_frame(rattle, frame, ATV::md::Phase::final_integrate);
    rattle.set_comm_mode(cpu::md::FixRattle::CommMode::VP);
    int list[] = {0};
    int pbc[] = {1, 0, 0};
    double buf[3] = {};
    EXPECT_EQ(rattle.pack_forward_comm(1, list, buf, 1, pbc), 3);
    EXPECT_DOUBLE_EQ(buf[0], 0.4);
    EXPECT_DOUBLE_EQ(buf[1], 0.0);
    EXPECT_DOUBLE_EQ(buf[2], 0.0);
}

TEST(TestFixRattlePacker, VDoesNotApplyPbc) {
    cpu::md::FixRattle rattle{1.0};
    cpu::Frame frame = one_atom_box();
    bind_frame(rattle, frame, ATV::md::Phase::final_integrate);
    rattle.set_comm_mode(cpu::md::FixRattle::CommMode::V);
    int list[] = {0};
    int pbc[] = {1, 0, 0};
    double buf[3] = {};
    EXPECT_EQ(rattle.pack_forward_comm(1, list, buf, 1, pbc), 3);
    EXPECT_DOUBLE_EQ(buf[0], 0.4);
}

TEST(TestFixRattlePacker, MdPlanHasNoConstraintOrChargeRefreshOp) {
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{1.0});
    sim.set_evaluator(cpu::PairLJCut{0.0103, 3.4, 2.5});
    sim.set_neighbor(cpu::md::NeighborList<>{2.5, 0.3});
    StubConstraint constraint;
    StubCharge charge;
    sim.on(ATV::md::Phase::post_integrate).add_ref(constraint);
    sim.on(ATV::md::Phase::pre_force).add_ref(charge);
    sim.build();
    EXPECT_FALSE(plan_has_name(sim, "halo_refresh_x"));
    EXPECT_FALSE(plan_has_name(sim, "halo_refresh_v"));
    EXPECT_FALSE(plan_has_name(sim, "halo_refresh_q"));
    EXPECT_FALSE(plan_has_name(sim, "halo_refresh"));
}

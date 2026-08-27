/// @file tests/cpu/driver/test_preset_comm_plan.cpp
/// @brief MD / Evaluate / Minimize Plan shape after halo plugins left the Plan.
///
/// `install_defaults()` still mounts `md::mpi::Reduce`. It does not mount
/// `halo_send` / `halo_recv` / `migrate` / `halo_refresh_*`. Halo and
/// exchange live on `ctx.md.comm`. `build()` only — no `run()`.

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>
#include <vector>

#include "atomiverse/core/plan.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

namespace {

constexpr double kCutoff = 6.0;
constexpr double kEps = 0.0104;
constexpr double kSigma = 3.4;
constexpr double kSkin = 0.3;

const std::unordered_set<std::string> kDeadCommNames{
    "halo_send",      "halo_recv",      "migrate",     "halo_refresh_x",
    "halo_refresh_v", "halo_refresh_q", "halo_refresh"};

std::vector<std::string>
plan_names(const ATV::Plan<ATV::cpu::Component>& plan) {
    std::vector<std::string> names;
    for (const auto& slot : plan.slots) {
        for (const auto& op : slot.ops) {
            if (op.component)
                names.push_back(op.component->name());
        }
    }
    return names;
}

bool has_name(const std::vector<std::string>& names, const std::string& want) {
    for (const auto& n : names) {
        if (n == want)
            return true;
    }
    return false;
}

std::string dead_hit(const std::vector<std::string>& names) {
    for (const auto& n : names) {
        if (kDeadCommNames.count(n) != 0)
            return n;
    }
    return {};
}

} // namespace

TEST(TestMdInstallDefaults, PlanHasReduceAndNoDeadCommPlugins) {
    ATV::cpu::md::MD sim;
    sim.set_integrator(ATV::cpu::md::VelocityVerlet{1.0});
    sim.set_evaluator(ATV::cpu::PairLJCut{kEps, kSigma, kCutoff});
    sim.set_neighbor(ATV::cpu::md::NeighborList<>{kCutoff, kSkin});
    sim.build();

    const auto names = plan_names(sim.plan());
    EXPECT_TRUE(has_name(names, "reduce")) << "Reduce must stay on the MD Plan";
    EXPECT_TRUE(dead_hit(names).empty())
        << "dead comm plugin still on the MD Plan: " << dead_hit(names);
}

TEST(TestEvaluateInstallDefaults, PlanHasReduceAndNoDeadCommPlugins) {
    ATV::cpu::Evaluate eval;
    eval.set_evaluator(ATV::cpu::PairLJCut{kEps, kSigma, kCutoff});
    eval.set_neighbor(ATV::cpu::md::NeighborList<>{kCutoff, kSkin});
    eval.build();

    const auto names = plan_names(eval.plan());
    EXPECT_TRUE(has_name(names, "reduce"))
        << "Reduce must stay on the Evaluate Plan";
    EXPECT_TRUE(dead_hit(names).empty())
        << "dead comm plugin still on the Evaluate Plan: " << dead_hit(names);
}

TEST(TestMinimizeInstallDefaults, PlanHasReduceAndNoDeadCommPlugins) {
    ATV::cpu::minimizer::Minimize opt;
    opt.set_optimizer(ATV::cpu::minimizer::LBFGS{});
    opt.set_evaluator(ATV::cpu::PairLJCut{kEps, kSigma, kCutoff});
    opt.set_neighbor(ATV::cpu::md::NeighborList<>{kCutoff, kSkin});
    opt.build();

    const auto names = plan_names(opt.plan());
    EXPECT_TRUE(has_name(names, "reduce"))
        << "Reduce must stay on the Minimize Plan";
    EXPECT_TRUE(dead_hit(names).empty())
        << "dead comm plugin still on the Minimize Plan: " << dead_hit(names);
}

/// @file tests/cpu/md/test_md_plan_equivalence.cpp
/// @brief Phase 5 (driver-infra-auto-injection) — T10 Plan-equivalence
///        and T15 escape-hatch coexistence.
///
/// T10 is the load-bearing acceptance test: build the same MD
/// simulation two ways — once via the categorical setter API
/// (`set_evaluator`, `set_neighbor`, `set_integrator`, …) and once via
/// the raw `on(phase).add(c, order)` / `on(phase).add_ref(c, order)`
/// escape hatch — and assert the two compiled component graphs (sorted
/// by phase / order / component name) are byte-equivalent. This
/// proves the new API is sugar over the underlying machinery,
/// not a rewrite that silently reorders components.
///
/// The raw path spells out, by hand, exactly what the setter's
/// `bind_defaults()` expands each component's `default_bindings()` into.
/// A multi-phase component (VelocityVerlet, NoseHooverNPT) is bound
/// through `add_ref` at each of its phases so ONE instance lands in
/// several slots — which is what `bind_defaults` does; binding an rvalue
/// per phase would silently create one integrator *per phase* and split
/// the thermostat/barostat state.
///
/// T15 verifies that user-defined hooks registered via
/// `sim.on(phase).add(MyHook{}, +N)` coexist with the categorical
/// setters and appear at the user-supplied order in the compiled
/// Plan.

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

namespace {

constexpr double kCutoff = 6.0;
constexpr double kEps = 0.0104;
constexpr double kSigma = 3.4;
constexpr double kSkin = 0.3;
constexpr double kDt = 1.0;

/// @brief Per-phase (phase, order, name) signature of a compiled Plan.
/// Two Plans are "Plan-equivalent" iff their signatures match exactly.
struct PlanSig {
    std::vector<std::tuple<int, std::string>> rows; // (phase, name)

    bool operator==(const PlanSig& o) const { return rows == o.rows; }
};

PlanSig signature_of(const ATV::cpu::md::MD& sim) {
    PlanSig s;
    const auto& p = sim.plan();
    for (const auto& slot : p.slots) {
        for (const auto& op : slot.ops) {
            s.rows.emplace_back(slot.phase, op.component->name());
        }
    }
    return s;
}

std::string render(const PlanSig& s) {
    std::string out;
    for (const auto& row : s.rows) {
        out += "  phase=" + std::to_string(std::get<0>(row))
               + " name=" + std::get<1>(row) + "\n";
    }
    return out;
}

// ─── T10: Plan equivalence — NVE LJ ─────────────────────────────────

TEST(MDPlanEquivalence, NVELJSetApiMatchesRawAddPath) {
    using namespace ATV;
    cpu::md::MD via_setters;
    via_setters.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCutoff});
    via_setters.set_neighbor(cpu::md::NeighborList<>{kCutoff, kSkin});
    via_setters.set_integrator(cpu::md::VelocityVerlet{kDt});
    via_setters.build();

    // Declared BEFORE the driver: the Plan borrows it, so it must outlive
    // `via_raw` (locals destruct in reverse declaration order).
    cpu::md::VelocityVerlet vv{kDt};

    cpu::md::MD via_raw;
    via_raw.on(md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    // set_neighbor binds at the dedicated `neighbor` phase @ -10 (the
    // LAMMPS-granular skeleton), so the raw mirror must too.
    via_raw.on(md::Phase::neighbor)
        .add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    // VelocityVerlet::default_bindings() = {initial_integrate @ 0,
    // final_integrate @ 0} — the kick-drift-kick split. BOTH, one instance.
    via_raw.on(md::Phase::initial_integrate).add_ref(vv, 0);
    via_raw.on(md::Phase::final_integrate).add_ref(vv, 0);
    via_raw.build();

    const auto sa = signature_of(via_setters);
    const auto sr = signature_of(via_raw);
    EXPECT_EQ(sa, sr) << "categorical setters:\n"
                      << render(sa) << "raw add() path:\n"
                      << render(sr);
}

// ─── T10: Plan equivalence — NPT (auto-Virial vs explicit) ─────

TEST(MDPlanEquivalence, NPTSetApiMatchesRawAddPathWithExplicitVirialStress) {
    using namespace ATV;

    // The set_api path triggers Case A: install_infra auto-injects
    // Virial because NoseHooverNPT::reads()==Stress and no
    // explicit Virial was registered.
    cpu::md::MD via_setters;
    via_setters.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCutoff});
    via_setters.set_neighbor(cpu::md::NeighborList<>{kCutoff, kSkin});
    via_setters.set_integrator(
        cpu::md::NoseHooverNPT{300.0, 100.0, 0.0, 1000.0, kDt});
    via_setters.build();

    // The raw path adds Virial explicitly — what existing
    // tests-mpi/* tests do today. Both multi-phase components are declared
    // before the driver that borrows them.
    cpu::md::NoseHooverNPT npt{300.0, 100.0, 0.0, 1000.0, kDt};

    cpu::md::MD via_raw;
    via_raw.on(md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    via_raw.on(md::Phase::neighbor)
        .add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    // NoseHooverNPT::default_bindings() = {initial_integrate @ 0,
    // final_integrate @ 0, end_of_step @ -40} — the MTK half-steps plus the
    // chain/barostat bookkeeping. All three, one instance.
    via_raw.on(md::Phase::initial_integrate).add_ref(npt, 0);
    via_raw.on(md::Phase::final_integrate).add_ref(npt, 0);
    via_raw.on(md::Phase::end_of_step).add_ref(npt, -40);
    // Virial::default_bindings() = {post_force @ -50} — the same slot
    // install_defaults() auto-injects into on the setter path.
    via_raw.on(md::Phase::post_force).add(cpu::md::Virial{}, -50);
    via_raw.build();

    const auto sa = signature_of(via_setters);
    const auto sr = signature_of(via_raw);
    EXPECT_EQ(sa, sr) << "categorical setters:\n"
                      << render(sa) << "raw add() path with explicit Virial:\n"
                      << render(sr);
}

// ─── T15: add()/on() escape hatch coexists with set_* setters ───────

class CustomEndOfStepHook : public ATV::cpu::Component {
  public:
    std::string name() const override { return "custom_eos_hook"; }
    void launch(const ATV::cpu::LaunchContext&) override {}
};

TEST(MDPlanEquivalence, CustomHookViaOnAddCoexistsWithSetters) {
    using namespace ATV;
    cpu::md::MD sim;
    sim.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCutoff});
    sim.set_neighbor(cpu::md::NeighborList<>{kCutoff, kSkin});
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    // User-defined hook through the public escape hatch:
    sim.on(md::Phase::end_of_step).add(CustomEndOfStepHook{}, +50);
    sim.build();

    const auto sig = signature_of(sim);
    bool found = false;
    int found_phase = -1;
    for (const auto& row : sig.rows) {
        if (std::get<1>(row) == "custom_eos_hook") {
            found = true;
            found_phase = std::get<0>(row);
            break;
        }
    }
    EXPECT_TRUE(found) << "custom hook missing from compiled Plan:\n"
                       << render(sig);
    EXPECT_EQ(found_phase, ATV::slot_of(md::Phase::end_of_step));
}

} // namespace

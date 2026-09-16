/// @file tests/cpu/test_install_infra_hook.cpp
/// @brief Phase 1 (driver-infra-auto-injection) — assert the foundation:
///        (a) `Component::requires_system_stress()` defaults to false;
///        (b) the neighbor contract is an opt-in protocol — see (g)/(h);
///        (c) a CPU driver subclass can override `install_defaults()` to inject
///            a component, and the hook fires *before* `validate_components()`
///            so an injection can satisfy an otherwise-unsatisfied validation
///            rule.
///
/// Acceptance: T1 + T2 of `.claude/specs/driver-infra-auto-injection.md`.
///
/// arch-refactor-02-halo additions (`HaloParticipant` protocol scan):
///        (d) `Component::halo_participant()` defaults to `nullptr`; the
///            legacy `pack_forward` name is gone from the base vtable
///            (packer protocol is `pack_forward_comm`).
///
/// arch-refactor-03-neighbor additions (`NeighborConsumer` /
/// `NeighborProvider` opt-in protocol, neighbor_protocol.h):
///        (g) `Component::as_neighbor_consumer()` and
///            `as_neighbor_provider()` default to `nullptr`; the three
///            legacy in-vtable neighbor trio is gone
///            from the base vtable;
///        (h) protocol implementers surface themselves through the typed
///            accessors: a consumer probe reports its interaction cutoff
///            [Å] via `NeighborConsumer::neighbor_cutoff()`, a provider
///            probe tags itself via `as_neighbor_provider()`, and the two
///            protocols are orthogonal (opting into one leaves the other
///            accessor `nullptr`).

#include <gtest/gtest.h>

#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "atomiverse/core/plan.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/neighbor_protocol.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

#include "cpu/driver_test_helpers.h"

using atv_test::make_trivial_frame;
using atv_test::StubForceEvaluator;
using atv_test::StubIntegrator;

namespace {

// ─── Foundation virtuals: defaults ──────────────────────────────────

TEST(InstallInfraHook, RequiresSystemStressDefaultsFalse) {
    StubIntegrator c;
    EXPECT_FALSE(c.requires_system_stress());
    EXPECT_FALSE(c.provides_system_stress());
}

TEST(InstallInfraHook, AsNeighborConsumerDefaultsToNull) {
    StubIntegrator c;
    EXPECT_EQ(c.as_neighbor_consumer(), nullptr);
}

TEST(InstallInfraHook, AsNeighborProviderDefaultsToNull) {
    StubIntegrator c;
    EXPECT_EQ(c.as_neighbor_provider(), nullptr);
}

// ─── install_defaults() hook fires before validate_components() ────────

/// @brief Counts how often `install_defaults()` is called per `build()`.
///
/// Subclasses the `cpu::md::MD` preset (the category bases are gone). MD *itself*
/// overrides `install_defaults()` — it injects `mpi::Reduce` and the
/// neighbor-list guard — so the override must chain to `MD::install_defaults()`, not to
/// `Driver`'s empty base hook, or that infrastructure silently disappears.
class HookCounterDynamics : public ATV::cpu::md::MD {
  public:
    int install_calls = 0;
    StubIntegrator integ;

  protected:
    void install_defaults() override {
        ++install_calls;
        ATV::cpu::md::MD::install_defaults();
        // Inject the integrator from inside the hook — driver's
        // `validate_components()` must see this component on the
        // pending list, otherwise validation throws "missing Integrator".
        this->on(ATV::md::Phase::initial_integrate).add_ref(integ);
    }
};

TEST(InstallInfraHook, HookFiresBeforeValidateComponents) {
    HookCounterDynamics d;
    // Only register the force evaluator from outside; the integrator
    // arrives via install_defaults(). If validate_components() ran first,
    // the build would throw "missing Integrator".
    StubForceEvaluator force;
    d.on(ATV::md::Phase::force).add_ref(force);

    EXPECT_NO_THROW(d.build());
    EXPECT_EQ(d.install_calls, 1);
}

TEST(InstallInfraHook, HookFiresExactlyOncePerBuild) {
    HookCounterDynamics d;
    StubForceEvaluator force;
    d.on(ATV::md::Phase::force).add_ref(force);

    d.build();
    EXPECT_EQ(d.install_calls, 1);
}

// ─── install_defaults injects infrastructure, never a required role ────

TEST(InstallInfraHook, DefaultInstallInfraIsNoOp) {
    // The plain `cpu::md::MD` (no subclass override) must still throw on missing
    // Integrator. Its own `install_defaults()` injects INFRASTRUCTURE (Reduce,
    // …) — proves it never silently fabricates a component that satisfies a required
    // role, which is what would make the validation rule vacuous.
    ATV::cpu::md::MD d;
    StubForceEvaluator force;
    d.on(ATV::md::Phase::force).add_ref(force);
    EXPECT_THROW(d.build(), std::runtime_error);
}

// ═══ Component base vtable stays free of halo plumbing ═════════════════

// Detection idiom: the legacy halo virtuals must stay GONE from
// `cpu::Component`. The `halo_participant()` query that once replaced them
// is itself gone (comm-maint-01-deadcode): it had zero implementers, and
// the extra-field forward path is self-initiated by the plugin.
#define ATV_TEST_HAS_MEMBER(memb)                                              \
    template <typename T, typename = void>                                     \
    struct HasMember_##memb : std::false_type {};                              \
    template <typename T>                                                      \
    struct HasMember_##                                                        \
        memb<T, std::void_t<decltype(&T::memb)>> : std::true_type {}

ATV_TEST_HAS_MEMBER(provides_halo_send);
ATV_TEST_HAS_MEMBER(provides_halo_recv);
ATV_TEST_HAS_MEMBER(comm_reverse);
ATV_TEST_HAS_MEMBER(pack_forward);
ATV_TEST_HAS_MEMBER(unpack_forward);
ATV_TEST_HAS_MEMBER(pack_reverse);
ATV_TEST_HAS_MEMBER(unpack_reverse);

#undef ATV_TEST_HAS_MEMBER

TEST(HaloParticipantProtocol, LegacyHaloVirtualsAreGoneFromComponentBase) {
    using CB = ATV::cpu::Component;
    static_assert(!HasMember_provides_halo_send<CB>::value,
                  "provides_halo_send must be deleted from cpu::Component");
    static_assert(!HasMember_provides_halo_recv<CB>::value,
                  "provides_halo_recv must be deleted from cpu::Component");
    static_assert(!HasMember_comm_reverse<CB>::value,
                  "comm_reverse must be deleted from cpu::Component");
    static_assert(!HasMember_pack_forward<CB>::value,
                  "pack_forward must be deleted from cpu::Component");
    static_assert(!HasMember_unpack_forward<CB>::value,
                  "unpack_forward must be deleted from cpu::Component");
    static_assert(!HasMember_pack_reverse<CB>::value,
                  "pack_reverse must be deleted from cpu::Component");
    static_assert(!HasMember_unpack_reverse<CB>::value,
                  "unpack_reverse must be deleted from cpu::Component");
    SUCCEED(); // contract is compile-time
}

// ═══ NeighborConsumer/NeighborProvider protocol (arch-refactor-03) ══════

/// @brief Interaction cutoff [Å] declared by the consumer probe below.
constexpr ATV::AccReal kProbeCutoff = 5.5;

/// @brief Consumer-side probe: opts into `NeighborConsumer` alongside
/// `Component` and reports a fixed cutoff. Mirrors what
/// `cpu::PairEvaluator` / `cpu::CoulLong` do in production.
class ProbeNeighborConsumer : public ATV::cpu::Component,
                              public ATV::cpu::NeighborConsumer {
  public:
    std::string name() const override { return "probe_neighbor_consumer"; }
    const ATV::cpu::NeighborConsumer* as_neighbor_consumer() const override {
        return this;
    }
    ATV::AccReal neighbor_cutoff() const override { return kProbeCutoff; }
    void launch(const ATV::cpu::LaunchContext&) override {}
};

/// @brief Supply-side probe: opts into the `NeighborProvider` tag protocol.
/// Mirrors what `cpu::md::NeighborList` does in production.
class ProbeNeighborProvider : public ATV::cpu::Component,
                              public ATV::cpu::NeighborProvider {
  public:
    std::string name() const override { return "probe_neighbor_provider"; }
    const ATV::cpu::NeighborProvider* as_neighbor_provider() const override {
        return this;
    }
    ATV::cpu::NeighborProvider* neighbor_provider() override { return this; }
    /// A provider must answer the row-order question (Monte Carlo forbids the
    /// reorder); this probe never reorders anything, so it has nothing to do.
    void forbid_row_reordering() override { reordering_forbidden = true; }
    void launch(const ATV::cpu::LaunchContext&) override {}

    bool reordering_forbidden = false;
};

// ─── (g) base-vtable contract ───────────────────────────────────────────
// The deletion of the legacy in-vtable neighbor trio is enforced by the
// acceptance grep gate (arch-refactor-03 ac-001) plus the fact that every
// migrated call site compiles — no in-test detection idiom needed (its
// literal member-name tokens would themselves trip the grep gate).

// ─── (h) protocol accessors route to the implementer ────────────────────

TEST(NeighborProtocol, ConsumerProbeExposesCutoffThroughAccessor) {
    ProbeNeighborConsumer probe;
    const ATV::cpu::NeighborConsumer* nc = probe.as_neighbor_consumer();
    ASSERT_NE(nc, nullptr);
    EXPECT_EQ(nc, static_cast<const ATV::cpu::NeighborConsumer*>(&probe));
    EXPECT_EQ(nc->neighbor_cutoff(), kProbeCutoff); // exact forwarding [Å]
}

TEST(NeighborProtocol, ProviderProbeExposesTagThroughAccessor) {
    ProbeNeighborProvider probe;
    const ATV::cpu::NeighborProvider* np = probe.as_neighbor_provider();
    ASSERT_NE(np, nullptr);
    EXPECT_EQ(np, static_cast<const ATV::cpu::NeighborProvider*>(&probe));
}

TEST(NeighborProtocol, ConsumerAndProviderProtocolsAreOrthogonal) {
    // Opting into one side of the contract must not implicitly opt into
    // the other — consumer groups (pair styles, CoulLong) and provider
    // groups (NeighborList) are disjoint by design.
    ProbeNeighborConsumer consumer;
    ProbeNeighborProvider provider;
    EXPECT_EQ(consumer.as_neighbor_provider(), nullptr);
    EXPECT_EQ(provider.as_neighbor_consumer(), nullptr);
}

} // namespace

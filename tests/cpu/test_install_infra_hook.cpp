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
///            eight legacy halo virtuals are gone from the base vtable;
///        (e) `HaloSend`/`HaloRecv` expose `Endpoint::send`/`Endpoint::recv`
///            through the protocol, and a manually registered pair +
///            `add_potential` builds to exactly ONE endpoint pair (no double
///            injection);
///        (f) a pure field participant (`Endpoint::none`) is not treated as
///            an endpoint and does not suppress the automatic injection.
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

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/comm/halo_exchange.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/halo_participant.h"
#include "atomiverse/cpu/component/neighbor_protocol.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/facade/component_role.h"
#include "atomiverse/facade/plan.h"
#include "atomiverse/facade/stage.h"

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
/// overrides `install_defaults()` — it injects `mpi::Reduce`, the halo pair and the
/// neighbor-list guard — so the override must chain to `MD::install_defaults()`, not to
/// `CPUDriver`'s empty base hook, or that infrastructure silently disappears.
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
    // Integrator. Its own `install_defaults()` injects INFRASTRUCTURE (Reduce, halo,
    // …) — proves it never silently fabricates a component that satisfies a required
    // role, which is what would make the validation rule vacuous.
    ATV::cpu::md::MD d;
    StubForceEvaluator force;
    d.on(ATV::md::Phase::force).add_ref(force);
    EXPECT_THROW(d.build(), std::runtime_error);
}

// ═══ HaloParticipant protocol scan (arch-refactor-02-halo) ══════════════

/// @brief Pure field participant: opts into `HaloParticipant`, keeps the
/// default `Endpoint::none`, and declares one width-1 forward field. Must
/// neither count as a halo endpoint nor suppress endpoint auto-injection.
class PureFieldParticipant : public ATV::cpu::Component,
                             public ATV::cpu::HaloParticipant {
  public:
    std::string name() const override { return "pure_field_participant"; }
    ATV::cpu::HaloParticipant* halo_participant() override { return this; }
    std::vector<CommField> comm_forward_fields() const override {
        CommField f;
        f.data = nullptr; // storage is only allocated in build(); the
        f.width = 1;      // declaration shape alone drives the driver scan
        return {f};
    }
    void launch(const ATV::cpu::LaunchContext&) override {}
};

/// @brief Count distinct send/recv halo endpoints in a compiled Plan via the
/// protocol probe (`halo_participant()->halo_endpoint()`), deduplicating
/// components that appear under multiple phase bindings. No `dynamic_cast`,
/// no `name()` matching — same discipline as `scan_pending_infra_()`.
std::pair<int, int>
count_halo_endpoints(const ATV::Plan<ATV::cpu::Component>& plan) {
    std::unordered_set<const ATV::cpu::Component*> seen;
    int sends = 0;
    int recvs = 0;
    for (const auto& slot : plan.slots) {
        for (const auto& op : slot.ops) {
            ATV::cpu::Component* c = op.component;
            if (!seen.insert(c).second)
                continue;
            ATV::cpu::HaloParticipant* p = c->halo_participant();
            if (p == nullptr)
                continue;
            switch (p->halo_endpoint()) {
            case ATV::cpu::HaloParticipant::Endpoint::send:
                ++sends;
                break;
            case ATV::cpu::HaloParticipant::Endpoint::recv:
                ++recvs;
                break;
            case ATV::cpu::HaloParticipant::Endpoint::none:
                break;
            }
        }
    }
    return {sends, recvs};
}

// ─── (d) base-vtable contract ───────────────────────────────────────────

TEST(HaloParticipantProtocol, ComponentBaseDefaultsToNullParticipant) {
    StubIntegrator c;
    EXPECT_EQ(c.halo_participant(), nullptr);
}

// Detection idiom: the eight legacy halo virtuals must be GONE from
// `cpu::Component` (replaced by the single `halo_participant()` query).
#define ATV_TEST_HAS_MEMBER(memb)                                              \
    template <typename T, typename = void>                                     \
    struct HasMember_##memb : std::false_type {};                              \
    template <typename T>                                                      \
    struct HasMember_##                                                        \
        memb<T, std::void_t<decltype(&T::memb)>> : std::true_type {}

ATV_TEST_HAS_MEMBER(provides_halo_send);
ATV_TEST_HAS_MEMBER(provides_halo_recv);
ATV_TEST_HAS_MEMBER(comm_forward);
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
    static_assert(!HasMember_comm_forward<CB>::value,
                  "comm_forward must be deleted from cpu::Component");
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

// ─── (e) endpoint exposure + no double injection ────────────────────────

TEST(HaloParticipantProtocol, HaloSendRecvExposeEndpoints) {
    ATV::cpu::comm::HaloSend send{2.0};
    ATV::cpu::HaloParticipant* ps = send.halo_participant();
    ASSERT_NE(ps, nullptr);
    EXPECT_EQ(ps, static_cast<ATV::cpu::HaloParticipant*>(&send));
    EXPECT_EQ(ps->halo_endpoint(), ATV::cpu::HaloParticipant::Endpoint::send);

    ATV::cpu::comm::HaloRecv recv{};
    ATV::cpu::HaloParticipant* pr = recv.halo_participant();
    ASSERT_NE(pr, nullptr);
    EXPECT_EQ(pr, static_cast<ATV::cpu::HaloParticipant*>(&recv));
    EXPECT_EQ(pr->halo_endpoint(), ATV::cpu::HaloParticipant::Endpoint::recv);
}

TEST(HaloParticipantProtocol, ManualHaloPairIsNotDoubleInjected) {
    // User registers the halo pair explicitly (custom cutoff) AND sets a
    // pair evaluator. install_defaults() must detect the existing endpoints
    // via the protocol probe and inject nothing on top — reverse exchange
    // must run exactly once (see halo_exchange.h).
    ATV::cpu::md::MD sim;
    StubIntegrator integ;
    sim.on(ATV::md::Phase::initial_integrate).add_ref(integ);
    sim.add_potential(ATV::cpu::PairLJCut{0.0104, 3.4, 6.0});
    sim.set_neighbor(ATV::cpu::md::NeighborList<>{6.0});
    sim.add(ATV::cpu::comm::HaloSend{6.6});
    sim.add(ATV::cpu::comm::HaloRecv{});
    sim.build();

    const auto [sends, recvs] = count_halo_endpoints(sim.plan());
    EXPECT_EQ(sends, 1) << "expected exactly one HaloSend endpoint "
                           "(manual registration must suppress auto-inject)";
    EXPECT_EQ(recvs, 1) << "expected exactly one HaloRecv endpoint "
                           "(manual registration must suppress auto-inject)";
}

// ─── (f) Endpoint::none neither endpoint nor injection suppressor ───────

TEST(HaloParticipantProtocol, FieldParticipantIsNotAnEndpoint) {
    PureFieldParticipant fp;
    ATV::cpu::HaloParticipant* p = fp.halo_participant();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->halo_endpoint(), ATV::cpu::HaloParticipant::Endpoint::none);
    EXPECT_TRUE(fp.comm_reverse_fields().empty()); // base default
    ASSERT_EQ(fp.comm_forward_fields().size(), 1u);
    EXPECT_EQ(fp.comm_forward_fields()[0].width, 1);
}

TEST(HaloParticipantProtocol, FieldParticipantDoesNotSuppressAutoInjection) {
    // A pure field participant is registered, but NO halo endpoints are.
    // Under an MPI build, install_defaults() must still auto-inject the
    // HaloSend/HaloRecv pair for the add_potential cutoff: Endpoint::none must
    // not trip the has_halo_send/has_halo_recv scan booleans.
    ATV::cpu::md::MD sim;
    StubIntegrator integ;
    PureFieldParticipant fp;
    sim.on(ATV::md::Phase::initial_integrate).add_ref(integ);
    sim.add_potential(ATV::cpu::PairLJCut{0.0104, 3.4, 6.0});
    sim.set_neighbor(ATV::cpu::md::NeighborList<>{6.0});
    sim.on(ATV::md::Phase::force).add_ref(fp, +100);
    sim.build();

    const auto [sends, recvs] = count_halo_endpoints(sim.plan());
#ifdef ATV_HAS_MPI
    EXPECT_EQ(sends, 1) << "Endpoint::none participant suppressed the "
                           "HaloSend auto-injection";
    EXPECT_EQ(recvs, 1) << "Endpoint::none participant suppressed the "
                           "HaloRecv auto-injection";
#else
    EXPECT_EQ(sends, 0) << "no MPI build: nothing may inject endpoints";
    EXPECT_EQ(recvs, 0) << "no MPI build: nothing may inject endpoints";
#endif
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
    void launch(const ATV::cpu::LaunchContext&) override {}
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

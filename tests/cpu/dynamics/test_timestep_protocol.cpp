/// @file tests/cpu/dynamics/test_timestep_protocol.cpp
/// @brief esmm-cpmd-02-timestep-protocol — dt is single-sourced from the
///        integrator through the `TimestepProvider` / `TimestepConsumer`
///        opt-in protocol pair, pushed by `cpu::md::MD` at build time.
///
/// The deleted `GPWEvaluator` electronic-dt member duplicated
/// `VelocityVerlet`'s private `dt_` on faith: "must match the nuclear dt" was
/// a docs-only contract with no runtime check. This file pins the replacement
/// contract on the MD side, with local stubs so nothing here depends on GPW:
///
///   - a provider publishes its ctor dt [fs] through `as_timestep_provider()`
///     (`VelocityVerlet` and `NVELimit` — two independently-shaped ctors, so
///     the accessor cannot be special-cased on one class);
///   - two providers whose dt disagree → `MD::build()` throws NAMING BOTH
///     (a broken setup whether or not anything consumes the dt);
///   - a consumer with `timestep_required()` and no provider → throw naming
///     the consumer and the protocol it needs;
///   - the push lands on every consumer BEFORE component `build()` runs, so
///     build-time validation inside a consumer can already see the dt;
///   - a non-required consumer with no provider is silently fine (and is
///     never handed a dt);
///   - a dt that is not finite and > 0 surfaces as a build-time throw and
///     never reaches a consumer as a usable value.
///
/// Determinism: no frame physics is asserted — the single `run(frame, 1)` in
/// `ConsumerReceivesPushBeforeBuild` exists only to fire the component
/// `build()` pass (MD defers it to the first run), and all components are
/// no-ops.
///
/// The GPW-side twin of this contract (a cpmd evaluator refusing to build with
/// no pushed dt, and `set_timestep` validation) is in
/// tests/cpu/gpw/test_gpw_cpmd.cpp.
///
/// Spec: `.claude/specs/esmm-cpmd-02-timestep-protocol/spec.md` (Design).

#include "atomiverse/cpu/dynamics/barostat.h"
#include "atomiverse/cpu/dynamics/berendsen.h"
#include "atomiverse/cpu/dynamics/deform.h"
#include "atomiverse/cpu/dynamics/fix_move.h"
#include "atomiverse/cpu/dynamics/fix_oscillating_efield.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/integrator.h"
#include "atomiverse/cpu/component/timestep_protocol.h"
#include "atomiverse/cpu/driver/dof.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nve_limit.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"

#include "cpu/driver_test_helpers.h"

namespace {

constexpr double kDt = 0.5;      // [fs] — the nuclear timestep under test
constexpr double kOtherDt = 1.0; // [fs] — a DIFFERENT, conflicting one
constexpr double kXmax = 0.1;    // [Å]  — NVELimit displacement cap

TEST(IntegrationImages, VerletLeavesCrossingUnwrappedUntilNeighborRebuild) {
    ATV::cpu::Frame frame(1);
    frame.box = ATV::cpu::Box::cubic(10);
    frame.span(ATV::kAtom_x)[0] = 9.9;
    frame.span(ATV::kAtom_vx)[0] = 1;
    frame.span(ATV::kAtom_mass)[0] = 1;
    frame.grow(4);
    frame.set_nghost(1);
    frame.span(ATV::kAtom_x)[1] = -.1;
    ATV::cpu::Context context;
    ATV::cpu::md::VelocityVerlet integrator(.2);
    integrator.launch(
        {context, frame, ATV::slot_of(ATV::md::Phase::initial_integrate), 0});
    EXPECT_NEAR(frame.span(ATV::kAtom_x)[0], 10.1, 1e-14);
    EXPECT_DOUBLE_EQ(frame.span(ATV::kAtom_x)[1], -.1);
}

bool message_mentions(const std::exception& e, const std::string& needle) {
    return std::string(e.what()).find(needle) != std::string::npos;
}

/// @brief Test-only integrator that publishes a timestep [fs] through the
/// protocol. Deliberately not one of the shipping integrators: the push walk
/// must key on the typed accessor, never on a concrete class or a name.
class TimestepProviderStub final : public ATV::cpu::Integrator,
                                   public ATV::cpu::TimestepProvider {
  public:
    TimestepProviderStub(std::string name, double dt_fs)
        : name_(std::move(name)), dt_fs_(dt_fs) {}

    std::string name() const override { return name_; }
    const ATV::cpu::TimestepProvider* as_timestep_provider() const override {
        return this;
    }
    double timestep() const override { return dt_fs_; }
    std::vector<ATV::DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 0)};
    }
    void launch(const ATV::cpu::LaunchContext&) override {}

  private:
    std::string name_;
    double dt_fs_;
};

/// @brief Test-only consumer: records the pushed dt [fs] and whether it had
/// already arrived by the time its own component `build()` ran.
///
/// `set_timestep` reproduces the documented contract of the real consumers
/// (`GPWEvaluator`, later `MeanFieldMethod`): a dt that is not finite and > 0
/// is rejected where it arrives, so a bad value can never be stored.
class TimestepConsumerStub final : public ATV::cpu::Component,
                                   public ATV::cpu::TimestepConsumer {
  public:
    TimestepConsumerStub(std::string name, bool required)
        : name_(std::move(name)), required_(required) {}

    int set_calls = 0;
    int build_calls = 0;
    double received_dt_fs = 0.0;
    bool dt_known_at_build = false;

    std::string name() const override { return name_; }
    ATV::cpu::TimestepConsumer* as_timestep_consumer() override { return this; }
    bool timestep_required() const override { return required_; }

    void set_timestep(double dt_fs) override {
        if (!(dt_fs > 0.0) || !std::isfinite(dt_fs))
            throw std::invalid_argument(
                name_ + ": timestep must be finite and > 0 [fs], got "
                + std::to_string(dt_fs));
        ++set_calls;
        received_dt_fs = dt_fs;
    }

    /// @brief True once a usable nuclear dt has been received.
    bool holds_usable_dt() const {
        return received_dt_fs > 0.0 && std::isfinite(received_dt_fs);
    }

    std::vector<ATV::DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, 0)};
    }
    void build(const ATV::cpu::BuildContext&) override {
        ++build_calls;
        dt_known_at_build = holds_usable_dt();
    }
    void launch(const ATV::cpu::LaunchContext&) override {}

  private:
    std::string name_;
    bool required_;
};

} // namespace

// ─── Provider side: the ctor dt is published, not duplicated ────────────

TEST(TimestepProtocol, VelocityVerletPublishesCtorTimestep) {
    ATV::cpu::md::VelocityVerlet vv{kDt};
    const ATV::cpu::TimestepProvider* p = vv.as_timestep_provider();
    ASSERT_NE(p, nullptr) << "VelocityVerlet must be a TimestepProvider";
    EXPECT_DOUBLE_EQ(p->timestep(), kDt);
}

/// The same one-line override, on a differently-shaped ctor `(dt, xmax)`:
/// the protocol is a property of every integrator that owns the step, not a
/// VelocityVerlet special case.
TEST(TimestepProtocol, NVELimitPublishesCtorTimestep) {
    ATV::cpu::md::NVELimit nve{kDt, kXmax};
    const ATV::cpu::TimestepProvider* p = nve.as_timestep_provider();
    ASSERT_NE(p, nullptr) << "NVELimit must be a TimestepProvider";
    EXPECT_DOUBLE_EQ(p->timestep(), kDt);
}

// ─── Push-site rules on cpu::md::MD ────────────────────────────────────

/// Two integrators publishing different dt is a broken setup — one of them is
/// wrong and the engine cannot know which. The throw must name BOTH, because
/// naming one leaves the user diffing the whole component set for the other.
/// Unconditional: no consumer is registered here, so this is not "the push
/// failed", it is "the clock is ambiguous".
TEST(TimestepProtocol, ConflictingProvidersThrowNamingBoth) {
    TimestepProviderStub fast{"stub_integrator_dt_half", kDt};
    TimestepProviderStub slow{"stub_integrator_dt_one", kOtherDt};
    atv_test::NoOpForce force;
    ATV::cpu::md::MD sim; // declared last: the driver destructs first

    sim.on(ATV::md::Phase::initial_integrate).add_ref(fast, 0);
    sim.on(ATV::md::Phase::initial_integrate).add_ref(slow, 10);
    sim.set_observer_ref(force);

    try {
        sim.build();
        FAIL() << "expected MD::build() to reject two disagreeing timesteps";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "stub_integrator_dt_half")) << e.what();
        EXPECT_TRUE(message_mentions(e, "stub_integrator_dt_one")) << e.what();
    }
}

/// Equal timestep values do not make two independent integrators one owner.
TEST(TimestepProtocol, DistinctIntegratorsWithEqualTimestepsAreRejected) {
    TimestepProviderStub one{"stub_integrator_a", kDt};
    TimestepProviderStub two{"stub_integrator_b", kDt};
    atv_test::NoOpForce force;
    TimestepConsumerStub consumer{"stub_dt_consumer", /*required=*/true};
    ATV::cpu::md::MD sim;

    sim.on(ATV::md::Phase::initial_integrate).add_ref(one, 0);
    sim.on(ATV::md::Phase::initial_integrate).add_ref(two, 10);
    sim.set_observer_ref(force);
    sim.set_observer_ref(consumer);

    EXPECT_THROW(sim.build(), std::runtime_error);
    EXPECT_EQ(consumer.set_calls, 0);
}

/// A consumer that cannot run without a dt (`cpmd == true`) and no provider is
/// the exact silent-wrong-answer this protocol exists to kill. The message must
/// name the consumer AND the protocol, so the fix ("add an MD integrator") is
/// readable from the error alone.
TEST(TimestepProtocol, RequiredConsumerWithoutProviderThrows) {
    atv_test::StubIntegrator integrator; // an Integrator, but NO provider
    atv_test::NoOpForce force;
    TimestepConsumerStub consumer{"stub_dt_consumer", /*required=*/true};
    ATV::cpu::md::MD sim;

    sim.on(ATV::md::Phase::initial_integrate).add_ref(integrator);
    sim.set_observer_ref(force);
    sim.set_observer_ref(consumer);

    try {
        sim.build();
        FAIL() << "expected MD::build() to reject a required timestep consumer "
                  "with no TimestepProvider";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "stub_dt_consumer")) << e.what();
        EXPECT_TRUE(message_mentions(e, "TimestepProvider")) << e.what();
    }
    EXPECT_EQ(consumer.set_calls, 0);
}

/// The push runs at the END of `install_defaults()`, i.e. inside driver
/// `build()` and strictly before any component `build()` — that ordering is
/// what lets a consumer validate its own dt at build time.
TEST(TimestepProtocol, ConsumerReceivesPushBeforeBuild) {
    atv_test::NoOpForce force;
    TimestepConsumerStub consumer{"stub_dt_consumer", /*required=*/true};
    ATV::cpu::md::MD sim;

    sim.set_integrator(ATV::cpu::md::VelocityVerlet{kDt});
    sim.set_observer_ref(force);
    sim.set_observer_ref(consumer);
    sim.build();

    EXPECT_EQ(consumer.set_calls, 1);
    EXPECT_DOUBLE_EQ(consumer.received_dt_fs, kDt);
    EXPECT_EQ(consumer.build_calls, 0)
        << "driver build() must push before the component build() pass";

    // MD defers the component build() pass to the first run; one no-op step is
    // enough to fire it and read back what the consumer saw.
    auto frame = atv_test::make_trivial_frame();
    sim.run(frame, /*n_steps=*/1);
    EXPECT_EQ(consumer.build_calls, 1);
    EXPECT_TRUE(consumer.dt_known_at_build)
        << "the pushed dt must be visible inside the consumer's build()";
}

/// Opt-in means opt-in: a consumer that declares it can run without a dt is
/// not a reason to demand an integrator protocol, and it is never handed one
/// it did not ask for when no provider exists.
TEST(TimestepProtocol, NonRequiredConsumerWithoutProviderIsFine) {
    atv_test::StubIntegrator integrator; // no provider
    atv_test::NoOpForce force;
    TimestepConsumerStub consumer{"stub_dt_consumer", /*required=*/false};
    ATV::cpu::md::MD sim;

    sim.on(ATV::md::Phase::initial_integrate).add_ref(integrator);
    sim.set_observer_ref(force);
    sim.set_observer_ref(consumer);

    EXPECT_NO_THROW(sim.build());
    EXPECT_EQ(consumer.set_calls, 0);
    EXPECT_FALSE(consumer.holds_usable_dt());
}

// ─── dt validity surfaces at build time, never as a stored bad value ────
//
// The throw may come from either layer — the push may gate the provider's
// value, or the consumer's `set_timestep` may reject it (the spec documents
// `std::invalid_argument` there). Both are build-time and both are correct;
// what must never happen is a consumer left holding an unusable dt, so that
// is what these two tests assert alongside the throw.

TEST(TimestepProtocol, NonPositiveProviderTimestepSurfacesAtBuild) {
    TimestepProviderStub zero{"stub_integrator_dt_zero", 0.0};
    atv_test::NoOpForce force;
    TimestepConsumerStub consumer{"stub_dt_consumer", /*required=*/true};
    ATV::cpu::md::MD sim;

    sim.set_observer_ref(zero);
    sim.set_observer_ref(force);
    sim.set_observer_ref(consumer);

    try {
        sim.build();
        FAIL() << "expected MD::build() to reject a published dt of 0 fs";
    } catch (const std::exception& e) {
        EXPECT_EQ(consumer.set_calls, 0) << e.what();
        EXPECT_FALSE(consumer.holds_usable_dt()) << e.what();
    }
}

TEST(TimestepProtocol, NonFiniteProviderTimestepSurfacesAtBuild) {
    TimestepProviderStub inf{"stub_integrator_dt_inf",
                             std::numeric_limits<double>::infinity()};
    atv_test::NoOpForce force;
    TimestepConsumerStub consumer{"stub_dt_consumer", /*required=*/true};
    ATV::cpu::md::MD sim;

    sim.set_observer_ref(inf);
    sim.set_observer_ref(force);
    sim.set_observer_ref(consumer);

    try {
        sim.build();
        FAIL() << "expected MD::build() to reject a non-finite published dt";
    } catch (const std::exception& e) {
        EXPECT_EQ(consumer.set_calls, 0) << e.what();
        EXPECT_FALSE(consumer.holds_usable_dt()) << e.what();
    }
}

TEST(TimestepConsumers, ConcreteOwnersUsePublishedTimestep) {
    using namespace ATV;
    auto check = [](auto make, ATV::md::Phase phase) {
        cpu::Frame frame(2);
        frame.box = cpu::Box::cubic(10);
        for (int i = 0; i < 2; ++i) {
            frame.span(kAtom_tag)[i] = i + 1;
            frame.span(kAtom_type)[i] = 7;
            frame.span(kAtom_mass)[i] = 39.9;
            frame.span(kAtom_charge)[i] = i ? -1 : 1;
            frame.span(kAtom_x)[i] = i + 1;
            frame.span(kAtom_vx)[i] = i ? -0.01 : 0.01;
        }
        frame.meta.get(kStress).fill(10);
        auto reference_frame = frame;
        auto old_frame = frame;
        auto consumer = make(0.5);
        auto reference = make(1.0);
        auto old = make(0.5);
        SCOPED_TRACE(consumer.name());
        ASSERT_NE(consumer.as_timestep_consumer(), nullptr);
        consumer.as_timestep_consumer()->set_timestep(1.0);
        EXPECT_THROW(consumer.as_timestep_consumer()->set_timestep(0),
                     std::invalid_argument);
        EXPECT_THROW(consumer.as_timestep_consumer()->set_timestep(
                         std::numeric_limits<double>::infinity()),
                     std::invalid_argument);
        cpu::Context context;
        // Direct owner calls need the same constraint-aware DOF publication
        // that Driver performs before component build (two atoms => 3 DOF).
        cpu::publish_global_dof(context, frame, 0);
        ASSERT_EQ(context.md.n_dof_global, 3);
        consumer.build({context, frame});
        reference.build({context, reference_frame});
        old.build({context, old_frame});
        consumer.launch({context, frame, slot_of(phase), 3});
        reference.launch({context, reference_frame, slot_of(phase), 3});
        old.launch({context, old_frame, slot_of(phase), 3});
        double changed = std::abs(frame.box.volume() - old_frame.box.volume());
        EXPECT_DOUBLE_EQ(frame.box.volume(), reference_frame.box.volume());
        for (auto column :
             {kAtom_x, kAtom_y, kAtom_z, kAtom_vx, kAtom_vy, kAtom_vz})
            for (int i = 0; i < 2; ++i) {
                EXPECT_DOUBLE_EQ(frame.span(column)[i],
                                 reference_frame.span(column)[i]);
                changed +=
                    std::abs(frame.span(column)[i] - old_frame.span(column)[i]);
            }
        for (auto column : {kAtom_fx, kAtom_fy, kAtom_fz})
            for (int i = 0; i < 2; ++i) {
                EXPECT_DOUBLE_EQ(frame.span(column)[i],
                                 reference_frame.span(column)[i]);
                changed +=
                    std::abs(frame.span(column)[i] - old_frame.span(column)[i]);
            }
        EXPECT_GT(changed, 1e-12);
        consumer.destroy({context, frame});
        reference.destroy({context, reference_frame});
        old.destroy({context, old_frame});
    };
    check([](double dt) { return cpu::md::Berendsen(300, 100, dt); },
          md::Phase::end_of_step);
    check([](double dt) { return cpu::md::NoseHoover(300, 100, dt); },
          md::Phase::initial_integrate);
    check([](double dt) { return cpu::md::Barostat(0, 100, dt); },
          md::Phase::end_of_step);
    check(
        [](double dt) {
            return cpu::md::Deform(cpu::md::Deform::Mode::erate, 0.01, dt);
        },
        md::Phase::end_of_step);
    check([](double dt) { return cpu::md::FixMove(0.1, 0.2, 0.3, 7, dt); },
          md::Phase::initial_integrate);
    check(
        [](double dt) {
            return cpu::md::FixOscillatingEfield(1, 2, 3, 10, dt);
        },
        md::Phase::post_force);
}

TEST(FixMoveIdentity, RowPermutationPreservesOriginsAndForceVeto) {
    using namespace ATV;
    cpu::Frame frame(2);
    frame.box = cpu::Box::cubic(10);
    frame.span(kAtom_tag)[0] = 11;
    frame.span(kAtom_tag)[1] = 22;
    frame.span(kAtom_type)[0] = frame.span(kAtom_type)[1] = 7;
    frame.span(kAtom_x)[0] = 2;
    frame.span(kAtom_x)[1] = 4;
    cpu::Context context;
    cpu::md::FixMove move(1, 0, 0, 7, 0.5);
    move.build({context, frame});
    const auto original = frame;
    frame.copy_row_from(original, 0, 1);
    frame.copy_row_from(original, 1, 0);
    move.launch({context, frame, slot_of(md::Phase::initial_integrate), 3});
    EXPECT_EQ(frame.span(kAtom_x)[0], 5.5);
    EXPECT_EQ(frame.span(kAtom_x)[1], 3.5);
    frame.span(kAtom_fx)[0] = frame.span(kAtom_fx)[1] = 99;
    move.launch(
        {context, frame, slot_of(md::Phase::post_force), 3, false, false});
    EXPECT_EQ(frame.span(kAtom_fx)[0], 99);
    EXPECT_EQ(frame.span(kAtom_fx)[1], 99);
    frame.span(kAtom_tag)[1] = 33;
    EXPECT_THROW(
        move.launch({context, frame, slot_of(md::Phase::initial_integrate), 4}),
        std::runtime_error);
    move.destroy({context, frame});
}

TEST(NoseHooverBoundary, ZeroPublishedDofPreservesVelocitiesAndChain) {
    using namespace ATV;
    cpu::Context context;
    cpu::Frame single(1);
    single.span(kAtom_mass)[0] = 39.9;
    single.span(kAtom_vx)[0] = 0.01;
    single.span(kAtom_vy)[0] = -0.02;
    cpu::publish_global_dof(context, single, 0);
    ASSERT_EQ(context.md.n_dof_global, 0);
    cpu::md::NoseHoover thermostat(300, 100, 1);
    thermostat.build({context, single});
    for (auto phase :
         {md::Phase::initial_integrate, md::Phase::final_integrate}) {
        thermostat.launch({context, single, slot_of(phase), 0});
        EXPECT_EQ(single.span(kAtom_vx)[0], 0.01);
        EXPECT_EQ(single.span(kAtom_vy)[0], -0.02);
        EXPECT_EQ(single.span(kAtom_vz)[0], 0);
    }

    // A later valid DOF publication must see an unpoisoned thermostat chain.
    cpu::Frame pair(2);
    for (int i = 0; i < 2; ++i) {
        pair.span(kAtom_mass)[i] = 39.9;
        pair.span(kAtom_vx)[i] = i ? -0.01 : 0.01;
    }
    auto reference_frame = pair;
    cpu::publish_global_dof(context, pair, 0);
    cpu::md::NoseHoover reference(300, 100, 1);
    reference.build({context, reference_frame});
    thermostat.launch(
        {context, pair, slot_of(md::Phase::initial_integrate), 1});
    reference.launch(
        {context, reference_frame, slot_of(md::Phase::initial_integrate), 1});
    for (int i = 0; i < 2; ++i) {
        EXPECT_TRUE(std::isfinite(pair.span(kAtom_vx)[i]));
        EXPECT_DOUBLE_EQ(pair.span(kAtom_vx)[i],
                         reference_frame.span(kAtom_vx)[i]);
    }
    EXPECT_GT(std::abs(pair.span(kAtom_vx)[0] - 0.01), 1e-12);
}

TEST(TimestepProtocol, SameIntegratorMayOwnBothIntegrationPhases) {
    TimestepProviderStub clock{"clock", kDt};
    TimestepConsumerStub consumer{"consumer", true};
    atv_test::NoOpForce force;
    ATV::cpu::md::MD sim;
    sim.on(ATV::md::Phase::initial_integrate).add_ref(clock);
    sim.on(ATV::md::Phase::final_integrate).add_ref(clock);
    sim.set_observer_ref(force);
    sim.set_observer_ref(consumer);
    EXPECT_NO_THROW(sim.build());
    EXPECT_EQ(consumer.set_calls, 1);
}

TEST(TimestepProtocol, LiveConsumerReceivesClockBeforeItsBuild) {
    atv_test::NoOpForce force;
    TimestepConsumerStub consumer{"live_consumer", true};
    ATV::cpu::md::MD sim;
    sim.set_integrator(ATV::cpu::md::VelocityVerlet{kDt});
    sim.set_observer_ref(force);
    auto frame = atv_test::make_trivial_frame();
    sim.run(frame, 0);
    sim.set_observer_ref(consumer);
    sim.run(frame, 0);
    EXPECT_EQ(consumer.build_calls, 1);
    EXPECT_TRUE(consumer.dt_known_at_build);
    EXPECT_DOUBLE_EQ(consumer.received_dt_fs, kDt);
}

TEST(TimestepProtocol, LiveSecondIntegratorIsRejectedBeforeDispatch) {
    TimestepProviderStub first{"first", kDt}, second{"second", kDt};
    atv_test::NoOpForce force;
    ATV::cpu::md::MD sim;
    sim.on(ATV::md::Phase::initial_integrate).add_ref(first);
    sim.set_observer_ref(force);
    auto frame = atv_test::make_trivial_frame();
    sim.run(frame, 0);
    sim.on(ATV::md::Phase::final_integrate).add_ref(second);
    EXPECT_THROW(sim.run(frame, 1), std::runtime_error);
}

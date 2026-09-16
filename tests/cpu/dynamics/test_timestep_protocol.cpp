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
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nve_limit.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"

#include "cpu/driver_test_helpers.h"

namespace {

constexpr double kDt = 0.5;      // [fs] — the nuclear timestep under test
constexpr double kOtherDt = 1.0; // [fs] — a DIFFERENT, conflicting one
constexpr double kXmax = 0.1;    // [Å]  — NVELimit displacement cap

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

/// Agreement is not a conflict: two providers on the same clock build fine,
/// and the agreed dt still reaches the consumer.
TEST(TimestepProtocol, AgreeingProvidersPushTheSharedTimestep) {
    TimestepProviderStub one{"stub_integrator_a", kDt};
    TimestepProviderStub two{"stub_integrator_b", kDt};
    atv_test::NoOpForce force;
    TimestepConsumerStub consumer{"stub_dt_consumer", /*required=*/true};
    ATV::cpu::md::MD sim;

    sim.on(ATV::md::Phase::initial_integrate).add_ref(one, 0);
    sim.on(ATV::md::Phase::initial_integrate).add_ref(two, 10);
    sim.set_observer_ref(force);
    sim.set_observer_ref(consumer);

    ASSERT_NO_THROW(sim.build());
    EXPECT_DOUBLE_EQ(consumer.received_dt_fs, kDt);
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

/// @file tests/cpu/driver/test_borrowed_lifetime.cpp
/// @brief A borrowed component destroyed before its driver is released by
///        the driver's teardown without being called into.

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"

#include "cpu/driver_test_helpers.h"

using atv_test::make_trivial_frame;
using atv_test::StubIntegrator;
using atv_test::StubProbeForce;

namespace {

/// Observer that records destroy() into a counter it does not own.
class DestroyCounter : public ATV::cpu::Component {
  public:
    explicit DestroyCounter(int& destroys) : destroys_(destroys) {}
    std::string name() const override { return "destroy_counter"; }
    void launch(const ATV::cpu::LaunchContext&) override {}
    void destroy(const ATV::cpu::DestroyContext&) override { ++destroys_; }

  private:
    int& destroys_;
};

} // namespace

TEST(BorrowedLifetime, LiveBorrowedComponentIsDestroyedOnce) {
    int destroys = 0;
    DestroyCounter counter(destroys);
    {
        ATV::cpu::md::MD md;
        StubIntegrator integ;
        StubProbeForce force;
        md.on(ATV::md::Phase::initial_integrate).add_ref(integ);
        md.on(ATV::md::Phase::force).add_ref(force);
        md.on(ATV::md::Phase::end_of_step).add_ref(counter);
        md.build();
        auto frame = make_trivial_frame();
        md.run(frame, 1);
        EXPECT_EQ(destroys, 0); // continuable run: destroy deferred to ~MD
    }
    EXPECT_EQ(destroys, 1);
}

TEST(BorrowedLifetime, ComponentDestroyedBeforeDriverIsNotCalledInto) {
    int destroys = 0;
    StubIntegrator integ;
    StubProbeForce force;
    auto frame = make_trivial_frame();
    {
        ATV::cpu::md::MD md;
        auto counter = std::make_unique<DestroyCounter>(destroys);
        md.on(ATV::md::Phase::initial_integrate).add_ref(integ);
        md.on(ATV::md::Phase::force).add_ref(force);
        md.on(ATV::md::Phase::end_of_step).add_ref(*counter);
        md.build();
        md.run(frame, 1);
        counter.reset(); // the borrowed object dies first
        // ~MD must not dereference it (destroy() or nested_node()).
    }
    EXPECT_EQ(destroys, 0);
}

TEST(BorrowedLifetime, CopiesCarryTheirOwnLifetime) {
    int destroys = 0;
    StubIntegrator integ;
    StubProbeForce force;
    auto frame = make_trivial_frame();
    DestroyCounter original(destroys);
    {
        ATV::cpu::md::MD md;
        auto copy = std::make_unique<DestroyCounter>(original);
        md.on(ATV::md::Phase::initial_integrate).add_ref(integ);
        md.on(ATV::md::Phase::force).add_ref(force);
        md.on(ATV::md::Phase::end_of_step).add_ref(original);
        md.build();
        md.run(frame, 1);
        copy.reset(); // destroying a copy must not mark the original dead
    }
    EXPECT_EQ(destroys, 1);
}

/// @file tests/cpu/communication/test_comm.cpp
/// @brief Unit tests for `cpu::Comm` / `cpu::comm::CommBrick` shape.
///
/// Interface only: abstract base, private constructor, Packer overload.
/// No MD.run. Transport goldens live in `test_comm_brick.cpp`.

#include <gtest/gtest.h>

#include <type_traits>

#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/communication/comm_brick.h"
#include "atomiverse/cpu/communication/domain_runtime.h"
#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"

using namespace ATV;

static_assert(std::is_abstract_v<cpu::Comm>,
              "cpu::Comm is the abstract communicator");
static_assert(std::is_base_of_v<cpu::Comm, cpu::comm::CommBrick>,
              "CommBrick implements Comm");
static_assert(
    !std::is_constructible_v<cpu::comm::CommBrick>,
    "CommBrick ctor is private; users obtain Comm* from DomainRuntime");
static_assert(!std::is_base_of_v<cpu::Component, cpu::comm::CommBrick>,
              "CommBrick is not a Component");

TEST(TestComm, PackerIsForwardPacker) {
    static_assert(std::is_same_v<cpu::Packer, cpu::ForwardPacker>,
                  "Packer is ForwardPacker");
    SUCCEED();
}

TEST(TestComm, DomainRuntimePublishesComm) {
    cpu::Context ctx;
    cpu::communication::DomainRuntime domain;
    domain.allocate(ctx);
    ASSERT_NE(ctx.md.comm, nullptr);
    EXPECT_EQ(ctx.md.comm, domain.comm());
    cpu::Packer* packer = nullptr;
    (void)packer;
}

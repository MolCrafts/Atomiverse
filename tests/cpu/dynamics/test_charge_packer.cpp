/// @file tests/cpu/dynamics/test_charge_packer.cpp
/// @brief ChargePacker ignores PBC. q = 0.25 e with pbc = +x packs 0.25 e.

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/charge_packer.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

TEST(TestChargePacker, IgnoresPbc) {
    cpu::Frame frame(1);
    frame.span(kAtom_charge)[0] = 0.25;
    frame.box = cpu::Box::cubic(10.0);
    cpu::ChargePacker packer{frame};
    EXPECT_EQ(packer.comm_forward(), 1);
    int list[] = {0};
    int pbc[] = {1, 0, 0};
    double buf[1] = {0.0};
    EXPECT_EQ(packer.pack_forward_comm(1, list, buf, 1, pbc), 1);
    EXPECT_DOUBLE_EQ(buf[0], 0.25);
}

TEST(TestChargePacker, UnpackWritesGhostCharge) {
    cpu::Frame frame(2);
    frame.span(kAtom_charge)[0] = 0.25;
    frame.span(kAtom_charge)[1] = 0.0;
    cpu::ChargePacker packer{frame};
    double buf[1] = {0.25};
    packer.unpack_forward_comm(1, 1, buf);
    EXPECT_DOUBLE_EQ(static_cast<double>(frame.span(kAtom_charge)[1]), 0.25);
}

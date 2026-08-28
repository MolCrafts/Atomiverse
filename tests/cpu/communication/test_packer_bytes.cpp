/// @file tests/cpu/communication/test_packer_bytes.cpp
/// @brief Extra-field packer payload size is n_atoms × nsize doubles.
///
/// `CommBrick::forward_comm(Packer&)` must Irecv that many bytes. nsize=1
/// (ChargePacker) hid a `n_recv` not `n_recv * nsize` post; RATTLE/SETTLE
/// use nsize=3.

#include <gtest/gtest.h>

#include "atomiverse/cpu/communication/atom_pack.h"

using ATV::cpu::comm::packer_bytes;

TEST(TestPackerBytes, ChargePackerIsOneDoublePerAtom) {
    EXPECT_EQ(packer_bytes(7, 1), 7u * sizeof(double));
}

TEST(TestPackerBytes, RattleSettleIsThreeDoublesPerAtom) {
    EXPECT_EQ(packer_bytes(5, 3), 15u * sizeof(double));
}

TEST(TestPackerBytes, EmptyReceiveIsZero) { EXPECT_EQ(packer_bytes(0, 3), 0u); }

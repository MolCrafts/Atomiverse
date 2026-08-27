/// @file tests/cpu/component/test_pack_forward_comm.cpp
/// @brief Unit tests for Component / ForwardPacker defaults and Comm
///        packer forward (serial no-op). No MD loop.

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/communication/domain_runtime.h"
#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/charge_packer.h"
#include "atomiverse/cpu/dynamics/fix_settle.h"

using namespace ATV;

namespace {

class StubComponent : public cpu::Component {
  public:
    std::string name() const override { return "stub"; }
    void launch(const cpu::LaunchContext&) override {}
};

class DefaultPacker : public cpu::ForwardPacker {};

class CountingPacker : public cpu::ForwardPacker {
  public:
    int n_pack = 0;
    int n_unpack = 0;
    int pack_forward_comm(int n, const int* list, double* buf, int pbc_flag,
                          const int* pbc) override {
        (void)n;
        (void)list;
        (void)buf;
        (void)pbc_flag;
        (void)pbc;
        ++n_pack;
        return 0;
    }
    void unpack_forward_comm(int n, int first, const double* buf) override {
        (void)n;
        (void)first;
        (void)buf;
        ++n_unpack;
    }
};

#define ATV_TEST_HAS_MEMBER(memb)                                              \
    template <typename T, typename = void>                                     \
    struct HasMember_##memb : std::false_type {};                              \
    template <typename T>                                                      \
    struct HasMember_##                                                        \
        memb<T, std::void_t<decltype(&T::memb)>> : std::true_type {}

ATV_TEST_HAS_MEMBER(pack_forward);
ATV_TEST_HAS_MEMBER(unpack_forward);
ATV_TEST_HAS_MEMBER(pack_forward_comm);
ATV_TEST_HAS_MEMBER(unpack_forward_comm);
ATV_TEST_HAS_MEMBER(comm_forward);

#undef ATV_TEST_HAS_MEMBER

} // namespace

TEST(TestPackForwardComm, PackerDefaultCommForwardIsZero) {
    DefaultPacker p;
    EXPECT_EQ(p.comm_forward(), 0);
}

TEST(TestPackForwardComm, PackerDefaultPackReturnsZero) {
    DefaultPacker p;
    int list[] = {0};
    int pbc[] = {1, 0, 0};
    double buf[3] = {1.0, 2.0, 3.0};
    EXPECT_EQ(p.pack_forward_comm(1, list, buf, 1, pbc), 0);
    EXPECT_DOUBLE_EQ(buf[0], 1.0);
}

TEST(TestPackForwardComm, PackerDefaultUnpackIsNoOp) {
    DefaultPacker p;
    double buf[3] = {9.0, 8.0, 7.0};
    p.unpack_forward_comm(1, 0, buf);
    EXPECT_DOUBLE_EQ(buf[0], 9.0);
}

TEST(TestPackForwardComm, OldPackForwardNameIsAbsent) {
    using CB = cpu::ForwardPacker;
    static_assert(!HasMember_pack_forward<CB>::value,
                  "pack_forward must stay deleted; the LAMMPS name is "
                  "pack_forward_comm");
    static_assert(!HasMember_unpack_forward<CB>::value,
                  "unpack_forward must stay deleted; the LAMMPS name is "
                  "unpack_forward_comm");
    static_assert(HasMember_pack_forward_comm<CB>::value,
                  "pack_forward_comm is the ForwardPacker hook");
    static_assert(HasMember_unpack_forward_comm<CB>::value,
                  "unpack_forward_comm is the ForwardPacker hook");
    static_assert(HasMember_comm_forward<CB>::value,
                  "comm_forward is the ForwardPacker size");
    SUCCEED();
}

TEST(TestPackForwardComm, ComponentIsNotAForwardPacker) {
    // comm-maint-01-deadcode: the forward path is self-initiated by the
    // plugin, so nothing scans for a packer and Component carries none of
    // its virtuals. Only the classes that actually pack name the protocol.
    static_assert(!std::is_base_of_v<cpu::ForwardPacker, cpu::Component>,
                  "Component must not carry the packer trio in its vtable");
    static_assert(std::is_base_of_v<cpu::ForwardPacker, cpu::md::FixSettle>,
                  "FixSettle names ForwardPacker itself");
    static_assert(std::is_base_of_v<cpu::ForwardPacker, cpu::ChargePacker>,
                  "ChargePacker is a standalone ForwardPacker");
    static_assert(std::is_same_v<cpu::Packer, cpu::ForwardPacker>,
                  "Comm::Packer is ForwardPacker");
}

TEST(TestPackForwardComm, SingleRankForwardCommIsNoOp) {
    cpu::Context ctx;
    cpu::communication::DomainRuntime domain;
    domain.allocate(ctx);
    ASSERT_NE(ctx.md.comm, nullptr);
    CountingPacker packer;
    ctx.md.comm->forward_comm(packer);
    EXPECT_EQ(packer.n_pack, 0);
    EXPECT_EQ(packer.n_unpack, 0);
}

/// @file tests/cpu/multiscale/test_qm_subsystem.cpp
/// @brief Unit: QmSubsystem mask / frame / engine (esmm-qmmm-03).
///
/// Multi-iter private-frame launch energy pins: regressions/multiscale/*
/// (e2e-in-tests wave-2).

#ifdef ATV_HAS_HF

#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/column.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/multiscale/internal/qm_subsystem.h"
#include "atomiverse/cpu/multiscale/qm_mask.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"

using namespace ATV;
using ATV::cpu::multiscale::internal::QmSubsystem;

namespace {

void insert_qm_z(cpu::Frame& f) {
    f.blocks.at("atoms").insert(
        "qm_z",
        cpu::ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone,
                        kDynamicFieldId});
}

cpu::Frame make_h2o_ar() {
    cpu::Frame f(4);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_z)[0] = 0.1173;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_y)[1] = 0.7572;
    f.span(kAtom_z)[1] = -0.4692;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_y)[2] = -0.7572;
    f.span(kAtom_z)[2] = -0.4692;
    f.span(kAtom_type)[3] = 2;
    f.span(kAtom_z)[3] = 3.0;
    f.span(kAtom_charge)[3] = 0.5;
    for (int i = 0; i < 4; ++i)
        f.span(kAtom_tag)[i] = i + 1;
    f.box = cpu::Box::cubic(20.0);
    insert_qm_z(f);
    auto qm = f.span(cpu::kAtomQmZ);
    qm[0] = 8;
    qm[1] = 1;
    qm[2] = 1;
    qm[3] = 0;
    return f;
}

} // namespace

TEST(QmSubsystem, MissingMaskThrowsHowTo) {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    QmSubsystem sub;
    cpu::scf::RHF rhf;
    sub.set_engine(std::move(rhf));
    EXPECT_THROW(
        {
            try {
                sub.build(f, 1);
            } catch (const std::runtime_error& e) {
                EXPECT_NE(std::string(e.what()).find("qm_z"), std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}

TEST(QmSubsystem, AllZeroMaskThrows) {
    auto f = make_h2o_ar();
    auto qm = f.span(cpu::kAtomQmZ);
    qm[0] = qm[1] = qm[2] = 0;
    QmSubsystem sub;
    cpu::scf::RHF rhf;
    sub.set_engine(std::move(rhf));
    EXPECT_THROW(sub.build(f, 1), std::runtime_error);
}

TEST(QmSubsystem, NonzeroQmChargeThrows) {
    auto f = make_h2o_ar();
    f.span(kAtom_charge)[0] = 0.1;
    QmSubsystem sub;
    cpu::scf::RHF rhf;
    sub.set_engine(std::move(rhf));
    EXPECT_THROW(sub.build(f, 1), std::runtime_error);
}

TEST(QmSubsystem, Am1EngineRejectedByName) {
    auto f = make_h2o_ar();
    QmSubsystem sub;
    cpu::semiempirical::AM1 am1;
    sub.set_engine(std::move(am1));
    EXPECT_THROW(
        {
            try {
                sub.build(f, 1);
            } catch (const std::runtime_error& e) {
                EXPECT_NE(std::string(e.what()).find("am1"), std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}

TEST(QmSubsystem, MpiSizeGreaterThanOneThrows) {
    auto f = make_h2o_ar();
    QmSubsystem sub;
    cpu::scf::RHF rhf;
    sub.set_engine(std::move(rhf));
    EXPECT_THROW(sub.build(f, 2), std::runtime_error);
}

TEST(QmSubsystem, PrivateFrameAllocatedOnBuild) {
    auto f = make_h2o_ar();
    QmSubsystem sub;
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    sub.set_engine(std::move(rhf));
    sub.embed_cutoff() = 20.0;
    sub.build(f, 1);
    ASSERT_NE(sub.private_frame(), nullptr);
    EXPECT_GE(sub.private_frame()->nlocal(), 1);
}

#else
TEST(QmSubsystem, SkippedWithoutHF) { GTEST_SKIP() << "ATV_HAS_HF off"; }
#endif

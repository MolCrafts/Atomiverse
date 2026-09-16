/// @file tests/cpu/multiscale/test_qmmm_mechanical.cpp
#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/column.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/multiscale/qm_mask.h"
#include "atomiverse/cpu/multiscale/qmmm_mechanical.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

void insert_qm_z(cpu::Frame& f) {
    f.blocks.at("atoms").insert(
        "qm_z",
        cpu::ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone,
                        kDynamicFieldId});
}

} // namespace

TEST(QmmmMechanical, RequiresEngine) {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = 0.74;
    f.box = cpu::Box::cubic(20);
    insert_qm_z(f);
    f.span(cpu::kAtomQmZ)[0] = 1;
    f.span(cpu::kAtomQmZ)[1] = 1;
    cpu::QmmmMechanical m{{{1, 1, 0.01, 3.4}}, 10.0};
    cpu::Context ctx;
    EXPECT_THROW(m.build({ctx, f}), std::runtime_error);
}

TEST(QmmmMechanical, H2PlusArWithDftb2) {
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_x)[1] = 0.74;
    f.span(kAtom_x)[2] = 5.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 18;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.span(kAtom_tag)[2] = 3;
    f.span(kAtom_mass)[0] = 1;
    f.span(kAtom_mass)[1] = 1;
    f.span(kAtom_mass)[2] = 40;
    f.box = cpu::Box::cubic(30);
    f.meta.get(kEnergy) = 0;
    insert_qm_z(f);
    f.span(cpu::kAtomQmZ)[0] = 1;
    f.span(cpu::kAtomQmZ)[1] = 1;
    f.span(cpu::kAtomQmZ)[2] = 0;

    cpu::Context ctx;
    cpu::QmmmMechanical qmmm{{{1, 1, 0.01, 3.4}, {18, 18, 0.01, 3.4}}, 10.0};
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = false;
    qmmm.set_engine(std::move(dftb));
    qmmm.build({ctx, f});
    cpu::LaunchContext lc{ctx, f, 0, 0};
    lc.forces_requested = false;
    qmmm.launch(lc);
    EXPECT_TRUE(std::isfinite(qmmm.last_qm_energy()));
    EXPECT_TRUE(std::isfinite(qmmm.last_mm_energy()));
    EXPECT_NEAR(f.meta.get(kEnergy),
                qmmm.last_qm_energy() + qmmm.last_mm_energy(), 1e-10);
}

TEST(QmmmMechanical, ClassicalCoulombClosedForm) {
    // H2 QM (q=0 on both) + MM site at 2 Å with q=-1; put q=1 on first H.
    // Classical coul only from QM charges × MM charge.
    cpu::Frame f(3);
    f.span(kAtom_x)[1] = 0.74;
    f.span(kAtom_x)[2] = 2.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 18;
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_charge)[1] = 0.0;
    f.span(kAtom_charge)[2] = -1.0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.span(kAtom_tag)[2] = 3;
    f.box = cpu::Box::cubic(30);
    insert_qm_z(f);
    f.span(cpu::kAtomQmZ)[0] = 1;
    f.span(cpu::kAtomQmZ)[1] = 1;
    f.span(cpu::kAtomQmZ)[2] = 0;

    cpu::QmmmMechanical q2{{{1, 1, 0.0, 3.4}, {18, 18, 0.0, 3.4}}, 10.0};
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = false;
    q2.set_engine(std::move(dftb));
    cpu::Context ctx;
    q2.build({ctx, f});
    cpu::LaunchContext lc{ctx, f, 0, 0};
    lc.forces_requested = false;
    f.meta.get(kEnergy) = 0;
    q2.launch(lc);
    // only H0–MM at r=2: ke * 1 * (-1) / 2
    const double want = units::coulomb * (1.0) * (-1.0) / 2.0;
    EXPECT_NEAR(q2.last_coul_energy(), want, 1e-12);
}

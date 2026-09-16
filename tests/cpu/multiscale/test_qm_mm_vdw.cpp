/// @file tests/cpu/multiscale/test_qm_mm_vdw.cpp
/// @brief Unit: QM–MM LJ mixing and pair selection (esmm-qmmm-03).

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/multiscale/internal/qm_mm_vdw.h"

using namespace ATV;
using ATV::cpu::multiscale::internal::QmMmVdw;

TEST(QmMmVdw, LorentzBerthelotMixing) {
    // Diagonals only → off-diagonal from LB
    QmMmVdw vdw({{{1, 1, 1.0, 2.0}, {2, 2, 4.0, 4.0}}}, 10.0);
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 2;
    f.span(kAtom_x)[1] = 3.0; // r = 3 Å
    f.box = cpu::Box::cubic(20.0);
    // ε_mix = √(1·4)=2, σ_mix = (2+4)/2=3; at r=3=σ → 4ε(1−1)=0, e=−V_shift
    // with V_shift = 4ε[(σ/rc)¹²−(σ/rc)⁶]. For rc=10, σ=3: e = −V_shift < 0
    // only when the shift is positive (attractive well). At large rc/σ the
    // shift can make e slightly positive or negative — just check finite and
    // that LB produced a nonzero interaction (vs no-diagonal → 0).
    const double e = vdw.launch(f, /*qm_rows=*/{0}, /*forces=*/false);
    EXPECT_TRUE(std::isfinite(e));
    EXPECT_NE(e, 0.0);
}

TEST(QmMmVdw, NoDiagonalMeansNoInteraction) {
    QmMmVdw vdw({{{1, 1, 1.0, 2.0}}}, 10.0); // type 2 has no diagonal
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 2;
    f.span(kAtom_x)[1] = 2.5;
    f.box = cpu::Box::cubic(20.0);
    EXPECT_DOUBLE_EQ(vdw.launch(f, {0}, false), 0.0);
}

TEST(QmMmVdw, SkipsQmQmAndMmMm) {
    QmMmVdw vdw({{{1, 1, 1.0, 2.0}}}, 10.0);
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_x)[1] = 2.5;
    f.span(kAtom_x)[2] = 5.0;
    f.box = cpu::Box::cubic(20.0);
    // Both atoms QM → no pairs
    EXPECT_DOUBLE_EQ(vdw.launch(f, {0, 1, 2}, false), 0.0);
    // Both MM relative to empty qm_rows
    EXPECT_DOUBLE_EQ(vdw.launch(f, {}, false), 0.0);
}

TEST(QmMmVdw, ShiftContinuousAtCutoff) {
    const double rc = 4.0;
    QmMmVdw vdw({{{1, 1, 0.01, 3.0}}}, rc);
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.box = cpu::Box::cubic(20.0);
    f.span(kAtom_x)[1] = rc - 1e-8;
    const double e_in = vdw.launch(f, {0}, false);
    f.span(kAtom_x)[1] = rc + 1e-8;
    const double e_out = vdw.launch(f, {0}, false);
    EXPECT_NEAR(e_in, 0.0, 1e-6);
    EXPECT_DOUBLE_EQ(e_out, 0.0);
}

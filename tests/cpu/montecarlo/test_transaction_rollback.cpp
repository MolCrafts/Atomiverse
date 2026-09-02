/// @file tests/cpu/mc/test_transaction_rollback.cpp
/// @brief ac-002 of mc-gcmc-npt: `mc::Transaction` is a Frame patch, and
///        rollback restores the frame EXACTLY (bitwise, field-by-field
///        against a pre-move copy) for every verb — a rejected displacement,
///        insertion (no orphan row), deletion (full resurrection, topology
///        included), and volume move (box AND coordinates).

#include <gtest/gtest.h>

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/montecarlo/transaction.h"

using namespace ATV;
using Verb = cpu::mc::Transaction::Verb;

namespace {

cpu::Frame make_frame(int n) {
    cpu::Frame frame(n);
    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    auto vx = frame.span(kAtom_vx);
    auto vy = frame.span(kAtom_vy);
    auto vz = frame.span(kAtom_vz);
    auto fx = frame.span(kAtom_fx);
    auto fy = frame.span(kAtom_fy);
    auto fz = frame.span(kAtom_fz);
    auto mass = frame.span(kAtom_mass);
    auto charge = frame.span(kAtom_charge);
    auto type = frame.span(kAtom_type);
    auto image = frame.span(kAtom_image);
    for (int i = 0; i < n; ++i) {
        x[i] = 0.11 + i;
        y[i] = 0.22 + i;
        z[i] = 0.33 + i;
        vx[i] = 1.0 + 0.1 * i;
        vy[i] = 2.0 + 0.1 * i;
        vz[i] = 3.0 + 0.1 * i;
        fx[i] = -1.0 - i;
        fy[i] = -2.0 - i;
        fz[i] = -3.0 - i;
        mass[i] = 39.948;
        charge[i] = 0.5 - 0.25 * i;
        type[i] = 1 + (i % 2);
        image(i, 0) = static_cast<std::int8_t>(i % 3 - 1);
    }
    // A bond and an angle so a deleted row carries real topology.
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, n);
    frame.add_bond(/*ti=*/0, /*tj=*/1, /*type=*/3);
    frame.add_angle(/*ti=*/0, /*tj=*/1, /*tk=*/2, /*type=*/4);
    frame.box.set_cell(Mat3::diagonal(10.0, 12.0, 14.0));
    frame.meta.get(ATV::kEnergy) = -1.25;
    return frame;
}

/// @brief Bitwise equality of every per-atom field and the box, rows
///        `[0, a.nlocal)`.
void expect_frames_identical(const cpu::Frame& f, const cpu::Frame& ref) {
    ASSERT_EQ(f.nlocal(), ref.nlocal());
    const auto ax = f.span(kAtom_x), bx = ref.span(kAtom_x);
    const auto ay = f.span(kAtom_y), by = ref.span(kAtom_y);
    const auto az = f.span(kAtom_z), bz = ref.span(kAtom_z);
    const auto avx = f.span(kAtom_vx), bvx = ref.span(kAtom_vx);
    const auto avy = f.span(kAtom_vy), bvy = ref.span(kAtom_vy);
    const auto avz = f.span(kAtom_vz), bvz = ref.span(kAtom_vz);
    const auto afx = f.span(kAtom_fx), bfx = ref.span(kAtom_fx);
    const auto afy = f.span(kAtom_fy), bfy = ref.span(kAtom_fy);
    const auto afz = f.span(kAtom_fz), bfz = ref.span(kAtom_fz);
    const auto amass = f.span(kAtom_mass), bmass = ref.span(kAtom_mass);
    const auto aq = f.span(kAtom_charge), bq = ref.span(kAtom_charge);
    const auto atype = f.span(kAtom_type), btype = ref.span(kAtom_type);
    const auto atag = f.span(kAtom_tag), btag = ref.span(kAtom_tag);
    const auto aimage = f.span(kAtom_image), bimage = ref.span(kAtom_image);
    const auto anbond = f.span(kAtom_num_bond);
    const auto bnbond = ref.span(kAtom_num_bond);
    const auto abond_atom = f.span(kAtom_bond_atom);
    const auto bbond_atom = ref.span(kAtom_bond_atom);
    const auto abond_type = f.span(kAtom_bond_type);
    const auto bbond_type = ref.span(kAtom_bond_type);
    const auto anangle = f.span(kAtom_num_angle);
    const auto bnangle = ref.span(kAtom_num_angle);
    const auto aangle_atom = f.span(kAtom_angle_atom);
    const auto bangle_atom = ref.span(kAtom_angle_atom);
    const auto aangle_type = f.span(kAtom_angle_type);
    const auto bangle_type = ref.span(kAtom_angle_type);
    const auto anspecial = f.span(kAtom_nspecial);
    const auto bnspecial = ref.span(kAtom_nspecial);
    const auto aspecial = f.span(kAtom_special);
    const auto bspecial = ref.span(kAtom_special);
    for (int i = 0; i < ref.nlocal(); ++i) {
        EXPECT_EQ(ax[i], bx[i]) << "row " << i;
        EXPECT_EQ(ay[i], by[i]);
        EXPECT_EQ(az[i], bz[i]);
        EXPECT_EQ(avx[i], bvx[i]);
        EXPECT_EQ(avy[i], bvy[i]);
        EXPECT_EQ(avz[i], bvz[i]);
        EXPECT_EQ(afx[i], bfx[i]);
        EXPECT_EQ(afy[i], bfy[i]);
        EXPECT_EQ(afz[i], bfz[i]);
        EXPECT_EQ(amass[i], bmass[i]);
        EXPECT_EQ(aq[i], bq[i]);
        EXPECT_EQ(atype[i], btype[i]);
        EXPECT_EQ(atag[i], btag[i]) << "row " << i;
        for (int d = 0; d < 3; ++d)
            EXPECT_EQ(aimage(i, d), bimage(i, d));
        EXPECT_EQ(anbond[i], bnbond[i]) << "row " << i;
        for (int m = 0; m < bnbond[i]; ++m) {
            EXPECT_EQ(abond_atom(i, m), bbond_atom(i, m));
            EXPECT_EQ(abond_type(i, m), bbond_type(i, m));
        }
        EXPECT_EQ(anangle[i], bnangle[i]);
        for (int m = 0; m < bnangle[i]; ++m) {
            EXPECT_EQ(aangle_atom(i, 2 * m + 0), bangle_atom(i, 2 * m + 0));
            EXPECT_EQ(aangle_atom(i, 2 * m + 1), bangle_atom(i, 2 * m + 1));
            EXPECT_EQ(aangle_type(i, m), bangle_type(i, m));
        }
        for (int d = 0; d < 3; ++d)
            EXPECT_EQ(anspecial(i, d), bnspecial(i, d));
        for (int s = 0; s < bnspecial(i, 2); ++s)
            EXPECT_EQ(aspecial(i, s), bspecial(i, s));
    }
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_EQ(f.box.cell()(r, c), ref.box.cell()(r, c));
}

} // namespace

TEST(TransactionRollback, RejectedDisplacementRestoresTheRow) {
    cpu::Frame frame = make_frame(4);
    const cpu::Frame before = frame;

    cpu::mc::Transaction t;
    t.begin(Verb::Displace, frame);
    t.record_touch(frame, 2);
    frame.span(ATV::kAtom_x)[2] += 0.7;
    frame.span(ATV::kAtom_y)[2] -= 0.4;
    frame.span(ATV::kAtom_z)[2] += 0.1;

    t.rollback(frame);
    expect_frames_identical(frame, before);
}

TEST(TransactionRollback, RejectedInsertionLeavesNoOrphanRow) {
    cpu::Frame frame = make_frame(3);
    const cpu::Frame before = frame;

    cpu::mc::Transaction t;
    t.begin(Verb::Insert, frame);
    const int row = frame.nlocal();
    if (row + 1 > frame.capacity())
        frame.grow(row + 1);
    frame.span(kAtom_x)[row] = 5.0;
    frame.span(kAtom_y)[row] = 6.0;
    frame.span(kAtom_z)[row] = 7.0;
    frame.span(kAtom_mass)[row] = 39.948;
    frame.span(kAtom_type)[row] = 1;
    frame.span(kAtom_tag)[row] = frame.acquire_tag();
    frame.set_nlocal(row + 1);
    ASSERT_EQ(frame.nlocal(), 4);

    t.rollback(frame);
    expect_frames_identical(frame, before);
}

TEST(TransactionRollback, RejectedDeletionResurrectsTheAtomIntact) {
    cpu::Frame frame = make_frame(5);
    const cpu::Frame before = frame;

    // GCMC deletion primitive: snapshot the victim, compact last into its
    // slot, shrink.
    const int victim = 1; // owns bond 0-1's partner entry via atom 0; carries
                          // angle topology as the center of 0-1-2
    cpu::mc::Transaction t;
    t.begin(Verb::Delete, frame);
    t.record_touch(frame, victim);
    frame.copy_row(victim, frame.nlocal() - 1);
    frame.set_nlocal(frame.nlocal() - 1);
    ASSERT_EQ(frame.nlocal(), 4);

    t.rollback(frame);
    expect_frames_identical(frame, before);
}

TEST(TransactionRollback, RejectedVolumeMoveRestoresBoxAndCoordinates) {
    cpu::Frame frame = make_frame(4);
    const cpu::Frame before = frame;

    cpu::mc::Transaction t;
    t.begin(Verb::Box, frame);
    t.record_box(frame);
    const double s = 1.03;
    for (int i = 0; i < frame.nlocal(); ++i) {
        t.record_touch(frame, i);
        frame.span(ATV::kAtom_x)[i] *= s;
        frame.span(ATV::kAtom_y)[i] *= s;
        frame.span(ATV::kAtom_z)[i] *= s;
    }
    frame.box.scale_isotropic(s);

    t.rollback(frame);
    // s*x/s != x in floating point — only a stored-row splice passes this.
    expect_frames_identical(frame, before);
}

/// A volume move records every owned row, so a per-row schema resolution
/// would be O(N) descriptor rebuilds per trial. The patch is a
/// `clone_structure` of the source and rows move by `copy_row_from`, which
/// walks the Block's own column map — so the whole operation resolves
/// NOTHING, at any atom count.
TEST(TransactionRollback, VolumePatchResolutionIsConstantInAtomCount) {
    cpu::Frame frame = make_frame(256);
    cpu::mc::Transaction t;

    frame.begin_resolution_window(1);
    t.begin(Verb::Box, frame);
    for (int i = 0; i < frame.nlocal(); ++i)
        t.record_touch(frame, i);
    EXPECT_EQ(frame.resolution_count(), 0u)
        << "recording a whole-frame patch must not resolve a column plan at "
           "all, let alone one per row";
    EXPECT_NO_THROW(frame.end_resolution_window("transaction_record_all"));

    // Grow the main frame: the atoms Block's structural generation changes,
    // so the patch is re-cloned at the operation boundary — and the row loop
    // still resolves nothing.
    frame.grow(frame.capacity() + 1);
    frame.begin_resolution_window(1);
    t.rollback(frame);
    EXPECT_EQ(frame.resolution_count(), 0u);
    EXPECT_NO_THROW(frame.end_resolution_window("transaction_rollback_all"));
}

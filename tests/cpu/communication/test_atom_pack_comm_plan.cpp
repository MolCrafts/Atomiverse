/// @file tests/cpu/communication/test_atom_pack_comm_plan.cpp
/// @brief Unit tests for `make_comm_plan` / `make_border_plan` / `comm_bytes`.
///
/// Spec `.claude/specs/lammps-comm-port.md` D2 (`comm_x_only`): the forward
/// comm plan is a membership correction, not a new switch. Canonical schema
/// (`cpu::Frame(n)`, n >= 1) yields exactly the coordinate trio; ghost v/q
/// stay on the border plan (full rebuild), matching LAMMPS `pack_comm` vs
/// `pack_border`.
///
/// Surface pinned (no MPI, no halo, no Driver):
/// @code{.cpp}
///   PackPlan make_comm_plan(const Frame& frame);
///   PackPlan make_border_plan(const Frame& frame);
///   std::size_t comm_bytes(const PackPlan& plan, int n);
/// @endcode
///
/// Categories: basics (column keys / wire_ids), domain golden (exactly 3
/// columns; 3 * sizeof(StorageReal) = 24 B at the default double storage
/// tier).
///
/// References for the hard-coded membership:
///   * LAMMPS `src/atom_vec.cpp` `AtomVec::pack_comm` (x/y/z only) vs
///     `pack_border` (v / mass / q / type / tag / image), baseline
///     `/nobackup/proj/disk/teoroo/personal/jicli594/work/lammps-src`
///     @ 78a7ddf (2026-08-25).
///   * A. P. Thompson et al., Comput. Phys. Commun. **271**, 108171 (2022),
///     DOI 10.1016/j.cpc.2021.108171 — `forward_comm` / `borders` payload.

#include <gtest/gtest.h>

#include <cstdint>
#include <string_view>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/communication/atom_pack.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

bool plan_has_key(const cpu::comm::PackPlan& plan, std::string_view key) {
    for (const cpu::ColumnRef& field : plan.fields) {
        if (field.key == key)
            return true;
    }
    return false;
}

} // namespace

// ── 1. basics / domain golden: comm-plan membership ───────────────────

/// Canonical atoms Block: `kColComm` is exactly x, y, z (ATV_ATOM_FIELDS
/// order). Count + ordered wire_id/key; a 7-column plan (the pre-port
/// vx/vy/vz/charge extras) must fail here.
TEST(TestCommPlanMembership, CommPlanIsExactlyXyz) {
    const cpu::Frame frame(1);
    const cpu::comm::PackPlan plan = cpu::comm::make_comm_plan(frame);

    ASSERT_EQ(plan.fields.size(), 3u);
    EXPECT_EQ(plan.fields[0].wire_id,
              static_cast<std::uint16_t>(AtomFieldId::x));
    EXPECT_EQ(plan.fields[1].wire_id,
              static_cast<std::uint16_t>(AtomFieldId::y));
    EXPECT_EQ(plan.fields[2].wire_id,
              static_cast<std::uint16_t>(AtomFieldId::z));
    EXPECT_EQ(plan.fields[0].key, "x");
    EXPECT_EQ(plan.fields[1].key, "y");
    EXPECT_EQ(plan.fields[2].key, "z");
}

// ── 2. domain golden: 24 B / ghost at default double storage ──────────

TEST(TestCommPlanMembership, CommBytesIsThreeStorageRealsPerRow) {
    static_assert(sizeof(StorageReal) == sizeof(double),
                  "default storage tier is double (24 B = 3 * 8 B)");
    EXPECT_EQ(sizeof(StorageReal), 8u);

    const cpu::Frame frame(1);
    const cpu::comm::PackPlan plan = cpu::comm::make_comm_plan(frame);
    EXPECT_EQ(plan.bytes(1), 3 * sizeof(StorageReal));
}

// ── 3. basics: ghost v/q moved to border, not deleted ─────────────────

/// `kColBorder` still ships the LAMMPS `pack_border` identity/state set.
/// Presence-only: do not compare against the comm plan (kColComm is xyz-only;
/// a superset check would not require v/q/mass/type/tag/image on border).
TEST(TestCommPlanMembership, BorderPlanStillCarriesGhostVelocityAndCharge) {
    const cpu::Frame frame(1);
    const cpu::comm::PackPlan plan = cpu::comm::make_border_plan(frame);

    for (const char* key :
         {"vx", "vy", "vz", "mass", "charge", "type", "tag", "image"}) {
        EXPECT_TRUE(plan_has_key(plan, key)) << "missing border key " << key;
    }
}

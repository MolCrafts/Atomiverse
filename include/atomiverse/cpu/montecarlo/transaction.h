/// @file cpu/montecarlo/transaction.h
/// @brief cpu::mc::Transaction — the Frame-patch changeset for Monte Carlo
///        trial moves.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace mc {

/// @brief Changeset for one Monte Carlo trial move, recorded as a **Frame
///        patch**: the touched atoms' ORIGINAL rows held in a small
///        `cpu::Frame` (plus the original `Box`), and the pre-trial `nlocal`.
///        Rollback splices the patch back into the main Frame.
///
/// This is deliberately NOT a set of parallel coordinate arrays (the user
/// directive: Frame is the only I/O currency — no bare `old_xyz`/`new_xyz`).
/// Holding whole rows is what makes every trial verb the same thing on
/// rejection — "here is a Frame to restore":
///
///   - **Displace** — patch = the moved atom's original row.
///   - **Delete**   — patch = the deleted atom's FULL row, topology and all
///                    (which a bare xyz array could never have carried). The
///                    compaction survivor that was copied into the victim's
///                    slot is simply overwritten back.
///   - **Insert**   — patch empty; restoring the recorded `nlocal` truncates
///                    the appended rows. (The tag allocator is monotonic and
///                    never rolled back — uniqueness needs no reuse.)
///   - **Box**      — patch carries the original `Box`; every rescaled row is
///                    recorded so coordinates restore bitwise.
///
/// Rows travel by `Frame::copy_row_from` — the cross-Frame sibling of the
/// row copy migration already uses to compact its owner array, carrying the
/// same `kColMigrate` set. The patch is a `clone_structure` of the source,
/// so a whole atom (topology included) moves as a per-column memcpy: `mc/`
/// owns no atom-row serializer, and it does not borrow the migration WIRE
/// either. Nothing here is communication, so nothing here compiles a
/// `comm::PackPlan`.
///
/// It is pure data + its own splice logic; it never reaches into the neighbor
/// list, and it is never a member of `cpu::Frame` (Frame-purity). The driver
/// owns it and publishes a non-owning pointer on `ctx.mc.proposal`; the Mover
/// records, the driver's `OutcomePolicy` decides, and whoever recorded — this
/// object — undoes.
class Transaction {
  public:
    /// @brief What kind of trial this proposal is. Read by grand-canonical /
    ///        isobaric acceptance policies, whose acceptance rule differs by
    ///        verb.
    enum class Verb { None, Displace, Insert, Delete, Box };

    /// @brief The trial verb recorded by the Mover for this proposal.
    Verb verb = Verb::None;

    /// @brief Construct an empty changeset (the driver owns it; a Mover
    ///        fills it each `propose` phase).
    Transaction() = default;

    /// @brief Start recording a proposal: clears the previous patch, stamps
    ///        the verb, and captures the pre-trial `nlocal` so rollback can
    ///        restore the population count for any verb.
    /// @param v The trial verb.
    /// @param f The main frame, BEFORE any mutation.
    void begin(Verb v, const cpu::Frame& f);

    /// @brief Record row `row`'s current (pre-mutation) content into the
    ///        patch. Call BEFORE mutating that row.
    /// @param f The main frame.
    /// @param row Owned row index to snapshot.
    void record_touch(const cpu::Frame& f, int row);

    /// @brief Record the current (pre-mutation) box into the patch. Call
    ///        BEFORE mutating the box.
    /// @param f The main frame.
    void record_box(const cpu::Frame& f);

    /// @brief Splice the patch back: restore `nlocal`, every touched row
    ///        (field-exact, topology included), and the box if recorded.
    /// @param f The main frame to restore.
    void rollback(cpu::Frame& f);

    /// @brief Number of touched (snapshotted) rows in the patch.
    std::size_t size() const noexcept { return touched_.size(); }

    /// @brief Main-frame row that patch row `k` restores into.
    LocalInt touched(std::size_t k) const { return touched_[k]; }

    /// @brief The patch itself: touched atoms' original rows (+ box).
    const cpu::Frame& patch() const noexcept { return patch_; }

    /// @brief Reset to an empty changeset, retaining backing storage so a
    ///        Mover reuses it across moves without per-move reallocation.
    void clear();

  private:
    /// Bring the patch Frame's structure back in step with the source and
    /// reserve enough patch rows for the announced operation, once at the
    /// transaction boundary. Per-row record/rollback loops are then plain
    /// column memcpys.
    void prepare_patch_(const cpu::Frame& source, int required_rows);

    cpu::Frame patch_{0};           ///< Original rows of touched atoms + box.
    std::vector<LocalInt> touched_; ///< Main-frame row of patch row k.
    int nlocal_before_ = -1;        ///< `nlocal` when `begin()` ran.
    bool box_recorded_ = false;     ///< Patch box is meaningful.
    const cpu::Frame* source_frame_ = nullptr;
    /// The atoms Block the patch was last cloned from, and its structural
    /// generation. A schema insert, a capacity change or a topology
    /// re-stride bumps that generation, and the patch is re-cloned — which
    /// is what keeps `copy_row_from`'s stride equality an invariant rather
    /// than a hope.
    const cpu::Block* cloned_from_ = nullptr;
    std::uint64_t cloned_generation_ = 0;
};

} // namespace mc
} // namespace cpu
} // namespace ATV

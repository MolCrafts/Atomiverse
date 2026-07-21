/// @file cpu/mc/insert_delete_move.h
/// @brief cpu::mc::InsertDeleteMove — grand-canonical insertion/deletion
///        Mover (single species).
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/component/mover.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cpu {
namespace mc {

/// @brief Monte Carlo trial move for μVT: with equal probability, either
///        **insert** one atom at a uniform position in the cell or **delete**
///        one uniformly chosen atom (Frenkel & Smit §5.6.1, single species).
///
/// Insertion appends a fully zeroed row at `nlocal` (velocity, force, charge,
/// image, topology counts — rows past `nlocal` may hold stale content from
/// earlier deletions), stamps the configured type/mass, and takes its tag
/// from `Frame::acquire_tag()` — never from identity seeding, which would
/// duplicate a survivor's tag after a delete-compaction. Deletion is the
/// migration compaction primitive: snapshot the victim into the Frame patch
/// (full row, topology included), `copy_row(victim, nlocal-1)`, `--nlocal`.
///
/// A deletion drawn on an empty frame stamps an empty (`Verb::None`)
/// proposal, which the policy rejects — the standard "trial deletion from
/// N = 0 is simply rejected".
///
/// Role `Mover`, capabilities `None`. Bound to `sampling::Phase::propose`.
class InsertDeleteMove : public cpu::Mover {
  public:
    /// @param type Atom type stamped on inserted atoms (must be > 0).
    /// @param mass_amu Mass [amu] stamped on inserted atoms (must be > 0;
    ///        use the same species mass the `GcmcMetropolis` Λ was built
    ///        from).
    /// @throws std::invalid_argument on a non-positive type or mass.
    InsertDeleteMove(int type, double mass_amu);

    /// @brief Component identifier.
    /// @return "insert_delete".
    std::string name() const override { return "insert_delete"; }

    /// @brief Structural role.
    /// @return `ComponentRole::Mover`.
    /* role via Mover base */

    /// @brief Default binding: the `propose` phase, order 0.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::sampling::Phase::propose, 0)};
    }

    /// @brief Propose one insertion or deletion (nothing on the bootstrap
    ///        pass).
    /// @param ctx Launch context; reads `runtime.mc.rng`, writes the frame
    ///        and `runtime.mc.proposal`.
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    int type_;        ///< Type stamped on inserted atoms.
    double mass_amu_; ///< Mass [amu] stamped on inserted atoms.
};

} // namespace mc
} // namespace cpu
} // namespace ATV

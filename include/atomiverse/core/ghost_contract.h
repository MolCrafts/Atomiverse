/// @file core/ghost_contract.h
/// @brief `GhostField` / `GhostContract` — what a component needs of the
///        ghost copies it uses, and the folded union of every such need.
///
/// **A data-lifetime contract, not a communication detail.** A component
/// states two orthogonal facts about each field it touches on a ghost row:
///
/// | Axis | Question | Constructor |
/// | --- | --- | --- |
/// | freshness (owner → ghost) | how current must this be when I read it? | `read_stable` / `read_each_step` |
/// | data flow (ghost → owner) | what do I tally that has to reach the owner? | `reduce_to_owner` |
///
/// Nothing here names a wire, a pass, or a timestep. The mapping from a
/// contract to the halo's creation / refresh / reverse passes lives once, in
/// the communication layer that owns those passes
/// (`cpu/communication/atom_pack.cpp`). `read_each_step` is a **freshness
/// level**, not a lifecycle event: a component that wants to *act* at a
/// point in the loop binds to a `Phase` instead.
///
/// **This is a Component-author surface, not a user surface.** Nobody
/// writing a simulation script constructs a `GhostField`; it is declared
/// once, `static constexpr`, alongside the style that reads the data — the
/// same side of the fence as `launch()` and `default_bindings()`.
///
/// **The per-component declaration is a compile-time constant.** A style
/// hands back a `std::span` over its own `static constexpr` table, so the
/// only work at `build()` is the union across mounted components — no
/// allocation per component, no builder object, and no way to hand back a
/// half-stated field, because the named constructors are the only way to
/// make one.
///
/// The relationship to `Frame` is **validation, not construction**. A Frame
/// arrives with whatever its source gave it (a data file, `Block::insert`,
/// another rank); nothing builds a Frame from a contract.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "atomiverse/core/frame_schema.h"

namespace ATV {

/// @brief Which direction a component moves a field across the owner/ghost
///        boundary.
enum class GhostAccess : std::uint8_t {
    /// @brief The component reads the owner's value on a ghost row.
    read,
    /// @brief The component tallies onto a ghost row; the sum belongs to the
    ///        owner.
    reduce,
};

/// @brief How current a ghost copy has to be when its reader runs.
///
/// A validity guarantee, not a schedule. `step` is ordered above `stable`:
/// when two components ask for the same field the stricter guarantee wins,
/// so the folded contract does not depend on declaration order.
enum class Freshness : std::uint8_t {
    /// @brief Constant for the life of a ghost incarnation — a fixed charge,
    ///        a type. One copy per rebuild is enough.
    stable,
    /// @brief Must be the owner's current value every time the reader runs —
    ///        a position, or a neighbour's velocity.
    step,
};

/// @brief One field a component needs on ghost rows, with its guarantee.
///
/// A literal type: declare the table `static constexpr` next to the style
/// that reads it. The named constructors are the only way to build one, so
/// "read this field" without saying how fresh, or "accumulate this" without
/// saying where it goes, are not expressible.
struct GhostField {
    /// @brief Column key in the atoms Block.
    std::string_view name;
    /// @brief Element type the declarer expects.
    ColDType dtype = ColDType::Storage;
    /// @brief Components per item; 1 for a scalar, 3 for a fixed triple.
    int cols = 1;
    /// @brief Direction across the owner/ghost boundary.
    GhostAccess access = GhostAccess::read;
    /// @brief Owner→ghost guarantee. Not consulted for a reduction.
    Freshness freshness = Freshness::stable;
    /// @brief Who asked for it. Turns "field missing" into "PairDpd needs
    ///        this and it is not there".
    std::string_view required_by;

    /// @brief Read a field that does not change while a ghost lives.
    /// @param name Column key.
    /// @param dtype Element type.
    /// @param cols Components per item.
    /// @param required_by Declaring component's name.
    /// @return The declaration.
    static constexpr GhostField read_stable(std::string_view name,
                                            ColDType dtype, int cols,
                                            std::string_view required_by) {
        return GhostField{
            name,       dtype, cols, GhostAccess::read, Freshness::stable,
            required_by};
    }
    /// @brief Read a field that does not change while a ghost lives.
    /// @param key Typed column key.
    /// @param required_by Declaring component's name.
    /// @return The declaration.
    template <typename T>
    static constexpr GhostField read_stable(const ColKey<T>& key,
                                            std::string_view required_by) {
        return read_stable(key.key, key.dtype, 1, required_by);
    }

    /// @brief Read a field that must be the owner's current value.
    /// @param name Column key.
    /// @param dtype Element type.
    /// @param cols Components per item.
    /// @param required_by Declaring component's name.
    /// @return The declaration.
    static constexpr GhostField read_each_step(std::string_view name,
                                               ColDType dtype, int cols,
                                               std::string_view required_by) {
        return GhostField{
            name, dtype, cols, GhostAccess::read, Freshness::step, required_by};
    }
    /// @brief Read a field that must be the owner's current value.
    /// @param key Typed column key.
    /// @param required_by Declaring component's name.
    /// @return The declaration.
    template <typename T>
    static constexpr GhostField read_each_step(const ColKey<T>& key,
                                               std::string_view required_by) {
        return read_each_step(key.key, key.dtype, 1, required_by);
    }

    /// @brief Tally onto ghost rows; the sum belongs to the owner.
    /// @param name Column key.
    /// @param dtype Element type.
    /// @param cols Components per item.
    /// @param required_by Declaring component's name.
    /// @return The declaration.
    static constexpr GhostField reduce_to_owner(std::string_view name,
                                                ColDType dtype, int cols,
                                                std::string_view required_by) {
        return GhostField{
            name,       dtype, cols, GhostAccess::reduce, Freshness::stable,
            required_by};
    }
    /// @brief Tally onto ghost rows; the sum belongs to the owner.
    /// @param key Typed column key.
    /// @param required_by Declaring component's name.
    /// @return The declaration.
    template <typename T>
    static constexpr GhostField reduce_to_owner(const ColKey<T>& key,
                                                std::string_view required_by) {
        return reduce_to_owner(key.key, key.dtype, 1, required_by);
    }
};

/// @brief One contradiction found while folding or validating.
struct SchemaError {
    /// @brief Field the problem is about.
    std::string_view field;
    /// @brief Component that declared it.
    std::string_view required_by;
    /// @brief What is wrong, phrased so it can be acted on.
    std::string reason;
};

/// @brief The union of every mounted component's ghost declaration.
///
/// The **only** runtime object between a component's constexpr table and the
/// compiled `PackPlan`. There is no intermediate per-wire bag: a pass asks
/// this contract which fields it carries, using the `access` / `freshness`
/// pair, and resolves them against the Frame in one step.
///
/// Declaration order is Plan order, identical on every rank, so two ranks
/// folding the same component set produce the same contract without sorting
/// and without a collective.
class GhostContract {
  public:
    /// @brief Fold one component's declaration into the union.
    ///
    /// Idempotent for an identical redeclaration — two styles needing
    /// `charge` is normal. Same field at different strengths merges to the
    /// **stricter** freshness whichever order they arrive in. A
    /// redeclaration at a different type, width or direction is a
    /// contradiction between two components and is reported by
    /// `conflicts()`; it is not silently merged.
    ///
    /// @param fields The component's `declare_ghosts()` table.
    void declare(std::span<const GhostField> fields) {
        for (const GhostField& f : fields)
            declare_one_(f);
    }

    /// @brief Declared fields, in declaration order.
    /// @return The folded union.
    const std::vector<GhostField>& fields() const noexcept { return fields_; }

    /// @brief Contradictions found while folding.
    /// @return One entry per disagreement.
    const std::vector<SchemaError>& conflicts() const noexcept {
        return conflicts_;
    }

    /// @brief The declaration for `name`, or null.
    /// @param name Column key.
    /// @return Pointer into the field list; invalidated by further declares.
    const GhostField* find(std::string_view name) const {
        auto it =
            std::find_if(fields_.begin(), fields_.end(),
                         [&](const GhostField& f) { return f.name == name; });
        return it == fields_.end() ? nullptr : &*it;
    }

    /// @brief Whether the contract carries `name`.
    /// @param name Column key.
    /// @return True when some component declared it.
    bool contains(std::string_view name) const { return find(name) != nullptr; }

    /// @brief Number of declared fields.
    std::size_t size() const noexcept { return fields_.size(); }
    /// @brief Whether nothing has been declared.
    bool empty() const noexcept { return fields_.empty(); }
    /// @brief Iteration over the folded union.
    auto begin() const noexcept { return fields_.begin(); }
    /// @brief Iteration over the folded union.
    auto end() const noexcept { return fields_.end(); }

    /// @brief Drop everything, keeping heap capacity.
    void clear() noexcept {
        fields_.clear();
        conflicts_.clear();
    }

    /// @brief Order-sensitive equality — two ranks must agree on the ORDER,
    ///        not merely the membership, or one packs what the other cannot
    ///        read.
    /// @param other Contract to compare with.
    /// @return True when both carry the same fields in the same order.
    bool operator==(const GhostContract& other) const {
        if (fields_.size() != other.fields_.size())
            return false;
        for (std::size_t i = 0; i < fields_.size(); ++i) {
            const GhostField& a = fields_[i];
            const GhostField& b = other.fields_[i];
            if (a.name != b.name || a.dtype != b.dtype || a.cols != b.cols
                || a.access != b.access || a.freshness != b.freshness)
                return false;
        }
        return true;
    }
    /// @brief Negation of `operator==`.
    /// @param other Contract to compare with.
    /// @return True when they differ.
    bool operator!=(const GhostContract& other) const {
        return !(*this == other);
    }

    /// @brief Human name of an element type, for messages.
    /// @param d Element type.
    /// @return Static name.
    static std::string_view dtype_name(ColDType d) {
        switch (d) {
        case ColDType::Storage:
            return "StorageReal";
        case ColDType::Acc:
            return "AccReal";
        case ColDType::Tag:
            return "TagInt";
        case ColDType::Int:
            return "int";
        case ColDType::Int8:
            return "int8";
        }
        return "?";
    }

  private:
    void declare_one_(const GhostField& f) {
        for (GhostField& seen : fields_) {
            if (seen.name != f.name)
                continue;
            if (seen.dtype != f.dtype || seen.cols != f.cols) {
                conflicts_.push_back(SchemaError{
                    f.name, f.required_by,
                    "declared as " + std::string(dtype_name(f.dtype)) + "["
                        + std::to_string(f.cols) + "] here but as "
                        + std::string(dtype_name(seen.dtype)) + "["
                        + std::to_string(seen.cols) + "] by "
                        + std::string(seen.required_by)
                        + "; two components disagree about what this field "
                          "is"});
                return;
            }
            if (seen.access != f.access) {
                conflicts_.push_back(SchemaError{
                    f.name, f.required_by,
                    "read by one component and accumulated by another; a "
                    "field that genuinely travels both ways needs two "
                    "columns, not one"});
                return;
            }
            // Strictest guarantee wins, in either arrival order.
            if (f.freshness == Freshness::step)
                seen.freshness = Freshness::step;
            return;
        }
        fields_.push_back(f);
    }

    std::vector<GhostField> fields_;
    std::vector<SchemaError> conflicts_;
};

/// @brief What the halo owes every run, whatever the components declare.
///
/// The same idea as LAMMPS' `AtomVec::default_comm` / `default_border` /
/// `default_reverse` (`src/atom_vec.cpp:36-40`, checkout `78a7ddf`): a few
/// fields are the engine's own product rather than any component's request.
/// Stated in the SAME vocabulary, as the SAME kind of constexpr table — there
/// is no privileged second way to put something on a wire.
///
/// Copyright (2003) Sandia Corporation. GPL-2.0-only.
namespace base_fields {

/// @brief The engine's own ghost declaration.
///
/// Positions are read fresh every step — producing current ghost positions is
/// what the halo is for. Tag (bonded resolution) and type (every pair style
/// indexes by it) do not change while a ghost lives. Forces are what the
/// reverse pass folds back. Folded before any component declares, so the
/// halo's own product keeps the leading wire positions whatever is mounted.
inline constexpr GhostField kEngine[] = {
    GhostField::read_each_step("x", ColDType::Storage, 1, "halo"),
    GhostField::read_each_step("y", ColDType::Storage, 1, "halo"),
    GhostField::read_each_step("z", ColDType::Storage, 1, "halo"),
    GhostField::read_stable("tag", ColDType::Tag, 1, "halo"),
    GhostField::read_stable("type", ColDType::Int, 1, "halo"),
    GhostField::reduce_to_owner("fx", ColDType::Acc, 1, "halo"),
    GhostField::reduce_to_owner("fy", ColDType::Acc, 1, "halo"),
    GhostField::reduce_to_owner("fz", ColDType::Acc, 1, "halo"),
};

/// @brief The engine's declaration, already folded.
///
/// For a caller with no component set — the pack/unpack tests, and a `Comm`
/// reached without a Driver fold.
/// @return Contract holding exactly `kEngine`.
inline GhostContract engine_contract() {
    GhostContract c;
    c.declare(kEngine);
    return c;
}

} // namespace base_fields
} // namespace ATV

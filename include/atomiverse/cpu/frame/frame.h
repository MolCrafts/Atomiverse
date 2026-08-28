/// @file cpu/frame/frame.h
/// @brief CPU Frame physical mapping: blocks + typed meta + box.
///
/// This is the hard-cutover representation.  There is no Atom aggregate,
/// SystemScalars mirror, `simbox` alias, or compatibility accessor.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "atomiverse/core/ghost_contract.h"
#include "atomiverse/cpu/frame/block.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/meta.h"

namespace ATV {
namespace cpu {

inline constexpr MetaKey<TagIndex> kTagIndex{"tag_to_row", false};

class Frame {
  public:
    explicit Frame(int n_atoms = 0);
    Frame(const Frame& other);
    Frame& operator=(const Frame& other);
    Frame(Frame&& other);
    Frame& operator=(Frame&& other);

    // The complete physical representation.  Backend-local Columns own all
    // per-row buffers; all bookkeeping and scalar results live in typed meta.
    BlockMap blocks;
    MetaMap meta;
    Box box;

    Block& block(std::string_view key);
    const Block& block(std::string_view key) const;

    template <typename T>
    Span<T> span(const ColKey<T>& key) {
        static_assert(!std::is_const_v<T>);
        return block(key.block).template span<T>(key);
    }

    template <typename T>
    Span<const T> span(const ColKey<T>& key) const {
        return block(key.block).span(key);
    }

    bool has(std::string_view block_key, std::string_view column_key) const;

    void begin_resolution_window(std::size_t budget) const noexcept;
    void end_resolution_window(const char* component) const;
    std::size_t resolution_count() const noexcept;
    ResolutionWindow resolution_window() const noexcept;
    void restore_resolution_window(ResolutionWindow window) const noexcept;

    int nlocal() const { return static_cast<int>(meta.get(kNLocal)); }
    int nghost() const { return static_cast<int>(meta.get(kNGhost)); }
    int capacity() const { return static_cast<int>(meta.get(kNMax)); }
    int n_total() const { return nlocal() + nghost(); }

    void set_nlocal(int n);
    void set_nghost(int n);

    /// LAMMPS-shaped chunked capacity growth over the canonical atoms Block.
    void grow(int target_capacity);
    void set_capacity(int new_capacity);
    void copy_row(int dst, int src);

    bool has_bonded() const { return meta.get(kHasBonded) != 0; }
    void ensure_bonded_storage();
    void grow_bond_per_atom(int target);
    void grow_angle_per_atom(int target);
    void grow_dihedral_per_atom(int target);
    void grow_improper_per_atom(int target);
    void grow_maxspecial(int target);
    void reserve_topology(int bond, int angle, int dihedral, int special_n);
    void reserve_topology(int bond, int angle, int dihedral, int improper,
                          int special_n);
    int group_capacity(ColGroup group) const;
    void grow_group(ColGroup group, int count);

    TagInt acquire_tag();
    void clear_tag_index();
    void release_tag_index();
    void rebuild_tag_index(int begin, int end);

    void add_bond(TagInt ti, TagInt tj, int type);
    void add_angle(TagInt ti, TagInt tj, TagInt tk, int type);
    void add_dihedral(TagInt ti, TagInt tj, TagInt tk, TagInt tl, int type);
    /// @brief Append one improper owned by atom `tj` (LAMMPS central atom).
    void add_improper(TagInt ti, TagInt tj, TagInt tk, TagInt tl, int type);
    void build_special_list();
    void reshuffle_topology_to_tags();

    std::size_t memory_bytes() const noexcept;

    /// Create an independent Frame with the same exact Block/Column and
    /// serializable-meta schema. Per-row values and backend-local counts/index
    /// are fresh; topology strides and box are preserved.
    Frame clone_structure(int rows) const;

  private:
    static int cols_for_(const FieldSpec& spec, const MetaMap& meta);
    void bind_resolution_state_() const noexcept;
    Block& atoms_block_() { return block("atoms"); }
    const Block& atoms_block_() const { return block("atoms"); }
};

/// @brief Validate a Frame against the folded ghost contract, pydantic
///        `model_validate` style: report EVERY problem, not the first.
///
/// The contract-to-Frame relationship is validation, not construction. A
/// Frame is a mapping and arrives with whatever its source gave it (a data
/// file, `Block::insert`, another rank); nothing builds one from a contract.
///
/// Reports a field that is absent, and a field that is present with the
/// wrong element type or width — the second is the case a bare name list
/// cannot see, and it is the one that corrupts a wire silently. Freshness
/// and direction are not checked here: they say what the engine must do with
/// the column, not what the Frame has to look like.
///
/// @param required Fields the mounted components declared.
/// @param frame Frame to check.
/// @return One entry per problem, each naming the field and its declarer.
///         Empty when the Frame satisfies the contract.
std::vector<SchemaError> validate_ghosts(const GhostContract& required,
                                         const Frame& frame);

/// @brief `validate_ghosts`, but throws a single message listing every
///        problem. The systematic form of CLAUDE.md's per-component rule:
///        a missing required field throws with a how-to, never defaults.
/// @param required Fields the mounted components declared.
/// @param frame Frame to check.
/// @param who Caller name, used in the message.
/// @throws std::runtime_error when the Frame does not satisfy the contract.
void check_ghosts(const GhostContract& required, const Frame& frame,
                  std::string_view who);

} // namespace cpu
} // namespace ATV

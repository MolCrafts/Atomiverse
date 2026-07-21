/// @file cpu/frame/frame.h
/// @brief CPU Frame physical mapping: blocks + typed meta + simbox.
///
/// This is the hard-cutover representation.  There is no Atom aggregate,
/// SystemScalars mirror, named box alias, or compatibility accessor.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

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
    Box simbox;

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
    /// are fresh; topology strides and simbox are preserved.
    Frame clone_structure(int rows) const;

  private:
    static int cols_for_(const FieldSpec& spec, const MetaMap& meta);
    void bind_resolution_state_() const noexcept;
    Block& atoms_block_() { return block("atoms"); }
    const Block& atoms_block_() const { return block("atoms"); }
};

} // namespace cpu
} // namespace ATV

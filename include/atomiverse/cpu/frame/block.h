/// @file cpu/frame/block.h
/// @brief Owning CPU Block: deterministic mapping from column key to Column.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "atomiverse/cpu/frame/column.h"

namespace ATV {
namespace cpu {

class Frame;

class Block {
  public:
    using Columns = std::map<std::string, Column, std::less<>>;

    explicit Block(int rows = 0) : rows_(rows) {}
    Block(const Block& other);
    Block& operator=(const Block& other);
    Block(Block&& other) noexcept;
    Block& operator=(Block&& other) noexcept;

    int rows() const noexcept { return rows_; }
    std::uint64_t generation() const noexcept { return generation_; }

    bool contains(std::string_view key) const noexcept {
        note_resolution_();
        return columns_.find(key) != columns_.end();
    }

    /// Insert one canonical or runtime column.  Runtime columns use the same
    /// Column representation and structural paths as schema columns.
    void insert(std::string key, ColumnSpec spec);
    void erase(std::string_view key);

    /// Reallocate every column to exactly new_rows, preserving the common
    /// prefix.  Every resolved Span/ColumnRef becomes stale.
    void set_rows(int new_rows);

    /// Change one column's row width, preserving each row prefix.
    void restride(std::string_view key, int new_cols);

    /// Copy all columns tagged kColMigrate from src to dst.
    void copy_row(int dst, int src);

    std::size_t memory_bytes() const noexcept;
    Block clone_structure(int rows) const;

    template <typename T>
    Span<T> span(const ColKey<T>& key) {
        static_assert(!std::is_const_v<T>);
        note_resolution_();
        using U = std::remove_const_t<T>;
        auto it = columns_.find(key.key);
        if (it == columns_.end())
            throw column_error("Frame block has no column '"
                               + std::string(key.key) + "'");
        Column& col = it->second;
        if (key.wire_id != kDynamicFieldId && col.wire_id() != key.wire_id)
            throw column_error("Frame column wire-id mismatch for '"
                               + std::string(key.key) + "'");
        if (col.dtype() != key.dtype || !dtype_matches_<U>(key.dtype))
            throw column_error("Frame column dtype mismatch for '"
                               + std::string(key.key) + "'");
        return Span<T>{static_cast<T*>(col.data()), col.cols(), &generation_,
                       generation_};
    }

    template <typename T>
    Span<const T> span(const ColKey<T>& key) const {
        note_resolution_();
        auto it = columns_.find(key.key);
        if (it == columns_.end())
            throw column_error("Frame block has no column '"
                               + std::string(key.key) + "'");
        const Column& col = it->second;
        if (key.wire_id != kDynamicFieldId && col.wire_id() != key.wire_id)
            throw column_error("Frame column wire-id mismatch for '"
                               + std::string(key.key) + "'");
        if (col.dtype() != key.dtype || !dtype_matches_<T>(key.dtype))
            throw column_error("Frame column dtype mismatch for '"
                               + std::string(key.key) + "'");
        return Span<const T>{static_cast<const T*>(col.data()), col.cols(),
                             &generation_, generation_};
    }

    /// Resolve a deterministic descriptor vector.  Allocation occurs while a
    /// structural plan is built/rebuilt, never per row.
    std::vector<ColumnRef> resolve_columns(unsigned required_flags = 0);
    std::vector<ColumnRef> resolve_columns(unsigned required_flags = 0) const;

    /// @brief Resolve exactly the fields a `Schema` names, in the Schema's
    ///        declaration order.
    ///
    /// The flag overloads above answer "which columns claim to belong on
    /// wire X"; this answers "which columns did the mounted components ask
    /// for". The second question is the one with an answer that depends on
    /// the run, which is why the wire is built from it.
    ///
    /// Declaration order is the wire order, and it is the same on every rank
    /// because it is Plan order. `check_schema` has already run at build(),
    /// so a named field being absent here is an engine bug, not user error,
    /// and throws as such.
    /// @param schema Fields to resolve.
    /// @return One ColumnRef per declared field, in declaration order.
    /// @throws column_error if a declared field is not in this Block.
    std::vector<ColumnRef> resolve_columns(const Schema& schema);
    std::vector<ColumnRef> resolve_columns(const Schema& schema) const;

    const Columns& columns() const noexcept { return columns_; }

  private:
    friend class Frame;

    template <typename T>
    static bool dtype_matches_(ColDType dtype) noexcept {
        // Precision/ID roles may share one C++ representation in a configured
        // build (StorageReal == AccReal in double mode; TagInt == int in
        // smallbig mode).  The key's explicit semantic dtype above
        // disambiguates the role; this check only proves that T is the native
        // representation selected for that role.
        using U = std::remove_const_t<T>;
        switch (dtype) {
        case ColDType::Storage:
            return std::is_same_v<U, StorageReal>;
        case ColDType::Acc:
            return std::is_same_v<U, AccReal>;
        case ColDType::Tag:
            return std::is_same_v<U, TagInt>;
        case ColDType::Int:
            return std::is_same_v<U, int>;
        case ColDType::Int8:
            return std::is_same_v<U, std::int8_t>;
        }
        return false;
    }

    void bump_generation_() noexcept { ++generation_; }
    void note_resolution_() const noexcept { resolution_state_->note(); }
    void bind_resolution_state_(
        const std::shared_ptr<ResolutionState>& state) const noexcept {
        resolution_state_ = state;
    }

    Columns columns_;
    int rows_ = 0;
    std::uint64_t generation_ = 1;
    mutable std::shared_ptr<ResolutionState> resolution_state_ =
        std::make_shared<ResolutionState>();
};

using BlockMap = std::map<std::string, Block, std::less<>>;

} // namespace cpu
} // namespace ATV

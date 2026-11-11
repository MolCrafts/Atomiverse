/// @file cpu/frame/input_snapshot.h
/// @brief Exact host-input checkpoint shared by CPU and CUDA run boundaries.
#pragma once

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "atomiverse/core/input_change.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV::cpu {

/// Stores selected physical input columns, never a complete Frame or results.
/// Capture only after a successful run/write-back; compare before continuation.
/// CPU selects state columns; CUDA selects only fields its transfer supports.
/// Predicate signature: bool(std::string_view block, const ColumnRef& column).
class FrameInputSnapshot {
  public:
    static bool state_input(std::string_view block, const ColumnRef& column) {
        return (column.flags & kColMigrate)
               && !(block == "atoms"
                    && (column.key == "fx" || column.key == "fy"
                        || column.key == "fz"));
    }

    template <class Select>
    void capture(const Frame& frame, Select select) {
        valid_ = false;
        nlocal_ = frame.nlocal();
        bonded_ = frame.has_bonded();
        for (std::size_t i = 0; i < topology_keys_.size(); ++i) {
            topology_present_[i] = frame.meta.contains(topology_keys_[i].key);
            topology_[i] =
                topology_present_[i] ? frame.meta.get(topology_keys_[i]) : 0;
        }
        topology_selected_.fill(false);
        std::size_t index = 0;
        for (const auto& [block_name, block] : frame.blocks) {
            for (const auto& column : block.resolve_columns()) {
                if (!select(block_name, column))
                    continue;
                if (block_name == "atoms")
                    for (std::size_t i = 0; i < topology_columns_.size(); ++i)
                        topology_selected_[i] =
                            topology_selected_[i]
                            || column.key == topology_columns_[i];
                if (index == columns_.size())
                    columns_.emplace_back();
                auto& saved = columns_[index++];
                saved.block = block_name;
                saved.key = column.key;
                saved.dtype = column.dtype;
                saved.cols = column.cols;
                const int rows =
                    block_name == "atoms" ? frame.nlocal() : block.rows();
                saved.bytes.resize(static_cast<std::size_t>(rows)
                                   * column.row_bytes());
                if (!saved.bytes.empty())
                    std::memcpy(saved.bytes.data(), column.bytes(),
                                saved.bytes.size());
            }
        }
        columns_.resize(index);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cell_[3 * i + j] = frame.box.cell()(i, j);
        valid_ = true;
    }

    template <class Select>
    InputChange changes(const Frame& frame, Select select) const {
        if (!valid_)
            return InputChange::None;
        InputChange result = InputChange::None;
        if (frame.nlocal() != nlocal_ || frame.has_bonded() != bonded_)
            result |= InputChange::Identity;
        for (std::size_t i = 0; i < topology_keys_.size(); ++i) {
            const bool present = frame.meta.contains(topology_keys_[i].key);
            if (topology_selected_[i]
                && (present != topology_present_[i]
                    || (present
                        && frame.meta.get(topology_keys_[i]) != topology_[i])))
                result |= InputChange::Identity;
        }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (cell_[3 * i + j] != frame.box.cell()(i, j))
                    result |= InputChange::Cell;
        std::size_t index = 0;
        for (const auto& [block_name, block] : frame.blocks) {
            for (const auto& column : block.resolve_columns()) {
                if (!select(block_name, column))
                    continue;
                if (index == columns_.size())
                    return result | InputChange::Identity;
                const auto& saved = columns_[index++];
                const int rows =
                    block_name == "atoms" ? frame.nlocal() : block.rows();
                const auto bytes =
                    static_cast<std::size_t>(rows) * column.row_bytes();
                if (saved.block != block_name || saved.key != column.key
                    || saved.dtype != column.dtype || saved.cols != column.cols
                    || saved.bytes.size() != bytes) {
                    result |= InputChange::Identity;
                } else if (bytes
                           && std::memcmp(saved.bytes.data(), column.bytes(),
                                          bytes)) {
                    result |= kind_(block_name, column.key);
                }
            }
        }
        if (index != columns_.size())
            result |= InputChange::Identity;
        return result;
    }

    void clear() noexcept { valid_ = false; }

  private:
    static InputChange kind_(std::string_view block, std::string_view key) {
        if (block == "atoms") {
            if (key == "x" || key == "y" || key == "z" || key == "image")
                return InputChange::Geometry;
            if (key == "vx" || key == "vy" || key == "vz")
                return InputChange::Velocity;
            if (key == "charge")
                return InputChange::Charge;
        }
        return InputChange::Identity;
    }

    struct ColumnInput {
        std::string block, key;
        ColDType dtype{};
        int cols = 0;
        std::vector<std::byte> bytes;
    };
    static constexpr std::array topology_keys_{kBondPerAtom, kAnglePerAtom,
                                               kDihedralPerAtom,
                                               kImproperPerAtom, kMaxSpecial};
    static constexpr std::array<std::string_view, 5> topology_columns_{
        "bond_atom", "angle_atom", "dihedral_atom", "improper_atom", "special"};
    std::array<bool, topology_keys_.size()> topology_selected_{};
    std::array<std::int64_t, topology_keys_.size()> topology_{};
    std::array<bool, topology_keys_.size()> topology_present_{};
    std::vector<ColumnInput> columns_;
    std::array<AccReal, 9> cell_{};
    int nlocal_ = 0;
    bool bonded_ = false, valid_ = false;
};

} // namespace ATV::cpu

/// @file cpu/frame/column.h
/// @brief Owning CPU Column plus generation-aware typed resolved spans.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/frame_schema.h"

namespace ATV {
namespace cpu {

inline constexpr std::size_t kResolveBudgetPerLaunch =
    static_cast<std::size_t>(kAtomFieldCount) * 2;

struct ResolutionWindow {
    std::size_t count = 0;
    std::size_t budget = kResolveBudgetPerLaunch;
    bool active = false;
};

/// Debug/performance instrumentation shared by MetaMap and all Blocks in one
/// Frame. It is bookkeeping owned by the typed-meta pillar, never a fourth
/// physical Frame field.
class ResolutionState {
  public:
    void note() noexcept {
        if (active_.load(std::memory_order_relaxed))
            count_.fetch_add(1, std::memory_order_relaxed);
    }

    void begin(std::size_t budget) noexcept {
        count_.store(0, std::memory_order_relaxed);
        budget_.store(budget, std::memory_order_relaxed);
        active_.store(true, std::memory_order_relaxed);
    }

    ResolutionWindow snapshot() const noexcept {
        return ResolutionWindow{count_.load(std::memory_order_relaxed),
                                budget_.load(std::memory_order_relaxed),
                                active_.load(std::memory_order_relaxed)};
    }

    void restore(ResolutionWindow window) noexcept {
        count_.store(window.count, std::memory_order_relaxed);
        budget_.store(window.budget, std::memory_order_relaxed);
        active_.store(window.active, std::memory_order_relaxed);
    }

    void deactivate() noexcept {
        active_.store(false, std::memory_order_relaxed);
    }

  private:
    std::atomic<std::size_t> count_{0};
    std::atomic<std::size_t> budget_{kResolveBudgetPerLaunch};
    std::atomic<bool> active_{false};
};

class stale_handle_error : public std::logic_error {
  public:
    using std::logic_error::logic_error;
};

class column_error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

/// A resolved contiguous column view.  Extracting/indexing validates the
/// Block generation captured at resolution.  Hot loops should call data()
/// once at launch entry and iterate the returned pointer; no check occurs per
/// particle after that explicit boundary.
template <typename T>
class Span {
  public:
    using element_type = T;

    Span() = default;
    Span(T* ptr, int cols, const std::uint64_t* generation,
         std::uint64_t captured) noexcept
        : ptr_(ptr), cols_(cols), generation_(generation), captured_(captured) {
    }

    void validate() const {
        if (generation_ != nullptr && *generation_ != captured_)
            throw stale_handle_error(
                "resolved Frame span is stale after a structural mutation");
    }

    bool stale() const noexcept {
        return generation_ != nullptr && *generation_ != captured_;
    }

    T* data() const {
        validate();
        return ptr_;
    }

    int cols() const noexcept { return cols_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }
    operator T*() const { return data(); }

    T& operator[](std::size_t i) const {
        validate();
        return ptr_[i];
    }

    T& operator()(int row, int col) const {
        validate();
        return ptr_[static_cast<std::size_t>(row) * cols_
                    + static_cast<std::size_t>(col)];
    }

  private:
    T* ptr_ = nullptr;
    int cols_ = 0;
    const std::uint64_t* generation_ = nullptr;
    std::uint64_t captured_ = 0;
};

struct ColumnSpec {
    ColDType dtype = ColDType::Storage;
    int cols = 1;
    ColGroup group = ColGroup::Core;
    int entries_per_item = 0;
    unsigned flags = kColNone;
    std::uint16_t wire_id = kDynamicFieldId;
};

/// Native, type-erased resolved descriptor used to compile structural plans.
/// It never widens through double.  read_data and write_data point at the same
/// native contiguous storage; write_data is null for a const resolution.
struct ColumnRef {
    std::string_view key;
    ColDType dtype = ColDType::Storage;
    ColGroup group = ColGroup::Core;
    int entries_per_item = 0;
    unsigned flags = kColNone;
    std::uint16_t wire_id = kDynamicFieldId;
    const void* read_data = nullptr;
    void* write_data = nullptr;
    int cols = 0;
    std::size_t element_size = 0;
    bool writable = false;
    const std::uint64_t* generation = nullptr;
    std::uint64_t captured_generation = 0;

    void validate() const {
        if (generation != nullptr && *generation != captured_generation)
            throw stale_handle_error("resolved Frame column plan is stale "
                                     "after a structural mutation");
    }

    const std::byte* bytes() const {
        validate();
        return static_cast<const std::byte*>(read_data);
    }

    std::byte* mutable_bytes() const {
        validate();
        if (!writable)
            throw column_error("attempt to write through a const ColumnRef");
        return static_cast<std::byte*>(write_data);
    }

    std::size_t row_bytes() const noexcept {
        return static_cast<std::size_t>(cols) * element_size;
    }
};

template <typename T, ColDType DType>
struct ColumnBuffer {
    using value_type = T;
    static constexpr ColDType dtype = DType;
    xt::xtensor<T, 2> values;
};

/// One owning CPU column.  All columns are rank-2 `[capacity, cols]`; a scalar
/// column simply has `cols == 1`.  That uniform layout makes structural
/// traversal independent of a named C++ member or xtensor rank.
class Column {
  public:
    using Storage = std::variant<ColumnBuffer<StorageReal, ColDType::Storage>,
                                 ColumnBuffer<AccReal, ColDType::Acc>,
                                 ColumnBuffer<TagInt, ColDType::Tag>,
                                 ColumnBuffer<int, ColDType::Int>,
                                 ColumnBuffer<std::int8_t, ColDType::Int8>>;

    Column() = default;
    Column(ColumnSpec spec, int rows);

    ColDType dtype() const noexcept { return spec_.dtype; }
    ColGroup group() const noexcept { return spec_.group; }
    int cols() const noexcept { return spec_.cols; }
    int entries_per_item() const noexcept { return spec_.entries_per_item; }
    unsigned flags() const noexcept { return spec_.flags; }
    std::uint16_t wire_id() const noexcept { return spec_.wire_id; }
    ColumnSpec spec() const noexcept { return spec_; }
    std::size_t element_size() const noexcept {
        return col_dtype_size(spec_.dtype);
    }
    std::size_t memory_bytes() const noexcept;

  private:
    friend class Block;

    void resize_rows(int old_rows, int new_rows);
    void restride(int rows, int new_cols);
    void copy_row(int dst, int src);
    void copy_row_from(const Column& source, int dst, int src);
    const void* data() const noexcept;
    void* data() noexcept;

    ColumnSpec spec_{};
    Storage storage_{};
};

} // namespace cpu
} // namespace ATV

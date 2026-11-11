/// @file src/cpu/communication/atom_pack.cpp
/// @brief Native-dtype schema-compiled Frame row packing.
///
/// The forward and border gather/scatter follow LAMMPS AtomVec
/// (src/atom_vec.cpp, e8c4c278a8), adapted to Atomiverse's column storage.
/// Copyright (2003) Sandia Corporation. Under the terms of Contract
/// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
/// certain rights in this software. Distributed under the GNU General
/// Public License (GPL-2.0-only).

#include "atomiverse/cpu/communication/atom_pack.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ATV {
namespace cpu {
namespace comm {

namespace {

constexpr std::array<ColGroup, 5> kGroups{
    ColGroup::Bond, ColGroup::Angle, ColGroup::Dihedral, ColGroup::Improper,
    ColGroup::Special};

class Writer {
  public:
    Writer(std::byte* data, std::size_t capacity)
        : data_(data), capacity_(capacity) {}

    void append(const void* src, std::size_t bytes) {
        if (bytes > capacity_ - size_)
            throw std::length_error("Frame pack output buffer is too small");
        if (bytes != 0)
            std::memcpy(data_ + size_, src, bytes);
        size_ += bytes;
    }

    std::size_t size() const noexcept { return size_; }

  private:
    std::byte* data_;
    std::size_t capacity_;
    std::size_t size_ = 0;
};

class Reader {
  public:
    Reader(const std::byte* data, std::size_t size)
        : data_(data), size_(size) {}

    const std::byte* take(std::size_t bytes) {
        if (bytes > size_ - offset_)
            throw std::length_error("Frame pack input payload is truncated");
        const std::byte* out = data_ + offset_;
        offset_ += bytes;
        return out;
    }

    void skip(std::size_t bytes) { (void)take(bytes); }
    std::size_t offset() const noexcept { return offset_; }

  private:
    const std::byte* data_;
    std::size_t size_;
    std::size_t offset_ = 0;
};

std::size_t checked_mul(std::size_t a, std::size_t b) {
    return comm::checked_mul(a, b, "Frame pack byte count");
}

std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("Frame pack byte count overflow");
    return a + b;
}

void hash_bytes(std::uint64_t& hash, const void* data, std::size_t n) {
    constexpr std::uint64_t prime = 1099511628211ull;
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        hash ^= bytes[i];
        hash *= prime;
    }
}

template <typename T>
void hash_value(std::uint64_t& hash, const T& value) {
    hash_bytes(hash, &value, sizeof(value));
}

void hash_fields(std::uint64_t& hash, const std::vector<ColumnRef>& fields) {
    if (fields.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("Frame pack schema has too many columns");
    const std::uint32_t count = static_cast<std::uint32_t>(fields.size());
    hash_value(hash, count);
    for (const ColumnRef& field : fields) {
        hash_bytes(hash, field.key.data(), field.key.size());
        const unsigned char nul = 0;
        hash_value(hash, nul);
        hash_value(hash, field.wire_id);
        hash_value(hash, field.dtype);
        hash_value(hash, field.element_size);
        hash_value(hash, field.group);
        hash_value(hash, field.entries_per_item);
        hash_value(hash, field.flags);
        // Count and fixed/core columns transmit the full row.  Variable
        // topology columns transmit count*entries_per_item, so their local
        // reserve stride is intentionally not part of rank agreement.
        if (field.group == ColGroup::Core || (field.flags & kColCount))
            hash_value(hash, field.cols);
    }
}

std::uint64_t fingerprint(PackKind kind, const std::vector<ColumnRef>& fields) {
    std::uint64_t hash = 1469598103934665603ull;
    hash_value(hash, kind);
    hash_fields(hash, fields);
    return hash;
}

/// @brief Does this declaration ride the pass `kind` drives?
///
/// The compile step, in one function. Reading a field means the ghost
/// carries it from the moment it exists, so every read is on the border
/// pass; only a `step` guarantee also needs re-sending each step; a
/// reduction travels the other way. Exchange is not contract-driven.
///
/// @param field One folded declaration.
/// @param kind Pass being compiled.
/// @return True when the pass carries this field.
bool on_pass(const GhostField& field, PackKind kind) {
    switch (kind) {
    case PackKind::Border:
        return field.access == GhostAccess::read;
    case PackKind::Comm:
        return field.access == GhostAccess::read
               && field.freshness == Freshness::step;
    case PackKind::Reverse:
        return field.access == GhostAccess::reduce;
    case PackKind::Exchange:
        break;
    }
    throw std::invalid_argument(
        "the exchange wire is flag-driven; it is not compiled from the "
        "ghost contract");
}

/// @brief Compile a plan for the fields the contract puts on this pass.
///
/// The wire is what the mounted components asked for, resolved in
/// declaration order. Only the exchange wire still selects by flag, because
/// migration carries what an atom IS rather than what some consumer reads —
/// that is a property of the Frame, not of the component set.
template <typename FrameT>
PackPlan make_plan(FrameT& frame, PackKind kind,
                   const GhostContract& contract) {
    auto& block = frame.block("atoms");
    PackPlan plan;
    plan.kind = kind;
    plan.frame_generation = frame.meta.generation();
    plan.block_generation = block.generation();
    plan.frame_owner = &frame;
    plan.owner = &block;
    for (const GhostField& field : contract)
        if (on_pass(field, kind))
            plan.fields.push_back(block.resolve_column(field));
    if (plan.fields.empty())
        throw column_error(
            std::string("the ") + pack_kind_name(kind)
            + " schema named no fields for this wire; a wire with nothing on "
              "it is a build-time mistake, not a run-time state");
    plan.writable =
        std::all_of(plan.fields.begin(), plan.fields.end(),
                    [](const ColumnRef& field) { return field.writable; });
    plan.fingerprint = fingerprint(kind, plan.fields);
    return plan;
}

template <typename FrameT>
PackPlan make_plan(FrameT& frame, PackKind kind) {
    auto& block = frame.block("atoms");
    const unsigned required_flags = kColExchange;

    PackPlan plan;
    plan.kind = kind;
    plan.frame_generation = frame.meta.generation();
    plan.block_generation = block.generation();
    plan.frame_owner = &frame;
    plan.owner = &block;
    plan.fields = block.resolve_columns(required_flags);
    if (plan.fields.empty())
        throw column_error("Frame pack schema selected no columns");
    plan.writable =
        std::all_of(plan.fields.begin(), plan.fields.end(),
                    [](const ColumnRef& field) { return field.writable; });
    plan.fingerprint = fingerprint(kind, plan.fields);
    return plan;
}

const std::byte* row_ptr(const ColumnRef& column, int row) {
    return column.bytes() + static_cast<std::size_t>(row) * column.row_bytes();
}

std::byte* mutable_row_ptr(const ColumnRef& column, int row) {
    return column.mutable_bytes()
           + static_cast<std::size_t>(row) * column.row_bytes();
}

int group_count_from_column(const ColumnRef& count_column, int row) {
    if (count_column.dtype != ColDType::Int
        || count_column.element_size != sizeof(int) || count_column.cols < 1)
        throw column_error("exchange count column must be native int");
    int count = 0;
    const std::byte* src =
        row_ptr(count_column, row)
        + static_cast<std::size_t>(count_column.cols - 1) * sizeof(int);
    std::memcpy(&count, src, sizeof(count));
    if (count < 0)
        throw std::runtime_error("negative topology count in Frame row");
    return count;
}

std::size_t payload_bytes_per_item(const PackPlan& plan, ColGroup group) {
    std::size_t bytes = 0;
    for (const ColumnRef& col : plan.fields)
        if (col.group == group && !(col.flags & kColCount))
            bytes = checked_add(bytes, checked_mul(static_cast<std::size_t>(
                                                       col.entries_per_item),
                                                   col.element_size));
    return bytes;
}

void validate_row(const Frame& frame, int row) {
    if (row < 0 || row >= frame.capacity())
        throw std::out_of_range("Frame pack row is outside capacity");
}

/// Coordinate columns are marked `kColPbcShift` on the field table, not
/// inferred from `wire_id <= z`. A plugin Storage column — even a scalar
/// one, even one whose wire_id happens to be small — must travel unshifted.
void append_field(Writer& writer, const ColumnRef& col, int row,
                  const double shift[3]) {
    if ((col.flags & kColPbcShift) == 0) {
        writer.append(row_ptr(col, row), col.row_bytes());
        return;
    }
    if (col.dtype != ColDType::Storage || col.cols != 1 || col.wire_id > 2)
        throw column_error("position halo column has invalid schema");
    StorageReal value;
    std::memcpy(&value, row_ptr(col, row), sizeof(value));
    value = static_cast<StorageReal>(static_cast<double>(value)
                                     + shift[col.wire_id]);
    writer.append(&value, sizeof(value));
}

} // namespace

bool PackPlan::valid_for(const Frame& frame) const noexcept {
    return frame_owner == &frame && owner != nullptr
           && frame.meta.generation() == frame_generation
           && owner->generation() == block_generation;
}

void PackPlan::validate(const Frame& frame) const {
    if (frame_owner != &frame || owner == nullptr)
        throw stale_handle_error("PackPlan belongs to a different Frame");
    if (frame.meta.generation() != frame_generation
        || owner->generation() != block_generation)
        throw stale_handle_error(
            "PackPlan is stale after a Frame schema/capacity mutation");
    for (const ColumnRef& field_ref : fields)
        field_ref.validate();
}

PackPlan make_wire_plan(Frame& frame, PackKind kind,
                        const GhostContract& contract) {
    return make_plan(frame, kind, contract);
}

PackPlan make_wire_plan(const Frame& frame, PackKind kind,
                        const GhostContract& contract) {
    return make_plan(frame, kind, contract);
}

std::uint64_t column_schema_fingerprint(const std::vector<ColumnRef>& fields) {
    std::uint64_t hash = 1469598103934665603ull;
    hash_fields(hash, fields);
    return hash;
}

PackPlan make_exchange_plan(const Frame& frame) {
    return make_plan(frame, PackKind::Exchange);
}
PackPlan make_exchange_plan(Frame& frame) {
    return make_plan(frame, PackKind::Exchange);
}
void require_schema_fingerprint(const PackPlan& local,
                                std::uint64_t remote_fingerprint) {
    if (remote_fingerprint == local.fingerprint)
        return;
    std::ostringstream message;
    message << "Frame wire schema mismatch: local 0x" << std::hex
            << local.fingerprint << ", remote 0x" << remote_fingerprint;
    throw std::runtime_error(message.str());
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* operation) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error(std::string(operation) + ": size overflow");
    return a * b;
}

std::size_t packer_bytes(int n_atoms, int nsize) {
    if (n_atoms < 0 || nsize < 0)
        throw std::invalid_argument("packer_bytes: negative count");
    return checked_mul(checked_mul(static_cast<std::size_t>(n_atoms),
                                   static_cast<std::size_t>(nsize),
                                   "packer wire"),
                       sizeof(double), "packer wire");
}

static std::size_t max_exchange_bytes(const Frame& frame,
                                      const PackPlan& plan) {
    plan.validate(frame);
    if (plan.kind != PackKind::Exchange)
        throw std::invalid_argument("max_exchange_bytes needs Exchange plan");
    std::size_t bytes = 0;
    for (const ColumnRef& col : plan.fields)
        if (col.group == ColGroup::Core || (col.flags & kColCount))
            bytes = checked_add(bytes, col.row_bytes());
    for (ColGroup group : kGroups)
        bytes = checked_add(
            bytes,
            checked_mul(static_cast<std::size_t>(frame.group_capacity(group)),
                        payload_bytes_per_item(plan, group)));
    return bytes;
}

namespace {

/// @brief True when every column on the wire is a scalar (`cols == 1`).
///
/// Forward and border pack/unpack use the same layout rule. Scalar fields
/// travel field-major so each field unpacks into one contiguous ghost run.
/// A multi-component field puts the whole wire in row-major order.
inline bool plan_is_all_scalar(const PackPlan& plan) {
    if (plan.fields.empty())
        return false;
    for (const ColumnRef& col : plan.fields)
        if (col.cols != 1)
            return false;
    return true;
}

template <std::size_t Width>
void gather_scalar(const std::byte* src, const int* rows, int n,
                   std::byte* dst) {
    for (int k = 0; k < n; ++k)
        std::memcpy(dst + static_cast<std::size_t>(k) * Width,
                    src + static_cast<std::size_t>(rows[k]) * Width, Width);
}

/// Field-major gather for the scalar forward and border wires.
void pack_scalar_field_major(const Frame& frame, const PackPlan& plan,
                             const int* rows, int n, const double shift[3],
                             std::byte* out) {
    const int first = rows[0];
    const int capacity = frame.capacity();
    bool valid = true;
    bool contiguous = true;
    // Check the indices once for all fields. Matching endpoints alone does
    // not prove that the indices between them are ordered or even in range.
    for (int k = 0; k < n; ++k) {
        const int row = rows[k];
        valid &= row >= 0 && row < capacity;
        contiguous &= static_cast<std::int64_t>(row) - first == k;
    }
    if (!valid)
        throw std::out_of_range("Frame pack row is outside capacity");

    for (const ColumnRef& col : plan.fields) {
        const std::size_t elem = col.row_bytes();
        // PackPlan::pack validated every descriptor before selecting the layout.
        const auto* src = static_cast<const std::byte*>(col.read_data);
        const auto run_bytes = static_cast<std::size_t>(n) * elem;
        if ((col.flags & kColPbcShift) != 0) {
            if (col.dtype != ColDType::Storage || col.wire_id > 2)
                throw column_error("position halo column has invalid schema");
            const auto* positions =
                static_cast<const StorageReal*>(col.read_data);
            const auto offset = static_cast<StorageReal>(shift[col.wire_id]);
            if (contiguous && offset == StorageReal{0}) {
                std::memcpy(out, positions + first, run_bytes);
            } else {
                for (int k = 0; k < n; ++k) {
                    const StorageReal value =
                        positions[contiguous ? first + k : rows[k]] + offset;
                    // Earlier fields can leave a native value unaligned.
                    std::memcpy(
                        out + static_cast<std::size_t>(k) * sizeof(value),
                        &value, sizeof(value));
                }
            }
        } else if (contiguous) {
            std::memcpy(out, src + static_cast<std::size_t>(first) * elem,
                        run_bytes);
        } else if (elem == sizeof(std::uint32_t)) {
            gather_scalar<sizeof(std::uint32_t)>(src, rows, n, out);
        } else if (elem == sizeof(std::uint64_t)) {
            gather_scalar<sizeof(std::uint64_t)>(src, rows, n, out);
        } else {
            for (int k = 0; k < n; ++k)
                std::memcpy(out + static_cast<std::size_t>(k) * elem,
                            src + static_cast<std::size_t>(rows[k]) * elem,
                            elem);
        }
        out += run_bytes;
    }
}

/// @brief Scatter the field-major wire into contiguous ghost slots.
/// @param plan Compiled all-scalar wire (writable).
/// @param n Row count.
/// @param first First ghost row to write.
/// @param data Wire buffer.
void unpack_scalar_field_major(const PackPlan& plan, int n, int first,
                               const std::byte* data) {
    const std::byte* cursor = data;
    for (const ColumnRef& col : plan.fields) {
        const std::size_t run = static_cast<std::size_t>(n) * col.row_bytes();
        if (run != 0)
            std::memcpy(mutable_row_ptr(col, first), cursor, run);
        cursor += run;
    }
}

} // namespace

std::size_t pack_exchange_wire(const Frame& frame, const PackPlan& plan,
                               int row, std::byte* out, std::size_t capacity) {
    plan.validate(frame);
    if (plan.kind != PackKind::Exchange)
        throw std::invalid_argument("pack_exchange needs Exchange plan");
    validate_row(frame, row);
    Writer writer(out, capacity);
    for (const ColumnRef& col : plan.fields)
        if (col.group == ColGroup::Core)
            writer.append(row_ptr(col, row), col.row_bytes());
    for (ColGroup group : kGroups) {
        int count = -1;
        for (const ColumnRef& col : plan.fields) {
            if (col.group != group || !(col.flags & kColCount))
                continue;
            writer.append(row_ptr(col, row), col.row_bytes());
            count = group_count_from_column(col, row);
        }
        if (count < 0)
            throw column_error("exchange group has no count column");
        for (const ColumnRef& col : plan.fields) {
            if (col.group != group || (col.flags & kColCount))
                continue;
            const std::size_t required =
                checked_mul(static_cast<std::size_t>(count),
                            static_cast<std::size_t>(col.entries_per_item));
            if (required > static_cast<std::size_t>(col.cols))
                throw std::length_error(
                    "exchange topology count exceeds column stride");
            writer.append(row_ptr(col, row),
                          checked_mul(required, col.element_size));
        }
    }
    return writer.size();
}

std::size_t unpack_exchange_wire(Frame& frame, PackPlan& plan, int row,
                                 const std::byte* data, std::size_t size) {
    plan.validate(frame);
    if (plan.kind != PackKind::Exchange)
        throw std::invalid_argument("unpack_exchange needs Exchange plan");
    validate_row(frame, row);

    // Pre-scan counts before writing.  Any required re-stride happens once,
    // then the generation-bound plan is rebuilt before a pointer is touched.
    Reader scan(data, size);
    for (const ColumnRef& col : plan.fields)
        if (col.group == ColGroup::Core)
            scan.skip(col.row_bytes());
    // Must match kGroups.size() (Bond, Angle, Dihedral, Improper, Special).
    // A too-short array was a silent OOB when Improper was added — arm64
    // stack layout sometimes papered over it; Linux x86_64 did not.
    std::array<int, kGroups.size()> counts{};
    for (std::size_t gi = 0; gi < kGroups.size(); ++gi) {
        const ColGroup group = kGroups[gi];
        int count = -1;
        for (const ColumnRef& col : plan.fields) {
            if (col.group != group || !(col.flags & kColCount))
                continue;
            if (col.dtype != ColDType::Int || col.element_size != sizeof(int))
                throw column_error("exchange count column must be native int");
            if (col.cols < 1)
                throw column_error("exchange count column has zero width");
            const std::byte* count_bytes = scan.take(col.row_bytes());
            std::memcpy(&count,
                        count_bytes
                            + static_cast<std::size_t>(col.cols - 1)
                                  * sizeof(int),
                        sizeof(int));
        }
        if (count < 0)
            throw std::runtime_error("invalid/negative exchange group count");
        counts[gi] = count;
        scan.skip(checked_mul(static_cast<std::size_t>(count),
                              payload_bytes_per_item(plan, group)));
    }
    const std::size_t consumed = scan.offset();

    bool grew = false;
    for (std::size_t gi = 0; gi < kGroups.size(); ++gi) {
        if (counts[gi] > frame.group_capacity(kGroups[gi])) {
            frame.grow_group(kGroups[gi], counts[gi]);
            grew = true;
        }
    }
    if (grew) {
        const auto expected = plan.fingerprint;
        plan = make_exchange_plan(frame);
        if (plan.fingerprint != expected)
            throw std::runtime_error(
                "topology re-stride changed the semantic exchange schema");
    }

    Reader reader(data, consumed);
    for (const ColumnRef& col : plan.fields) {
        if (col.group != ColGroup::Core)
            continue;
        std::memcpy(mutable_row_ptr(col, row), reader.take(col.row_bytes()),
                    col.row_bytes());
    }
    for (std::size_t gi = 0; gi < kGroups.size(); ++gi) {
        const ColGroup group = kGroups[gi];
        const int count = counts[gi];
        for (const ColumnRef& col : plan.fields) {
            if (col.group != group || !(col.flags & kColCount))
                continue;
            std::memcpy(mutable_row_ptr(col, row), reader.take(col.row_bytes()),
                        col.row_bytes());
        }
        for (const ColumnRef& col : plan.fields) {
            if (col.group != group || (col.flags & kColCount))
                continue;
            std::byte* dst = mutable_row_ptr(col, row);
            if (col.row_bytes() != 0)
                std::memset(dst, 0, col.row_bytes());
            const std::size_t active = checked_mul(
                checked_mul(static_cast<std::size_t>(count),
                            static_cast<std::size_t>(col.entries_per_item)),
                col.element_size);
            const std::byte* src = reader.take(active);
            if (active != 0)
                std::memcpy(dst, src, active);
        }
    }
    return reader.offset();
}

/// @brief True when the plan is the standard force wire: exactly the three
/// scalar `AccReal` columns fx, fy, fz. What the reverse pass resolves
/// to on every current Frame; anything else falls back to the generic loops.
bool plan_is_fxyz_reverse(const PackPlan& plan) {
    if (plan.fields.size() != 3)
        return false;
    for (const ColumnRef& col : plan.fields)
        if (col.dtype != ColDType::Acc || col.cols != 1)
            return false;
    return true;
}

void pack_reverse_wire(const PackPlan& plan, int first, int count,
                       std::vector<std::byte>& buffer) {
    buffer.resize(plan.bytes(count));
    if (!plan_is_fxyz_reverse(plan)) {
        // General wire: row-major, columns in plan order. A plugin-declared
        // runtime column on the reverse pass rides this arm — exercised
        // by tests/mpi/test_halo_exchange.cpp.
        std::size_t offset = 0;
        for (const ColumnRef& field : plan.fields) {
            for (int k = 0; k < count; ++k) {
                const std::size_t bytes = field.row_bytes();
                if (bytes != 0)
                    std::memcpy(buffer.data() + offset,
                                field.bytes()
                                    + static_cast<std::size_t>(first + k)
                                          * bytes,
                                bytes);
                offset += bytes;
            }
        }
        return;
    }
    // Ghost force rows are contiguous per column: three block copies
    // instead of a per-row memcpy chain. Field-major layout, identical
    // bytes to the generic path.
    std::size_t offset = 0;
    for (const ColumnRef& field : plan.fields) {
        field.validate();
        const std::size_t bytes =
            static_cast<std::size_t>(count) * sizeof(AccReal);
        std::memcpy(buffer.data() + offset,
                    static_cast<const AccReal*>(field.read_data) + first,
                    bytes);
        offset += bytes;
    }
    return;
}

namespace {

/// @brief Fold one column's contiguous payload run onto `rows`, `+=`.
template <typename T>
void accumulate_column(const ColumnRef& field, const std::vector<int>& rows,
                       const std::byte* source) {
    if (!field.writable)
        throw column_error("halo reverse plan is const");
    T* destination = static_cast<T*>(field.write_data);
    for (std::size_t k = 0; k < rows.size(); ++k) {
        for (int c = 0; c < field.cols; ++c) {
            T value{};
            const std::size_t value_index =
                k * static_cast<std::size_t>(field.cols)
                + static_cast<std::size_t>(c);
            std::memcpy(&value, source + value_index * sizeof(T), sizeof(T));
            destination[static_cast<std::size_t>(rows[k]) * field.cols + c] +=
                value;
        }
    }
}

} // namespace

void accumulate_reverse_wire(const PackPlan& plan, const std::vector<int>& rows,
                             const std::vector<std::byte>& buffer) {
    if (rows.empty()) {
        if (!buffer.empty())
            throw std::runtime_error(
                "accumulate_reverse_rows: no owner rows but the payload is "
                "not empty; sender and receiver disagree on the row count.");
        return;
    }
    if (!plan.writable)
        throw column_error(
            "accumulate_reverse_rows needs a writable plan; this one was "
            "compiled from a const Frame. Rebuild it with "
            "make_wire_plan(Frame&, kind, schema).");
    const std::size_t expected = plan.bytes(static_cast<int>(rows.size()));
    if (buffer.size() != expected)
        throw std::length_error(
            "accumulate_reverse_rows: payload is "
            + std::to_string(buffer.size()) + " B but "
            + std::to_string(rows.size()) + " owner rows need "
            + std::to_string(expected)
            + " B; sender and receiver disagree on the row count.");

    if (!plan_is_fxyz_reverse(plan)) {
        // General wire: row-major, mirroring pack_reverse_rows' generic arm.
        std::size_t offset = 0;
        for (const ColumnRef& field : plan.fields) {
            const std::size_t bytes =
                checked_mul(rows.size(), field.row_bytes());
            switch (field.dtype) {
            case ColDType::Storage:
                accumulate_column<StorageReal>(field, rows,
                                               buffer.data() + offset);
                break;
            case ColDType::Acc:
                accumulate_column<AccReal>(field, rows, buffer.data() + offset);
                break;
            case ColDType::Tag:
                accumulate_column<TagInt>(field, rows, buffer.data() + offset);
                break;
            case ColDType::Int:
                accumulate_column<int>(field, rows, buffer.data() + offset);
                break;
            case ColDType::Image:
                accumulate_column<ImageInt>(field, rows,
                                            buffer.data() + offset);
                break;
            }
            offset += bytes;
        }
        return;
    }

    // Field-major wire (see pack_reverse_rows): one contiguous run per
    // component, scattered onto the owner rows. `+=`, not `=` — several
    // ghosts of one owner fold in across swaps and periodic images.
    const auto* __restrict src =
        reinterpret_cast<const AccReal*>(buffer.data());
    const std::size_t n = rows.size();
    const int* __restrict rowp = rows.data();
    for (int f = 0; f < 3; ++f) {
        plan.fields[static_cast<std::size_t>(f)].validate();
        auto* __restrict dst = static_cast<AccReal*>(
            plan.fields[static_cast<std::size_t>(f)].write_data);
        const AccReal* __restrict col = src + f * n;
        for (std::size_t k = 0; k < n; ++k)
            dst[rowp[k]] += col[k];
    }
}

// ── PackPlan: the verbs ───────────────────────────────────────────────

std::size_t PackPlan::row_bytes() const {
    std::size_t total = 0;
    for (const ColumnRef& field : fields) {
        if (field.row_bytes() > std::numeric_limits<std::size_t>::max() - total)
            throw std::overflow_error("wire plan row size overflow");
        total += field.row_bytes();
    }
    return total;
}

std::size_t PackPlan::bytes(int n) const {
    if (kind == PackKind::Exchange)
        throw std::invalid_argument(
            "PackPlan::bytes: an exchange row is variable length; size it with "
            "max_row_bytes() and pack one row at a time.");
    if (n < 0)
        throw std::invalid_argument("PackPlan::bytes: negative row count");
    return checked_mul(static_cast<std::size_t>(n), row_bytes());
}

std::size_t PackPlan::max_row_bytes() const {
    if (frame_owner == nullptr)
        throw std::runtime_error(
            "PackPlan::max_row_bytes: this plan has no owning Frame; it was "
            "default-constructed or its Frame was destroyed.");
    return max_exchange_bytes(*frame_owner, *this);
}

std::uint64_t PackPlan::wire_fingerprint(std::uint64_t discriminator) const {
    std::uint64_t hash = 1469598103934665603ull;
    hash_value(hash, fingerprint);
    hash_value(hash, discriminator);
    return hash;
}

std::size_t PackPlan::pack(const Frame& frame, const int* rows, int n,
                           const double shift[3], std::byte* out,
                           std::size_t capacity) const {
    if (kind != PackKind::Comm && kind != PackKind::Border)
        throw std::invalid_argument(
            "PackPlan::pack drives the fixed-stride comm and border wires; a "
            "reverse plan uses pack_rows and an exchange plan uses pack_row.");
    validate(frame);
    const auto size = bytes(n);
    if (size > capacity)
        throw std::length_error("Frame pack output buffer is too small");
    if (n == 0)
        return 0;
    if (plan_is_all_scalar(*this)) {
        pack_scalar_field_major(frame, *this, rows, n, shift, out);
        return size;
    }

    Writer writer(out, capacity);
    for (int k = 0; k < n; ++k) {
        const int row = rows[k];
        validate_row(frame, row);
        for (const ColumnRef& col : fields)
            append_field(writer, col, row, shift);
    }
    return writer.size();
}

void PackPlan::unpack(Frame& frame, int n, int first, const std::byte* data,
                      std::size_t size) const {
    if (kind != PackKind::Comm && kind != PackKind::Border)
        throw std::invalid_argument(
            "PackPlan::unpack drives the fixed-stride comm and border wires; a "
            "reverse plan uses accumulate and an exchange plan uses "
            "unpack_row.");
    validate(frame);
    if (n < 0 || first < 0 || first > frame.capacity() - n)
        throw std::invalid_argument(
            "PackPlan::unpack: invalid destination range");
    if (size != bytes(n))
        throw std::length_error("PackPlan::unpack: payload size mismatch");
    if (!writable)
        throw column_error(
            "PackPlan::unpack needs a writable plan. Rebuild it with "
            "make_wire_plan(Frame&, kind, contract).");
    if (n == 0)
        return;
    if (plan_is_all_scalar(*this)) {
        unpack_scalar_field_major(*this, n, first, data);
        return;
    }

    Reader reader(data, size);
    for (int k = 0; k < n; ++k) {
        for (const ColumnRef& col : fields) {
            const std::size_t width = col.row_bytes();
            std::byte* dst = mutable_row_ptr(col, first + k);
            const std::byte* src = reader.take(width);
            if (width != 0)
                std::memcpy(dst, src, width);
        }
    }
}

void PackPlan::pack_rows(int first, int count,
                         std::vector<std::byte>& out) const {
    pack_reverse_wire(*this, first, count, out);
}

void PackPlan::accumulate(const std::vector<int>& rows,
                          const std::vector<std::byte>& in) const {
    accumulate_reverse_wire(*this, rows, in);
}

std::size_t PackPlan::pack_row(const Frame& frame, int row, std::byte* out,
                               std::size_t capacity) const {
    return pack_exchange_wire(frame, *this, row, out, capacity);
}

std::size_t PackPlan::unpack_row(Frame& frame, int row, const std::byte* data,
                                 std::size_t size) {
    return unpack_exchange_wire(frame, *this, row, data, size);
}

} // namespace comm
} // namespace cpu
} // namespace ATV

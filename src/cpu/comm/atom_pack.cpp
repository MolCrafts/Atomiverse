/// @file src/cpu/comm/atom_pack.cpp
/// @brief Native-dtype schema-compiled Frame row packing.

#include "atomiverse/cpu/comm/atom_pack.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
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
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("Frame pack byte count overflow");
    return a * b;
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

std::uint64_t fingerprint(PackKind kind, const std::vector<ColumnRef>& fields) {
    std::uint64_t hash = 1469598103934665603ull;
    hash_value(hash, kAtomWireVersion);
    hash_value(hash, kind);
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
    return hash;
}

template <typename FrameT>
PackPlan make_plan(FrameT& frame, PackKind kind) {
    auto& block = frame.block("atoms");
    const unsigned required_flags =
        kind == PackKind::Comm
            ? kColComm
            : (kind == PackKind::Border
                   ? kColBorder
                   : (kind == PackKind::Reverse ? kColReverse : kColExchange));

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

} // namespace

bool PackPlan::valid_for(const Frame& frame) const noexcept {
    return frame_owner == &frame && owner != nullptr
           && version == kAtomWireVersion
           && frame.meta.generation() == frame_generation
           && owner->generation() == block_generation;
}

void PackPlan::validate(const Frame& frame) const {
    if (frame_owner != &frame || owner == nullptr)
        throw stale_handle_error("PackPlan belongs to a different Frame");
    if (version != kAtomWireVersion
        || frame.meta.generation() != frame_generation
        || owner->generation() != block_generation)
        throw stale_handle_error(
            "PackPlan is stale after a Frame schema/capacity mutation");
    for (const ColumnRef& field_ref : fields)
        field_ref.validate();
}

PackPlan make_comm_plan(const Frame& frame) {
    return make_plan(frame, PackKind::Comm);
}
PackPlan make_comm_plan(Frame& frame) {
    return make_plan(frame, PackKind::Comm);
}
PackPlan make_border_plan(const Frame& frame) {
    return make_plan(frame, PackKind::Border);
}
PackPlan make_border_plan(Frame& frame) {
    return make_plan(frame, PackKind::Border);
}
PackPlan make_exchange_plan(const Frame& frame) {
    return make_plan(frame, PackKind::Exchange);
}
PackPlan make_exchange_plan(Frame& frame) {
    return make_plan(frame, PackKind::Exchange);
}
PackPlan make_reverse_plan(const Frame& frame) {
    return make_plan(frame, PackKind::Reverse);
}
PackPlan make_reverse_plan(Frame& frame) {
    return make_plan(frame, PackKind::Reverse);
}

void require_schema_fingerprint(const PackPlan& local,
                                std::uint64_t remote_fingerprint,
                                std::uint32_t remote_version) {
    if (remote_version == local.version
        && remote_fingerprint == local.fingerprint)
        return;
    std::ostringstream message;
    message << "Frame wire schema mismatch: local v" << local.version << "/0x"
            << std::hex << local.fingerprint << ", remote v" << std::dec
            << remote_version << "/0x" << std::hex << remote_fingerprint;
    throw std::runtime_error(message.str());
}

std::size_t comm_bytes(const PackPlan& plan, int n) {
    if (plan.kind != PackKind::Comm || n < 0)
        throw std::invalid_argument("comm_bytes received the wrong plan/count");
    std::size_t per_row = 0;
    for (const ColumnRef& col : plan.fields)
        per_row = checked_add(per_row, col.row_bytes());
    return checked_mul(static_cast<std::size_t>(n), per_row);
}

std::size_t border_bytes(const PackPlan& plan, int n) {
    if (plan.kind != PackKind::Border || n < 0)
        throw std::invalid_argument(
            "border_bytes received the wrong plan/count");
    std::size_t per_row = 0;
    for (const ColumnRef& col : plan.fields)
        per_row = checked_add(per_row, col.row_bytes());
    return checked_mul(static_cast<std::size_t>(n), per_row);
}

std::size_t max_exchange_bytes(const Frame& frame, const PackPlan& plan) {
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

std::size_t pack_comm(const Frame& frame, const PackPlan& plan,
                      const int* sendlist, int n, double sx, double sy,
                      double sz, std::byte* out, std::size_t capacity) {
    plan.validate(frame);
    if (plan.kind != PackKind::Comm || n < 0)
        throw std::invalid_argument("pack_comm received wrong plan/count");
    Writer writer(out, capacity);
    const std::array<double, 3> shift{sx, sy, sz};
    for (int k = 0; k < n; ++k) {
        const int row = sendlist[k];
        validate_row(frame, row);
        for (const ColumnRef& col : plan.fields) {
            if (col.wire_id <= static_cast<std::uint16_t>(AtomFieldId::z)) {
                if (col.dtype != ColDType::Storage || col.cols != 1)
                    throw column_error(
                        "position halo column has invalid schema");
                StorageReal value;
                std::memcpy(&value, row_ptr(col, row), sizeof(value));
                value = static_cast<StorageReal>(static_cast<double>(value)
                                                 + shift[col.wire_id]);
                writer.append(&value, sizeof(value));
            } else {
                writer.append(row_ptr(col, row), col.row_bytes());
            }
        }
    }
    return writer.size();
}

void unpack_comm(Frame& frame, const PackPlan& plan, int n, int first,
                 const std::byte* data, std::size_t size) {
    plan.validate(frame);
    if (plan.kind != PackKind::Comm || n < 0 || first < 0
        || first + n > frame.capacity())
        throw std::invalid_argument("unpack_comm received invalid range/plan");
    if (size != comm_bytes(plan, n))
        throw std::length_error("unpack_comm payload size mismatch");
    Reader reader(data, size);
    for (int k = 0; k < n; ++k) {
        for (const ColumnRef& col : plan.fields) {
            const std::size_t bytes = col.row_bytes();
            std::byte* dst = mutable_row_ptr(col, first + k);
            const std::byte* src = reader.take(bytes);
            if (bytes != 0)
                std::memcpy(dst, src, bytes);
        }
    }
}

std::size_t pack_border(const Frame& frame, const PackPlan& plan,
                        const int* sendlist, int n, double sx, double sy,
                        double sz, std::byte* out, std::size_t capacity) {
    plan.validate(frame);
    if (plan.kind != PackKind::Border || n < 0)
        throw std::invalid_argument("pack_border received wrong plan/count");
    Writer writer(out, capacity);
    const std::array<double, 3> shift{sx, sy, sz};
    for (int k = 0; k < n; ++k) {
        const int row = sendlist[k];
        validate_row(frame, row);
        for (const ColumnRef& col : plan.fields) {
            if (col.wire_id <= static_cast<std::uint16_t>(AtomFieldId::z)) {
                if (col.dtype != ColDType::Storage || col.cols != 1)
                    throw column_error(
                        "position border column has invalid schema");
                StorageReal value;
                std::memcpy(&value, row_ptr(col, row), sizeof(value));
                value = static_cast<StorageReal>(static_cast<double>(value)
                                                 + shift[col.wire_id]);
                writer.append(&value, sizeof(value));
            } else {
                writer.append(row_ptr(col, row), col.row_bytes());
            }
        }
    }
    return writer.size();
}

void unpack_border(Frame& frame, const PackPlan& plan, int n, int first,
                   const std::byte* data, std::size_t size) {
    plan.validate(frame);
    if (plan.kind != PackKind::Border || n < 0 || first < 0
        || first + n > frame.capacity())
        throw std::invalid_argument(
            "unpack_border received invalid range/plan");
    if (size != border_bytes(plan, n))
        throw std::length_error("unpack_border payload size mismatch");
    Reader reader(data, size);
    for (int k = 0; k < n; ++k) {
        const int row = first + k;
        for (const ColumnRef& col : plan.fields) {
            const std::size_t bytes = col.row_bytes();
            std::byte* dst = mutable_row_ptr(col, row);
            const std::byte* src = reader.take(bytes);
            if (bytes != 0)
                std::memcpy(dst, src, bytes);
        }
    }
}

std::size_t pack_exchange(const Frame& frame, const PackPlan& plan, int row,
                          std::byte* out, std::size_t capacity) {
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

std::size_t unpack_exchange(Frame& frame, PackPlan& plan, int row,
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

} // namespace comm
} // namespace cpu
} // namespace ATV

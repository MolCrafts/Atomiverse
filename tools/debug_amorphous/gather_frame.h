/// @file tools/debug_amorphous/gather_frame.h
/// @brief Native-schema inter-stage all-gather for multi-rank CPU Frames.
///
/// After a distributed `cpu::md::MD::run()`, each rank owns only its first
/// `n_local` rows.  A second driver must start from one complete replicated
/// Frame, including runtime columns and per-row bonded topology.  This helper
/// therefore gathers `cpu::comm::PackPlan::Exchange` rows as native bytes,
/// sorts them by the native `TagInt`, and atomically replaces the distributed
/// Frame with the complete sorted result on every rank.
///
/// No field is enumerated here, and no value is widened through `double`.
#pragma once

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/communication/atom_pack.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/schema_mpi.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace debug_amorphous {

/// @brief Capture-n_local helper: a Component that records `dd->n_local` to
/// a user-supplied int at every end_of_step, leaving the int holding the
/// rank's owned-count after the driver returns.
class CaptureNLocal : public cpu::Component {
  public:
    explicit CaptureNLocal(int* sink) : sink_(sink) {}
    std::string name() const override { return "capture_n_local"; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, 30)};
    }
    void launch(const cpu::LaunchContext& ctx) override {
        const auto* dd = ctx.runtime.md.dd;
        *sink_ =
            (dd && ctx.runtime.mpi_size > 1) ? dd->n_local : ctx.frame.nlocal();
    }

  private:
    int* sink_;
};

namespace detail {

struct PackedRow {
    std::size_t offset = 0;
    std::size_t size = 0;
    TagInt tag = 0;
};

inline std::size_t checked_add(std::size_t lhs, std::size_t rhs,
                               const char* operation) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
        throw std::overflow_error(std::string(operation)
                                  + ": byte count overflow");
    return lhs + rhs;
}

inline std::size_t packed_tag_offset(const cpu::comm::PackPlan& plan) {
    std::size_t offset = 0;
    for (const cpu::ColumnRef& field : plan.fields) {
        if (field.group != ColGroup::Core)
            continue;
        if (field.wire_id == static_cast<std::uint16_t>(AtomFieldId::tag)) {
            if (field.dtype != ColDType::Tag || field.cols != 1
                || field.element_size != sizeof(TagInt))
                throw cpu::column_error(
                    "exchange plan has a non-native tag column");
            return offset;
        }
        offset = checked_add(offset, field.row_bytes(),
                             "gather_full_frame tag offset");
    }
    throw cpu::column_error("exchange plan has no tag column");
}

} // namespace detail

inline void gather_full_frame_inplace(cpu::Frame& frame, int n_local,
                                      MPI_Comm comm) {
    int world_size = 0;
    MPI_Comm_size(comm, &world_size);
    if (world_size <= 1)
        return;

    const int local_range_valid =
        n_local >= 0 && n_local <= frame.capacity() ? 1 : 0;
    int all_ranges_valid = 0;
    MPI_Allreduce(&local_range_valid, &all_ranges_valid, 1, MPI_INT, MPI_MIN,
                  comm);
    if (!all_ranges_valid)
        throw std::invalid_argument(
            "gather_full_frame: an owned-row range is invalid");

    cpu::comm::PackPlan source_plan = cpu::comm::make_exchange_plan(frame);
    cpu::comm::require_collective_schema(source_plan, comm,
                                         "gather_full_frame");
    const std::size_t tag_offset = detail::packed_tag_offset(source_plan);

    // Frame each variable-size exchange row with one fixed-width byte count.
    // The row itself remains exactly the native PackPlan payload.
    const std::size_t row_capacity =
        cpu::comm::max_exchange_bytes(frame, source_plan);
    std::vector<std::byte> row_buffer(row_capacity);
    std::vector<std::byte> send_buffer;
    for (int row = 0; row < n_local; ++row) {
        const std::size_t payload_size = cpu::comm::pack_exchange(
            frame, source_plan, row, row_buffer.data(), row_buffer.size());
        if (payload_size > std::numeric_limits<std::uint64_t>::max())
            throw std::length_error(
                "gather_full_frame: one row exceeds the wire envelope");
        const std::size_t envelope_offset = send_buffer.size();
        const std::size_t payload_offset = detail::checked_add(
            envelope_offset, sizeof(std::uint64_t), "gather_full_frame");
        send_buffer.resize(detail::checked_add(payload_offset, payload_size,
                                               "gather_full_frame"));
        const auto wire_size = static_cast<std::uint64_t>(payload_size);
        std::memcpy(send_buffer.data() + envelope_offset, &wire_size,
                    sizeof(wire_size));
        if (payload_size != 0)
            std::memcpy(send_buffer.data() + payload_offset, row_buffer.data(),
                        payload_size);
    }

    std::vector<int> owner_counts(static_cast<std::size_t>(world_size));
    MPI_Allgather(&n_local, 1, MPI_INT, owner_counts.data(), 1, MPI_INT, comm);

    const int send_bytes = cpu::comm::checked_mpi_byte_count(
        send_buffer.size(), "gather_full_frame");
    std::vector<int> recv_counts(static_cast<std::size_t>(world_size));
    MPI_Allgather(&send_bytes, 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                  comm);

    std::vector<int> displacements(static_cast<std::size_t>(world_size), 0);
    int total_bytes = 0;
    int n_total = 0;
    for (int rank = 0; rank < world_size; ++rank) {
        if (recv_counts[rank] < 0 || owner_counts[rank] < 0
            || recv_counts[rank] > std::numeric_limits<int>::max() - total_bytes
            || owner_counts[rank] > std::numeric_limits<int>::max() - n_total)
            throw std::length_error(
                "gather_full_frame: collective payload exceeds INT_MAX");
        displacements[rank] = total_bytes;
        total_bytes += recv_counts[rank];
        n_total += owner_counts[rank];
    }

    std::vector<std::byte> receive_buffer(
        static_cast<std::size_t>(total_bytes));
    MPI_Allgatherv(send_buffer.data(), send_bytes, MPI_BYTE,
                   receive_buffer.data(), recv_counts.data(),
                   displacements.data(), MPI_BYTE, comm);

    std::vector<detail::PackedRow> rows;
    rows.reserve(static_cast<std::size_t>(n_total));
    for (int rank = 0; rank < world_size; ++rank) {
        std::size_t cursor = static_cast<std::size_t>(displacements[rank]);
        const std::size_t end = detail::checked_add(
            cursor, static_cast<std::size_t>(recv_counts[rank]),
            "gather_full_frame rank segment");
        for (int row = 0; row < owner_counts[rank]; ++row) {
            if (sizeof(std::uint64_t) > end - cursor)
                throw std::length_error(
                    "gather_full_frame: truncated row envelope");
            std::uint64_t wire_size = 0;
            std::memcpy(&wire_size, receive_buffer.data() + cursor,
                        sizeof(wire_size));
            cursor += sizeof(wire_size);
            if (wire_size > static_cast<std::uint64_t>(end - cursor))
                throw std::length_error(
                    "gather_full_frame: truncated exchange row");
            const std::size_t payload_size =
                static_cast<std::size_t>(wire_size);
            if (tag_offset > payload_size
                || sizeof(TagInt) > payload_size - tag_offset)
                throw std::length_error(
                    "gather_full_frame: exchange row has no complete tag");
            TagInt tag = 0;
            std::memcpy(&tag, receive_buffer.data() + cursor + tag_offset,
                        sizeof(tag));
            rows.push_back(detail::PackedRow{cursor, payload_size, tag});
            cursor += payload_size;
        }
        if (cursor != end)
            throw std::length_error(
                "gather_full_frame: rank segment has trailing bytes");
    }
    if (rows.size() != static_cast<std::size_t>(n_total))
        throw std::length_error(
            "gather_full_frame: owner and row counts disagree");

    std::stable_sort(
        rows.begin(), rows.end(),
        [](const detail::PackedRow& lhs, const detail::PackedRow& rhs) {
            return lhs.tag < rhs.tag;
        });

    cpu::Frame gathered = frame.clone_structure(n_total);
    cpu::comm::PackPlan destination_plan =
        cpu::comm::make_exchange_plan(gathered);
    cpu::comm::require_schema_fingerprint(
        destination_plan, source_plan.fingerprint, source_plan.version);
    for (int destination = 0; destination < n_total; ++destination) {
        const auto& row = rows[static_cast<std::size_t>(destination)];
        const std::size_t consumed = cpu::comm::unpack_exchange(
            gathered, destination_plan, destination,
            receive_buffer.data() + row.offset, row.size);
        if (consumed != row.size)
            throw std::length_error(
                "gather_full_frame: exchange decoder left trailing bytes");
    }
    gathered.set_nlocal(n_total);
    gathered.set_nghost(0);
    gathered.clear_tag_index();
    frame = std::move(gathered);
}

} // namespace debug_amorphous
} // namespace ATV

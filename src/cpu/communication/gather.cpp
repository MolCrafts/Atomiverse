/// @file cpu/communication/gather.cpp
/// @brief Native-dtype owner gather with schema-driven runtime columns.

#include "atomiverse/cpu/communication/gather.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "atomiverse/cpu/communication/atom_pack.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/frame/frame.h"

#ifdef ATV_HAS_MPI
#include "atomiverse/cpu/communication/schema_mpi.h"
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace md {
namespace mpi {

namespace {

std::size_t checked_product(std::size_t a, std::size_t b,
                            const char* operation) {
    return comm::checked_mul(a, b, operation);
}

/// @brief The atoms-Block columns a gathered frame carries.
///
/// The Block's own flag selector, not a compiled wire plan: a gather is a
/// bulk column-major collective, so it wants the column *set* and never any
/// of `PackPlan`'s per-row pack verbs. `kColExchange` is the same "what an
/// atom IS" question migration asks, which is exactly what a single-rank
/// view of the system has to reproduce.
std::vector<ColumnRef> migrated_columns(const Frame& frame) {
    return frame.block("atoms").resolve_columns(kColExchange);
}

void copy_runtime_columns(GatheredFrame& out, const Frame& frame, int n_local,
                          const std::vector<ColumnRef>& fields) {
    for (const ColumnRef& field : fields) {
        if (field.wire_id != kDynamicFieldId)
            continue;
        GatheredColumn column;
        column.key = field.key;
        column.dtype = field.dtype;
        column.cols = field.cols;
        column.group = field.group;
        column.entries_per_item = field.entries_per_item;
        column.flags = field.flags;
        const std::size_t bytes =
            checked_product(static_cast<std::size_t>(n_local),
                            field.row_bytes(), "gather/runtime-column");
        column.bytes.resize(bytes);
        if (bytes != 0)
            std::memcpy(column.bytes.data(), field.bytes(), bytes);
        out.runtime_columns.push_back(std::move(column));
    }
}

void copy_owners_into(GatheredFrame& out, const Frame& frame, int n_local) {
    if (n_local < 0 || n_local > frame.capacity())
        throw std::out_of_range(
            "gather: owner range is outside Frame capacity");
    out.n_total = n_local;
    out.box = frame.box.cell();
    const StorageReal* x = frame.span(kAtom_x).data();
    const StorageReal* y = frame.span(kAtom_y).data();
    const StorageReal* z = frame.span(kAtom_z).data();
    const StorageReal* vx = frame.span(kAtom_vx).data();
    const StorageReal* vy = frame.span(kAtom_vy).data();
    const StorageReal* vz = frame.span(kAtom_vz).data();
    const AccReal* fx = frame.span(kAtom_fx).data();
    const AccReal* fy = frame.span(kAtom_fy).data();
    const AccReal* fz = frame.span(kAtom_fz).data();
    const StorageReal* mass = frame.span(kAtom_mass).data();
    const StorageReal* charge = frame.span(kAtom_charge).data();
    const int* type = frame.span(kAtom_type).data();
    const TagInt* tag = frame.span(kAtom_tag).data();
    const std::int8_t* image = frame.span(kAtom_image).data();
    auto copy = [](auto& destination, const auto* source, std::size_t count) {
        destination.resize(count);
        if (count != 0)
            std::copy_n(source, count, destination.data());
    };
    const std::size_t rows = static_cast<std::size_t>(n_local);
    copy(out.x, x, rows);
    copy(out.y, y, rows);
    copy(out.z, z, rows);
    copy(out.vx, vx, rows);
    copy(out.vy, vy, rows);
    copy(out.vz, vz, rows);
    copy(out.fx, fx, rows);
    copy(out.fy, fy, rows);
    copy(out.fz, fz, rows);
    copy(out.mass, mass, rows);
    copy(out.charge, charge, rows);
    copy(out.type, type, rows);
    copy(out.tag, tag, rows);
    copy(out.image, image,
         checked_product(rows, std::size_t{3}, "gather/image"));
    copy_runtime_columns(out, frame, n_local, migrated_columns(frame));
}

void sort_by_tag(GatheredFrame& gathered) {
    const int n = gathered.n_total;
    if (n <= 1)
        return;
    std::vector<int> permutation(static_cast<std::size_t>(n));
    std::iota(permutation.begin(), permutation.end(), 0);
    std::stable_sort(permutation.begin(), permutation.end(), [&](int a, int b) {
        return gathered.tag[static_cast<std::size_t>(a)]
               < gathered.tag[static_cast<std::size_t>(b)];
    });

    auto permute = [&](auto& values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        std::vector<T> reordered(values.size());
        for (int i = 0; i < n; ++i)
            reordered[static_cast<std::size_t>(i)] =
                values[static_cast<std::size_t>(permutation[i])];
        values.swap(reordered);
    };
    permute(gathered.x);
    permute(gathered.y);
    permute(gathered.z);
    permute(gathered.vx);
    permute(gathered.vy);
    permute(gathered.vz);
    permute(gathered.fx);
    permute(gathered.fy);
    permute(gathered.fz);
    permute(gathered.mass);
    permute(gathered.charge);
    permute(gathered.type);
    permute(gathered.tag);

    std::vector<std::int8_t> reordered_image(gathered.image.size());
    for (int i = 0; i < n; ++i) {
        const int source = permutation[i];
        std::memcpy(reordered_image.data() + 3 * i,
                    gathered.image.data() + 3 * source,
                    3 * sizeof(std::int8_t));
    }
    gathered.image.swap(reordered_image);

    for (GatheredColumn& column : gathered.runtime_columns) {
        if (column.bytes.size() % static_cast<std::size_t>(n) != 0)
            throw std::runtime_error(
                "gather: runtime-column payload is not row aligned");
        const std::size_t row_bytes =
            column.bytes.size() / static_cast<std::size_t>(n);
        std::vector<std::byte> reordered(column.bytes.size());
        for (int i = 0; i < n; ++i)
            if (row_bytes != 0)
                std::memcpy(
                    reordered.data() + static_cast<std::size_t>(i) * row_bytes,
                    column.bytes.data()
                        + static_cast<std::size_t>(permutation[i]) * row_bytes,
                    row_bytes);
        column.bytes.swap(reordered);
    }
}

} // namespace

#ifdef ATV_HAS_MPI

GatheredFrame gather_to_root(const cpu::Frame& frame,
                             const DomainDecomposition& dd, int root) {
    GatheredFrame out;
    out.root_rank = root;
    out.box = frame.box.cell();

    if (dd.cart_comm == MPI_COMM_NULL) {
        copy_owners_into(out, frame, dd.n_local);
        return out;
    }

    int world_size = 0;
    int rank = 0;
    MPI_Comm_size(dd.cart_comm, &world_size);
    MPI_Comm_rank(dd.cart_comm, &rank);
    int root_min = 0;
    int root_max = 0;
    MPI_Allreduce(&root, &root_min, 1, MPI_INT, MPI_MIN, dd.cart_comm);
    MPI_Allreduce(&root, &root_max, 1, MPI_INT, MPI_MAX, dd.cart_comm);
    if (root_min != root_max || root < 0 || root >= world_size)
        throw std::invalid_argument(
            "gather_to_root: root rank is invalid or differs across ranks");
    if (world_size <= 1) {
        copy_owners_into(out, frame, dd.n_local);
        return out;
    }

    const std::vector<ColumnRef> fields = migrated_columns(frame);
    comm::require_collective_fingerprint(
        comm::kAtomWireVersion, comm::column_schema_fingerprint(fields),
        dd.cart_comm, "gather_to_root");

    const int local_range_valid =
        dd.n_local >= 0 && dd.n_local <= frame.capacity();
    int all_ranges_valid = 0;
    MPI_Allreduce(&local_range_valid, &all_ranges_valid, 1, MPI_INT, MPI_MIN,
                  dd.cart_comm);
    if (!all_ranges_valid)
        throw std::out_of_range(
            "gather_to_root: owner range is outside Frame capacity");

    const std::int64_t local_count = dd.n_local;
    std::int64_t global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT64_T, MPI_SUM,
                  dd.cart_comm);
    if (global_count < 0 || global_count > std::numeric_limits<int>::max() / 3)
        throw std::length_error("gather_to_root: global atom count overflow");
    const int n_total = static_cast<int>(global_count);
    out.n_total = n_total;

    std::vector<int> counts;
    if (rank == root)
        counts.resize(static_cast<std::size_t>(world_size));
    const int n_local = dd.n_local;
    MPI_Gather(&n_local, 1, MPI_INT, rank == root ? counts.data() : nullptr, 1,
               MPI_INT, root, dd.cart_comm);

    std::vector<int> displacements;
    if (rank == root) {
        displacements.resize(static_cast<std::size_t>(world_size));
        int cursor = 0;
        for (int r = 0; r < world_size; ++r) {
            displacements[static_cast<std::size_t>(r)] = cursor;
            cursor += counts[static_cast<std::size_t>(r)];
        }
        if (cursor != n_total)
            throw std::runtime_error(
                "gather_to_root: owner-count disagreement");
        out.x.resize(n_total);
        out.y.resize(n_total);
        out.z.resize(n_total);
        out.vx.resize(n_total);
        out.vy.resize(n_total);
        out.vz.resize(n_total);
        out.fx.resize(n_total);
        out.fy.resize(n_total);
        out.fz.resize(n_total);
        out.mass.resize(n_total);
        out.charge.resize(n_total);
        out.type.resize(n_total);
        out.tag.resize(n_total);
        out.image.resize(static_cast<std::size_t>(n_total) * 3);
    }

    const MPI_Datatype storage_mpi =
        std::is_same_v<StorageReal, float> ? MPI_FLOAT : MPI_DOUBLE;
    const MPI_Datatype tag_mpi =
        std::is_same_v<TagInt, std::int32_t> ? MPI_INT32_T : MPI_INT64_T;

    auto gather_native = [&](const auto* source, auto& destination,
                             MPI_Datatype dtype) {
        using T = typename std::decay_t<decltype(destination)>::value_type;
        T* receive = rank == root ? destination.data() : nullptr;
        MPI_Gatherv(source, n_local, dtype, receive,
                    rank == root ? counts.data() : nullptr,
                    rank == root ? displacements.data() : nullptr, dtype, root,
                    dd.cart_comm);
    };

    gather_native(frame.span(kAtom_x).data(), out.x, storage_mpi);
    gather_native(frame.span(kAtom_y).data(), out.y, storage_mpi);
    gather_native(frame.span(kAtom_z).data(), out.z, storage_mpi);
    gather_native(frame.span(kAtom_vx).data(), out.vx, storage_mpi);
    gather_native(frame.span(kAtom_vy).data(), out.vy, storage_mpi);
    gather_native(frame.span(kAtom_vz).data(), out.vz, storage_mpi);
    gather_native(frame.span(kAtom_fx).data(), out.fx, MPI_DOUBLE);
    gather_native(frame.span(kAtom_fy).data(), out.fy, MPI_DOUBLE);
    gather_native(frame.span(kAtom_fz).data(), out.fz, MPI_DOUBLE);
    gather_native(frame.span(kAtom_mass).data(), out.mass, storage_mpi);
    gather_native(frame.span(kAtom_charge).data(), out.charge, storage_mpi);
    gather_native(frame.span(kAtom_type).data(), out.type, MPI_INT);
    gather_native(frame.span(kAtom_tag).data(), out.tag, tag_mpi);

    std::vector<int> image_counts;
    std::vector<int> image_displacements;
    if (rank == root) {
        image_counts.resize(static_cast<std::size_t>(world_size));
        image_displacements.resize(static_cast<std::size_t>(world_size));
        for (int r = 0; r < world_size; ++r) {
            image_counts[static_cast<std::size_t>(r)] =
                3 * counts[static_cast<std::size_t>(r)];
            image_displacements[static_cast<std::size_t>(r)] =
                3 * displacements[static_cast<std::size_t>(r)];
        }
    }
    MPI_Gatherv(frame.span(kAtom_image).data(), 3 * n_local, MPI_INT8_T,
                rank == root ? out.image.data() : nullptr,
                rank == root ? image_counts.data() : nullptr,
                rank == root ? image_displacements.data() : nullptr, MPI_INT8_T,
                root, dd.cart_comm);

    for (const ColumnRef& field : fields) {
        if (field.wire_id != kDynamicFieldId)
            continue;
        comm::require_collective_fingerprint(
            comm::kAtomWireVersion,
            static_cast<std::uint64_t>(field.row_bytes()), dd.cart_comm,
            "gather_to_root/runtime-width");
        const std::size_t row_bytes = field.row_bytes();
        const std::size_t local_bytes =
            checked_product(static_cast<std::size_t>(n_local), row_bytes,
                            "gather/runtime-local");
        const std::size_t total_bytes =
            checked_product(static_cast<std::size_t>(n_total), row_bytes,
                            "gather/runtime-global");
        const int send_count =
            comm::checked_mpi_byte_count(local_bytes, "gather/runtime-send");
        if (total_bytes
            > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::length_error("gather: runtime payload exceeds INT_MAX");

        std::vector<int> byte_counts;
        std::vector<int> byte_displacements;
        GatheredColumn* target = nullptr;
        if (rank == root) {
            byte_counts.resize(static_cast<std::size_t>(world_size));
            byte_displacements.resize(static_cast<std::size_t>(world_size));
            for (int r = 0; r < world_size; ++r) {
                byte_counts[static_cast<std::size_t>(r)] =
                    comm::checked_mpi_byte_count(
                        checked_product(static_cast<std::size_t>(counts[r]),
                                        row_bytes, "gather/runtime-rank"),
                        "gather/runtime-rank");
                byte_displacements[static_cast<std::size_t>(r)] =
                    comm::checked_mpi_byte_count(
                        checked_product(
                            static_cast<std::size_t>(displacements[r]),
                            row_bytes, "gather/runtime-displacement"),
                        "gather/runtime-displacement");
            }
            out.runtime_columns.push_back(
                GatheredColumn{std::string(field.key), field.dtype, field.cols,
                               field.group, field.entries_per_item, field.flags,
                               std::vector<std::byte>(total_bytes)});
            target = &out.runtime_columns.back();
        }
        MPI_Gatherv(field.bytes(), send_count, MPI_BYTE,
                    target ? target->bytes.data() : nullptr,
                    rank == root ? byte_counts.data() : nullptr,
                    rank == root ? byte_displacements.data() : nullptr,
                    MPI_BYTE, root, dd.cart_comm);
    }

    if (rank == root)
        sort_by_tag(out);
    return out;
}

#else // !ATV_HAS_MPI

GatheredFrame gather_to_root(const cpu::Frame& frame,
                             const DomainDecomposition& dd, int root) {
    GatheredFrame out;
    out.root_rank = root;
    copy_owners_into(out, frame, dd.n_local);
    return out;
}

#endif // ATV_HAS_MPI

cpu::Frame to_frame(const GatheredFrame& gathered) {
    if (gathered.x.size()
        > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("to_frame: gathered atom count overflow");
    const int n = static_cast<int>(gathered.x.size());
    auto require_rows = [&](std::size_t size, const char* key) {
        if (size != static_cast<std::size_t>(n))
            throw std::invalid_argument(std::string("to_frame: column '") + key
                                        + "' has inconsistent rows");
    };
    require_rows(gathered.y.size(), "y");
    require_rows(gathered.z.size(), "z");
    require_rows(gathered.vx.size(), "vx");
    require_rows(gathered.vy.size(), "vy");
    require_rows(gathered.vz.size(), "vz");
    require_rows(gathered.fx.size(), "fx");
    require_rows(gathered.fy.size(), "fy");
    require_rows(gathered.fz.size(), "fz");
    require_rows(gathered.mass.size(), "mass");
    require_rows(gathered.charge.size(), "charge");
    require_rows(gathered.type.size(), "type");
    require_rows(gathered.tag.size(), "tag");
    if (gathered.image.size() != static_cast<std::size_t>(n) * 3)
        throw std::invalid_argument("to_frame: image has inconsistent rows");

    cpu::Frame frame(n);
    frame.box.set_cell(gathered.box);
    for (const GatheredColumn& column : gathered.runtime_columns) {
        frame.block("atoms").insert(
            column.key,
            ColumnSpec{column.dtype, column.cols, column.group,
                       column.entries_per_item, column.flags, kDynamicFieldId});
    }

    std::copy(gathered.x.begin(), gathered.x.end(), frame.span(kAtom_x).data());
    std::copy(gathered.y.begin(), gathered.y.end(), frame.span(kAtom_y).data());
    std::copy(gathered.z.begin(), gathered.z.end(), frame.span(kAtom_z).data());
    std::copy(gathered.vx.begin(), gathered.vx.end(),
              frame.span(kAtom_vx).data());
    std::copy(gathered.vy.begin(), gathered.vy.end(),
              frame.span(kAtom_vy).data());
    std::copy(gathered.vz.begin(), gathered.vz.end(),
              frame.span(kAtom_vz).data());
    std::copy(gathered.fx.begin(), gathered.fx.end(),
              frame.span(kAtom_fx).data());
    std::copy(gathered.fy.begin(), gathered.fy.end(),
              frame.span(kAtom_fy).data());
    std::copy(gathered.fz.begin(), gathered.fz.end(),
              frame.span(kAtom_fz).data());
    std::copy(gathered.mass.begin(), gathered.mass.end(),
              frame.span(kAtom_mass).data());
    std::copy(gathered.charge.begin(), gathered.charge.end(),
              frame.span(kAtom_charge).data());
    std::copy(gathered.type.begin(), gathered.type.end(),
              frame.span(kAtom_type).data());
    std::copy(gathered.tag.begin(), gathered.tag.end(),
              frame.span(kAtom_tag).data());
    std::copy(gathered.image.begin(), gathered.image.end(),
              frame.span(kAtom_image).data());

    auto refs = frame.block("atoms").resolve_columns();
    for (const GatheredColumn& column : gathered.runtime_columns) {
        auto it =
            std::find_if(refs.begin(), refs.end(), [&](const ColumnRef& ref) {
                return ref.wire_id == kDynamicFieldId && ref.key == column.key;
            });
        if (it == refs.end())
            throw column_error("to_frame: missing runtime column '" + column.key
                               + "'");
        const std::size_t expected =
            checked_product(static_cast<std::size_t>(n), it->row_bytes(),
                            "to_frame/runtime-column");
        if (column.bytes.size() != expected)
            throw std::invalid_argument(
                "to_frame: runtime-column payload size mismatch");
        if (expected != 0)
            std::memcpy(it->mutable_bytes(), column.bytes.data(), expected);
    }
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, n);
    return frame;
}

} // namespace mpi
} // namespace md
} // namespace cpu
} // namespace ATV

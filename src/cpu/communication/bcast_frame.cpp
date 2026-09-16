/// @file cpu/communication/bcast_frame.cpp
/// @brief Whole-Frame broadcast: native column-major atom payload plus a
///        typed-meta byte stream.

#include "atomiverse/cpu/communication/bcast_frame.h"

#include "atomiverse/cpu/frame/frame.h"

#ifdef ATV_HAS_MPI

#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/communication/atom_pack.h"
#include "atomiverse/cpu/communication/schema_mpi.h"

namespace ATV {
namespace cpu {
namespace comm {

namespace {

inline constexpr std::uint32_t kBcastWireVersion = 1;
inline constexpr std::uint32_t kMetaWireVersion = 1;

class ByteWriter {
  public:
    template <typename T>
    void pod(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        bytes(&value, sizeof(T));
    }

    void bytes(const void* source, std::size_t count) {
        const auto* first = static_cast<const std::byte*>(source);
        data_.insert(data_.end(), first, first + count);
    }

    std::vector<std::byte> take() { return std::move(data_); }

  private:
    std::vector<std::byte> data_;
};

class ByteReader {
  public:
    explicit ByteReader(const std::vector<std::byte>& data)
        : data_(data.data()), size_(data.size()) {}

    template <typename T>
    T pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value{};
        std::memcpy(&value, take(sizeof(T)), sizeof(T));
        return value;
    }

    const std::byte* take(std::size_t count) {
        if (count > size_ - offset_)
            throw std::length_error(
                "bcast_frame: typed-meta payload truncated");
        const std::byte* result = data_ + offset_;
        offset_ += count;
        return result;
    }

    bool done() const noexcept { return offset_ == size_; }

  private:
    const std::byte* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t offset_ = 0;
};

template <typename T, std::size_t N>
void write_array(ByteWriter& writer, const std::array<T, N>& values) {
    writer.bytes(values.data(), N * sizeof(T));
}

template <typename T, std::size_t N>
std::array<T, N> read_array(ByteReader& reader) {
    std::array<T, N> values{};
    std::memcpy(values.data(), reader.take(N * sizeof(T)), N * sizeof(T));
    return values;
}

void write_meta_value(ByteWriter& writer, MetaDType dtype,
                      const MetaValue& value) {
    switch (dtype) {
    case MetaDType::Bool:
        writer.pod<std::uint8_t>(std::get<bool>(value) ? 1u : 0u);
        break;
    case MetaDType::I32:
        writer.pod(std::get<std::int32_t>(value));
        break;
    case MetaDType::I64:
        writer.pod(std::get<std::int64_t>(value));
        break;
    case MetaDType::U32:
        writer.pod(std::get<std::uint32_t>(value));
        break;
    case MetaDType::U64:
        writer.pod(std::get<std::uint64_t>(value));
        break;
    case MetaDType::F32:
        writer.pod(std::get<float>(value));
        break;
    case MetaDType::F64:
        writer.pod(std::get<double>(value));
        break;
    case MetaDType::String: {
        const auto& text = std::get<std::string>(value);
        writer.pod(static_cast<std::uint64_t>(text.size()));
        writer.bytes(text.data(), text.size());
        break;
    }
    case MetaDType::Bool3: {
        const auto& values = std::get<MetaBool3>(value);
        for (bool item : values)
            writer.pod<std::uint8_t>(item ? 1u : 0u);
        break;
    }
    case MetaDType::I32x3:
        write_array(writer, std::get<MetaI32x3>(value));
        break;
    case MetaDType::I64x3:
        write_array(writer, std::get<MetaI64x3>(value));
        break;
    case MetaDType::U32x3:
        write_array(writer, std::get<MetaU32x3>(value));
        break;
    case MetaDType::U64x3:
        write_array(writer, std::get<MetaU64x3>(value));
        break;
    case MetaDType::F32x3:
        write_array(writer, std::get<MetaF32x3>(value));
        break;
    case MetaDType::F64x3:
        write_array(writer, std::get<MetaF64x3>(value));
        break;
    case MetaDType::F32x6:
        write_array(writer, std::get<MetaF32x6>(value));
        break;
    case MetaDType::F64x6:
        write_array(writer, std::get<MetaF64x6>(value));
        break;
    case MetaDType::F32x9:
        write_array(writer, std::get<MetaF32x9>(value));
        break;
    case MetaDType::F64x9:
        write_array(writer, std::get<MetaF64x9>(value));
        break;
    case MetaDType::TagIndexLocal:
        throw meta_error(
            "bcast_frame: backend-local meta cannot be serialized");
    }
}

MetaValue read_meta_value(ByteReader& reader, MetaDType dtype) {
    switch (dtype) {
    case MetaDType::Bool:
        return MetaValue{reader.pod<std::uint8_t>() != 0};
    case MetaDType::I32:
        return MetaValue{reader.pod<std::int32_t>()};
    case MetaDType::I64:
        return MetaValue{reader.pod<std::int64_t>()};
    case MetaDType::U32:
        return MetaValue{reader.pod<std::uint32_t>()};
    case MetaDType::U64:
        return MetaValue{reader.pod<std::uint64_t>()};
    case MetaDType::F32:
        return MetaValue{reader.pod<float>()};
    case MetaDType::F64:
        return MetaValue{reader.pod<double>()};
    case MetaDType::String: {
        const std::uint64_t count = reader.pod<std::uint64_t>();
        if (count > std::numeric_limits<std::size_t>::max())
            throw std::length_error("bcast_frame: meta string is too large");
        const auto* bytes = reader.take(static_cast<std::size_t>(count));
        return MetaValue{std::string(reinterpret_cast<const char*>(bytes),
                                     static_cast<std::size_t>(count))};
    }
    case MetaDType::Bool3: {
        MetaBool3 values{};
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = reader.pod<std::uint8_t>() != 0;
        return MetaValue{values};
    }
    case MetaDType::I32x3:
        return MetaValue{read_array<std::int32_t, 3>(reader)};
    case MetaDType::I64x3:
        return MetaValue{read_array<std::int64_t, 3>(reader)};
    case MetaDType::U32x3:
        return MetaValue{read_array<std::uint32_t, 3>(reader)};
    case MetaDType::U64x3:
        return MetaValue{read_array<std::uint64_t, 3>(reader)};
    case MetaDType::F32x3:
        return MetaValue{read_array<float, 3>(reader)};
    case MetaDType::F64x3:
        return MetaValue{read_array<double, 3>(reader)};
    case MetaDType::F32x6:
        return MetaValue{read_array<float, 6>(reader)};
    case MetaDType::F64x6:
        return MetaValue{read_array<double, 6>(reader)};
    case MetaDType::F32x9:
        return MetaValue{read_array<float, 9>(reader)};
    case MetaDType::F64x9:
        return MetaValue{read_array<double, 9>(reader)};
    case MetaDType::TagIndexLocal:
        throw meta_error("bcast_frame: received backend-local meta dtype");
    }
    throw meta_error("bcast_frame: unknown typed-meta dtype");
}

std::vector<std::byte> serialize_meta(const MetaMap& meta) {
    ByteWriter writer;
    const auto entries = meta.serializable_entries();
    if (entries.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("bcast_frame: too many typed-meta entries");
    writer.pod(static_cast<std::uint32_t>(entries.size()));
    for (const SerializableMetaRef& entry : entries) {
        if (entry.key.size() > std::numeric_limits<std::uint32_t>::max())
            throw std::length_error("bcast_frame: typed-meta key is too long");
        writer.pod(static_cast<std::uint32_t>(entry.key.size()));
        writer.bytes(entry.key.data(), entry.key.size());
        writer.pod(static_cast<std::uint8_t>(entry.dtype));
        write_meta_value(writer, entry.dtype, *entry.value);
    }
    return writer.take();
}

void deserialize_meta(const std::vector<std::byte>& payload, MetaMap& meta) {
    ByteReader reader(payload);
    const std::uint32_t count = reader.pod<std::uint32_t>();
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t key_size = reader.pod<std::uint32_t>();
        const auto* key_bytes = reader.take(key_size);
        std::string key(reinterpret_cast<const char*>(key_bytes), key_size);
        const auto dtype = static_cast<MetaDType>(reader.pod<std::uint8_t>());
        meta.upsert_serializable(std::move(key),
                                 read_meta_value(reader, dtype));
    }
    if (!reader.done())
        throw std::length_error(
            "bcast_frame: typed-meta payload has trailing bytes");
}

std::size_t checked_product(std::size_t a, std::size_t b) {
    return comm::checked_mul(a, b, "bcast_frame payload");
}

} // namespace

void bcast_frame(Frame& frame, int root) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
        return;

    int rank = 0;
    int world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world <= 1)
        return;
    int root_min = 0;
    int root_max = 0;
    MPI_Allreduce(&root, &root_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&root, &root_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (root_min != root_max || root < 0 || root >= world)
        throw std::invalid_argument(
            "bcast_frame: root rank is invalid or differs across ranks");

    // The atoms Block's own column selection, not a compiled wire plan. A
    // broadcast is a bulk column-major collective: it wants the column set
    // and none of `PackPlan`'s per-row pack verbs, and shipping whole
    // columns is both fewer MPI calls and fewer bytes touched than a
    // per-row pack loop over the same data.
    std::vector<ColumnRef> fields =
        frame.block("atoms").resolve_columns(kColExchange);
    require_collective_fingerprint(kAtomWireVersion,
                                   column_schema_fingerprint(fields),
                                   MPI_COMM_WORLD, "bcast_frame/atoms");
    require_collective_fingerprint(kMetaWireVersion,
                                   frame.meta.serializable_fingerprint(),
                                   MPI_COMM_WORLD, "bcast_frame/meta");

    std::vector<std::byte> meta_payload;
    std::array<double, 9> cell{};
    // 0 wire version, 1 atom count, 2 meta bytes, 3 meta fingerprint,
    // 4 has_bonded, 5 column count, 6..10 per-atom topology capacities.
    std::array<std::uint64_t, 11> header{};

    if (rank == root) {
        meta_payload = serialize_meta(frame.meta);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cell[static_cast<std::size_t>(3 * i + j)] =
                    frame.box.cell()(i, j);
        header = {
            kBcastWireVersion,
            static_cast<std::uint64_t>(frame.nlocal()),
            static_cast<std::uint64_t>(meta_payload.size()),
            frame.meta.serializable_fingerprint(),
            frame.has_bonded() ? 1u : 0u,
            static_cast<std::uint64_t>(fields.size()),
            static_cast<std::uint64_t>(frame.group_capacity(ColGroup::Bond)),
            static_cast<std::uint64_t>(frame.group_capacity(ColGroup::Angle)),
            static_cast<std::uint64_t>(
                frame.group_capacity(ColGroup::Dihedral)),
            static_cast<std::uint64_t>(
                frame.group_capacity(ColGroup::Improper)),
            static_cast<std::uint64_t>(
                frame.group_capacity(ColGroup::Special))};
    }

    MPI_Bcast(header.data(), static_cast<int>(header.size()), MPI_UINT64_T,
              root, MPI_COMM_WORLD);
    if (header[0] != kBcastWireVersion)
        throw std::runtime_error("bcast_frame: unsupported wire version");
    if (header[1] > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::length_error(
            "bcast_frame: atom count exceeds local int range");
    if (header[2] > std::numeric_limits<std::size_t>::max())
        throw std::length_error(
            "bcast_frame: payload exceeds addressable size");
    if (header[3] != frame.meta.serializable_fingerprint())
        throw std::runtime_error(
            "bcast_frame: typed-meta schema changed during broadcast");
    if (header[5] != static_cast<std::uint64_t>(fields.size()))
        throw std::runtime_error(
            "bcast_frame: rank column counts disagree after the schema "
            "digest matched");

    const int atom_count = static_cast<int>(header[1]);
    const std::size_t meta_bytes = static_cast<std::size_t>(header[2]);
    const int meta_mpi_count =
        checked_mpi_byte_count(meta_bytes, "bcast_frame/meta");

    // The root's per-column row stride. Every rank agrees on dtype,
    // entries/item and core width (the schema digest above), but a variable
    // topology column's local reserve stride is deliberately outside that
    // agreement — so it travels explicitly and the receiver adapts.
    std::vector<std::uint64_t> wire_row_bytes(fields.size());
    if (rank == root)
        for (std::size_t i = 0; i < fields.size(); ++i)
            wire_row_bytes[i] =
                static_cast<std::uint64_t>(fields[i].row_bytes());
    if (!wire_row_bytes.empty())
        MPI_Bcast(wire_row_bytes.data(),
                  static_cast<int>(wire_row_bytes.size()), MPI_UINT64_T, root,
                  MPI_COMM_WORLD);

    if (rank != root)
        meta_payload.resize(meta_bytes);
    MPI_Bcast(cell.data(), static_cast<int>(cell.size()), MPI_DOUBLE, root,
              MPI_COMM_WORLD);
    if (meta_mpi_count != 0)
        MPI_Bcast(meta_payload.data(), meta_mpi_count, MPI_BYTE, root,
                  MPI_COMM_WORLD);

    if (rank != root) {
        deserialize_meta(meta_payload, frame.meta);
        if (frame.meta.serializable_fingerprint() != header[3])
            throw std::runtime_error(
                "bcast_frame: decoded typed-meta schema mismatch");

        Mat3 matrix{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                matrix(i, j) = cell[static_cast<std::size_t>(3 * i + j)];
        frame.box.set_cell(matrix);

        frame.set_nghost(0);
        frame.set_nlocal(0);
        frame.set_capacity(atom_count);
        // Adopt the root's topology widths where they are wider; a receiver
        // that already reserved more keeps its own stride and takes the
        // staged path below.
        static constexpr std::array<ColGroup, 5> kGroups{
            ColGroup::Bond, ColGroup::Angle, ColGroup::Dihedral,
            ColGroup::Improper, ColGroup::Special};
        for (std::size_t gi = 0; gi < kGroups.size(); ++gi)
            frame.grow_group(kGroups[gi], static_cast<int>(header[6 + gi]));
        frame.meta.set(kHasBonded, static_cast<std::int64_t>(header[4] != 0));
        // Every mutation above bumps the Block generation, so the refs have
        // to be resolved after the last of them.
        fields = frame.block("atoms").resolve_columns(kColExchange);
        if (header[5] != static_cast<std::uint64_t>(fields.size()))
            throw std::runtime_error(
                "bcast_frame: the receiving Frame's column set changed while "
                "it was being resized");
    }

    // One Bcast per column, `atom_count` rows of the ROOT's stride.
    std::vector<std::byte> staging;
    for (std::size_t i = 0; i < fields.size(); ++i) {
        const std::size_t wire_row =
            static_cast<std::size_t>(wire_row_bytes[i]);
        const std::size_t wire_bytes =
            checked_product(static_cast<std::size_t>(atom_count), wire_row);
        const int mpi_count =
            checked_mpi_byte_count(wire_bytes, "bcast_frame/atoms");
        const std::size_t local_row = fields[i].row_bytes();
        if (mpi_count == 0) {
            // The root reserved nothing for this column (an unallocated
            // topology column is legally zero-width). A receiver that
            // reserved more must not keep whatever its own rows held.
            if (rank != root && local_row != 0 && atom_count != 0)
                std::memset(
                    fields[i].mutable_bytes(), 0,
                    checked_product(static_cast<std::size_t>(atom_count),
                                    local_row));
            continue;
        }
        if (rank == root) {
            MPI_Bcast(fields[i].mutable_bytes(), mpi_count, MPI_BYTE, root,
                      MPI_COMM_WORLD);
            continue;
        }
        if (local_row == wire_row) {
            MPI_Bcast(fields[i].mutable_bytes(), mpi_count, MPI_BYTE, root,
                      MPI_COMM_WORLD);
            continue;
        }
        // Wider local reserve (a topology column this rank had already
        // grown past the root's): land the payload aside, then scatter it
        // row by row and zero the unused tail.
        staging.resize(wire_bytes);
        MPI_Bcast(staging.data(), mpi_count, MPI_BYTE, root, MPI_COMM_WORLD);
        if (local_row < wire_row)
            throw std::runtime_error(
                "bcast_frame: a topology column is narrower on this rank "
                "than on the root after adopting the root's reserve widths");
        std::byte* destination = fields[i].mutable_bytes();
        for (int row = 0; row < atom_count; ++row) {
            std::byte* out =
                destination + static_cast<std::size_t>(row) * local_row;
            std::memcpy(
                out, staging.data() + static_cast<std::size_t>(row) * wire_row,
                wire_row);
            std::memset(out + wire_row, 0, local_row - wire_row);
        }
    }

    if (rank == root)
        return;
    frame.set_nlocal(atom_count);
    frame.set_nghost(0);
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, atom_count);
}

} // namespace comm
} // namespace cpu
} // namespace ATV

#else // !ATV_HAS_MPI

namespace ATV {
namespace cpu {
namespace comm {

void bcast_frame(Frame& /*frame*/, int /*root*/) {}

} // namespace comm
} // namespace cpu
} // namespace ATV

#endif // ATV_HAS_MPI

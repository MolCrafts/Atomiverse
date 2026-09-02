/// @file src/cpu/frame/frame.cpp
/// @brief Owning Column/Block/Frame implementation (no cpu::Atom type).

#include "atomiverse/cpu/frame/frame.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <xtensor/generators/xbuilder.hpp>

namespace ATV {
namespace cpu {

// --------------------------------------------------------------------------
// Typed metadata

MetaDType meta_dtype(const MetaValue& value) noexcept {
    if (std::holds_alternative<bool>(value))
        return MetaDType::Bool;
    if (std::holds_alternative<std::int32_t>(value))
        return MetaDType::I32;
    if (std::holds_alternative<std::int64_t>(value))
        return MetaDType::I64;
    if (std::holds_alternative<std::uint32_t>(value))
        return MetaDType::U32;
    if (std::holds_alternative<std::uint64_t>(value))
        return MetaDType::U64;
    if (std::holds_alternative<float>(value))
        return MetaDType::F32;
    if (std::holds_alternative<double>(value))
        return MetaDType::F64;
    if (std::holds_alternative<std::string>(value))
        return MetaDType::String;
    if (std::holds_alternative<MetaBool3>(value))
        return MetaDType::Bool3;
    if (std::holds_alternative<MetaI32x3>(value))
        return MetaDType::I32x3;
    if (std::holds_alternative<MetaI64x3>(value))
        return MetaDType::I64x3;
    if (std::holds_alternative<MetaU32x3>(value))
        return MetaDType::U32x3;
    if (std::holds_alternative<MetaU64x3>(value))
        return MetaDType::U64x3;
    if (std::holds_alternative<MetaF32x3>(value))
        return MetaDType::F32x3;
    if (std::holds_alternative<MetaF64x3>(value))
        return MetaDType::F64x3;
    if (std::holds_alternative<MetaF32x6>(value))
        return MetaDType::F32x6;
    if (std::holds_alternative<MetaF64x6>(value))
        return MetaDType::F64x6;
    if (std::holds_alternative<MetaF32x9>(value))
        return MetaDType::F32x9;
    if (std::holds_alternative<MetaF64x9>(value))
        return MetaDType::F64x9;
    return MetaDType::TagIndexLocal;
}

MetaMap::MetaMap(const MetaMap& other) : entries_(other.entries_) {}

MetaMap& MetaMap::operator=(const MetaMap& other) {
    if (this == &other)
        return *this;
    entries_ = other.entries_;
    ++generation_;
    return *this;
}

MetaMap::MetaMap(MetaMap&& other) noexcept
    : entries_(std::move(other.entries_)) {
    ++other.generation_;
}

MetaMap& MetaMap::operator=(MetaMap&& other) noexcept {
    if (this == &other)
        return *this;
    entries_ = std::move(other.entries_);
    ++generation_;
    ++other.generation_;
    return *this;
}

std::vector<SerializableMetaRef> MetaMap::serializable_entries() const {
    std::vector<SerializableMetaRef> plan;
    for (const auto& [key, entry] : entries_) {
        if (!entry.serializable)
            continue;
        const MetaDType dtype = meta_dtype(entry.value);
        if (dtype == MetaDType::TagIndexLocal)
            throw meta_error("backend-local tag index cannot be serializable");
        plan.push_back(SerializableMetaRef{key, dtype, &entry.value});
    }
    return plan;
}

std::uint64_t MetaMap::serializable_fingerprint() const {
    constexpr std::uint64_t offset = 1469598103934665603ull;
    constexpr std::uint64_t prime = 1099511628211ull;
    std::uint64_t hash = offset;
    const auto plan = serializable_entries();
    for (const auto& field : plan) {
        for (unsigned char byte : field.key) {
            hash ^= byte;
            hash *= prime;
        }
        hash ^= 0;
        hash *= prime;
        hash ^= static_cast<std::uint8_t>(field.dtype);
        hash *= prime;
    }
    return hash;
}

void MetaMap::copy_serializable_from(const MetaMap& source) {
    // Destination schema entries are invariants: a missing or differently
    // typed source entry is an exact-contract violation, never a default.
    for (auto& [key, destination] : entries_) {
        if (!destination.serializable)
            continue;
        auto source_it = source.entries_.find(key);
        if (source_it == source.entries_.end()
            || !source_it->second.serializable)
            throw meta_error("source Frame meta is missing serializable key '"
                             + key + "'");
        if (meta_dtype(destination.value)
            != meta_dtype(source_it->second.value))
            throw meta_error("Frame meta dtype mismatch for serializable key '"
                             + key + "'");
        destination.value = source_it->second.value;
    }
    // Runtime serializable metadata is part of the current source schema.
    for (const auto& [key, source_entry] : source.entries_) {
        if (!source_entry.serializable || entries_.find(key) != entries_.end())
            continue;
        if (meta_dtype(source_entry.value) == MetaDType::TagIndexLocal)
            throw meta_error("tag index cannot cross a Frame boundary");
        entries_.emplace(key, source_entry);
        ++generation_;
    }
}

void MetaMap::upsert_serializable(std::string key, MetaValue value) {
    const MetaDType incoming_dtype = meta_dtype(value);
    if (incoming_dtype == MetaDType::TagIndexLocal)
        throw meta_error(
            "backend-local tag index cannot cross a Frame boundary");

    auto it = entries_.find(key);
    if (it == entries_.end()) {
        insert_impl_(std::move(key), std::move(value), true);
        return;
    }
    if (!it->second.serializable)
        throw meta_error("Frame meta key '" + it->first
                         + "' is backend-local and cannot be restored");
    if (meta_dtype(it->second.value) != incoming_dtype)
        throw meta_error("Frame meta dtype mismatch for serializable key '"
                         + it->first + "'");
    it->second.value = std::move(value);
}

// --------------------------------------------------------------------------
// Column

Column::Column(ColumnSpec spec, int rows) : spec_(spec) {
    if (rows < 0 || spec.cols < 0)
        throw std::invalid_argument("Column dimensions must be non-negative");
    const std::size_t n = static_cast<std::size_t>(rows);
    const std::size_t c = static_cast<std::size_t>(spec.cols);
    switch (spec.dtype) {
    case ColDType::Storage:
        storage_.emplace<0>(ColumnBuffer<StorageReal, ColDType::Storage>{
            xt::zeros<StorageReal>({n, c})});
        break;
    case ColDType::Acc:
        storage_.emplace<1>(
            ColumnBuffer<AccReal, ColDType::Acc>{xt::zeros<AccReal>({n, c})});
        break;
    case ColDType::Tag:
        storage_.emplace<2>(
            ColumnBuffer<TagInt, ColDType::Tag>{xt::zeros<TagInt>({n, c})});
        break;
    case ColDType::Int:
        storage_.emplace<3>(
            ColumnBuffer<int, ColDType::Int>{xt::zeros<int>({n, c})});
        break;
    case ColDType::Int8:
        storage_.emplace<4>(ColumnBuffer<std::int8_t, ColDType::Int8>{
            xt::zeros<std::int8_t>({n, c})});
        break;
    }
    if ((spec_.flags & kColIdentityTail) && spec_.cols == 1) {
        auto& tags = std::get<2>(storage_).values;
        for (int i = 0; i < rows; ++i)
            tags(static_cast<std::size_t>(i), 0) = static_cast<TagInt>(i);
    }
}

const void* Column::data() const noexcept {
    return std::visit(
        [](const auto& buffer) -> const void* { return buffer.values.data(); },
        storage_);
}

void* Column::data() noexcept {
    return std::visit(
        [](auto& buffer) -> void* { return buffer.values.data(); }, storage_);
}

std::size_t Column::memory_bytes() const noexcept {
    return std::visit(
        [](const auto& buffer) {
            using T = typename std::decay_t<decltype(buffer)>::value_type;
            return buffer.values.size() * sizeof(T);
        },
        storage_);
}

void Column::resize_rows(int old_rows, int new_rows) {
    std::visit(
        [&](auto& buffer) {
            using T = typename std::decay_t<decltype(buffer)>::value_type;
            auto& values = buffer.values;
            using Tensor = std::decay_t<decltype(values)>;
            Tensor fresh = xt::zeros<T>({static_cast<std::size_t>(new_rows),
                                         static_cast<std::size_t>(spec_.cols)});
            const int keep = std::min(old_rows, new_rows);
            const std::size_t bytes = static_cast<std::size_t>(keep)
                                      * static_cast<std::size_t>(spec_.cols)
                                      * sizeof(T);
            if (bytes != 0)
                std::memcpy(fresh.data(), values.data(), bytes);
            if ((spec_.flags & kColIdentityTail) && spec_.cols == 1) {
                for (int i = keep; i < new_rows; ++i)
                    fresh(static_cast<std::size_t>(i), 0) = static_cast<T>(i);
            }
            values = std::move(fresh);
        },
        storage_);
}

void Column::restride(int rows, int new_cols) {
    if (new_cols < 0)
        throw std::invalid_argument("Column stride must be non-negative");
    const int old_cols = spec_.cols;
    if (new_cols == old_cols)
        return;
    std::visit(
        [&](auto& buffer) {
            using T = typename std::decay_t<decltype(buffer)>::value_type;
            auto& values = buffer.values;
            using Tensor = std::decay_t<decltype(values)>;
            Tensor fresh = xt::zeros<T>({static_cast<std::size_t>(rows),
                                         static_cast<std::size_t>(new_cols)});
            const int keep_cols = std::min(old_cols, new_cols);
            for (int row = 0; row < rows; ++row) {
                if (keep_cols != 0)
                    std::memcpy(
                        fresh.data() + static_cast<std::size_t>(row) * new_cols,
                        values.data()
                            + static_cast<std::size_t>(row) * old_cols,
                        static_cast<std::size_t>(keep_cols) * sizeof(T));
            }
            values = std::move(fresh);
        },
        storage_);
    spec_.cols = new_cols;
}

void Column::copy_row(int dst, int src) {
    if (dst == src || spec_.cols == 0)
        return;
    const std::size_t bytes =
        static_cast<std::size_t>(spec_.cols) * element_size();
    auto* base = static_cast<std::byte*>(data());
    std::memmove(base + static_cast<std::size_t>(dst) * bytes,
                 base + static_cast<std::size_t>(src) * bytes, bytes);
}

void Column::copy_row_from(const Column& source, int dst, int src) {
    if (spec_.cols == 0)
        return;
    const std::size_t bytes =
        static_cast<std::size_t>(spec_.cols) * element_size();
    // memmove, like copy_row: a caller is free to pass the same Block on
    // both sides, and a same-row copy would otherwise be an overlap.
    std::memmove(static_cast<std::byte*>(data())
                     + static_cast<std::size_t>(dst) * bytes,
                 static_cast<const std::byte*>(source.data())
                     + static_cast<std::size_t>(src) * bytes,
                 bytes);
}

// --------------------------------------------------------------------------
// Block

Block::Block(const Block& other)
    : columns_(other.columns_), rows_(other.rows_),
      resolution_state_(std::make_shared<ResolutionState>()) {}

Block& Block::operator=(const Block& other) {
    if (this == &other)
        return *this;
    columns_ = other.columns_;
    rows_ = other.rows_;
    bump_generation_();
    return *this;
}

Block::Block(Block&& other) noexcept
    : columns_(std::move(other.columns_)), rows_(other.rows_) {
    other.bump_generation_();
    other.rows_ = 0;
}

Block& Block::operator=(Block&& other) noexcept {
    if (this == &other)
        return *this;
    columns_ = std::move(other.columns_);
    rows_ = other.rows_;
    bump_generation_();
    other.bump_generation_();
    other.rows_ = 0;
    return *this;
}

void Block::insert(std::string key, ColumnSpec spec) {
    if (key.empty())
        throw std::invalid_argument("Block column key must not be empty");
    if (spec.cols < 0 || (spec.cols == 0 && !(spec.flags & kColTopology)))
        throw std::invalid_argument(
            "zero-width columns are reserved for unallocated topology");
    auto [it, inserted] = columns_.emplace(std::move(key), Column(spec, rows_));
    if (!inserted)
        throw column_error("duplicate Frame column '" + it->first + "'");
    bump_generation_();
}

void Block::erase(std::string_view key) {
    auto it = columns_.find(key);
    if (it == columns_.end())
        throw column_error("Frame block has no column '" + std::string(key)
                           + "'");
    columns_.erase(it);
    bump_generation_();
}

void Block::set_rows(int new_rows) {
    if (new_rows < 0)
        throw std::invalid_argument("Block row count must be non-negative");
    if (new_rows == rows_)
        return;
    for (auto& [key, column] : columns_) {
        (void)key;
        column.resize_rows(rows_, new_rows);
    }
    rows_ = new_rows;
    bump_generation_();
}

void Block::restride(std::string_view key, int new_cols) {
    auto it = columns_.find(key);
    if (it == columns_.end())
        throw column_error("Frame block has no column '" + std::string(key)
                           + "'");
    if (it->second.cols() == new_cols)
        return;
    it->second.restride(rows_, new_cols);
    bump_generation_();
}

void Block::copy_row(int dst, int src) {
    if (dst < 0 || src < 0 || dst >= rows_ || src >= rows_)
        throw std::out_of_range("Block::copy_row row is outside capacity");
    for (auto& [key, column] : columns_) {
        (void)key;
        if (column.flags() & kColMigrate)
            column.copy_row(dst, src);
    }
}

void Block::copy_row_from(const Block& source, int dst, int src) {
    if (dst < 0 || dst >= rows_)
        throw std::out_of_range(
            "Block::copy_row_from destination row is outside capacity");
    if (src < 0 || src >= source.rows_)
        throw std::out_of_range(
            "Block::copy_row_from source row is outside the source Block");
    for (auto& [key, column] : columns_) {
        if (!(column.flags() & kColMigrate))
            continue;
        const auto it = source.columns_.find(key);
        if (it == source.columns_.end())
            throw column_error(
                "Block::copy_row_from: the source Block has no column '" + key
                + "'. Both Blocks must carry the same migrating columns; "
                  "build the destination with Frame::clone_structure().");
        const Column& from = it->second;
        if (from.dtype() != column.dtype() || from.cols() != column.cols())
            throw column_error(
                "Block::copy_row_from: column '" + key
                + "' has a different dtype or stride in the source Block. "
                  "Rebuild the destination with Frame::clone_structure() "
                  "after the source's schema or topology width changed.");
        column.copy_row_from(from, dst, src);
    }
}

std::size_t Block::memory_bytes() const noexcept {
    std::size_t bytes = 0;
    for (const auto& [key, column] : columns_)
        bytes += key.capacity() + column.memory_bytes();
    return bytes;
}

Block Block::clone_structure(int rows) const {
    if (rows < 0)
        throw std::invalid_argument(
            "Block clone row count must be non-negative");
    Block out(rows);
    for (const auto& [key, column] : columns_)
        out.insert(key, column.spec());
    return out;
}

namespace {

bool ref_order(const ColumnRef& a, const ColumnRef& b) {
    const bool ad = a.wire_id == kDynamicFieldId;
    const bool bd = b.wire_id == kDynamicFieldId;
    if (ad != bd)
        return !ad;
    if (!ad)
        return a.wire_id < b.wire_id;
    return a.key < b.key;
}

} // namespace

ColumnRef Block::resolve_column(const GhostField& field) {
    note_resolution_();
    const auto it = columns_.find(field.name);
    if (it == columns_.end())
        throw column_error(
            "the wire needs column '" + std::string(field.name)
            + "' (required by " + std::string(field.required_by)
            + ") but this Frame's atoms Block does not carry it. The "
              "Driver validates the Frame against the ghost contract at "
              "build(); reaching here means the Frame changed shape "
              "afterwards.");
    Column& column = it->second;
    return ColumnRef{it->first,      column.dtype(),
                     column.group(), column.entries_per_item(),
                     column.flags(), column.wire_id(),
                     column.data(),  column.data(),
                     column.cols(),  column.element_size(),
                     true,           &generation_,
                     generation_};
}

ColumnRef Block::resolve_column(const GhostField& field) const {
    note_resolution_();
    const auto it = columns_.find(field.name);
    if (it == columns_.end())
        throw column_error(
            "the wire needs column '" + std::string(field.name)
            + "' (required by " + std::string(field.required_by)
            + ") but this Frame's atoms Block does not carry it.");
    const Column& column = it->second;
    return ColumnRef{it->first,      column.dtype(),
                     column.group(), column.entries_per_item(),
                     column.flags(), column.wire_id(),
                     column.data(),  nullptr,
                     column.cols(),  column.element_size(),
                     false,          const_cast<std::uint64_t*>(&generation_),
                     generation_};
}

std::vector<ColumnRef> Block::resolve_columns(unsigned required_flags) {
    note_resolution_();
    std::vector<ColumnRef> refs;
    refs.reserve(columns_.size());
    for (auto& [key, column] : columns_) {
        if ((column.flags() & required_flags) != required_flags)
            continue;
        refs.push_back(ColumnRef{key, column.dtype(), column.group(),
                                 column.entries_per_item(), column.flags(),
                                 column.wire_id(), column.data(), column.data(),
                                 column.cols(), column.element_size(), true,
                                 &generation_, generation_});
    }
    std::sort(refs.begin(), refs.end(), ref_order);
    return refs;
}

std::vector<ColumnRef> Block::resolve_columns(unsigned required_flags) const {
    note_resolution_();
    std::vector<ColumnRef> refs;
    refs.reserve(columns_.size());
    for (const auto& [key, column] : columns_) {
        if ((column.flags() & required_flags) != required_flags)
            continue;
        refs.push_back(ColumnRef{key, column.dtype(), column.group(),
                                 column.entries_per_item(), column.flags(),
                                 column.wire_id(), column.data(), nullptr,
                                 column.cols(), column.element_size(), false,
                                 &generation_, generation_});
    }
    std::sort(refs.begin(), refs.end(), ref_order);
    return refs;
}

// --------------------------------------------------------------------------
// Frame

int Frame::cols_for_(const FieldSpec& spec, const MetaMap& meta) {
    switch (spec.stride) {
    case ColStride::One:
        return 1;
    case ColStride::Fixed3:
        return 3;
    case ColStride::BondPA:
        return static_cast<int>(meta.get(kBondPerAtom));
    case ColStride::AnglePA2:
        return 2 * static_cast<int>(meta.get(kAnglePerAtom));
    case ColStride::AnglePA:
        return static_cast<int>(meta.get(kAnglePerAtom));
    case ColStride::DihedralPA3:
        return 3 * static_cast<int>(meta.get(kDihedralPerAtom));
    case ColStride::DihedralPA:
        return static_cast<int>(meta.get(kDihedralPerAtom));
    case ColStride::ImproperPA3:
        return 3 * static_cast<int>(meta.get(kImproperPerAtom));
    case ColStride::ImproperPA:
        return static_cast<int>(meta.get(kImproperPerAtom));
    case ColStride::MaxSpecial:
        return static_cast<int>(meta.get(kMaxSpecial));
    }
    return 0;
}

Frame::Frame(int n_atoms) {
    if (n_atoms < 0)
        throw std::invalid_argument("Frame atom count must be non-negative");

    meta.insert(kNLocal, static_cast<std::int64_t>(n_atoms));
    meta.insert(kNGhost, std::int64_t{0});
    meta.insert(kNMax, static_cast<std::int64_t>(n_atoms));
    meta.insert(kBondPerAtom, std::int64_t{0});
    meta.insert(kAnglePerAtom, std::int64_t{0});
    meta.insert(kDihedralPerAtom, std::int64_t{0});
    meta.insert(kImproperPerAtom, std::int64_t{0});
    meta.insert(kMaxSpecial, std::int64_t{0});
    meta.insert(kHasBonded, std::int64_t{0});
    meta.insert(kNextTag, std::int64_t{-1});
    meta.insert(kStep, std::int64_t{0});
    meta.insert(kEnergy, 0.0);
    meta.insert(kEnergyPair, 0.0);
    meta.insert(kEnergyBond, 0.0);
    meta.insert(kEnergyAngle, 0.0);
    meta.insert(kEnergyDihedral, 0.0);
    meta.insert(kEnergyImproper, 0.0);
    meta.insert(kEnergyChargingWork, 0.0);
    meta.insert(kEnergyElectronic, 0.0);
    meta.insert(kKinetic, 0.0);
    meta.insert(kConserved, 0.0);
    meta.insert(kStress, MetaF64x6{});
    meta.insert(kStressKinetic, MetaF64x6{});
    meta.insert(kTagIndex, TagIndex{});

    auto [it, inserted] = blocks.emplace("atoms", Block(n_atoms));
    (void)inserted;
    for (const FieldSpec& field : kAtomFieldSpecs) {
        it->second.insert(field.key,
                          ColumnSpec{field.dtype, cols_for_(field, meta),
                                     field.group, field.epi, field.flags,
                                     field.wire_id});
    }
    bind_resolution_state_();
}

Frame::Frame(const Frame& other)
    : blocks(other.blocks), meta(other.meta), box(other.box) {
    bind_resolution_state_();
}

Frame& Frame::operator=(const Frame& other) {
    if (this == &other)
        return *this;
    blocks = other.blocks;
    meta = other.meta;
    box = other.box;
    bind_resolution_state_();
    return *this;
}

Frame::Frame(Frame&& other)
    : blocks(std::move(other.blocks)), meta(std::move(other.meta)),
      box(std::move(other.box)) {
    bind_resolution_state_();
}

Frame& Frame::operator=(Frame&& other) {
    if (this == &other)
        return *this;
    blocks = std::move(other.blocks);
    meta = std::move(other.meta);
    box = std::move(other.box);
    bind_resolution_state_();
    return *this;
}

void Frame::bind_resolution_state_() const noexcept {
    for (const auto& [key, value] : blocks) {
        (void)key;
        value.bind_resolution_state_(meta.resolution_state());
    }
}

Block& Frame::block(std::string_view key) {
    auto it = blocks.find(key);
    if (it == blocks.end())
        throw column_error("Frame has no block '" + std::string(key) + "'");
    it->second.bind_resolution_state_(meta.resolution_state());
    return it->second;
}

const Block& Frame::block(std::string_view key) const {
    auto it = blocks.find(key);
    if (it == blocks.end())
        throw column_error("Frame has no block '" + std::string(key) + "'");
    it->second.bind_resolution_state_(meta.resolution_state());
    return it->second;
}

bool Frame::has(std::string_view block_key, std::string_view column_key) const {
    auto it = blocks.find(block_key);
    if (it == blocks.end()) {
        meta.resolution_state()->note();
        return false;
    }
    it->second.bind_resolution_state_(meta.resolution_state());
    return it->second.contains(column_key);
}

void Frame::begin_resolution_window(std::size_t budget) const noexcept {
    meta.resolution_state()->begin(budget);
}

void Frame::end_resolution_window(const char* component) const {
    const ResolutionWindow window = meta.resolution_state()->snapshot();
    meta.resolution_state()->deactivate();
    if (window.count <= window.budget)
        return;
    throw std::logic_error(
        "Frame resolution budget exceeded by component '"
        + std::string(component == nullptr ? "<unknown>" : component)
        + "': " + std::to_string(window.count) + " lookups > "
        + std::to_string(window.budget));
}

std::size_t Frame::resolution_count() const noexcept {
    return meta.resolution_state()->snapshot().count;
}

ResolutionWindow Frame::resolution_window() const noexcept {
    return meta.resolution_state()->snapshot();
}

void Frame::restore_resolution_window(ResolutionWindow window) const noexcept {
    meta.resolution_state()->restore(window);
}

void Frame::set_nlocal(int n) {
    if (n < 0 || n + nghost() > capacity())
        throw std::out_of_range("Frame nlocal violates capacity");
    meta.set(kNLocal, static_cast<std::int64_t>(n));
}

void Frame::set_nghost(int n) {
    if (n < 0 || nlocal() + n > capacity())
        throw std::out_of_range("Frame nghost violates capacity");
    meta.set(kNGhost, static_cast<std::int64_t>(n));
}

void Frame::grow(int target_capacity) {
    if (target_capacity <= capacity())
        return;
    const int old = capacity();
    const int next =
        old == 0 ? std::max(target_capacity, 64)
                 : std::max(target_capacity, old + std::max(old / 8, 4096));
    set_capacity(next);
}

void Frame::set_capacity(int new_capacity) {
    if (new_capacity < n_total())
        throw std::invalid_argument(
            "Frame capacity cannot be smaller than live owner+ghost rows");
    if (new_capacity == capacity())
        return;
    atoms_block_().set_rows(new_capacity);
    meta.set(kNMax, static_cast<std::int64_t>(new_capacity));
}

void Frame::copy_row(int dst, int src) { atoms_block_().copy_row(dst, src); }

void Frame::copy_row_from(const Frame& source, int dst, int src) {
    atoms_block_().copy_row_from(source.atoms_block_(), dst, src);
}

void Frame::ensure_bonded_storage() { meta.set(kHasBonded, std::int64_t{1}); }

void Frame::grow_bond_per_atom(int target) {
    const int old = static_cast<int>(meta.get(kBondPerAtom));
    if (target <= old)
        return;
    ensure_bonded_storage();
    auto& atoms = atoms_block_();
    atoms.restride("bond_atom", target);
    atoms.restride("bond_type", target);
    meta.set(kBondPerAtom, static_cast<std::int64_t>(target));
}

void Frame::grow_angle_per_atom(int target) {
    const int old = static_cast<int>(meta.get(kAnglePerAtom));
    if (target <= old)
        return;
    ensure_bonded_storage();
    auto& atoms = atoms_block_();
    atoms.restride("angle_atom", 2 * target);
    atoms.restride("angle_type", target);
    meta.set(kAnglePerAtom, static_cast<std::int64_t>(target));
}

void Frame::grow_dihedral_per_atom(int target) {
    const int old = static_cast<int>(meta.get(kDihedralPerAtom));
    if (target <= old)
        return;
    ensure_bonded_storage();
    auto& atoms = atoms_block_();
    atoms.restride("dihedral_atom", 3 * target);
    atoms.restride("dihedral_type", target);
    meta.set(kDihedralPerAtom, static_cast<std::int64_t>(target));
}

void Frame::grow_improper_per_atom(int target) {
    const int old = static_cast<int>(meta.get(kImproperPerAtom));
    if (target <= old)
        return;
    ensure_bonded_storage();
    auto& atoms = atoms_block_();
    atoms.restride("improper_atom", 3 * target);
    atoms.restride("improper_type", target);
    meta.set(kImproperPerAtom, static_cast<std::int64_t>(target));
}

void Frame::grow_maxspecial(int target) {
    const int old = static_cast<int>(meta.get(kMaxSpecial));
    if (target <= old)
        return;
    ensure_bonded_storage();
    atoms_block_().restride("special", target);
    meta.set(kMaxSpecial, static_cast<std::int64_t>(target));
}

void Frame::reserve_topology(int bond, int angle, int dihedral, int special_n) {
    reserve_topology(bond, angle, dihedral, /*improper=*/0, special_n);
}

void Frame::reserve_topology(int bond, int angle, int dihedral, int improper,
                             int special_n) {
    ensure_bonded_storage();
    grow_bond_per_atom(bond);
    grow_angle_per_atom(angle);
    grow_dihedral_per_atom(dihedral);
    grow_improper_per_atom(improper);
    grow_maxspecial(special_n);
}

int Frame::group_capacity(ColGroup group) const {
    switch (group) {
    case ColGroup::Bond:
        return static_cast<int>(meta.get(kBondPerAtom));
    case ColGroup::Angle:
        return static_cast<int>(meta.get(kAnglePerAtom));
    case ColGroup::Dihedral:
        return static_cast<int>(meta.get(kDihedralPerAtom));
    case ColGroup::Improper:
        return static_cast<int>(meta.get(kImproperPerAtom));
    case ColGroup::Special:
        return static_cast<int>(meta.get(kMaxSpecial));
    case ColGroup::Core:
        return 0;
    }
    return 0;
}

void Frame::grow_group(ColGroup group, int count) {
    switch (group) {
    case ColGroup::Bond:
        grow_bond_per_atom(count);
        break;
    case ColGroup::Angle:
        grow_angle_per_atom(count);
        break;
    case ColGroup::Dihedral:
        grow_dihedral_per_atom(count);
        break;
    case ColGroup::Improper:
        grow_improper_per_atom(count);
        break;
    case ColGroup::Special:
        grow_maxspecial(count);
        break;
    case ColGroup::Core:
        break;
    }
}

TagInt Frame::acquire_tag() {
    auto next = meta.ref(kNextTag);
    if (next.get() < 0) {
        const TagInt* tags = span(kAtom_tag).data();
        TagInt top = -1;
        for (int i = 0; i < n_total(); ++i)
            top = std::max(top, tags[i]);
        next.get() = static_cast<std::int64_t>(top + 1);
    }
    return static_cast<TagInt>(next.get()++);
}

void Frame::clear_tag_index() { meta.get(kTagIndex).clear(); }

void Frame::release_tag_index() {
    TagIndex fresh;
    meta.get(kTagIndex).swap(fresh);
}

void Frame::rebuild_tag_index(int begin, int end) {
    if (begin < 0 || end < begin || end > n_total())
        throw std::out_of_range("Frame tag-index range is invalid");
    auto& index = meta.get(kTagIndex);
    const TagInt* tags = span(kAtom_tag).data();
    index.reserve(index.size() + static_cast<std::size_t>(end - begin));
    for (int i = begin; i < end; ++i)
        index.emplace(tags[i], i);
}

void Frame::add_bond(TagInt ti, TagInt tj, int type_value) {
    ensure_bonded_storage();
    const TagInt owner_tag = std::min(ti, tj);
    const TagInt partner_tag = std::max(ti, tj);
    const auto& index = meta.get(kTagIndex);
    auto it = index.find(owner_tag);
    if (it == index.end())
        return;
    const int row = it->second;
    const int slot = span(kAtom_num_bond)[static_cast<std::size_t>(row)];
    grow_bond_per_atom(slot + 1);
    auto num = span(kAtom_num_bond);
    auto partner = span(kAtom_bond_atom);
    auto kind = span(kAtom_bond_type);
    partner(row, slot) = partner_tag;
    kind(row, slot) = type_value;
    num[static_cast<std::size_t>(row)] = slot + 1;
}

void Frame::add_angle(TagInt ti, TagInt tj, TagInt tk, int type_value) {
    ensure_bonded_storage();
    const auto& index = meta.get(kTagIndex);
    auto it = index.find(tj);
    if (it == index.end())
        return;
    const int row = it->second;
    const int slot = span(kAtom_num_angle)[static_cast<std::size_t>(row)];
    grow_angle_per_atom(slot + 1);
    auto num = span(kAtom_num_angle);
    auto partner = span(kAtom_angle_atom);
    auto kind = span(kAtom_angle_type);
    partner(row, 2 * slot) = ti;
    partner(row, 2 * slot + 1) = tk;
    kind(row, slot) = type_value;
    num[static_cast<std::size_t>(row)] = slot + 1;
}

void Frame::add_dihedral(TagInt ti, TagInt tj, TagInt tk, TagInt tl,
                         int type_value) {
    ensure_bonded_storage();
    const auto& index = meta.get(kTagIndex);
    auto it = index.find(tj);
    if (it == index.end())
        return;
    const int row = it->second;
    const int slot = span(kAtom_num_dihedral)[static_cast<std::size_t>(row)];
    grow_dihedral_per_atom(slot + 1);
    auto num = span(kAtom_num_dihedral);
    auto partner = span(kAtom_dihedral_atom);
    auto kind = span(kAtom_dihedral_type);
    partner(row, 3 * slot) = ti;
    partner(row, 3 * slot + 1) = tk;
    partner(row, 3 * slot + 2) = tl;
    kind(row, slot) = type_value;
    num[static_cast<std::size_t>(row)] = slot + 1;
}

void Frame::add_improper(TagInt ti, TagInt tj, TagInt tk, TagInt tl,
                         int type_value) {
    ensure_bonded_storage();
    const auto& index = meta.get(kTagIndex);
    auto it = index.find(tj);
    if (it == index.end())
        return;
    const int row = it->second;
    const int slot = span(kAtom_num_improper)[static_cast<std::size_t>(row)];
    grow_improper_per_atom(slot + 1);
    auto num = span(kAtom_num_improper);
    auto partner = span(kAtom_improper_atom);
    auto kind = span(kAtom_improper_type);
    partner(row, 3 * slot) = ti;
    partner(row, 3 * slot + 1) = tk;
    partner(row, 3 * slot + 2) = tl;
    kind(row, slot) = type_value;
    num[static_cast<std::size_t>(row)] = slot + 1;
}

void Frame::build_special_list() {
    const int n = nlocal();
    if (!has_bonded() || n == 0)
        return;
    const TagInt* tags = span(kAtom_tag).data();
    const int* num_bond = span(kAtom_num_bond).data();
    const auto bonds = span(kAtom_bond_atom);
    const auto& index = meta.get(kTagIndex);

    std::vector<std::vector<int>> adjacency(static_cast<std::size_t>(n));
    for (int row = 0; row < n; ++row) {
        for (int b = 0; b < num_bond[row]; ++b) {
            auto it = index.find(bonds(row, b));
            if (it == index.end() || it->second >= n)
                continue;
            adjacency[static_cast<std::size_t>(row)].push_back(it->second);
            adjacency[static_cast<std::size_t>(it->second)].push_back(row);
        }
    }

    struct SpecialRow {
        std::vector<TagInt> values;
        int n12 = 0;
        int n13 = 0;
        int n14 = 0;
    };
    std::vector<SpecialRow> rows(static_cast<std::size_t>(n));
    int widest = 0;
    for (int root = 0; root < n; ++root) {
        std::vector<int> distance(static_cast<std::size_t>(n), -1);
        std::queue<int> pending;
        distance[static_cast<std::size_t>(root)] = 0;
        pending.push(root);
        while (!pending.empty()) {
            const int current = pending.front();
            pending.pop();
            if (distance[static_cast<std::size_t>(current)] == 3)
                continue;
            for (int next : adjacency[static_cast<std::size_t>(current)]) {
                if (distance[static_cast<std::size_t>(next)] >= 0)
                    continue;
                distance[static_cast<std::size_t>(next)] =
                    distance[static_cast<std::size_t>(current)] + 1;
                pending.push(next);
            }
        }
        auto& out = rows[static_cast<std::size_t>(root)];
        for (int depth = 1; depth <= 3; ++depth) {
            for (int row = 0; row < n; ++row)
                if (distance[static_cast<std::size_t>(row)] == depth)
                    out.values.push_back(tags[row]);
            if (depth == 1)
                out.n12 = static_cast<int>(out.values.size());
            else if (depth == 2)
                out.n13 = static_cast<int>(out.values.size());
            else
                out.n14 = static_cast<int>(out.values.size());
        }
        widest = std::max(widest, static_cast<int>(out.values.size()));
    }

    grow_maxspecial(widest);
    auto counts = span(kAtom_nspecial);
    auto values = span(kAtom_special);
    for (int row = 0; row < n; ++row) {
        const auto& src = rows[static_cast<std::size_t>(row)];
        counts(row, 0) = src.n12;
        counts(row, 1) = src.n13;
        counts(row, 2) = src.n14;
        for (std::size_t i = 0; i < src.values.size(); ++i)
            values(row, static_cast<int>(i)) = src.values[i];
    }
}

void Frame::reshuffle_topology_to_tags() {
    if (!has_bonded())
        return;
    const int n = nlocal();
    const TagInt* tags = span(kAtom_tag).data();
    std::vector<int> source(static_cast<std::size_t>(n), -1);
    for (int row = 0; row < n; ++row) {
        if (tags[row] < 0 || tags[row] >= n)
            throw std::runtime_error(
                "Frame topology reshuffle requires dense tags [0,n)");
        source[static_cast<std::size_t>(row)] = static_cast<int>(tags[row]);
    }

    auto refs = atoms_block_().resolve_columns(kColMigrate);
    struct Snapshot {
        ColumnRef ref;
        std::vector<std::byte> bytes;
    };
    std::vector<Snapshot> snapshots;
    for (const auto& ref : refs) {
        if (ref.group == ColGroup::Core)
            continue;
        Snapshot snap{ref, {}};
        snap.bytes.resize(static_cast<std::size_t>(n) * ref.row_bytes());
        if (!snap.bytes.empty())
            std::memcpy(snap.bytes.data(), ref.bytes(), snap.bytes.size());
        snapshots.push_back(std::move(snap));
    }
    for (auto& snap : snapshots) {
        const std::size_t width = snap.ref.row_bytes();
        auto* dst = snap.ref.mutable_bytes();
        for (int row = 0; row < n; ++row)
            std::memcpy(dst + static_cast<std::size_t>(row) * width,
                        snap.bytes.data()
                            + static_cast<std::size_t>(source[row]) * width,
                        width);
    }
}

std::size_t Frame::memory_bytes() const noexcept {
    std::size_t bytes = 0;
    for (const auto& [key, value] : blocks)
        bytes += key.capacity() + value.memory_bytes();
    const auto& index = meta.get(kTagIndex);
    bytes += index.size() * (sizeof(TagInt) + sizeof(int) + 2 * sizeof(void*));
    bytes += index.bucket_count() * sizeof(void*);
    for (const auto& [key, value] : meta.entries()) {
        bytes += key.capacity() + sizeof(value);
        if (const auto* text = std::get_if<std::string>(&value.value))
            bytes += text->capacity();
    }
    return bytes;
}

Frame Frame::clone_structure(int rows) const {
    if (rows < 0)
        throw std::invalid_argument(
            "Frame clone row count must be non-negative");
    Frame out(rows);
    out.box = box;

    // Match the source topology widths before copying runtime schema. The
    // canonical fixed fields already exist in the constructor.
    out.reserve_topology(static_cast<int>(meta.get(kBondPerAtom)),
                         static_cast<int>(meta.get(kAnglePerAtom)),
                         static_cast<int>(meta.get(kDihedralPerAtom)),
                         static_cast<int>(meta.get(kImproperPerAtom)),
                         static_cast<int>(meta.get(kMaxSpecial)));
    out.meta.set(kHasBonded, meta.get(kHasBonded));

    const Block& source_atoms = block("atoms");
    Block& target_atoms = out.block("atoms");
    for (const auto& [key, column] : source_atoms.columns()) {
        if (column.wire_id() == kDynamicFieldId)
            target_atoms.insert(key, column.spec());
    }
    for (const auto& [key, source_block] : blocks) {
        if (key == "atoms")
            continue;
        out.blocks.emplace(key,
                           source_block.clone_structure(source_block.rows()));
    }
    out.meta.copy_serializable_from(meta);
    out.set_nlocal(rows);
    out.set_nghost(0);
    out.meta.set(kNextTag, std::int64_t{-1});
    out.clear_tag_index();
    return out;
}

std::vector<SchemaError> validate_ghosts(const GhostContract& required,
                                         const Frame& frame) {
    std::vector<SchemaError> errors = required.conflicts();
    for (const GhostField& f : required) {
        if (!frame.has("atoms", f.name)) {
            errors.push_back(SchemaError{
                f.name, f.required_by,
                "the Frame has no such column. Add it before the run "
                "(Block::insert, or load a file that carries it), or unmount "
                "the component that needs it."});
            continue;
        }
        const auto& cols = frame.block("atoms").columns();
        const auto it = cols.find(f.name);
        if (it == cols.end())
            continue; // has() said otherwise; nothing to compare against
        const Column& col = it->second;
        if (col.dtype() != f.dtype)
            errors.push_back(SchemaError{
                f.name, f.required_by,
                "declared as " + std::string(GhostContract::dtype_name(f.dtype))
                    + " but the Frame carries "
                    + std::string(GhostContract::dtype_name(col.dtype()))});
        else if (col.cols() != f.cols)
            errors.push_back(
                SchemaError{f.name, f.required_by,
                            "declared with " + std::to_string(f.cols)
                                + " component(s) but the Frame carries "
                                + std::to_string(col.cols())});
    }
    return errors;
}

void check_ghosts(const GhostContract& required, const Frame& frame,
                  std::string_view who) {
    const std::vector<SchemaError> errors = validate_ghosts(required, frame);
    if (errors.empty())
        return;
    std::string msg = std::string(who)
                      + ": this Frame does not satisfy the "
                        "ghost contract ("
                      + std::to_string(errors.size()) + " problem(s)):";
    for (const SchemaError& e : errors) {
        msg += "\n  - " + std::string(e.field) + " (required by "
               + std::string(e.required_by) + "): " + e.reason;
    }
    throw std::runtime_error(msg);
}

} // namespace cpu
} // namespace ATV

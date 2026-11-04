/// @file src/cpu/frame/molpy_exchange.cpp
/// @brief molrs.Frame <-> canonical CPU block Frame.

#include "atomiverse/cpu/frame/molpy_exchange.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../io/molrs_meta.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "rust/cxx.h"

namespace ATV {
namespace cpu {

namespace {

bool contains(std::string_view value,
              std::initializer_list<std::string_view> allowed) {
    for (const std::string_view candidate : allowed)
        if (value == candidate)
            return true;
    return false;
}

std::string as_string(const rust::String& value) {
    return {value.data(), value.size()};
}

void validate_columns(const molrs::FrameRef& src, const char* block,
                      std::initializer_list<std::string_view> allowed) {
    if (!molrs::frame_has_block(src, block))
        return;
    for (const rust::String& raw : molrs::frame_block_columns(src, block)) {
        const std::string column = as_string(raw);
        if (!contains(column, allowed))
            throw std::runtime_error("cpu::from_molpy: unsupported column '"
                                     + std::string(block) + "." + column + "'");
    }
}

void validate_exchange_schema(const molrs::FrameRef& src) {
    for (const rust::String& raw : molrs::frame_block_names(src)) {
        const std::string block = as_string(raw);
        if (!contains(block,
                      {"atoms", "bonds", "angles", "dihedrals", "impropers"}))
            throw std::runtime_error("cpu::from_molpy: unsupported block '"
                                     + block + "'");
    }

    // molrs 0.13 split the two meanings that used to share `atoms.type`:
    // `atomic_number` (UInt) carries Z, while `type` is a String force-field
    // label the caller owns. The engine's per-atom kind is `atoms.type_id`
    // (UInt, the force-field atom kind a per-type pair table is keyed by)
    // when the frame states one, and Z from `atomic_number` otherwise;
    // `type` and `element` (the String symbol column molpy's
    // Atomistic.to_frame() exports) stay in the allow-list — a frame arriving
    // from a molrs tool may legitimately carry them — but are not imported.
    // So do the line-notation facts molpy states on a parsed or embedded
    // molecule (`formal_charge`, `h_count`, `is_aromatic`, and a bond's
    // `bond_type` / `bond_number` class): graph annotations, not physics.
    validate_columns(src, "atoms",
                     {"x",       "y",           "z",       "vx",
                      "vy",      "vz",          "fx",      "fy",
                      "fz",      "mass",        "charge",  "atomic_number",
                      "type_id", "type",        "element", "formal_charge",
                      "h_count", "is_aromatic", "tag",     "ix",
                      "iy",      "iz"});
    validate_columns(
        src, "bonds",
        {"atomi", "atomj", "type_id", "type", "bond_type", "bond_number"});
    validate_columns(src, "angles",
                     {"atomi", "atomj", "atomk", "type_id", "type"});
    validate_columns(src, "dihedrals",
                     {"atomi", "atomj", "atomk", "atoml", "type_id", "type"});
    validate_columns(src, "impropers",
                     {"atomi", "atomj", "atomk", "atoml", "type_id", "type"});
}

bool has_column(const molrs::FrameRef& src, const char* block,
                const char* key) {
    for (const rust::String& raw : molrs::frame_block_columns(src, block))
        if (as_string(raw) == key)
            return true;
    return false;
}

::rust::Vec<double> require_f64(const molrs::FrameRef& src, const char* block,
                                const char* col) {
    auto values = molrs::frame_column_f64(src, block, col);
    if (values.empty())
        throw std::runtime_error("cpu::from_molpy: required field '"
                                 + std::string(block) + "." + col
                                 + "' is missing or empty");
    return values;
}

// The molrs domain index type `Idx` is u64 (`frame_column_u64`).
::rust::Vec<std::uint64_t> require_u64(const molrs::FrameRef& src,
                                       const char* block, const char* col) {
    auto values = molrs::frame_column_u64(src, block, col);
    if (values.empty())
        throw std::runtime_error("cpu::from_molpy: required field '"
                                 + std::string(block) + "." + col
                                 + "' is missing or empty");
    return values;
}

template <typename T>
void copy_optional_f64(const molrs::FrameRef& src, const char* key,
                       Span<T> destination, int n) {
    if (!has_column(src, "atoms", key))
        return;
    auto values = molrs::frame_column_f64(src, "atoms", key);
    if (static_cast<int>(values.size()) != n)
        throw std::runtime_error("cpu::from_molpy: optional field 'atoms."
                                 + std::string(key)
                                 + "' has the wrong dtype or row count");
    T* out = destination.data();
    for (int i = 0; i < n; ++i)
        out[i] = static_cast<T>(values[i]);
}

/// @brief Read one integer topology column (`atomi`/`atomj`/... or `type_id`).
///
/// molrs binds a key's dtype globally: `type` is String everywhere, `type_id`
/// is UInt. Atomiverse's topology kinds are integers, so they travel as
/// `type_id`; the connectivity indices stay signed and are read as i32.
std::vector<int> topology_kind_column(const molrs::FrameRef& src,
                                      const char* block, int n) {
    auto values = molrs::frame_column_u64(src, block, "type_id");
    if (static_cast<int>(values.size()) != n)
        throw std::runtime_error(
            "cpu::from_molpy: topology field '" + std::string(block)
            + ".type_id' is missing or has the wrong dtype/row count");
    std::vector<int> out(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        out[static_cast<std::size_t>(i)] = static_cast<int>(values[i]);
    return out;
}

/// @brief Read one topology connectivity column (`atomi`/`atomj`/`atomk`/
///        `atoml`).
///
/// molrs declares every identifier key UInt (u64 `Idx` since 0.14), so the
/// endpoint indices are unsigned — they are row numbers into the target node
/// block and are never negative.
/// Kinds travel separately as `type_id` (`topology_kind_column`); `type` is a
/// String force-field label this boundary does not import.
std::vector<int> topology_column(const molrs::FrameRef& src, const char* block,
                                 const char* key, int n) {
    auto values = molrs::frame_column_u64(src, block, key);
    if (static_cast<int>(values.size()) != n)
        throw std::runtime_error(
            "cpu::from_molpy: topology field '" + std::string(block) + "." + key
            + "' is missing or has the wrong dtype/row count");
    std::vector<int> out(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        out[static_cast<std::size_t>(i)] = static_cast<int>(values[i]);
    return out;
}

template <typename T>
void write_f64(molrs::FrameRef& dst, const char* key, const T* source, int n) {
    std::vector<double> values(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        values[static_cast<std::size_t>(i)] = static_cast<double>(source[i]);
    molrs::frame_set_column_f64(
        dst, "atoms", key,
        ::rust::Slice<const double>(values.data(), values.size()));
}

/// @brief The body of from_molpy; its molrs reads may raise `rust::Error`
/// (a column of another dtype), which from_molpy names.
Frame materialize(const molrs::FrameRef& src) {
    validate_exchange_schema(src);
    auto x = require_f64(src, "atoms", "x");
    auto y = require_f64(src, "atoms", "y");
    auto z = require_f64(src, "atoms", "z");
    // The per-atom kind: a force-field `type_id` when stated (a GAFF frame,
    // whose LJ table distinguishes c3 from ca), else the element. Components
    // that read Z (electronic methods, the XYZ recorder) need a frame
    // without `type_id`.
    // molrs converts species only on request (Element::fill_atomic_numbers);
    // a frame that spells symbols alone is named, not silently converted.
    const bool ff_kinds = has_column(src, "atoms", "type_id");
    if (!ff_kinds && !has_column(src, "atoms", "atomic_number")
        && has_column(src, "atoms", "element"))
        throw std::runtime_error(
            "cpu::from_molpy: atoms carry 'element' symbols but no "
            "'atomic_number'; convert them first (molrs "
            "Element::fill_atomic_numbers / frame_fill_atomic_numbers)");
    auto type =
        require_u64(src, "atoms", ff_kinds ? "type_id" : "atomic_number");
    const int n = static_cast<int>(x.size());
    if (static_cast<int>(y.size()) != n || static_cast<int>(z.size()) != n
        || static_cast<int>(type.size()) != n)
        throw std::runtime_error(
            "cpu::from_molpy: required atom columns have inconsistent rows");

    Frame frame(n);
    StorageReal* px = frame.span(kAtom_x).data();
    StorageReal* py = frame.span(kAtom_y).data();
    StorageReal* pz = frame.span(kAtom_z).data();
    int* ptype = frame.span(kAtom_type).data();
    for (int i = 0; i < n; ++i) {
        px[i] = static_cast<StorageReal>(x[i]);
        py[i] = static_cast<StorageReal>(y[i]);
        pz[i] = static_cast<StorageReal>(z[i]);
        ptype[i] = static_cast<int>(type[i]);
    }

    copy_optional_f64(src, "vx", frame.span(kAtom_vx), n);
    copy_optional_f64(src, "vy", frame.span(kAtom_vy), n);
    copy_optional_f64(src, "vz", frame.span(kAtom_vz), n);
    copy_optional_f64(src, "fx", frame.span(kAtom_fx), n);
    copy_optional_f64(src, "fy", frame.span(kAtom_fy), n);
    copy_optional_f64(src, "fz", frame.span(kAtom_fz), n);
    copy_optional_f64(src, "mass", frame.span(kAtom_mass), n);
    copy_optional_f64(src, "charge", frame.span(kAtom_charge), n);

    if (has_column(src, "atoms", "tag")) {
        auto tags = molrs::frame_column_i32(src, "atoms", "tag");
        if (static_cast<int>(tags.size()) != n)
            throw std::runtime_error(
                "cpu::from_molpy: optional field 'atoms.tag' has the wrong "
                "dtype or row count");
        TagInt* destination = frame.span(kAtom_tag).data();
        for (int i = 0; i < n; ++i)
            destination[i] = static_cast<TagInt>(tags[i]);
    }

    auto image = frame.span(kAtom_image);
    const char* axes[3] = {"ix", "iy", "iz"};
    for (int d = 0; d < 3; ++d) {
        if (!has_column(src, "atoms", axes[d]))
            continue;
        auto values = molrs::frame_column_i32(src, "atoms", axes[d]);
        if (static_cast<int>(values.size()) != n)
            throw std::runtime_error("cpu::from_molpy: optional field 'atoms."
                                     + std::string(axes[d])
                                     + "' has the wrong dtype or row count");
        for (int i = 0; i < n; ++i)
            image(i, d) = static_cast<std::int8_t>(values[i]);
    }

    frame.clear_tag_index();
    frame.rebuild_tag_index(0, n);
    const int nb =
        (molrs::frame_has_block(src, "bonds")
             ? static_cast<int>(molrs::frame_block_n_rows(src, "bonds"))
             : 0);
    if (nb > 0) {
        auto atomi = topology_column(src, "bonds", "atomi", nb);
        auto atomj = topology_column(src, "bonds", "atomj", nb);
        auto kind = topology_kind_column(src, "bonds", nb);
        for (int k = 0; k < nb; ++k)
            frame.add_bond(atomi[k], atomj[k], kind[k]);
    }
    const int na =
        (molrs::frame_has_block(src, "angles")
             ? static_cast<int>(molrs::frame_block_n_rows(src, "angles"))
             : 0);
    if (na > 0) {
        auto atomi = topology_column(src, "angles", "atomi", na);
        auto atomj = topology_column(src, "angles", "atomj", na);
        auto atomk = topology_column(src, "angles", "atomk", na);
        auto kind = topology_kind_column(src, "angles", na);
        for (int k = 0; k < na; ++k)
            frame.add_angle(atomi[k], atomj[k], atomk[k], kind[k]);
    }
    const int nd =
        (molrs::frame_has_block(src, "dihedrals")
             ? static_cast<int>(molrs::frame_block_n_rows(src, "dihedrals"))
             : 0);
    if (nd > 0) {
        auto atomi = topology_column(src, "dihedrals", "atomi", nd);
        auto atomj = topology_column(src, "dihedrals", "atomj", nd);
        auto atomk = topology_column(src, "dihedrals", "atomk", nd);
        auto atoml = topology_column(src, "dihedrals", "atoml", nd);
        auto kind = topology_kind_column(src, "dihedrals", nd);
        for (int k = 0; k < nd; ++k)
            frame.add_dihedral(atomi[k], atomj[k], atomk[k], atoml[k], kind[k]);
    }
    const int ni =
        (molrs::frame_has_block(src, "impropers")
             ? static_cast<int>(molrs::frame_block_n_rows(src, "impropers"))
             : 0);
    if (ni > 0) {
        auto atomi = topology_column(src, "impropers", "atomi", ni);
        auto atomj = topology_column(src, "impropers", "atomj", ni);
        auto atomk = topology_column(src, "impropers", "atomk", ni);
        auto atoml = topology_column(src, "impropers", "atoml", ni);
        auto kind = topology_kind_column(src, "impropers", ni);
        for (int k = 0; k < ni; ++k)
            frame.add_improper(atomi[k], atomj[k], atomk[k], atoml[k], kind[k]);
    }
    frame.build_special_list();

    if (molrs::frame_has_box(src)) {
        const auto box_data = molrs::frame_box(src);
        const auto& cell_values = box_data.h;
        if (cell_values.size() == 9u) {
            for (const auto value : box_data.origin)
                if (value != 0.0)
                    throw std::runtime_error(
                        "cpu::from_molpy: nonzero box origin is unsupported; "
                        "translate positions and set origin to zero");
            for (const auto periodic : box_data.pbc)
                if (!periodic)
                    throw std::runtime_error("cpu::from_molpy: a defined cell "
                                             "requires periodicity on "
                                             "all three axes");
            Mat3 cell;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    cell(i, j) = static_cast<AccReal>(cell_values[i * 3 + j]);
            frame.box.set_cell(cell);
        }
    }
    internal::decode_molrs_meta(src, frame.meta);
    return frame;
}

} // namespace

Frame from_molpy(const molrs::FrameRef& src) {
    try {
        return materialize(src);
    } catch (const ::rust::Error& e) {
        throw std::runtime_error(std::string("cpu::from_molpy: ") + e.what());
    }
}

void to_molpy(const Frame& src, molrs::FrameRef& dst) {
    const int n = src.nlocal();
    for (const auto& col : src.block("atoms").resolve_columns(kColWriteBack)) {
        if (col.cols != 1)
            throw std::runtime_error(
                "to_molpy: kColWriteBack column must be scalar");
        const std::string key{col.key};
        if (col.dtype == ColDType::Storage)
            write_f64(dst, key.c_str(),
                      static_cast<const StorageReal*>(col.read_data), n);
        else if (col.dtype == ColDType::Acc)
            write_f64(dst, key.c_str(),
                      static_cast<const AccReal*>(col.read_data), n);
        else
            throw std::runtime_error(
                "to_molpy: kColWriteBack column has unsupported dtype");
    }

    const auto image = src.span(kAtom_image);
    const TagInt* tags = src.span(kAtom_tag).data();
    std::vector<std::int32_t> ix(static_cast<std::size_t>(n));
    std::vector<std::int32_t> iy(static_cast<std::size_t>(n));
    std::vector<std::int32_t> iz(static_cast<std::size_t>(n));
    std::vector<std::int32_t> tag_values(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        ix[i] = image(i, 0);
        iy[i] = image(i, 1);
        iz[i] = image(i, 2);
        if (tags[i] > std::numeric_limits<std::int32_t>::max()
            || tags[i] < std::numeric_limits<std::int32_t>::min())
            throw std::runtime_error("to_molpy: atom tag does not fit i32");
        tag_values[i] = static_cast<std::int32_t>(tags[i]);
    }
    auto set_i32 = [&](const char* key,
                       const std::vector<std::int32_t>& values) {
        molrs::frame_set_column_i32(
            dst, "atoms", key,
            ::rust::Slice<const std::int32_t>(values.data(), values.size()));
    };
    set_i32("ix", ix);
    set_i32("iy", iy);
    set_i32("iz", iz);
    set_i32("tag", tag_values);

    // The cell is an OUTPUT of a run under a barostat / cell relaxation, so
    // it crosses back in the same molrs row-major H order the read side uses
    // (h[i * 3 + j] == cell(i, j)). A degenerate cell (default-constructed
    // Box, det == 0) describes no periodic system and molrs rejects it as a
    // singular matrix, so it is skipped rather than pushed across.
    if (src.box.volume() > 0.0) {
        const Mat3& cell = src.box.cell();
        std::array<double, 9> h{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                h[static_cast<std::size_t>(i * 3 + j)] =
                    static_cast<double>(cell(i, j));
        molrs::frame_set_box(
            dst, ::rust::Slice<const double>(h.data(), h.size()),
            ::rust::Slice<const double>(
                std::array<double, 3>{0.0, 0.0, 0.0}.data(), 3),
            ::rust::Slice<const bool>(
                std::array<bool, 3>{true, true, true}.data(), 3));
    }

    auto entries = internal::encode_molrs_meta(src.meta);
    for (auto& entry : entries)
        molrs::frame_set_meta(dst, std::move(entry));
}

} // namespace cpu
} // namespace ATV

/// @file src/cpu/frame/molrs_exchange.cpp
/// @brief molrs.Frame <-> canonical CPU block Frame.

#include "atomiverse/cpu/frame/molrs_exchange.h"

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
#include "atomiverse/core/mat3.h"

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
    for (const rust::String& raw : molrs::frame_block_columns(src, block)) {
        const std::string column = as_string(raw);
        if (!contains(column, allowed))
            throw std::runtime_error("cpu::from_molrs: unsupported column '"
                                     + std::string(block) + "." + column + "'");
    }
}

void validate_exchange_schema(const molrs::FrameRef& src) {
    for (const rust::String& raw : molrs::frame_block_names(src)) {
        const std::string block = as_string(raw);
        if (!contains(block,
                      {"atoms", "bonds", "angles", "dihedrals", "impropers"}))
            throw std::runtime_error("cpu::from_molrs: unsupported block '"
                                     + block + "'");
    }

    // molrs 0.13 split the two meanings that used to share `atoms.type`:
    // `atomic_number` (UInt) carries Z, while `type` is a String force-field
    // label the caller owns. Atomiverse keys on Z, so it reads and writes
    // `atomic_number`; `type` and `element` (the String symbol column
    // molpy's Atomistic.to_frame() exports) stay in the allow-list — a frame
    // arriving from a molrs tool may legitimately carry them — but are not
    // imported.
    validate_columns(src, "atoms",
                     {"x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz", "mass",
                      "charge", "atomic_number", "type", "element", "tag", "ix",
                      "iy", "iz"});
    validate_columns(src, "bonds", {"atomi", "atomj", "type_id", "type"});
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
        throw std::runtime_error("cpu::from_molrs: required field '"
                                 + std::string(block) + "." + col
                                 + "' is missing or empty");
    return values;
}

::rust::Vec<std::uint32_t> require_u32(const molrs::FrameRef& src,
                                       const char* block, const char* col) {
    auto values = molrs::frame_column_u32(src, block, col);
    if (values.empty())
        throw std::runtime_error("cpu::from_molrs: required field '"
                                 + std::string(block) + "." + col
                                 + "' is missing or empty");
    return values;
}

::rust::Vec<std::int32_t> require_i32(const molrs::FrameRef& src,
                                      const char* block, const char* col) {
    auto values = molrs::frame_column_i32(src, block, col);
    if (values.empty())
        throw std::runtime_error("cpu::from_molrs: required field '"
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
        throw std::runtime_error("cpu::from_molrs: optional field 'atoms."
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
    auto values = molrs::frame_column_u32(src, block, "type_id");
    if (static_cast<int>(values.size()) != n)
        throw std::runtime_error(
            "cpu::from_molrs: topology field '" + std::string(block)
            + ".type_id' is missing or has the wrong dtype/row count");
    std::vector<int> out(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        out[static_cast<std::size_t>(i)] = static_cast<int>(values[i]);
    return out;
}

/// @brief Read one topology connectivity column (`atomi`/`atomj`/`atomk`/
///        `atoml`).
///
/// molrs declares every identifier key UInt, so the endpoint indices are u32 —
/// they are row numbers into the target node block and are never negative.
/// Kinds travel separately as `type_id` (`topology_kind_column`); `type` is a
/// String force-field label this boundary does not import.
std::vector<int> topology_column(const molrs::FrameRef& src, const char* block,
                                 const char* key, int n) {
    auto values = molrs::frame_column_u32(src, block, key);
    if (static_cast<int>(values.size()) != n)
        throw std::runtime_error(
            "cpu::from_molrs: topology field '" + std::string(block) + "." + key
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

} // namespace

Frame from_molrs(const molrs::FrameRef& src) {
    internal::require_molrs_frame_schema_v2();
    validate_exchange_schema(src);
    auto x = require_f64(src, "atoms", "x");
    auto y = require_f64(src, "atoms", "y");
    auto z = require_f64(src, "atoms", "z");
    auto type = require_u32(src, "atoms", "atomic_number");
    const int n = static_cast<int>(x.size());
    if (static_cast<int>(y.size()) != n || static_cast<int>(z.size()) != n
        || static_cast<int>(type.size()) != n)
        throw std::runtime_error(
            "cpu::from_molrs: required atom columns have inconsistent rows");

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
                "cpu::from_molrs: optional field 'atoms.tag' has the wrong "
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
            throw std::runtime_error("cpu::from_molrs: optional field 'atoms."
                                     + std::string(axes[d])
                                     + "' has the wrong dtype or row count");
        for (int i = 0; i < n; ++i)
            image(i, d) = static_cast<std::int8_t>(values[i]);
    }

    frame.clear_tag_index();
    frame.rebuild_tag_index(0, n);
    const int nb = static_cast<int>(molrs::frame_block_nrows(src, "bonds"));
    if (nb > 0) {
        auto atomi = topology_column(src, "bonds", "atomi", nb);
        auto atomj = topology_column(src, "bonds", "atomj", nb);
        auto kind = topology_kind_column(src, "bonds", nb);
        for (int k = 0; k < nb; ++k)
            frame.add_bond(atomi[k], atomj[k], kind[k]);
    }
    const int na = static_cast<int>(molrs::frame_block_nrows(src, "angles"));
    if (na > 0) {
        auto atomi = topology_column(src, "angles", "atomi", na);
        auto atomj = topology_column(src, "angles", "atomj", na);
        auto atomk = topology_column(src, "angles", "atomk", na);
        auto kind = topology_kind_column(src, "angles", na);
        for (int k = 0; k < na; ++k)
            frame.add_angle(atomi[k], atomj[k], atomk[k], kind[k]);
    }
    const int nd = static_cast<int>(molrs::frame_block_nrows(src, "dihedrals"));
    if (nd > 0) {
        auto atomi = topology_column(src, "dihedrals", "atomi", nd);
        auto atomj = topology_column(src, "dihedrals", "atomj", nd);
        auto atomk = topology_column(src, "dihedrals", "atomk", nd);
        auto atoml = topology_column(src, "dihedrals", "atoml", nd);
        auto kind = topology_kind_column(src, "dihedrals", nd);
        for (int k = 0; k < nd; ++k)
            frame.add_dihedral(atomi[k], atomj[k], atomk[k], atoml[k], kind[k]);
    }
    const int ni = static_cast<int>(molrs::frame_block_nrows(src, "impropers"));
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

    auto cell_values = molrs::frame_box(src);
    if (cell_values.size() == 9u) {
        Mat3 cell;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cell(i, j) = static_cast<AccReal>(cell_values[i * 3 + j]);
        frame.box.set_cell(cell);
    }
    internal::decode_molrs_meta(molrs::frame_meta_entries(src), frame.meta);
    return frame;
}

void to_molrs(const Frame& src, molrs::FrameRef& dst) {
    internal::require_molrs_frame_schema_v2();
    const int n = src.nlocal();
    write_f64(dst, "x", src.span(kAtom_x).data(), n);
    write_f64(dst, "y", src.span(kAtom_y).data(), n);
    write_f64(dst, "z", src.span(kAtom_z).data(), n);
    write_f64(dst, "vx", src.span(kAtom_vx).data(), n);
    write_f64(dst, "vy", src.span(kAtom_vy).data(), n);
    write_f64(dst, "vz", src.span(kAtom_vz).data(), n);
    write_f64(dst, "fx", src.span(kAtom_fx).data(), n);
    write_f64(dst, "fy", src.span(kAtom_fy).data(), n);
    write_f64(dst, "fz", src.span(kAtom_fz).data(), n);
    write_f64(dst, "charge", src.span(kAtom_charge).data(), n);

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
            throw std::runtime_error("to_molrs: atom tag does not fit i32");
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
        molrs::frame_set_box(dst,
                             ::rust::Slice<const double>(h.data(), h.size()));
    }

    auto entries = internal::encode_molrs_meta(src.meta);
    for (auto& entry : entries)
        molrs::frame_set_meta_entry(dst, std::move(entry));
}

} // namespace cpu
} // namespace ATV

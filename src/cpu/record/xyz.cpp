/// @file cpu/record/xyz.cpp
/// @brief XYZ writer + Zarr reader — delegating to molrs FFI, one frame at a time.

#include "atomiverse/cpu/record/xyz.h"
#include "../io/molrs_meta.h"
#include "../io/readable_path.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/io/frame_writer.h"
#include "atomiverse/molrs.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace ATV {
namespace cpu {

namespace {
/// @brief Stopgap launch-boundary span resolution for a `const cpu::Frame&`.
///
/// Extract one frame's raw arrays (type_id, x, y, z, box) for the molrs writer.
void extract(const Frame& frame, std::vector<int>& type_id,
             std::vector<double>& x, std::vector<double>& y,
             std::vector<double>& z, std::array<double, 9>& box) {
    const int n = static_cast<int>(frame.nlocal());
    type_id.resize(n);
    x.resize(n);
    y.resize(n);
    z.resize(n);
    // Hoisted once before the loop -- the 13.7x rule.
    const int* __restrict sp_type = frame.span(kAtom_type);
    const StorageReal* __restrict sp_x = frame.span(kAtom_x);
    const StorageReal* __restrict sp_y = frame.span(kAtom_y);
    const StorageReal* __restrict sp_z = frame.span(kAtom_z);
    for (int i = 0; i < n; ++i) {
        type_id[i] = sp_type[i];
        x[i] = static_cast<double>(sp_x[i]);
        y[i] = static_cast<double>(sp_y[i]);
        z[i] = static_cast<double>(sp_z[i]);
    }
    const auto& cell = frame.box.cell();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            box[static_cast<std::size_t>(i * 3 + j)] =
                static_cast<double>(cell(i, j));
}
} // namespace

// ── XYZ ──────────────────────────────────────────────────────────────────────

XYZ::XYZ(const std::string& filename) : filename_(filename) {}

void XYZ::write(const Frame& frame) { write_impl(frame, /*append=*/false); }
void XYZ::append(const Frame& frame) { write_impl(frame, /*append=*/true); }

void XYZ::write_impl(const Frame& frame, bool append) {
    std::array<double, 9> box{};
    extract(frame, type_id_, x_, y_, z_, box);
    molrs::write_frame_xyz_typed(
        {filename_.data(), filename_.size()},
        {reinterpret_cast<const std::int32_t*>(type_id_.data()),
         type_id_.size()},
        {x_.data(), x_.size()}, {y_.data(), y_.size()}, {z_.data(), z_.size()},
        {box.data(), box.size()}, internal::encode_molrs_meta(frame.meta),
        append);
}

// ── ZarrReader ───────────────────────────────────────────────────────────────

ZarrReader::ZarrReader(const std::string& path) : path_(path) {}
ZarrReader::~ZarrReader() = default;

int ZarrReader::read_into(Frame& frame) {
    // Pre-flight the common "wrong path" case so the message is ours.
    internal::require_readable_store("cpu::ZarrReader::read_into", path_);

    // Get the first frame of the Zarr store as a molrs FrameRef. Column
    // names follow the convention established by `dump_frame_to_zarr()`
    // in the polyethylene_tension bench: f64 columns for positions /
    // velocities / mass / tag / image components, f64 column for type.
    // Fallible since molrs 0.13: a corrupt store raises `rust::Error` rather
    // than aborting, and the wrapper attaches the path the bridge omits.
    // molrs 0.14 renamed the door `read_frame_zarr_first` → `read_first_frame`
    // and reads it through the lazy sequence cursor (frame 0 only is decoded),
    // so it serves both the single-frame checkpoint and a streamed `*.mrec`.
    auto fref =
        internal::with_molrs_context("cpu::ZarrReader::read_into", path_, [&] {
            return molrs::read_first_frame({path_.data(), path_.size()});
        });
    auto& fr = *fref;

    auto x = molrs::frame_column_f64(fr, "atoms", "x");
    auto y = molrs::frame_column_f64(fr, "atoms", "y");
    auto z = molrs::frame_column_f64(fr, "atoms", "z");
    auto vx = molrs::frame_column_f64(fr, "atoms", "vx");
    auto vy = molrs::frame_column_f64(fr, "atoms", "vy");
    auto vz = molrs::frame_column_f64(fr, "atoms", "vz");
    auto mass = molrs::frame_column_f64(fr, "atoms", "mass");
    auto tag = molrs::frame_column_f64(fr, "atoms", "tag");
    auto img_x = molrs::frame_column_f64(fr, "atoms", "image_x");
    auto img_y = molrs::frame_column_f64(fr, "atoms", "image_y");
    auto img_z = molrs::frame_column_f64(fr, "atoms", "image_z");
    // molrs 0.13 renamed the element column; `type` is now a String
    // force-field label. The Zarr store writes numerics as f64.
    auto type = molrs::frame_column_f64(fr, "atoms", "atomic_number");
    auto box_h = molrs::frame_box(fr);

    const int n = static_cast<int>(x.size());
    if (n == 0)
        throw std::runtime_error("ZarrReader: empty Zarr at " + path_);
    // A column molrs cannot find is returned as an *empty* vector, not as an
    // error, so a store missing e.g. `vx` would be read past its end by the
    // per-atom loop below. Require every column to match the `x` row count.
    auto require_rows = [&](const char* col, std::size_t rows) {
        if (rows != x.size())
            throw std::runtime_error(
                "ZarrReader: atoms." + std::string(col) + " has "
                + std::to_string(rows) + " rows, expected "
                + std::to_string(x.size()) + " at " + path_);
    };
    require_rows("y", y.size());
    require_rows("z", z.size());
    require_rows("vx", vx.size());
    require_rows("vy", vy.size());
    require_rows("vz", vz.size());
    require_rows("mass", mass.size());
    require_rows("tag", tag.size());
    require_rows("image_x", img_x.size());
    require_rows("image_y", img_y.size());
    require_rows("image_z", img_z.size());
    require_rows("type", type.size());
    if (n > frame.nlocal())
        throw std::runtime_error(
            "ZarrReader: n_global=" + std::to_string(n) + " > frame.nlocal()="
            + std::to_string(frame.nlocal()) + " at " + path_);
    if (box_h.size() != 9)
        throw std::runtime_error("ZarrReader: box.size != 9 at " + path_);

    ATV::Mat3 cell;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            cell(r, c) = box_h[r * 3 + c];
    frame.box.set_cell(cell);

    // Hoisted once before the loop -- the 13.7x rule.
    StorageReal* __restrict sp_x = frame.span(kAtom_x);
    StorageReal* __restrict sp_y = frame.span(kAtom_y);
    StorageReal* __restrict sp_z = frame.span(kAtom_z);
    StorageReal* __restrict sp_vx = frame.span(kAtom_vx);
    StorageReal* __restrict sp_vy = frame.span(kAtom_vy);
    StorageReal* __restrict sp_vz = frame.span(kAtom_vz);
    StorageReal* __restrict sp_mass = frame.span(kAtom_mass);
    int* __restrict sp_type = frame.span(kAtom_type);
    TagInt* __restrict sp_tag = frame.span(kAtom_tag);
    Span<std::int8_t> sp_image = frame.span(kAtom_image);
    for (int i = 0; i < n; ++i) {
        sp_x[i] = static_cast<StorageReal>(x[i]);
        sp_y[i] = static_cast<StorageReal>(y[i]);
        sp_z[i] = static_cast<StorageReal>(z[i]);
        sp_vx[i] = static_cast<StorageReal>(vx[i]);
        sp_vy[i] = static_cast<StorageReal>(vy[i]);
        sp_vz[i] = static_cast<StorageReal>(vz[i]);
        sp_mass[i] = static_cast<StorageReal>(mass[i]);
        sp_type[i] = static_cast<int>(type[i]);
        sp_tag[i] = static_cast<TagInt>(static_cast<std::int64_t>(tag[i]));
        sp_image(i, 0) = static_cast<std::int8_t>(img_x[i]);
        sp_image(i, 1) = static_cast<std::int8_t>(img_y[i]);
        sp_image(i, 2) = static_cast<std::int8_t>(img_z[i]);
    }
    return n;
}

} // namespace cpu
} // namespace ATV

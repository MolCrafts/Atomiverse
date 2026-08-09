/// @file cpu/recorder/xyz.h
/// @brief XYZ and Zarr trajectory writers/readers — all backed by molrs.
///
/// The writers take a `cpu::Frame` and hand its raw per-atom arrays to molrs one
/// frame at a time (`cpu::write_frame_xyz`); there is no C++ record type.
///
/// @see cpu::write_frame_xyz / cpu::write_frame_zarr (cpu/io/frame_writer.h)
#pragma once

#include <string>
#include <vector>

namespace ATV {
namespace cpu {

class Frame; // fwd — CUDA-safe (this header is pulled by 02_rhf.cu)

/// @brief XYZ trajectory writer (standard element + coordinates).
///
/// `write`/`append` write one frame's geometry through molrs. Extraction
/// scratch is reused across calls, so per-frame appends on a long-lived
/// writer do not reallocate on the steady-state path.
class XYZ {
  public:
    /// @brief Construct.
    /// @param filename Output XYZ path.
    explicit XYZ(const std::string& filename);

    void write(const Frame& frame);  ///< truncate/create, write one frame
    void append(const Frame& frame); ///< append one frame

  private:
    void write_impl(const Frame& frame, bool append);

    std::string filename_;
    std::vector<int> type_id_; ///< per-frame extraction scratch (reused)
    std::vector<double> x_, y_, z_;
};

/// @brief Zarr V3 reader — loads the first frame of a Zarr-stored trajectory
///        into a cpu::Frame.
///
/// Used by checkpoint reload: a bench stage dumps its end-state via
/// `cpu::write_frame_zarr`, subsequent debug iterations call
/// `ZarrReader::read_into(frame)` to skip that stage.
///
/// The reader expects per-atom columns `x, y, z, vx, vy, vz, mass, tag,
/// image_x, image_y, image_z` (all f64) and an f64 `type` column in the
/// `atoms` block, plus a 9-element `box` (row-major h-matrix) — exactly the
/// fields `dump_frame_to_zarr()` writes (benchmarks/md/pe/.../polyethylene_tension.cpp).
class ZarrReader {
  public:
    explicit ZarrReader(const std::string& path);
    ~ZarrReader();
    ZarrReader(const ZarrReader&) = delete;
    ZarrReader& operator=(const ZarrReader&) = delete;

    /// @brief Read the first frame of the Zarr store into `frame`.
    /// Overwrites the owned atom Block rows and `frame.box`.
    /// @return Number of atoms loaded (== `n_global` from the file).
    /// @throws std::runtime_error if the store path does not exist or cannot be
    ///         read, the store is empty, an expected per-atom column is missing
    ///         or short, the box is not 9 elements, or
    ///         `n_global > frame.nlocal()`.
    int read_into(Frame& frame);

  private:
    std::string path_;
};

} // namespace cpu
} // namespace ATV

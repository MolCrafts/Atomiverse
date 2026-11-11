/// @file cpu/record/xyz.h
/// @brief XYZ trajectory writer — backed by molpy.
///
/// The writer takes a `cpu::Frame` and hands its raw per-atom arrays and
/// metadata to molpy one frame at a time (molpy `write_xyz`); there is
/// no C++ record type.
///
/// @see cpu::write_xyz_frame / cpu::write_mrec_frame (cpu/io/frame_writer.h)
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ATV {
namespace cpu {

class Frame; // fwd — keeps this header CUDA-safe

/// @brief XYZ trajectory writer (standard element + coordinates).
///
/// `write`/`append` write one frame's geometry through molpy. Species come
/// from `kAtom_atomic_number`, the atomic number Z (0 is the wildcard `*`);
/// molpy spells the element symbols. A frame without a species column
/// throws with the how-to. Extraction
/// scratch is reused across calls, so per-frame appends on a long-lived
/// writer do not reallocate on the steady-state path.
class XYZ {
  public:
    /// @brief Construct.
    /// @param filename Output XYZ path.
    explicit XYZ(const std::string& filename);

    /// @brief Truncate/create the file and write one frame.
    /// @param frame source of Z (`kAtom_atomic_number`), x/y/z [Å], box and meta.
    /// @throws std::invalid_argument on a negative Z; std::runtime_error
    /// (molpy, path attached) on Z > 118 or a write failure.
    void write(const Frame& frame);
    /// @brief Append one frame; same contract as `write`.
    /// @param frame source of Z (`kAtom_atomic_number`), x/y/z [Å], box and meta.
    void append(const Frame& frame);

  private:
    void write_impl(const Frame& frame, bool append);

    std::string filename_;
    std::vector<std::uint64_t> atomic_number_; ///< per-frame Z (reused)
    std::vector<double> x_, y_, z_;
};

} // namespace cpu
} // namespace ATV

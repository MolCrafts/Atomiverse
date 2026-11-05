/// @file cuda/record/recorder.cuh
/// @brief XyzRecorder -- header-only XYZ trajectory writer (an asynchronous
///        consumer of the shared capture buffer).
///
/// @see cpu::write_xyz_frame (cpu/io/frame_writer.h — deliberately CUDA-safe)
#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/io/frame_writer.h"
#include "atomiverse/cuda/asynchronous/async_consumer.cuh"

namespace ATV {
namespace cuda {

/// @brief XYZ trajectory recorder.
///
/// Reads the positions of each captured step from the driver's shared
/// capture buffer (`cuda::CaptureBuffer`): the frame written for step N is a
/// copy frozen on the compute stream at the end of step N (FP64, primary +
/// correction in mixed precision), never a view of device memory the next
/// step overwrites.
///
/// Frames go through the molpy-backed `cpu::write_xyz_frame` (extended XYZ
/// with a Lattice comment line): the species column is the element symbol of
/// each atom's species Z (`atoms.atomic_number`, Z = 0 is the wildcard). A
/// frame without a species column (a coarse-grained model) is refused at
/// build — a force-field type is never written as an element.
///
/// Step convention (LAMMPS `dump`, @see AsyncConsumer): a frame is the
/// state after a whole number of completed steps; a fresh `n`-step run at
/// `every` writes the states after steps 0, every, 2·every, …, up to and
/// including `n`, and a continued run never repeats the boundary state.
///
/// @code{.cpp}
/// sim.on(md::Phase::end_of_step).add(XyzRecorder{"traj.xyz", 10});
/// @endcode
class XyzRecorder : public AsyncRecorder {
  public:
    XyzRecorder(std::string filename, int every = 1)
        : filename_(std::move(filename)), every_(every) {
        if (every_ < 1)
            throw std::invalid_argument(
                "cuda::XyzRecorder: every must be >= 1 step, got "
                + std::to_string(every_));
    }

    ~XyzRecorder() override = default;
    XyzRecorder(XyzRecorder&&) noexcept = default;
    XyzRecorder& operator=(XyzRecorder&&) noexcept = default;

    std::string name() const override { return "xyz_recorder"; }
    std::string capture_name() const override {
        return "cuda::XyzRecorder: writing '" + filename_ + "'";
    }
    int frequency() const override { return every_; }

    CaptureColumns capture_columns() const override {
        CaptureColumns c;
        c.positions = true;
        return c;
    }

  protected:
    void on_build(const BuildContext& ctx) override {
        if (!ctx.frame.atoms.atomic_number)
            throw std::invalid_argument(
                std::string("cuda::XyzRecorder: device frame has no species "
                            "column; ")
                + atom_field_how_to("atomic_number"));
        // First frame of each build truncates any pre-existing file.
        wrote_any_ = false;
    }

    void consume(const CaptureView& v) override {
        const auto n = static_cast<std::size_t>(v.n_atoms);
        species_.assign(v.atomic_number, v.atomic_number + n);
        dx_.assign(v.x, v.x + n);
        dy_.assign(v.y, v.y + n);
        dz_.assign(v.z, v.z + n);
        for (std::size_t i = 0; i < 9; ++i)
            box_[i] = v.cell[i];
        // First write truncates any pre-existing file; later writes append.
        cpu::write_xyz_frame(filename_, species_, dx_, dy_, dz_, box_,
                             /*append=*/wrote_any_);
        wrote_any_ = true;
    }

  private:
    std::string filename_;
    int every_;
    std::vector<double> dx_, dy_, dz_; ///< writer scratch (worker thread)
    std::vector<int> species_;         ///< species Z per row
    std::array<double, 9> box_{};      ///< row-major cell [Å]
    bool wrote_any_ = false;           ///< false = next write truncates
};

} // namespace cuda
} // namespace ATV

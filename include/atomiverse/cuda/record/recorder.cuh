/// @file cuda/record/recorder.cuh
/// @brief Recorder -- header-only XYZ trajectory writer (async D2H bridge).
///
/// @see cpu::write_xyz_frame (cpu/io/frame_writer.h — deliberately CUDA-safe)
#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "atomiverse/cpu/io/frame_writer.h"
#include "atomiverse/cuda/asynchronous/async_bridge.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {

/// @brief XYZ trajectory recorder.
///
/// Freezes positions into a device-ring slot on the compute stream each sampled
/// step, so the frame written for step N holds step-N coordinates with no
/// tearing from the next step's integrator (a cross-stream race the previous
/// live-pointer D2H was exposed to).
///
/// Frames go through the molrs-backed `cpu::write_xyz_frame` (extended XYZ
/// with a Lattice comment line): the species column is the element symbol
/// derived from `atoms.type` interpreted as atomic number Z, so non-Z type
/// ids map to placeholder symbols (e.g. LJ type 0 -> "X"). This replaces the
/// former hand-rolled integer-species format with a per-frame `Step` comment.
///
/// @code{.cpp}
/// sim.on(md::Phase::end_of_step).add(XyzRecorder{"traj.xyz", 10});
/// @endcode
class XyzRecorder : public AsyncRecorder {
  public:
    XyzRecorder(std::string filename, int every = 1)
        : filename_(std::move(filename)), every_(every) {}

    ~XyzRecorder() override = default;
    XyzRecorder(XyzRecorder&&) noexcept = default;
    XyzRecorder& operator=(XyzRecorder&&) noexcept = default;

    std::string name() const override { return "xyz_recorder"; }
    /* role via AsyncRecorder / Recorder */
    int frequency() const override { return every_; }

  protected:
    void on_build(const BuildContext& ctx) override {
        auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
        auto& frame = ctx.frame;
        n_atoms_ = frame.n_atoms();
        auto n = static_cast<size_t>(n_atoms_);

        if (n == 0)
            return;

        // Frozen coordinates include the mixed-precision residual.
        for (int i = 0; i < kRingDepth; ++i) {
            ATV_CUDA_CHECK(cudaMalloc(&d_cell_[i], 9 * sizeof(prec_t)));
            ATV_CUDA_CHECK(cudaMallocHost(&h_cell_[i], 9 * sizeof(prec_t)));
            ATV_CUDA_CHECK(
                cudaMalloc(&d_x_[i], n * sizeof(motion_t))); // device-ring slot
            ATV_CUDA_CHECK(cudaMalloc(&d_y_[i], n * sizeof(motion_t)));
            ATV_CUDA_CHECK(cudaMalloc(&d_z_[i], n * sizeof(motion_t)));
            ATV_CUDA_CHECK(cudaMallocHost(
                &h_x_[i], n * sizeof(motion_t))); // pinned host slot
            ATV_CUDA_CHECK(cudaMallocHost(&h_y_[i], n * sizeof(motion_t)));
            ATV_CUDA_CHECK(cudaMallocHost(&h_z_[i], n * sizeof(motion_t)));
        }

        dx_.resize(n);
        dy_.resize(n);
        dz_.resize(n);

        // Types are immutable for the run -- D2H once at build (sync OK here).
        htype_.resize(n);
        ATV_CUDA_CHECK(cudaMemcpyAsync(htype_.data(), frame.atoms.type,
                                       n * sizeof(int), cudaMemcpyDeviceToHost,
                                       stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));

        // First sampled frame of each run truncates any pre-existing file.
        wrote_any_ = false;
    }

    bool can_run(const LaunchContext&) const override { return n_atoms_ > 0; }

    void freeze_slot(const LaunchContext& ctx, int slot) override {
        // D2D freeze of step-N positions on the compute stream, before the next
        // step's integrator overwrites them.
        auto ms = static_cast<cudaStream_t>(ctx.runtime.stream);
        capture_positions(ctx.frame, d_x_[slot], d_y_[slot], d_z_[slot], ms);
        ATV_CUDA_CHECK(cudaMemcpyAsync(d_cell_[slot], ctx.frame.box,
                                       9 * sizeof(prec_t),
                                       cudaMemcpyDeviceToDevice, ms));
    }

    void d2h_slot(const LaunchContext& ctx, int slot) override {
        auto ds = static_cast<cudaStream_t>(ctx.runtime.defer_stream);
        auto n = static_cast<size_t>(n_atoms_);
        ATV_CUDA_CHECK(cudaMemcpyAsync(h_cell_[slot], d_cell_[slot],
                                       9 * sizeof(prec_t),
                                       cudaMemcpyDeviceToHost, ds));
        ATV_CUDA_CHECK(cudaMemcpyAsync(h_x_[slot], d_x_[slot],
                                       n * sizeof(motion_t),
                                       cudaMemcpyDeviceToHost, ds));
        ATV_CUDA_CHECK(cudaMemcpyAsync(h_y_[slot], d_y_[slot],
                                       n * sizeof(motion_t),
                                       cudaMemcpyDeviceToHost, ds));
        ATV_CUDA_CHECK(cudaMemcpyAsync(h_z_[slot], d_z_[slot],
                                       n * sizeof(motion_t),
                                       cudaMemcpyDeviceToHost, ds));
    }

    void on_data_ready(const LaunchContext& ctx, int slot) override {
        // Widen motion-precision pinned slot to double for the molrs writer. Callbacks
        // are serialized on the defer stream, so the dx_/dy_/dz_/box_ scratch
        // is reused safely (one callback in flight at a time).
        for (int i = 0; i < n_atoms_; ++i) {
            dx_[i] = static_cast<double>(h_x_[slot][i]);
            dy_[i] = static_cast<double>(h_y_[slot][i]);
            dz_[i] = static_cast<double>(h_z_[slot][i]);
        }
        // Frozen step-N cell, independent of later barostat updates.
        for (int i = 0; i < 9; ++i)
            box_[static_cast<size_t>(i)] =
                static_cast<double>(h_cell_[slot][i]);
        // First write truncates any pre-existing file; later writes append.
        cpu::write_xyz_frame(filename_, htype_, dx_, dy_, dz_, box_,
                             /*append=*/wrote_any_);
        wrote_any_ = true;
    }

    void on_destroy(const DestroyContext&) override {
        for (int i = 0; i < kRingDepth; ++i) {
            ATV_CUDA_CHECK(cudaFree(d_cell_[i]));
            ATV_CUDA_CHECK(cudaFreeHost(h_cell_[i]));
            d_cell_[i] = nullptr;
            h_cell_[i] = nullptr;
            if (d_x_[i]) {
                ATV_CUDA_CHECK(cudaFree(d_x_[i]));
                d_x_[i] = nullptr;
            }
            if (d_y_[i]) {
                ATV_CUDA_CHECK(cudaFree(d_y_[i]));
                d_y_[i] = nullptr;
            }
            if (d_z_[i]) {
                ATV_CUDA_CHECK(cudaFree(d_z_[i]));
                d_z_[i] = nullptr;
            }
            if (h_x_[i]) {
                ATV_CUDA_CHECK(cudaFreeHost(h_x_[i]));
                h_x_[i] = nullptr;
            }
            if (h_y_[i]) {
                ATV_CUDA_CHECK(cudaFreeHost(h_y_[i]));
                h_y_[i] = nullptr;
            }
            if (h_z_[i]) {
                ATV_CUDA_CHECK(cudaFreeHost(h_z_[i]));
                h_z_[i] = nullptr;
            }
        }
    }

  private:
    std::string filename_;
    int every_;
    int n_atoms_ = 0;
    motion_t* d_x_[kRingDepth] = {}; ///< device-ring frozen x [Angstrom]
    motion_t* d_y_[kRingDepth] = {};
    motion_t* d_z_[kRingDepth] = {};
    motion_t* h_x_[kRingDepth] = {}; ///< pinned host x [Angstrom]
    motion_t* h_y_[kRingDepth] = {};
    motion_t* h_z_[kRingDepth] = {};
    std::vector<double> dx_, dy_, dz_; ///< widened-to-double scratch for writer
    std::vector<int> htype_;           ///< atom types (immutable, D2H once)
    prec_t* d_cell_[kRingDepth] = {};  ///< frozen step-N cell
    prec_t* h_cell_[kRingDepth] = {};  ///< pinned step-N cell
    std::array<double, 9> box_{};      ///< row-major cell scratch [Angstrom]
    bool wrote_any_ = false;           ///< false = next write truncates
};

} // namespace cuda
} // namespace ATV

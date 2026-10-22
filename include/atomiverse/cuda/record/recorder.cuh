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
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {

/// @brief XYZ trajectory recorder.
///
/// Freezes positions into a device-ring slot on the compute stream each sampled
/// step, so the frame written for step N holds step-N coordinates with no
/// tearing from the next step's integrator (a cross-stream race the previous
/// live-pointer D2H was exposed to). In mixed precision the correction
/// columns are frozen alongside the primary coordinates and the written
/// coordinate is the reconstructed `x + x_corr`.
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

        // frame.atoms.x/y/z are pos_t device buffers -- size by pos_t. The
        // blocked x|y|z correction slot exists only with kPositionCorrection.
        for (int i = 0; i < kRingDepth; ++i) {
            cudaMalloc(&d_box_[i], 9 * sizeof(geom_t));
            cudaMallocHost(&h_box_[i], 9 * sizeof(geom_t));
            cudaMalloc(&d_x_[i], n * sizeof(pos_t)); // device-ring slot
            cudaMalloc(&d_y_[i], n * sizeof(pos_t));
            cudaMalloc(&d_z_[i], n * sizeof(pos_t));
            cudaMallocHost(&h_x_[i], n * sizeof(pos_t)); // pinned host slot
            cudaMallocHost(&h_y_[i], n * sizeof(pos_t));
            cudaMallocHost(&h_z_[i], n * sizeof(pos_t));
            if constexpr (kPositionCorrection) {
                cudaMalloc(&d_corr_[i], 3 * n * sizeof(pos_t));
                cudaMallocHost(&h_corr_[i], 3 * n * sizeof(pos_t));
            }
        }

        dx_.resize(n);
        dy_.resize(n);
        dz_.resize(n);

        // Types are immutable for the run -- D2H once at build (sync OK here).
        htype_.resize(n);
        cudaMemcpyAsync(htype_.data(), frame.atoms.type, n * sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // First sampled frame of each run truncates any pre-existing file.
        wrote_any_ = false;
    }

    void freeze_slot(const LaunchContext& ctx, int slot) override {
        // D2D freeze of step-N positions on the compute stream, before the next
        // step's integrator overwrites them.
        auto ms = static_cast<cudaStream_t>(ctx.runtime.stream);
        auto n = static_cast<size_t>(n_atoms_);
        const auto& atoms = ctx.frame.atoms;
        cudaMemcpyAsync(d_box_[slot], ctx.frame.box, 9 * sizeof(geom_t),
                        cudaMemcpyDeviceToDevice, ms);
        cudaMemcpyAsync(d_x_[slot], atoms.x, n * sizeof(pos_t),
                        cudaMemcpyDeviceToDevice, ms);
        cudaMemcpyAsync(d_y_[slot], atoms.y, n * sizeof(pos_t),
                        cudaMemcpyDeviceToDevice, ms);
        cudaMemcpyAsync(d_z_[slot], atoms.z, n * sizeof(pos_t),
                        cudaMemcpyDeviceToDevice, ms);
        if constexpr (kPositionCorrection) {
            cudaMemcpyAsync(d_corr_[slot], atoms.x_corr, n * sizeof(pos_t),
                            cudaMemcpyDeviceToDevice, ms);
            cudaMemcpyAsync(d_corr_[slot] + n, atoms.y_corr, n * sizeof(pos_t),
                            cudaMemcpyDeviceToDevice, ms);
            cudaMemcpyAsync(d_corr_[slot] + 2 * n, atoms.z_corr,
                            n * sizeof(pos_t), cudaMemcpyDeviceToDevice, ms);
        }
    }

    void d2h_slot(const LaunchContext& ctx, int slot) override {
        auto ds = static_cast<cudaStream_t>(ctx.runtime.defer_stream);
        auto n = static_cast<size_t>(n_atoms_);
        cudaMemcpyAsync(h_box_[slot], d_box_[slot], 9 * sizeof(geom_t),
                        cudaMemcpyDeviceToHost, ds);
        cudaMemcpyAsync(h_x_[slot], d_x_[slot], n * sizeof(pos_t),
                        cudaMemcpyDeviceToHost, ds);
        cudaMemcpyAsync(h_y_[slot], d_y_[slot], n * sizeof(pos_t),
                        cudaMemcpyDeviceToHost, ds);
        cudaMemcpyAsync(h_z_[slot], d_z_[slot], n * sizeof(pos_t),
                        cudaMemcpyDeviceToHost, ds);
        if constexpr (kPositionCorrection)
            cudaMemcpyAsync(h_corr_[slot], d_corr_[slot], 3 * n * sizeof(pos_t),
                            cudaMemcpyDeviceToHost, ds);
    }

    void on_data_ready(const LaunchContext& ctx, int slot) override {
        // Reconstruct primary + correction and widen to double for the molrs
        // writer. Callbacks are serialized on the defer stream, so the
        // dx_/dy_/dz_/box_ scratch is reused safely (one callback in flight
        // at a time).
        const pos_t* corr = h_corr_[slot]; // null without correction
        const auto n = static_cast<size_t>(n_atoms_);
        const pos_t* cx = corr;
        const pos_t* cy = corr ? corr + n : nullptr;
        const pos_t* cz = corr ? corr + 2 * n : nullptr;
        for (int i = 0; i < n_atoms_; ++i) {
            dx_[i] = static_cast<double>(load_position(h_x_[slot], cx, i));
            dy_[i] = static_cast<double>(load_position(h_y_[slot], cy, i));
            dz_[i] = static_cast<double>(load_position(h_z_[slot], cz, i));
        }
        // Cell and positions are from the same frozen step.
        for (int i = 0; i < 9; ++i)
            box_[static_cast<size_t>(i)] = static_cast<double>(h_box_[slot][i]);
        // First write truncates any pre-existing file; later writes append.
        cpu::write_xyz_frame(filename_, htype_, dx_, dy_, dz_, box_,
                             /*append=*/wrote_any_);
        wrote_any_ = true;
    }

    void on_destroy(const DestroyContext&) override {
        for (int i = 0; i < kRingDepth; ++i) {
            cudaFree(d_box_[i]);
            cudaFreeHost(h_box_[i]);
            d_box_[i] = nullptr;
            h_box_[i] = nullptr;
            if (d_x_[i]) {
                cudaFree(d_x_[i]);
                d_x_[i] = nullptr;
            }
            if (d_y_[i]) {
                cudaFree(d_y_[i]);
                d_y_[i] = nullptr;
            }
            if (d_z_[i]) {
                cudaFree(d_z_[i]);
                d_z_[i] = nullptr;
            }
            if (h_x_[i]) {
                cudaFreeHost(h_x_[i]);
                h_x_[i] = nullptr;
            }
            if (h_y_[i]) {
                cudaFreeHost(h_y_[i]);
                h_y_[i] = nullptr;
            }
            if (h_z_[i]) {
                cudaFreeHost(h_z_[i]);
                h_z_[i] = nullptr;
            }
            if (d_corr_[i]) {
                cudaFree(d_corr_[i]);
                d_corr_[i] = nullptr;
            }
            if (h_corr_[i]) {
                cudaFreeHost(h_corr_[i]);
                h_corr_[i] = nullptr;
            }
        }
    }

  private:
    std::string filename_;
    int every_;
    int n_atoms_ = 0;
    geom_t* d_box_[kRingDepth] = {}; ///< device-ring frozen 3x3 cell [Angstrom]
    geom_t* h_box_[kRingDepth] = {}; ///< pinned host 3x3 cell [Angstrom]
    pos_t* d_x_[kRingDepth] = {};    ///< device-ring frozen x [Angstrom]
    pos_t* d_y_[kRingDepth] = {};
    pos_t* d_z_[kRingDepth] = {};
    pos_t* h_x_[kRingDepth] = {}; ///< pinned host x [Angstrom]
    pos_t* h_y_[kRingDepth] = {};
    pos_t* h_z_[kRingDepth] = {};
    /// device-ring frozen blocked x|y|z corrections [Angstrom]; null without
    /// kPositionCorrection
    pos_t* d_corr_[kRingDepth] = {};
    pos_t* h_corr_[kRingDepth] = {};   ///< pinned host corrections [Angstrom]
    std::vector<double> dx_, dy_, dz_; ///< widened-to-double scratch for writer
    std::vector<int> htype_;           ///< atom types (immutable, D2H once)
    std::array<double, 9> box_{};      ///< row-major cell scratch [Angstrom]
    bool wrote_any_ = false;           ///< false = next write truncates
};

} // namespace cuda
} // namespace ATV

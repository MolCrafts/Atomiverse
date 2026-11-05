/// @file cuda/observe/rdf.cuh
/// @brief RDF -- header-only radial distribution function observer (async D2H bridge).
///
/// Freezes frame positions + the per-frame cell into a device-ring slot and
/// streams every captured frame into the molpy g(r) accumulator
/// (`cpu::compute::RDFAccumulator`). All binning + shell-volume normalization
/// happens in molpy — no analysis math in C++; the normalized curve and its
/// molpy bin centers are written in destroy() via the single g(r) emitter
/// `cpu::rdf_write_curve`.
///
/// @see cpu::compute::RDFAccumulator (cpu/observe/trajectory_metrics.h)
/// @see cpu::rdf_write_curve (cpu/observe/rdf_curve.h)
#pragma once

#include <array>
#include <cstdio>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "atomiverse/cpu/observe/rdf_curve.h"
#include "atomiverse/cpu/observe/trajectory_metrics.h"
#include "atomiverse/cuda/asynchronous/async_bridge.cuh"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {

/// @brief Radial distribution function g(r) observer (molpy-backed).
///
/// Freezes the frame's position arrays (blocked `x|y|z`) plus the 3x3 cell
/// into a device-ring slot on the compute stream (before the next step mutates
/// them), so each accumulated frame is a self-consistent step-N snapshot. The
/// per-frame cell capture makes the accumulation NPT-correct (the cell is read
/// every capture, not once at build). Reads no neighbor/pair data — the
/// molpy accumulator builds its own LinkCell per frame.
///
/// @code{.cpp}
/// sim.on(md::Phase::end_of_step).add(RDF{3.0, 100, "rdf.dat"});
/// @endcode
class RDF : public AsyncObserver {
  public:
    /// @param r_max upper edge of the last bin [Å] (> 0). @param n_bins radial
    /// bin count (>= 1). @param filename g(r) output path.
    /// @throws std::invalid_argument on out-of-range parameters (fail-fast,
    /// no-fallback — refused by molpy `Rdf::new`).
    RDF(double r_max, int n_bins, std::string filename)
        : filename_(std::move(filename)),
          acc_(cpu::compute::RDF{n_bins, r_max, /*r_min=*/0.0}.accumulator()) {}

    ~RDF() override = default;
    RDF(RDF&&) noexcept = default;
    RDF& operator=(RDF&&) noexcept = default;

    std::string name() const override { return "rdf"; }
    /* role via AsyncObserver / Observer */

  protected:
    void on_build(const BuildContext& ctx) override {
        n_atoms_ = ctx.frame.n_atoms();
        if (n_atoms_ <= 0)
            return;
        const auto n = static_cast<size_t>(n_atoms_);
        for (int i = 0; i < kRingDepth; ++i) {
            cudaMalloc(&d_pos_[i], 3 * n * sizeof(pos_t));
            cudaMalloc(&d_cell_[i], 9 * sizeof(geom_t));
            cudaMallocHost(&h_pos_[i], 3 * n * sizeof(pos_t));
            cudaMallocHost(&h_cell_[i], 9 * sizeof(geom_t));
        }
        pos_.resize(3 * n);
    }

    bool can_run(const LaunchContext&) const override { return n_atoms_ > 0; }

    void freeze_slot(const LaunchContext& ctx, int slot) override {
        // D2D freeze of the step-N positions + cell on the compute stream,
        // before the next step's integrate/deform mutates them.
        auto ms = static_cast<cudaStream_t>(ctx.runtime.stream);
        const auto& atoms = ctx.frame.atoms;
        const auto n = static_cast<size_t>(n_atoms_);
        cudaMemcpyAsync(d_pos_[slot], atoms.x, n * sizeof(pos_t),
                        cudaMemcpyDeviceToDevice, ms);
        cudaMemcpyAsync(d_pos_[slot] + n, atoms.y, n * sizeof(pos_t),
                        cudaMemcpyDeviceToDevice, ms);
        cudaMemcpyAsync(d_pos_[slot] + 2 * n, atoms.z, n * sizeof(pos_t),
                        cudaMemcpyDeviceToDevice, ms);
        cudaMemcpyAsync(d_cell_[slot], ctx.frame.box, 9 * sizeof(geom_t),
                        cudaMemcpyDeviceToDevice, ms);
    }

    void d2h_slot(const LaunchContext& ctx, int slot) override {
        auto ds = static_cast<cudaStream_t>(ctx.runtime.defer_stream);
        const auto n = static_cast<size_t>(n_atoms_);
        cudaMemcpyAsync(h_pos_[slot], d_pos_[slot], 3 * n * sizeof(pos_t),
                        cudaMemcpyDeviceToHost, ds);
        cudaMemcpyAsync(h_cell_[slot], d_cell_[slot], 9 * sizeof(geom_t),
                        cudaMemcpyDeviceToHost, ds);
    }

    void on_data_ready(const LaunchContext&, int slot) override {
        // Widen the pinned pos_t/geom_t staging buffers to double for the
        // molpy accumulator (blocked x|y|z layout is preserved by the freeze).
        // Analysis reads the primary stored coordinate only; the mixed-mode
        // correction (< 1 ulp of FP32) is below any bin width.
        // pos_ scratch reuse is safe: callbacks run in defer-stream FIFO order.
        const int n3 = 3 * n_atoms_;
        for (int i = 0; i < n3; ++i)
            pos_[static_cast<size_t>(i)] = static_cast<double>(h_pos_[slot][i]);
        std::array<double, 9> cell{};
        for (int i = 0; i < 9; ++i)
            cell[static_cast<size_t>(i)] =
                static_cast<double>(h_cell_[slot][i]);
        // Host callback: a molpy refusal is counted for on_destroy(), never
        // thrown across the stream callback.
        try {
            acc_.push(pos_, cell);
        } catch (const std::exception&) {
            ++n_rejected_;
        }
    }

    void on_destroy(const DestroyContext&) override {
        if (n_rejected_ > 0) {
            // core/log.h is banned from .cuh/.cu TUs; stderr per the existing
            // CUDA host-side reporting precedent (comm/multi_gpu.cu).
            std::fprintf(stderr,
                         "[cuda::rdf] dropped %d frame(s) rejected by the "
                         "molpy accumulator (shape mismatch / degenerate "
                         "cell)\n",
                         n_rejected_);
        }
        if (acc_.n_frames() > 0) {
            try {
                const cpu::compute::RDFResult result = acc_.finish();
                cpu::rdf_write_curve(filename_, result.bin_centers, result.rdf);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[cuda::rdf] %s not written: %s\n",
                             filename_.c_str(), e.what());
            }
        }
        for (int i = 0; i < kRingDepth; ++i) {
            if (d_pos_[i]) {
                cudaFree(d_pos_[i]);
                d_pos_[i] = nullptr;
            }
            if (d_cell_[i]) {
                cudaFree(d_cell_[i]);
                d_cell_[i] = nullptr;
            }
            if (h_pos_[i]) {
                cudaFreeHost(h_pos_[i]);
                h_pos_[i] = nullptr;
            }
            if (h_cell_[i]) {
                cudaFreeHost(h_cell_[i]);
                h_cell_[i] = nullptr;
            }
        }
    }

  private:
    std::string filename_;
    cpu::compute::RDFAccumulator acc_; ///< molpy streaming g(r) state

    int n_atoms_ = 0;
    int n_rejected_ = 0;              ///< frames the accumulator refused
    pos_t* d_pos_[kRingDepth] = {};   ///< device-ring frozen blocked x|y|z [Å]
    geom_t* d_cell_[kRingDepth] = {}; ///< device-ring frozen 3x3 cell [Å]
    pos_t* h_pos_[kRingDepth] = {};   ///< pinned host blocked x|y|z [Å]
    geom_t* h_cell_[kRingDepth] = {}; ///< pinned host 3x3 cell [Å]
    std::vector<double> pos_;         ///< widened-to-double packing scratch
};

} // namespace cuda
} // namespace ATV

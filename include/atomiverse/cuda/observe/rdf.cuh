/// @file cuda/observe/rdf.cuh
/// @brief RDF -- header-only radial distribution function observer (an
///        asynchronous consumer of the shared capture buffer).
///
/// Streams every captured frame's positions + cell into the molpy g(r)
/// accumulator (`cpu::compute::RDFAccumulator`). All binning + shell-volume
/// normalization happens in molpy — no analysis math in C++; the normalized
/// curve and its molpy bin centers are written in on_destroy() via the
/// single g(r) emitter `cpu::rdf_write_curve`.
///
/// @see cpu::compute::RDFAccumulator (cpu/observe/trajectory_metrics.h)
/// @see cpu::rdf_write_curve (cpu/observe/rdf_curve.h)
#pragma once

#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/cpu/observe/rdf_curve.h"
#include "atomiverse/cpu/observe/trajectory_metrics.h"
#include "atomiverse/cuda/asynchronous/async_consumer.cuh"

namespace ATV {
namespace cuda {

/// @brief Radial distribution function g(r) observer (molpy-backed).
///
/// Reads the positions and the cell of each captured step from the driver's
/// shared capture buffer (`cuda::CaptureBuffer`), frozen on the compute
/// stream at the end of that step, so each accumulated frame is a
/// self-consistent step-N snapshot (NPT-correct: the cell travels with every
/// frame). Lossless: every step on the cadence is accumulated. Reads no
/// neighbor/pair data — the molpy accumulator builds its own LinkCell per
/// frame.
///
/// @code{.cpp}
/// sim.on(md::Phase::end_of_step).add(RDF{3.0, 100, "rdf.dat"});
/// @endcode
class RDF : public AsyncObserver {
  public:
    /// @param r_max upper edge of the last bin [Å] (> 0). @param n_bins radial
    /// bin count (>= 1). @param filename g(r) output path. @param every
    /// sample cadence [steps]: the state after every `every`-th completed
    /// step, step 0 included.
    /// @throws std::invalid_argument on out-of-range parameters (fail-fast,
    /// no-fallback — refused by molpy `Rdf::new`).
    RDF(double r_max, int n_bins, std::string filename, int every = 1)
        : filename_(std::move(filename)), every_(every),
          acc_(cpu::compute::RDF{n_bins, r_max, /*r_min=*/0.0}.accumulator()) {
        if (every_ < 1)
            throw std::invalid_argument(
                "cuda::RDF: every must be >= 1 step, got "
                + std::to_string(every_));
    }

    ~RDF() override = default;
    RDF(RDF&&) noexcept = default;
    RDF& operator=(RDF&&) noexcept = default;

    std::string name() const override { return "rdf"; }
    int frequency() const override { return every_; }
    CaptureColumns capture_columns() const override {
        CaptureColumns c;
        c.positions = true;
        return c;
    }

    /// @brief Frames accumulated so far [count].
    long long n_frames() const { return acc_.n_frames(); }
    /// @brief Frames the molpy accumulator refused [count].
    int n_rejected() const { return n_rejected_; }

  protected:
    void consume(const CaptureView& v) override {
        const auto n = static_cast<std::size_t>(v.n_atoms);
        pos_.resize(3 * n);
        std::copy(v.x, v.x + n, pos_.begin());
        std::copy(v.y, v.y + n, pos_.begin() + static_cast<long>(n));
        std::copy(v.z, v.z + n, pos_.begin() + static_cast<long>(2 * n));
        std::array<double, 9> cell{};
        std::copy(v.cell, v.cell + 9, cell.begin());
        // A frame molpy refuses (degenerate cell) is a property of the
        // trajectory: counted for on_destroy(), it does not stop the run.
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
                         "[cuda::rdf] %d frame(s) rejected by the molpy "
                         "accumulator (shape mismatch / degenerate cell)\n",
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
    }

  private:
    std::string filename_;
    int every_;
    cpu::compute::RDFAccumulator acc_; ///< molpy streaming g(r) state
    int n_rejected_ = 0;               ///< frames the accumulator refused
    std::vector<double> pos_;          ///< blocked x|y|z packing scratch [Å]
};

} // namespace cuda
} // namespace ATV

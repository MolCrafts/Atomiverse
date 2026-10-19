/// @file cpu/observe/vacf.h
/// @brief CPU async velocity-autocorrelation-function observer (molrs-backed,
/// streaming).
///
/// The engine measures (each captured `FrameSnapshot`'s velocities are packed
/// into a reusable scratch); molrs analyzes (`compute::Vacf` through its
/// `CorrelatorAccumulator`: streaming lagged-product sums with the exact
/// mean-subtraction correction, equal to the batch FFT-ACF to round-off).
/// Memory is O(resolution · n_atoms) for the lag ring — never O(trajectory).
/// No analysis math in C++. The VACF is the VDOS / Green-Kubo-diffusion input,
/// complementing the Einstein route in cpu::MSD.
///
/// @see cpu::MSD (observer/msd.h) for the streaming async-observer pattern.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/asynchronous/async_observer.h"
#include "atomiverse/cpu/asynchronous/frame_snapshot.h"
#include "atomiverse/cpu/observe/constant_count_gate.h"
#include "atomiverse/cpu/observe/trajectory_metrics.h"

namespace ATV {

/// @brief MD async velocity-autocorrelation observer.
///
/// Streams velocities from each captured `FrameSnapshot` into the molrs VACF
/// accumulator; `finalize()` reads the mean-subtracted, unbiased,
/// DOF-averaged VACF(t) over lags `0..=resolution` and writes it to file.
/// Requires a constant atom count across the captured window (single-rank /
/// no migration).
class VACF : public cpu::md::AsyncObserver {
  public:
    /// @param dt         time between captured frames [fs] (> 0).
    /// @param resolution max lag [captured frames] (>= 1); bounds the velocity
    ///                   ring to O(resolution · n_atoms) memory. The run must
    ///                   capture at least `resolution + 1` frames — molrs
    ///                   refuses a lag with no origin rather than clamping it.
    /// @param freq       capture every `freq` steps (0 = every snapshot).
    /// @param output_file VACF(t) output path.
    /// @throws std::invalid_argument when `dt <= 0` or `resolution < 1`.
    explicit VACF(double dt, int resolution = 256, int freq = 0,
                  std::string output_file = "vacf.dat");

    /// @brief This analysis consumes a complete system, not a spatial shard.
    bool requires_full_frame() const override { return true; }
    std::string name() const override { return "vacf_async"; }
    /* role via Observer base */
    int frequency() const override { return freq_; }

    cpu::CaptureMask capture_mask() const override {
        cpu::CaptureMask m;
        m.tags = true;
        m.velocities = true;
        return m;
    }

    void on_snapshot(const cpu::FrameSnapshot& snap) override;
    void finalize() override;

    /// @brief The VACF(t) curve [(Å/fs)²] over lags `0..=resolution`; empty
    /// until a successful `finalize()`. For inspection/tests.
    const std::vector<double>& curve() const { return vacf_curve_; }
    int n_frames() const { return n_frames_; }

  private:
    double dt_;
    int resolution_;
    int freq_;
    std::string output_file_;

    cpu::CorrelationInput input_;
    bool invalid_input_ = false;
    cpu::ConstantCountGate gate_;  ///< constant-atom-count admission gate
    cpu::RejectedFrames rejected_; ///< frames the molrs accumulator refused
    std::vector<double> scratch_;  ///< reused blocked vx|vy|vz buffer
    cpu::compute::CorrelatorAccumulator acc_; ///< molrs streaming VACF state
    int n_frames_ = 0;
    std::vector<double> vacf_curve_;
};

} // namespace ATV

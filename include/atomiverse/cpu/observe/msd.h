/// @file cpu/observe/msd.h
/// @brief CPU async mean-squared-displacement observer (molpy-backed,
/// streaming).
///
/// The C++ engine *measures* (each captured `FrameSnapshot`'s positions and
/// cell are packed into a reusable scratch on the analysis worker thread);
/// molpy *analyzes* (`compute::Msd` over all time origins — the windowed
/// estimator — folded frame-by-frame through its `CorrelatorAccumulator`, then
/// the Einstein slope through `compute::LinearFit`). Memory is
/// O(window · n_atoms) for the lag ring — never O(trajectory). No analysis
/// math and no record type live in C++ — see molpy crates/molrs/src/compute/time_series/.
///
/// Lossless: every step on the cadence reaches the accumulator through the
/// driver's shared capture buffer (`cpu::CaptureBuffer`); a slow analysis
/// makes the run wait, it never skips a frame.
///
/// @see cpu::VACF (observer/vacf.h) for the same streaming async pattern.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/asynchronous/async_observer.h"
#include "atomiverse/cpu/observe/constant_count_gate.h"
#include "atomiverse/cpu/observe/trajectory_metrics.h"

namespace ATV {

/// @brief MD async mean-squared-displacement observer.
///
/// Streams positions + cell from each captured `FrameSnapshot` into the molpy
/// MSD accumulator; `finalize()` reads the all-origins MSD(t) curve over lags
/// `0..=window`, composes the Einstein D = slope / (2·3) from a molpy
/// `LinearFit` over the 20–80 % lag window, and writes both to file.
///
/// The positions must be continuous across periodic boundaries: the cell is
/// passed with every frame, so molpy refuses a frame whose step from the
/// previous one exceeds half the cell height (a wrapped trajectory) and the
/// refusal is reported at `finalize()`.
///
/// Usage:
/// @code{.cpp}
/// cpu::md::MD sim;
/// sim.set_integrator(cpu::md::VelocityVerlet{0.001});
/// sim.set_evaluator(cpu::PairLJCut{0.0104, 3.4, 6.8});
/// MSD msd{/*dt_between_frames=*/0.1, /*freq=*/100, "msd.dat"};
/// sim.set_observer_ref(msd);
/// sim.build();  sim.run(frame, 100'000);
/// @endcode
class MSD : public cpu::md::AsyncObserver {
  public:
    /// @param dt          time between captured frames [fs] (= timestep * freq;
    ///                    > 0).
    /// @param freq        sample cadence [steps]: the state after every `freq`-th
    ///              completed step, step 0 included (0 = every step).
    /// @param output_file MSD(t) output path.
    /// @param window      max lag [captured frames] of the curve and the
    ///                    Einstein fit (>= 1); bounds the position ring to
    ///                    O(window · n_atoms) memory. The run must capture at
    ///                    least `window + 1` frames — molpy refuses a lag with
    ///                    no origin rather than clamping it.
    /// @throws std::invalid_argument when `dt <= 0` or `window < 1`.
    explicit MSD(double dt, int freq = 0, std::string output_file = "msd.dat",
                 int window = 256);

    /// @brief This analysis consumes a complete system, not a spatial shard.
    bool requires_full_frame() const override { return true; }
    std::string name() const override { return "msd_async"; }
    /* role via Observer base */
    int frequency() const override { return freq_; }

    CaptureColumns capture_columns() const override {
        CaptureColumns m;
        m.tags = true;
        m.positions = true;
        return m;
    }

    void on_snapshot(const cpu::FrameSnapshot& snap) override;
    void finalize() override;

    /// @brief The MSD(t) curve [Å²] over lags `0..=window` (index 0 = 0);
    /// empty until a successful `finalize()`. For inspection/tests.
    const std::vector<double>& curve() const { return msd_curve_; }
    /// @brief Einstein diffusion coefficient D [Å²/fs] from the last
    /// `finalize()`; NaN until one succeeds.
    double diffusion() const { return d_; }
    int n_frames() const { return n_frames_; }
    const std::string& output_file() const { return output_file_; }

  private:
    double dt_;
    int freq_;
    std::string output_file_;
    int window_;

    cpu::CorrelationInput input_;
    bool invalid_input_ = false;
    cpu::ConstantCountGate gate_;  ///< constant-atom-count admission gate
    cpu::RejectedFrames rejected_; ///< frames the molpy accumulator refused
    std::vector<double> scratch_;  ///< reused blocked x|y|z packing buffer
    cpu::compute::CorrelatorAccumulator acc_; ///< molpy streaming MSD state
    int n_frames_ = 0;
    std::vector<double> msd_curve_;
    double d_; ///< Einstein D from the last finalize() (NaN until then)
};

} // namespace ATV

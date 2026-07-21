/// @file cpu/observer/vacf.h
/// @brief CPU async velocity-autocorrelation-function observer (molrs-backed,
/// streaming).
///
/// The engine measures (each captured `FrameSnapshot`'s velocities are packed
/// into a reusable scratch); molrs analyzes (`VacfAccumulator`: streaming
/// lagged-product sums with exact mean-subtraction correction, matching the
/// batch FFT-ACF to round-off). Memory is O(resolution · n_atoms) for the lag
/// ring — never O(trajectory). No analysis math in C++. The VACF is the VDOS /
/// Green-Kubo-diffusion input, complementing the Einstein route in cpu::MSD.
///
/// @see cpu::MSD (observer/msd.h) for the streaming async-observer pattern.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/async/async_observer.h"
#include "atomiverse/cpu/async/frame_snapshot.h"
#include "atomiverse/cpu/observer/constant_count_gate.h"
#include "atomiverse/cpu/observer/trajectory_metrics.h"

namespace ATV {

/// @brief MD async velocity-autocorrelation observer.
///
/// Streams velocities from each captured `FrameSnapshot` into the molrs
/// `VacfAccumulator`; `finalize()` reads the DOF-averaged VACF(t) and writes
/// it to file. Requires a constant atom count across the captured window
/// (single-rank / no migration).
class VACF : public cpu::md::AsyncObserver {
  public:
    /// @param dt         time between captured frames [fs].
    /// @param resolution max lag [captured frames]; bounds the velocity ring
    ///                   to O(resolution · n_atoms) memory. Must be >= 1 — a
    ///                   streaming accumulator cannot defer "full trajectory"
    ///                   resolution (runs shorter than `resolution` frames
    ///                   resolve every lag they contain).
    /// @param freq       capture every `freq` steps (0 = every snapshot).
    /// @param output_file VACF(t) output path.
    /// @throws std::invalid_argument when `resolution < 1`.
    explicit VACF(double dt, int resolution = 256, int freq = 0,
                  std::string output_file = "vacf.dat");

    std::string name() const override { return "vacf_async"; }
    /* role via Observer base */
    int frequency() const override { return freq_; }

    cpu::CaptureMask capture_mask() const override {
        cpu::CaptureMask m;
        m.velocities = true;
        return m;
    }

    void on_snapshot(const cpu::FrameSnapshot& snap) override;
    void finalize() override;

    /// @brief The last computed VACF(t) curve. For inspection/tests.
    const std::vector<double>& curve() const { return vacf_curve_; }
    int n_frames() const { return n_frames_; }

  private:
    double dt_;
    int resolution_;
    int freq_;
    std::string output_file_;

    cpu::ConstantCountGate gate_;       ///< constant-atom-count admission gate
    std::vector<double> scratch_;       ///< reused blocked vx|vy|vz buffer
    cpu::compute::VACFAccumulator acc_; ///< molrs streaming VACF state
    int n_frames_ = 0;
    std::vector<double> vacf_curve_;
};

} // namespace ATV

/// @file cpu/observe/msd.h
/// @brief CPU async mean-squared-displacement observer (molrs-backed,
/// streaming).
///
/// LAMMPS `compute msd`-shaped: the C++ engine *measures* (each captured
/// `FrameSnapshot`'s positions are packed into a reusable scratch on the
/// analysis worker thread); molrs *analyzes* (`MsdAccumulator`: Direct-mode
/// curve with frame 0 as reference, plus windowed-MSD sums for the Einstein-D
/// fit, folded frame-by-frame). Memory is O(window · n_atoms) for the lag
/// ring — never O(trajectory). No analysis math and no record type live in
/// C++ — see molrs/src/compute/msd/.
///
/// Backend-uniform: regardless of the producing driver/backend, snapshots are
/// delivered to the CPU analysis executor's ring buffer (CUDA frames after
/// async D2H), so this observer is CPU-only by construction.
///
/// @see cpu::VACF (observer/vacf.h) for the same streaming async pattern.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/asynchronous/async_observer.h"
#include "atomiverse/cpu/asynchronous/frame_snapshot.h"
#include "atomiverse/cpu/observe/constant_count_gate.h"
#include "atomiverse/cpu/observe/trajectory_metrics.h"

namespace ATV {

/// @brief MD async mean-squared-displacement observer.
///
/// Streams positions from each captured `FrameSnapshot` into the molrs
/// `MsdAccumulator`; `finalize()` reads the Direct-mode MSD(t) curve and the
/// Einstein D (windowed-MSD slope fit) and writes them to file.
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
    /// @param dt          time between captured frames [fs] (= timestep * freq).
    /// @param freq        capture every `freq` steps (0 = every snapshot).
    /// @param output_file MSD(t) output path.
    /// @param window      max lag [captured frames] resolved by the windowed
    ///                    Einstein-D fit; bounds the position ring to
    ///                    O(window · n_atoms) memory. Runs shorter than
    ///                    `window` frames reproduce the full-trajectory
    ///                    windowed fit exactly. 0 disables the fit (D = NaN).
    /// @throws std::invalid_argument when `window` is negative.
    explicit MSD(double dt, int freq = 0, std::string output_file = "msd.dat",
                 int window = 256);

    std::string name() const override { return "msd_async"; }
    /* role via Observer base */
    int frequency() const override { return freq_; }

    cpu::CaptureMask capture_mask() const override {
        cpu::CaptureMask m;
        m.positions = true;
        return m;
    }

    void on_snapshot(const cpu::FrameSnapshot& snap) override;
    void finalize() override;

    /// @brief The last computed MSD(t) curve (index 0 = 0). For inspection/tests.
    const std::vector<double>& curve() const { return msd_curve_; }
    /// @brief Einstein diffusion coefficient D [Å²/fs] from the last
    /// finalize() (NaN if not yet computed / too few frames / window == 0).
    double diffusion() const { return d_; }
    int n_frames() const { return n_frames_; }
    const std::string& output_file() const { return output_file_; }

  private:
    double dt_;
    int freq_;
    std::string output_file_;
    int window_;

    cpu::ConstantCountGate gate_;      ///< constant-atom-count admission gate
    std::vector<double> scratch_;      ///< reused blocked x|y|z packing buffer
    cpu::compute::MSDAccumulator acc_; ///< molrs streaming MSD state
    int n_frames_ = 0;
    std::vector<double> msd_curve_;
    double d_ =
        0.0; ///< Einstein diffusion coefficient from the last finalize().
};

} // namespace ATV

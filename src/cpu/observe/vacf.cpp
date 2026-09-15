/// @file cpu/observe/vacf.cpp
/// @brief CPU async VACF observer — streams each snapshot into the molrs
/// VACF accumulator (bounded memory; all autocorrelation math in molrs).

#include "atomiverse/cpu/observe/vacf.h"

#include <cstddef>
#include <fstream>
#include <utility>

#include "atomiverse/core/log.h"

namespace ATV {

VACF::VACF(double dt, int resolution, int freq, std::string output_file)
    : dt_(dt), resolution_(resolution), freq_(freq),
      output_file_(std::move(output_file)), acc_(resolution) {}

void VACF::on_snapshot(const cpu::FrameSnapshot& snap) {
    const int n = snap.n_owned;
    if (n <= 0)
        return;
    if (!gate_.admit(n)) // constant atom count required (single-rank MVP)
        return;

    // Snapshot velocity fields are double (widened from StorageReal at capture).
    cpu::pack_blocked(scratch_, snap.vx, snap.vy, snap.vz, n);
    if (acc_.accumulate(scratch_))
        ++n_frames_;
}

void VACF::finalize() {
    if (gate_.dropped() > 0)
        ATV_WARN("[vacf] dropped {} snapshot(s) with changed atom count "
                 "(constant count required)",
                 gate_.dropped());
    if (n_frames_ < 2)
        return;

    vacf_curve_ = acc_.finalize(); // molrs (matches batch FFT-ACF)

    std::ofstream os(output_file_);
    os << "# VACF via molrs VacfAccumulator (DOF-averaged, streaming)\n";
    os << "# t[fs]  VACF   (frames=" << n_frames_
       << ", atoms=" << gate_.n_atoms() << ", max_lag<=" << resolution_
       << ")\n";
    for (std::size_t i = 0; i < vacf_curve_.size(); ++i)
        os << (dt_ * static_cast<double>(i)) << "  " << vacf_curve_[i] << '\n';
}

} // namespace ATV

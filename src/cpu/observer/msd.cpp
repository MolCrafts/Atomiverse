/// @file cpu/observer/msd.cpp
/// @brief CPU async MSD observer — streams each snapshot into the molrs
/// MSD accumulator (bounded memory; all analysis math in molrs).

#include "atomiverse/cpu/observer/msd.h"

#include <cstddef>
#include <fstream>
#include <utility>

#include "atomiverse/core/log.h"

namespace ATV {

MSD::MSD(double dt, int freq, std::string output_file, int window)
    : dt_(dt), freq_(freq), output_file_(std::move(output_file)),
      window_(window), acc_(window) {}

void MSD::on_snapshot(const cpu::FrameSnapshot& snap) {
    const int n = snap.n_owned;
    if (n <= 0)
        return;
    if (!gate_.admit(n)) // constant atom count required (single-rank MVP)
        return;

    // Snapshot position fields are double (widened from StorageReal at capture).
    cpu::pack_blocked(scratch_, snap.x, snap.y, snap.z, n);
    if (acc_.accumulate(scratch_))
        ++n_frames_;
}

void MSD::finalize() {
    if (gate_.dropped() > 0)
        ATV_WARN("[msd] dropped {} snapshot(s) with changed atom count "
                 "(constant count required)",
                 gate_.dropped());
    if (n_frames_ < 2)
        return;

    msd_curve_ = acc_.direct_curve(); // molrs, Direct mode (frame-0 reference)
    // Einstein D from the windowed-MSD slope over the 20-80% window of the max
    // resolved lag (<= window_; avoids the ballistic short-time and noisy
    // long-time tails). molrs owns the math.
    d_ = acc_.diffusion(dt_, /*dims=*/3, /*fit_lo=*/0.2, /*fit_hi=*/0.8);

    std::ofstream os(output_file_);
    os << "# MSD via molrs MsdAccumulator (Direct mode, frame-0 reference)\n";
    os << "# Einstein D (molrs, 20-80% fit of <=" << window_
       << "-frame lag window) = " << d_
       << " Angstrom^2/fs   (frames=" << n_frames_ << ")\n";
    os << "# t[fs]  MSD[Angstrom^2]\n";
    for (std::size_t i = 0; i < msd_curve_.size(); ++i)
        os << (dt_ * static_cast<double>(i)) << "  " << msd_curve_[i] << '\n';
}

} // namespace ATV

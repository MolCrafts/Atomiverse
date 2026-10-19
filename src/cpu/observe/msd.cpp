/// @file cpu/observe/msd.cpp
/// @brief CPU async MSD observer — streams each snapshot into the molrs
/// MSD accumulator (bounded memory; all analysis math in molrs).

#include "atomiverse/cpu/observe/msd.h"

#include <cstddef>
#include <exception>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

#include "atomiverse/core/log.h"

namespace ATV {

namespace {
/// Spatial dimensionality of the Einstein relation D = slope / (2·dims).
constexpr int kDims = 3;
/// Einstein fit window, fractions of the last lag: skips the ballistic
/// short-time head and the poorly averaged long-lag tail.
constexpr double kFitStart = 0.2;
constexpr double kFitEnd = 0.8;

double require_positive_dt(double dt) {
    if (!(dt > 0.0))
        throw std::invalid_argument(
            "MSD: dt (time between captured frames, fs) must be > 0");
    return dt;
}
} // namespace

MSD::MSD(double dt, int freq, std::string output_file, int window)
    : dt_(require_positive_dt(dt)), freq_(freq),
      output_file_(std::move(output_file)), window_(window),
      acc_(cpu::compute::MSD{window}.accumulator()),
      d_(std::numeric_limits<double>::quiet_NaN()) {}

void MSD::on_snapshot(const cpu::FrameSnapshot& snap) {
    if (invalid_input_)
        return;
    const int n = snap.n_owned;
    if (n <= 0)
        return;
    if (!gate_.admit(n)) // constant atom count required (single-rank MVP)
        return;

    // Snapshot position fields are double (widened from StorageReal at capture).

    // Worker thread: a molrs refusal (wrapped step, bad frame) is tallied for
    // finalize(), never thrown out of the analysis executor.
    try {
        input_.pack(scratch_, snap.tag, snap.x, snap.y, snap.z, n, snap.step,
                    freq_ > 0 ? freq_ : 1);
        acc_.push(scratch_, dt_, cpu::cell_matrix(snap.box));
        ++n_frames_;
    } catch (const std::exception& e) {
        rejected_.record(e);
        invalid_input_ = true;
    }
}

void MSD::finalize() {
    if (invalid_input_) {
        ATV_WARN("[msd] invalid trajectory input: {}; result not written",
                 rejected_.first_reason());
        return;
    }
    if (gate_.dropped() > 0)
        ATV_WARN("[msd] dropped {} snapshot(s) with changed atom count "
                 "(constant count required)",
                 gate_.dropped());
    if (rejected_.count() > 0)
        ATV_WARN("[msd] molrs refused {} snapshot(s); first: {}",
                 rejected_.count(), rejected_.first_reason());
    if (n_frames_ <= window_) {
        ATV_WARN("[msd] {} frame(s) accumulated; lags 0..{} need at least {} "
                 "— lower `window` or capture more frames. {} not written",
                 n_frames_, window_, window_ + 1, output_file_);
        return;
    }

    // finalize() runs from the Runtime's shutdown (a destructor path), so a
    // molrs error is reported, not thrown.
    try {
        const cpu::compute::TimeCorrelation curve = acc_.finish();
        // Einstein D: molrs fits the curve; the relation D = slope / (2·dims)
        // is the composition molrs' Msd documents for the caller.
        const cpu::compute::LinearFitResult fit =
            cpu::compute::LinearFit{kFitStart, kFitEnd}.compute(curve);
        msd_curve_ = curve.values();
        d_ = fit.slope / (2.0 * kDims);

        const std::vector<double> lag_times = curve.lag_times(); // [fs]
        std::ofstream os(output_file_);
        os << "# MSD via molrs Msd (all time origins, lags 0.." << window_
           << ")\n";
        os << "# Einstein D (molrs LinearFit over lags " << fit.fit_start
           << ".." << fit.fit_end << ") = " << d_
           << " Angstrom^2/fs   (frames=" << n_frames_ << ")\n";
        os << "# t[fs]  MSD[Angstrom^2]\n";
        for (std::size_t i = 0; i < msd_curve_.size(); ++i)
            os << lag_times[i] << "  " << msd_curve_[i] << '\n';
    } catch (const std::exception& e) {
        ATV_WARN("[msd] {} not written: {}", output_file_, e.what());
    }
}

} // namespace ATV

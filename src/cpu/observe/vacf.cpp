/// @file cpu/observe/vacf.cpp
/// @brief CPU async VACF observer — streams each snapshot into the molpy
/// VACF accumulator (bounded memory; all autocorrelation math in molpy).

#include "atomiverse/cpu/observe/vacf.h"

#include <cstddef>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <utility>

#include "atomiverse/core/log.h"

namespace ATV {

namespace {
double require_positive_dt(double dt) {
    if (!(dt > 0.0))
        throw std::invalid_argument(
            "VACF: dt (time between captured frames, fs) must be > 0");
    return dt;
}
} // namespace

VACF::VACF(double dt, int resolution, int freq, std::string output_file)
    : dt_(require_positive_dt(dt)), resolution_(resolution), freq_(freq),
      output_file_(std::move(output_file)),
      acc_(cpu::compute::VACF{resolution}.accumulator()) {}

void VACF::on_snapshot(const cpu::FrameSnapshot& snap) {
    if (invalid_input_)
        return;
    const int n = snap.n_owned;
    if (n <= 0)
        return;
    if (!gate_.admit(n)) // constant atom count required (single-rank MVP)
        return;

    // Snapshot velocity fields are double (widened from StorageReal at capture).

    // Worker thread: a molpy refusal is tallied for finalize(), not a
    // reason to stop the run.
    try {
        input_.pack(scratch_, snap.tag, snap.vx, snap.vy, snap.vz, n, snap.step,
                    freq_ > 0 ? freq_ : 1);
        acc_.push(scratch_, dt_);
        ++n_frames_;
    } catch (const std::exception& e) {
        rejected_.record(e);
        invalid_input_ = true;
    }
}

void VACF::finalize() {
    if (invalid_input_) {
        ATV_WARN("[vacf] invalid trajectory input: {}; result not written",
                 rejected_.first_reason());
        return;
    }
    if (gate_.rejected() > 0)
        ATV_WARN("[vacf] rejected {} snapshot(s) with a changed atom count "
                 "(constant count required)",
                 gate_.rejected());
    if (rejected_.count() > 0)
        ATV_WARN("[vacf] molpy refused {} snapshot(s); first: {}",
                 rejected_.count(), rejected_.first_reason());
    if (n_frames_ <= resolution_) {
        ATV_WARN("[vacf] {} frame(s) accumulated; lags 0..{} need at least {} "
                 "— lower `resolution` or capture more frames. {} not written",
                 n_frames_, resolution_, resolution_ + 1, output_file_);
        return;
    }

    // finalize() runs from the Runtime's shutdown (a destructor path), so a
    // molpy error is reported, not thrown.
    try {
        const cpu::compute::TimeCorrelation curve = acc_.finish();
        vacf_curve_ = curve.values();
        const std::vector<double> lag_times = curve.lag_times(); // [fs]

        std::ofstream os(output_file_);
        os << "# VACF via molpy Vacf (mean-subtracted, unbiased, "
              "DOF-averaged, streaming)\n";
        os << "# t[fs]  VACF   (frames=" << n_frames_
           << ", atoms=" << gate_.n_atoms() << ", max_lag=" << resolution_
           << ")\n";
        for (std::size_t i = 0; i < vacf_curve_.size(); ++i)
            os << lag_times[i] << "  " << vacf_curve_[i] << '\n';
    } catch (const std::exception& e) {
        ATV_WARN("[vacf] {} not written: {}", output_file_, e.what());
    }
}

} // namespace ATV

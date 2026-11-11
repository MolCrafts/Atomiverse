/// @file cpu/observe/rdf.cpp
/// @brief CPU async RDF observer — streams each snapshot into the molpy
/// g(r) accumulator (bounded memory; all binning + normalization in molpy).

#include "atomiverse/cpu/observe/rdf.h"

#include <exception>
#include <utility>

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/observe/rdf_curve.h"

namespace ATV {

RDF::RDF(double r_max, int n_bins, int freq, std::string output_file)
    : freq_(freq), output_file_(std::move(output_file)),
      acc_(cpu::compute::RDF{n_bins, r_max, /*r_min=*/0.0}.accumulator()) {}

void RDF::on_snapshot(const cpu::FrameSnapshot& snap) {
    const int n = snap.n_owned;
    if (n < 2)
        return;
    if (!gate_.admit(n)) // constant atom count required (single-rank MVP)
        return;

    cpu::pack_blocked(scratch_, snap.x, snap.y, snap.z, n);
    // Worker thread: a molpy refusal (bad frame, singular cell) is tallied
    // for finalize(), not a reason to stop the run.
    try {
        acc_.push(scratch_, cpu::cell_matrix(snap.box));
        ++n_frames_;
    } catch (const std::exception& e) {
        rejected_.record(e);
    }
}

void RDF::finalize() {
    if (gate_.rejected() > 0)
        ATV_WARN("[rdf] rejected {} snapshot(s) with a changed atom count "
                 "(constant count required)",
                 gate_.rejected());
    if (rejected_.count() > 0)
        ATV_WARN("[rdf] molpy refused {} snapshot(s); first: {}",
                 rejected_.count(), rejected_.first_reason());
    if (n_frames_ == 0)
        return;

    // All binning + shell-volume normalization happened in molpy. finalize()
    // runs from the Runtime's shutdown (a destructor path), so a molpy error
    // is reported, not thrown.
    try {
        const cpu::compute::RDFResult result = acc_.finish();
        gr_ = result.rdf;
        cpu::rdf_write_curve(output_file_, result.bin_centers, gr_);
    } catch (const std::exception& e) {
        ATV_WARN("[rdf] {} not written: {}", output_file_, e.what());
    }
}

} // namespace ATV

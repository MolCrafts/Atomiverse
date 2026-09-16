/// @file cpu/observe/rdf.cpp
/// @brief CPU async RDF observer — streams each snapshot into the molrs
/// g(r) accumulator (bounded memory; all binning + normalization in molrs).

#include "atomiverse/cpu/observe/rdf.h"

#include <array>

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/observe/rdf_curve.h"

namespace ATV {

RDF::RDF(double r_max, int n_bins, int freq, std::string output_file)
    : r_max_(r_max), n_bins_(n_bins), freq_(freq),
      output_file_(std::move(output_file)),
      bin_width_(r_max / static_cast<double>(n_bins)),
      acc_(n_bins, r_max, /*r_min=*/0.0) {}

void RDF::on_snapshot(const cpu::FrameSnapshot& snap) {
    const int n = snap.n_owned;
    if (n < 2)
        return;
    if (!gate_.admit(n)) // constant atom count required (single-rank MVP)
        return;

    cpu::pack_blocked(scratch_, snap.x, snap.y, snap.z, n);

    std::array<double, 9> cell{};
    const auto& c = snap.box.cell();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            cell[static_cast<std::size_t>(i * 3 + j)] =
                static_cast<double>(c(i, j));

    if (acc_.accumulate(scratch_, cell))
        ++n_frames_;
}

void RDF::finalize() {
    if (gate_.dropped() > 0)
        ATV_WARN("[rdf] dropped {} snapshot(s) with changed atom count "
                 "(constant count required)",
                 gate_.dropped());
    if (n_frames_ == 0)
        return;

    // All binning + shell-volume normalization happened in molrs.
    gr_ = acc_.finalize();
    if (gr_.empty())
        return;
    cpu::rdf_write_curve(output_file_, gr_, /*r_min=*/0.0, bin_width_, n_bins_);
}

} // namespace ATV

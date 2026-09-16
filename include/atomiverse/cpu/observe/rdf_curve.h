/// @file cpu/observe/rdf_curve.h
/// @brief rdf_write_curve — the single emitter of the g(r) file format.
///
/// std-only and CUDA-safe (no molrs / `rust/cxx.h`, no CPU async observer
/// stack), so `cuda/observer/rdf.cuh` can include it directly. Both backends'
/// RDF observers stream frames through the molrs accumulator
/// (`cpu::compute::RDFAccumulator` in observer/trajectory_metrics.h) and emit
/// the normalized curve through this one function — no g(r) analysis math
/// lives in C++.
#pragma once

#include <string>
#include <vector>

namespace ATV {
namespace cpu {

/// @brief Write a normalized g(r) curve to file — the single emitter of the
/// g(r) file format ("# r [Angstrom]   g(r)").
/// @param filename output path. @param gr normalized g(r), one value per bin
/// (missing tail bins are written as 0). @param r_min lower edge of bin 0 [Å].
/// @param bin_width radial bin width [Å]. @param n_bins number of lines to
/// write (bin-center radii `r_min + (i + 0.5)·bin_width`).
void rdf_write_curve(const std::string& filename, const std::vector<double>& gr,
                     double r_min, double bin_width, int n_bins);

} // namespace cpu
} // namespace ATV

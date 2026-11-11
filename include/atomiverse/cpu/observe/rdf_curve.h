/// @file cpu/observe/rdf_curve.h
/// @brief rdf_write_curve — the single emitter of the g(r) file format.
///
/// std-only and CUDA-safe (no molpy / `rust/cxx.h`, no CPU async observer
/// stack), so `cuda/observer/rdf.cuh` can include it directly. Both backends'
/// RDF observers stream frames through the molpy accumulator
/// (`cpu::compute::RDFAccumulator` in observe/trajectory_metrics.h) and emit
/// the normalized curve, with molpy' own bin centers, through this one
/// function — no g(r) analysis math lives in C++.
#pragma once

#include <string>
#include <vector>

namespace ATV {
namespace cpu {

/// @brief Write a normalized g(r) curve to file — the single emitter of the
/// g(r) file format ("# r [Angstrom]   g(r)").
/// @param filename output path. @param r bin-center radii [Å], as molpy
/// returns them (`RDFResult::bin_centers`). @param gr normalized g(r), one
/// value per bin (`RDFResult::rdf`).
/// @throws std::invalid_argument when `r` and `gr` differ in length.
void rdf_write_curve(const std::string& filename, const std::vector<double>& r,
                     const std::vector<double>& gr);

} // namespace cpu
} // namespace ATV

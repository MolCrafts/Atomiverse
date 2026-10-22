/// @file cpu/dynamics/internal/gauss_stream.h
/// @brief Private counter-keyed Gaussian stream shared by stochastic integrators.
#pragma once
#include "atomiverse/core/units.h"
#include <cmath>
#include <cstdint>

namespace ATV::cpu::md::internal {

/// @brief Per-atom Gaussian stream with Box-Muller pair caching.
///
/// Deliberately constructed once per atom inside the `launch()` loop so
/// each atom's kick is seeded from `(step, tag)` — two ranks that own
/// the same global atom (different row indices, same tag) produce the
/// same random force, preserving cross-rank reproducibility without
/// needing an MPI reduction. The construction cost is a single
/// `uint64_t` store on the stack; `next()`'s `log / cos / sqrt` triple
/// dominates per-atom cost.
///
/// Pair caching: a single Box-Muller call returns two IID gaussians
/// (the `sin` and `cos` partners); we consume one and stash the other,
/// halving the Box-Muller invocations per atom (3 gaussians needed ⇒
/// 2 invocations instead of 3). A dedicated HPC pass can drop the
/// `log / cos` entirely by switching to Marsaglia polar; keep the
/// current form here so the statistical properties stay identical to
/// the 12 existing Langevin tests.
struct GaussStream {
    std::uint64_t s;
    bool has_cached = false;
    double cached = 0.0;
    double next_u01() {
        s += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
    }
    double next() {
        if (has_cached) {
            has_cached = false;
            return cached;
        }
        const double u = next_u01();
        const double v = next_u01();
        const double r = std::sqrt(-2.0 * std::log(u > 1e-300 ? u : 1e-300));
        const double theta = 2.0 * ATV::units::pi * v;
        cached = r * std::sin(theta);
        has_cached = true;
        return r * std::cos(theta);
    }
};

} // namespace ATV::cpu::md::internal

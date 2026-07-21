/// @file cpu/scf/gradient_result.h
/// @brief `GradientResult` — the public forces-API return type (rehomed from the
///        since-deleted scf_result.h; the public gradient.h stays a
///        tombstone).
#pragma once

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Analytic / FD nuclear gradient result (public forces API).
struct GradientResult {
    xt::xtensor<double, 2> forces; ///< [n_atoms, 3] eV/Å
    double energy = 0;             ///< eV
};

} // namespace scf
} // namespace cpu
} // namespace ATV

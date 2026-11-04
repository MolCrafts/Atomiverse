/// @file cpu/minimize/vector_loop.h
/// @brief When an optimizer's per-element loop is worth an OpenMP team.
#pragma once

namespace ATV {
namespace cpu {
namespace minimizer {

/// Optimizer vector loops (positions, gradients, inner products, the L-BFGS
/// two-loop recursion) stream a few nanoseconds of work per element. Below
/// this many elements waking the team and joining it costs more than the
/// loop itself — and a minimizer runs a dozen such loops per iteration — so
/// they carry `if (n >= kParallelMinElements)` and stay on the calling thread.
inline constexpr int kParallelMinElements = 32768;

} // namespace minimizer
} // namespace cpu
} // namespace ATV

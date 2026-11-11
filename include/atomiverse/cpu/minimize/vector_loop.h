/// @file cpu/minimize/vector_loop.h
/// @brief Optimizer vector loops: when they are worth an OpenMP team, and the
///        owned-row reductions LBFGS / CG / FIRE / LbfgsHistory share.
///
/// Every reduction here is a per-rank partial; the caller combines it with
/// `global_sum` / `global_max` (`minimizer/global_reduce.h`).
///
/// **Reassociation.** The library builds with strict IEEE FP, under which a
/// `s += a*b` loop is one serial add chain the compiler may not reorder — about
/// 1 ns per element on Apple M4, slower than the force kernel of a cheap
/// potential. `omp simd reduction` licenses exactly that reordering and no
/// other: sums become a fixed lane-wise tree (deterministic for a given length
/// and build, a few ULP from the sequential order), maxima are exact. A sum and
/// a max are kept in separate loops because Clang vectorises neither when they
/// share one under strict FP; two vector passes cost ~3x less than one scalar
/// pass.
#pragma once

#include <cmath>

namespace ATV {
namespace cpu {
namespace minimizer {

/// Optimizer vector loops (positions, gradients, inner products, the L-BFGS
/// two-loop recursion) stream a few nanoseconds of work per element. Below
/// this many elements waking the team and joining it costs more than the
/// loop itself — and a minimizer runs a dozen such loops per iteration — so
/// they carry `if (parallel : n >= kParallelMinElements)` and stay on the
/// calling thread. The `parallel :` modifier matters: a bare `if` on a
/// combined `parallel for simd` also switches the `simd` part off.
inline constexpr int kParallelMinElements = 32768;

/// @brief Local inner product `a·b` over `[0, n)`.
inline double dot_local(const double* __restrict a, const double* __restrict b,
                        int n) {
    double s = 0;
#pragma omp parallel for simd schedule(static)                                 \
    reduction(+ : s) if (parallel : n >= kParallelMinElements)
    for (int j = 0; j < n; ++j)
        s += a[j] * b[j];
    return s;
}

/// @brief Local `max_j |a_j|` over `[0, n)`.
inline double max_abs_local(const double* __restrict a, int n) {
    double m = 0;
#pragma omp parallel for simd schedule(static)                                 \
    reduction(max : m) if (parallel : n >= kParallelMinElements)
    for (int j = 0; j < n; ++j)
        m = std::fmax(m, std::abs(a[j]));
    return m;
}

/// @brief Local `Σ|f_i|²` and `max_i |f_i|²` over rows `[0, n)` of three
///        force columns. `sqrt(max |f|²)` is bit-identical to `max |f|`.
template <class T>
inline void force_norms_local(const T* __restrict fx, const T* __restrict fy,
                              const T* __restrict fz, int n, double& sum_f2,
                              double& max_f2) {
    double s = 0, m = 0;
#pragma omp parallel for simd schedule(static)                                 \
    reduction(+ : s) if (parallel : n >= kParallelMinElements)
    for (int i = 0; i < n; ++i) {
        const double x = fx[i], y = fy[i], z = fz[i];
        s += x * x + y * y + z * z;
    }
#pragma omp parallel for simd schedule(static)                                 \
    reduction(max : m) if (parallel : n >= kParallelMinElements)
    for (int i = 0; i < n; ++i) {
        const double x = fx[i], y = fy[i], z = fz[i];
        m = std::fmax(m, x * x + y * y + z * z);
    }
    sum_f2 = s;
    max_f2 = m;
}

/// @brief Local `Σ|r_i - p_i|²` and the largest `|component|` of `r - p` over
///        rows `[0, n)`; `prev` is the interleaved `x0 y0 z0 x1 ...` origin.
template <class T>
inline void
displacement_norms_local(const T* __restrict x, const T* __restrict y,
                         const T* __restrict z, const double* __restrict prev,
                         int n, double& sum_d2, double& max_d) {
    double s = 0, m = 0;
#pragma omp parallel for simd schedule(static)                                 \
    reduction(+ : s) if (parallel : n >= kParallelMinElements)
    for (int i = 0; i < n; ++i) {
        const double dx = x[i] - prev[i * 3 + 0];
        const double dy = y[i] - prev[i * 3 + 1];
        const double dz = z[i] - prev[i * 3 + 2];
        s += dx * dx + dy * dy + dz * dz;
    }
#pragma omp parallel for simd schedule(static)                                 \
    reduction(max : m) if (parallel : n >= kParallelMinElements)
    for (int i = 0; i < n; ++i) {
        const double dx = x[i] - prev[i * 3 + 0];
        const double dy = y[i] - prev[i * 3 + 1];
        const double dz = z[i] - prev[i * 3 + 2];
        m = std::fmax(
            m, std::fmax(std::abs(dx), std::fmax(std::abs(dy), std::abs(dz))));
    }
    sum_d2 = s;
    max_d = m;
}

} // namespace minimizer
} // namespace cpu
} // namespace ATV

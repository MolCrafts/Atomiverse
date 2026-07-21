/// @file core/gto.h
/// @brief Gaussian-type orbital normalization functions.
///
/// Pure math, no backend dependency. Used by both CPU and CUDA code.
/// CUDA-free header (uses only the C++ `<cmath>` header).
#pragma once

#include <cmath>

namespace ATV {

/// @brief Physical GTO normalization: N = (2*alpha/pi)^{3/4} for s-type (l=0).
/// @details The standard textbook normalization ensuring <g|g> = 1 for a
///   single primitive Gaussian exp(-alpha*r^2).
/// @param alpha Gaussian exponent.
/// @return Normalization factor.
inline double phys_norm_s(double alpha) {
    return std::pow(2.0 * alpha / M_PI, 0.75);
}

/// @brief Physical GTO normalization for p-type (l=1).
/// @details N = (128*alpha^5/pi^3)^{1/4} ensuring <x*g|x*g> = 1.
/// @param alpha Gaussian exponent.
/// @return Normalization factor.
inline double phys_norm_p(double alpha) {
    return std::pow(128.0 * std::pow(alpha, 5) / (M_PI * M_PI * M_PI), 0.25);
}

/// @brief GTO normalization factor for Cartesian d shells (l=2).
/// @details Shared across all six Cartesian components (d_xx .. d_zz)
///   using libcint / PySCF convention — the "off-diagonal" norm where
///   <d_xy|d_xy> = 1 but <d_xx|d_xx> = 3 (extra factor (2*2-1)!! = 3).
///   SCF handles the non-identity overlap naturally via the generalized
///   eigenproblem, so the convention does not need to be "fixed" at
///   integral time; it does need to MATCH libcint/PySCF bit-for-bit.
///   N = (2*alpha/pi)^{3/4} * 4*alpha, i.e. the l_x=l_y=1 Cartesian
///   primitive normalization.
/// @param alpha Gaussian exponent.
/// @return Normalization factor.
inline double phys_norm_d(double alpha) {
    return std::pow(2.0 * alpha / M_PI, 0.75) * 4.0 * alpha;
}

/// @brief GTO normalization factor for Cartesian f shells (l=3).
/// @details Shared across all ten Cartesian components (f_xxx .. f_zzz)
///   using the libcint / PySCF convention — the shell normalization is
///   fixed by the "all-off-diagonal" component f_xyz (l_x=l_y=l_z=1),
///   whose triple product of double factorials (2*1-1)!!^3 = 1; the
///   diagonal components then carry the extra (2l_d-1)!! factors in
///   their self-overlap. Generalizes phys_norm_d:
///   N = (2*alpha/pi)^{3/4} * (4*alpha)^{l/2} with l = 3, i.e.
///   N = (2*alpha/pi)^{3/4} * (4*alpha)^{3/2}.
/// @param alpha Gaussian exponent.
/// @return Normalization factor.
inline double phys_norm_f(double alpha) {
    return std::pow(2.0 * alpha / M_PI, 0.75) * std::pow(4.0 * alpha, 1.5);
}

/// @brief GTO normalization factor for Cartesian g shells (l=4).
/// @details Shared across all fifteen Cartesian components
///   (g_xxxx .. g_zzzz), extending the libcint / PySCF cart convention
///   of phys_norm_d / phys_norm_f one rung up the
///   N = (2*alpha/pi)^{3/4} * (4*alpha)^{l/2} ladder with l = 4:
///   N = (2*alpha/pi)^{3/4} * (4*alpha)^2. Needed by the def2 JKFIT
///   auxiliary basis (density fitting), whose first/second-row rows
///   carry one g shell each; validated against PySCF `cart=True`
///   2-/3-center integrals in tests/cpu/scf/test_density_fitting.cpp.
/// @param alpha Gaussian exponent [Bohr^-2].
/// @return Normalization factor.
inline double phys_norm_g(double alpha) {
    return std::pow(2.0 * alpha / M_PI, 0.75) * 16.0 * alpha * alpha;
}

/// @brief GTO normalization factor (libcint convention, matches CINTgto_norm).
/// @details Includes the 2*sqrt(pi) spherical harmonic factor on top of phys_norm.
///   For validation against libcint only.
/// @param l Angular momentum quantum number.
/// @param alpha Gaussian exponent.
/// @return Normalization factor.
inline double gto_norm(int l, double alpha) {
    double n1 = (2 * l + 2 + 1) * 0.5;
    double gi = std::exp(std::lgamma(n1)) / (2.0 * std::pow(2.0 * alpha, n1));
    return 1.0 / std::sqrt(gi);
}

/// @brief Boys function F_0(x) = integral_0^1 exp(-x*t^2) dt.
/// @param x Non-negative argument.
/// @return F_0(x).
inline double boys_f0(double x) {
    if (x < 1e-12)
        return 1.0;
    double sx = std::sqrt(x);
    return 0.5 * std::sqrt(M_PI / x) * std::erf(sx);
}

/// @brief General Boys function F_n(x) = integral_0^1 t^{2n} exp(-x*t^2) dt.
/// @details Uses upward recurrence from F_0:
///   F_n(x) = [(2n-1)*F_{n-1}(x) - exp(-x)] / (2x)  for x > threshold.
///   F_n(0) = 1/(2n+1).
/// @param n Order (n >= 0).
/// @param x Non-negative argument.
/// @return F_n(x).
inline double boys_fn(int n, double x) {
    if (n == 0)
        return boys_f0(x);
    if (x < 1e-12)
        return 1.0 / (2 * n + 1);
    double ex = std::exp(-x);
    double inv2x = 0.5 / x;
    double fn = boys_f0(x);
    for (int k = 1; k <= n; ++k)
        fn = ((2 * k - 1) * fn - ex) * inv2x;
    return fn;
}

} // namespace ATV

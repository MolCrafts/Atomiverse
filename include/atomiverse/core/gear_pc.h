/// @file core/gear_pc.h
/// @brief Fifth-order Gear predictor–corrector for first-order ODEs
///        (Nordsieck form). Host and device share these functions.
///
/// One variable \(y(t)\) with \(\dot y = f(y, t)\) carries the scaled
/// Nordsieck vector
/// \[
///   z_k = \frac{h^k}{k!}\,\frac{\mathrm{d}^k y}{\mathrm{d}t^k},\qquad
///   k = 0,\dots,5,
/// \]
/// \(h\) the step. A step is
///  1. predict: \(z \leftarrow P z\), \(P\) the upper-triangular Pascal
///     matrix (exact Taylor shift of a quintic),
///  2. evaluate \(f\) at the predicted state,
///  3. correct: \(\Delta = h f - z_1\), \(z_k \leftarrow z_k + c_k \Delta\).
///
/// The corrector \(c = (95/288, 1, 25/24, 35/72, 5/48, 1/120)\) is Gear's
/// six-value set for FIRST-order equations (C. W. Gear, *Numerical Initial
/// Value Problems in Ordinary Differential Equations*, Prentice-Hall 1971;
/// Allen & Tildesley, *Computer Simulation of Liquids*, App. E): the
/// Nordsieck form of the fifth-order Adams–Moulton method, so the step
/// matrix \((I - c\,e_1^{T})P\) has the eigenvalues 1, 0, 0, 0, 0, 0.
/// It is not the second-order set (3/16, 251/360, 1, 11/18, 1/6, 1/60): the
/// equations of motion are integrated as the first-order system (r, v),
/// which the Gaussian and configurational thermostats need (their
/// multipliers enter \(\dot v\) and \(\dot r\) separately).
///
/// Global error is \(O(h^{6})\) only with an accurate start; the higher
/// derivatives are usually unknown at \(t = 0\) and start at zero, which
/// leaves a start-up offset of lower order (see `GearPC`).
#pragma once

#if defined(__CUDACC__)
#define ATV_MD_HD __host__ __device__
#else
#define ATV_MD_HD
#endif

namespace ATV::md {

/// @brief Number of Nordsieck values per variable (y and five derivatives).
inline constexpr int kGearValues = 6;

/// @brief Predictor: \(z \leftarrow P z\) in place (Pascal triangle).
/// @param z Scaled Nordsieck vector, \(z_0 = y\) … \(z_5\).
ATV_MD_HD inline void gear_predict(double* z) {
    for (int i = 0; i < kGearValues - 1; ++i)
        for (int j = kGearValues - 2; j >= i; --j)
            z[j] += z[j + 1];
}

/// @brief Corrector: \(z_k \leftarrow z_k + c_k (h f - z_1)\).
/// @param z      Predicted Nordsieck vector (corrected in place).
/// @param h_rate Step times the right-hand side at the predicted state,
///               \(h f(y_p)\) (same units as \(y\)).
///
/// Written out rather than looped over a coefficient table so device code
/// needs no constant-memory array.
ATV_MD_HD inline void gear_correct(double* z, double h_rate) {
    const double delta = h_rate - z[1];
    z[0] += (95.0 / 288.0) * delta;
    z[1] += delta;
    z[2] += (25.0 / 24.0) * delta;
    z[3] += (35.0 / 72.0) * delta;
    z[4] += (5.0 / 48.0) * delta;
    z[5] += (1.0 / 120.0) * delta;
}

} // namespace ATV::md

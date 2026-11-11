/// @file core/fene.h
/// @brief Kremer–Grest FENE, one formula for the CPU and CUDA bond kernels.
///
/// \(U=-\frac12 K R_0^2\ln(1-(r/R_0)^2)\) for \(r<R_0\).
/// `fpair` is \(F_i / (x_i-x_j) = -(dU/dr)/r = -K/(1-(r/R_0)^2)\).
#pragma once

#include <cmath>

namespace ATV {

template <typename T>
struct FeneTerm {
    T energy;
    T fpair;
    bool overstretched;
};

#if defined(__CUDACC__)
#define ATV_FENE_HD __host__ __device__
#else
#define ATV_FENE_HD
#endif

template <typename T>
ATV_FENE_HD inline FeneTerm<T> fene_term(T r, T K, T R0) {
    const T one_m = T(1) - (r / R0) * (r / R0);
    if (!(r < R0) || !(one_m > T(0)))
        return FeneTerm<T>{T(0), T(0), true};
#if defined(__CUDA_ARCH__)
    const T energy = T(-0.5) * K * R0 * R0 * log(one_m);
#else
    const T energy = T(-0.5) * K * R0 * R0 * std::log(one_m);
#endif
    // Algebraically -K/one_m, including the r→0 limit of -(dU/dr)/r.
    return FeneTerm<T>{energy, -K / one_m, false};
}

#undef ATV_FENE_HD

} // namespace ATV

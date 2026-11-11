/// @file cuda/semiempirical/nddo_derivative.cuh
/// @brief Bounded device-only Cartesian differentiation of NDDO pair laws.
#pragma once
#include <cmath>
#include <type_traits>
namespace ATV::cuda::semiempirical::internal::detail {
using ::exp;
using ::fabs;
using ::pow;
using ::sqrt;
// Trivial default construction permits statically allocated shared storage.
// Constants have zero derivatives; no tape, heap, or SCF differentiation.
struct PairDerivative {
    double value;
    double gradient[3];
    __device__ PairDerivative() = default;
    __device__ PairDerivative(double v) : value(v), gradient{} {}
    __device__ PairDerivative& operator+=(const PairDerivative& b) {
        value += b.value;
        for (int d = 0; d < 3; ++d)
            gradient[d] += b.gradient[d];
        return *this;
    }
    __device__ PairDerivative& operator-=(const PairDerivative& b) {
        value -= b.value;
        for (int d = 0; d < 3; ++d)
            gradient[d] -= b.gradient[d];
        return *this;
    }
    __device__ PairDerivative& operator*=(const PairDerivative& b) {
        for (int d = 0; d < 3; ++d)
            gradient[d] = gradient[d] * b.value + value * b.gradient[d];
        value *= b.value;
        return *this;
    }
    __device__ PairDerivative& operator/=(const PairDerivative& b) {
        const double q = value / b.value;
        for (int d = 0; d < 3; ++d)
            gradient[d] = (gradient[d] - q * b.gradient[d]) / b.value;
        value = q;
        return *this;
    }
};
static_assert(std::is_trivially_default_constructible_v<PairDerivative>);
static_assert(sizeof(PairDerivative) == 4 * sizeof(double));
__device__ inline PairDerivative operator+(PairDerivative a,
                                           const PairDerivative& b) {
    return a += b;
}
__device__ inline PairDerivative operator-(PairDerivative a,
                                           const PairDerivative& b) {
    return a -= b;
}
__device__ inline PairDerivative operator*(PairDerivative a,
                                           const PairDerivative& b) {
    return a *= b;
}
__device__ inline PairDerivative operator/(PairDerivative a,
                                           const PairDerivative& b) {
    return a /= b;
}
__device__ inline PairDerivative operator-(PairDerivative a) {
    return a *= -1.0;
}
__device__ inline bool operator<(const PairDerivative& a,
                                 const PairDerivative& b) {
    return a.value < b.value;
}
__device__ inline bool operator<=(const PairDerivative& a,
                                  const PairDerivative& b) {
    return a.value <= b.value;
}
__device__ inline bool operator>=(const PairDerivative& a,
                                  const PairDerivative& b) {
    return a.value >= b.value;
}
__device__ inline PairDerivative sqrt(const PairDerivative& a) {
    PairDerivative out(::sqrt(a.value));
    for (int d = 0; d < 3; ++d)
        out.gradient[d] =
            a.gradient[d] == 0.0 ? 0.0 : a.gradient[d] / (2 * out.value);
    return out;
}
__device__ inline PairDerivative exp(const PairDerivative& a) {
    PairDerivative out(::exp(a.value));
    for (int d = 0; d < 3; ++d)
        out.gradient[d] = out.value * a.gradient[d];
    return out;
}
__device__ inline PairDerivative fabs(const PairDerivative& a) {
    return a.value < 0 ? -a : a;
}
__device__ inline PairDerivative pow(const PairDerivative& a, int n) {
    PairDerivative out(::pow(a.value, n));
    const double slope = n == 0 ? 0.0 : n * ::pow(a.value, n - 1);
    for (int d = 0; d < 3; ++d)
        out.gradient[d] = slope * a.gradient[d];
    return out;
}
__device__ inline bool finite(const PairDerivative& a) {
    return isfinite(a.value) && isfinite(a.gradient[0])
           && isfinite(a.gradient[1]) && isfinite(a.gradient[2]);
}
} // namespace ATV::cuda::semiempirical::internal::detail

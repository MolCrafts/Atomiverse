/// @file
/// @brief Three Cartesian derivative directions for native NDDO pair kernels.
#pragma once
#include "atomiverse/cpu/semiempirical/nddo_hcore.h"
#include "atomiverse/cpu/semiempirical/nddo_overlap.h"
#include "atomiverse/cpu/semiempirical/nddo_two_electron.h"
#include <array>
#include <cmath>

namespace ATV::cpu::semiempirical::internal {
// Deliberately local to the scalar NDDO expressions: no tape, heap allocation,
// differentiation through SCF, or external AD runtime.
struct NDDODerivative {
    double value = 0;
    std::array<double, 3> gradient{};
    constexpr NDDODerivative() = default;
    constexpr NDDODerivative(double v) : value(v) {}
    constexpr NDDODerivative(double v, std::array<double, 3> d)
        : value(v), gradient(d) {}
    NDDODerivative& operator+=(const NDDODerivative& b) {
        value += b.value;
        for (int d = 0; d < 3; ++d)
            gradient[d] += b.gradient[d];
        return *this;
    }
    NDDODerivative& operator-=(const NDDODerivative& b) {
        value -= b.value;
        for (int d = 0; d < 3; ++d)
            gradient[d] -= b.gradient[d];
        return *this;
    }
    NDDODerivative& operator*=(const NDDODerivative& b) {
        for (int d = 0; d < 3; ++d)
            gradient[d] = gradient[d] * b.value + value * b.gradient[d];
        value *= b.value;
        return *this;
    }
    NDDODerivative& operator/=(const NDDODerivative& b) {
        const double quotient = value / b.value;
        for (int d = 0; d < 3; ++d)
            gradient[d] = (gradient[d] - quotient * b.gradient[d]) / b.value;
        value = quotient;
        return *this;
    }
};
inline NDDODerivative operator+(NDDODerivative a, const NDDODerivative& b) {
    return a += b;
}
inline NDDODerivative operator-(NDDODerivative a, const NDDODerivative& b) {
    return a -= b;
}
inline NDDODerivative operator*(NDDODerivative a, const NDDODerivative& b) {
    return a *= b;
}
inline NDDODerivative operator/(NDDODerivative a, const NDDODerivative& b) {
    return a /= b;
}
inline NDDODerivative operator-(NDDODerivative a) { return a *= -1.0; }
inline bool operator<(const NDDODerivative& a, const NDDODerivative& b) {
    return a.value < b.value;
}
inline bool operator>(const NDDODerivative& a, const NDDODerivative& b) {
    return a.value > b.value;
}
inline bool operator<=(const NDDODerivative& a, const NDDODerivative& b) {
    return a.value <= b.value;
}
inline bool operator>=(const NDDODerivative& a, const NDDODerivative& b) {
    return a.value >= b.value;
}
inline bool operator==(const NDDODerivative& a, const NDDODerivative& b) {
    return a.value == b.value;
}
inline bool operator!=(const NDDODerivative& a, const NDDODerivative& b) {
    return !(a == b);
}
inline NDDODerivative sqrt(const NDDODerivative& a) {
    const double v = std::sqrt(a.value);
    NDDODerivative out(v);
    for (int d = 0; d < 3; ++d)
        out.gradient[d] = a.gradient[d] == 0.0 ? 0.0 : a.gradient[d] / (2 * v);
    return out;
}
inline NDDODerivative exp(const NDDODerivative& a) {
    const double v = std::exp(a.value);
    NDDODerivative out(v);
    for (int d = 0; d < 3; ++d)
        out.gradient[d] = v * a.gradient[d];
    return out;
}
inline NDDODerivative abs(const NDDODerivative& a) {
    return a.value < 0 ? -a : a;
}
inline NDDODerivative pow(const NDDODerivative& a, int n) {
    const double v = std::pow(a.value, n);
    NDDODerivative out(v);
    const double slope = n == 0 ? 0.0 : n * std::pow(a.value, n - 1);
    for (int d = 0; d < 3; ++d)
        out.gradient[d] = slope * a.gradient[d];
    return out;
}

// The azimuth at the poles is arbitrary, but its discarded derivatives are
// not. Choose a nonparallel reference axis and differentiate an orthonormal
// frame; final s/p contractions are independent of its transverse gauge.
template <class Scalar>
std::array<std::array<Scalar, 3>, 3>
nddo_pair_axes(const std::array<Scalar, 3>& delta) {
    using std::abs;
    using std::sqrt;
    const Scalar r =
        sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
    std::array<std::array<Scalar, 3>, 3> p{};
    for (int d = 0; d < 3; ++d)
        p[0][d] = delta[d] / r;
    if (abs(p[0][2]) < 0.8) {
        p[2][0] = -p[0][1];
        p[2][1] = p[0][0];
    } else {
        p[2][0] = p[0][2];
        p[2][2] = -p[0][0];
    }
    const Scalar t =
        sqrt(p[2][0] * p[2][0] + p[2][1] * p[2][1] + p[2][2] * p[2][2]);
    for (int d = 0; d < 3; ++d)
        p[2][d] /= t;
    for (int d = 0; d < 3; ++d)
        p[1][d] = p[2][(d + 1) % 3] * p[0][(d + 2) % 3]
                  - p[2][(d + 2) % 3] * p[0][(d + 1) % 3];
    return p;
}

std::array<NDDOLocalMatrix9, 3>
one_electron_two_center_derivative(const ATV::semiempirical::AM1Parameter& pi,
                                   const ATV::semiempirical::AM1Parameter& pj,
                                   const std::array<double, 3>& xi,
                                   const std::array<double, 3>& xj);
// Two s/p atoms have at most eight orbitals and 36 packed matrix entries.
// This bounded result is scratch, not a method-sized matrix allocation.
struct NDDOPairHCoreDerivative {
    std::array<double, 36> packed_hcore{};
    double nuclear_repulsion_ev = 0.0;
};
std::array<NDDOPairHCoreDerivative, 3>
hcore_pair_derivative(const NDDOSystem& pair, const std::array<double, 3>& xi,
                      const std::array<double, 3>& xj);
// Each s/p center has at most ten orbital pairs. Values use the active
// second-center pair count as row stride; unused entries are zero.
std::array<std::array<double, 100>, 3>
two_center_integral_derivative(const NDDOSystem& system, const StorageReal* x,
                               const StorageReal* y, const StorageReal* z,
                               int i, int j);
} // namespace ATV::cpu::semiempirical::internal

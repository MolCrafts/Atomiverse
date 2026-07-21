/// @file core/real_harmonics.cpp
/// @brief Implementation of arbitrary-l real solid / spherical harmonics.
///
/// real_solid_harmonics(l) expands the complex solid harmonic Y_l^{|m|} of
/// Ribaldone & Desmarais (2025) Eq. (7) into monomials, then takes the real part
/// (m >= 0) or imaginary part (m < 0). real_spherical_harmonics_normalized(l)
/// divides each by its exact L2(S^2) norm computed from sphere_monomial_integral.

#include "atomiverse/core/real_harmonics.h"

#include <array>
#include <cmath>
#include <complex>
#include <map>

namespace ATV {

namespace {

constexpr double kPi = 3.14159265358979323846;

double factorial(int n) {
    double f = 1.0;
    for (int i = 2; i <= n; ++i)
        f *= i;
    return f;
}

/// (m)!! with (-1)!! = 0!! = 1.
double dfact(int m) {
    double f = 1.0;
    for (int k = m; k > 1; k -= 2)
        f *= k;
    return f;
}

double binom(int n, int k) {
    if (k < 0 || k > n)
        return 0.0;
    return factorial(n) / (factorial(k) * factorial(n - k));
}

/// i^j over the 4-cycle {1, i, -1, -i}.
std::complex<double> i_pow(int j) {
    switch (((j % 4) + 4) % 4) {
    case 0:
        return {1.0, 0.0};
    case 1:
        return {0.0, 1.0};
    case 2:
        return {-1.0, 0.0};
    default:
        return {0.0, -1.0};
    }
}

using Key = std::array<int, 3>;

/// Accumulate the complex monomial expansion of Y_l^{|m|} (Eq. 7).
std::map<Key, std::complex<double>> complex_solid(int l, int mabs) {
    std::map<Key, std::complex<double>> poly;
    const double scale = 1.0 / (std::pow(2.0, l) * factorial(l));
    const int sigma = (l - mabs) / 2; // floor((l-|m|)/2)
    // (x+iy)^{|m|} = sum_j C(|m|,j) i^j x^{|m|-j} y^j.
    for (int j = 0; j <= mabs; ++j) {
        const std::complex<double> base =
            std::complex<double>(binom(mabs, j), 0.0) * i_pow(j);
        const int bx = mabs - j, by = j;
        for (int k = 0; k <= sigma; ++k) {
            const int zpow = l - mabs - 2 * k;
            const double sk = binom(l, k) * std::pow(-1.0, k)
                              * factorial(2 * l - 2 * k) / factorial(zpow);
            // (x^2+y^2+z^2)^k = sum_{p+q+s=k} k!/(p!q!s!) x^{2p} y^{2q} z^{2s}.
            for (int p = 0; p <= k; ++p) {
                for (int q = 0; q <= k - p; ++q) {
                    const int s = k - p - q;
                    const double mult =
                        factorial(k)
                        / (factorial(p) * factorial(q) * factorial(s));
                    const Key key{bx + 2 * p, by + 2 * q, 2 * s + zpow};
                    poly[key] += base * (scale * sk * mult);
                }
            }
        }
    }
    return poly;
}

RealHarmonic to_real(const std::map<Key, std::complex<double>>& poly,
                     bool imag) {
    RealHarmonic out;
    for (const auto& kv : poly) {
        const double c = imag ? kv.second.imag() : kv.second.real();
        if (std::abs(c) > 1e-12)
            out.push_back({kv.first[0], kv.first[1], kv.first[2], c});
    }
    return out;
}

} // namespace

double sphere_monomial_integral(int i, int j, int k) {
    if ((i & 1) || (j & 1) || (k & 1))
        return 0.0;
    return 4.0 * kPi * dfact(i - 1) * dfact(j - 1) * dfact(k - 1)
           / dfact(i + j + k + 1);
}

std::vector<RealHarmonic> real_solid_harmonics(int l) {
    std::vector<RealHarmonic> out;
    out.reserve(static_cast<std::size_t>(2 * l + 1));
    // Order: m = 0, +1, -1, +2, -2, ... (ordering is irrelevant to both uses).
    for (int mabs = 0; mabs <= l; ++mabs) {
        const auto poly = complex_solid(l, mabs);
        out.push_back(to_real(poly, /*imag=*/false)); // X_l^{+|m|} (Re)
        if (mabs > 0)
            out.push_back(to_real(poly, /*imag=*/true)); // X_l^{-|m|} (Im)
    }
    return out;
}

std::vector<RealHarmonic> real_spherical_harmonics_normalized(int l) {
    std::vector<RealHarmonic> out = real_solid_harmonics(l);
    for (auto& harm : out) {
        // ||X/r^l||^2_{L2(S^2)} = sum_{t,t'} c_t c_t' int (mono_t mono_t')/r^{2l} dOmega.
        double n2 = 0.0;
        for (const auto& a : harm)
            for (const auto& b : harm)
                n2 += a.coeff * b.coeff
                      * sphere_monomial_integral(a.lx + b.lx, a.ly + b.ly,
                                                 a.lz + b.lz);
        const double inv = 1.0 / std::sqrt(n2);
        for (auto& t : harm)
            t.coeff *= inv;
    }
    return out;
}

} // namespace ATV

/// @file cpu/longrange/multipole_expansion.cpp
/// @brief Solid-harmonic multipole / local expansions (open-boundary Coulomb).

#include "atomiverse/cpu/longrange/multipole_expansion.h"

#include <cmath>
#include <utility>

namespace ATV {
namespace longrange {
namespace {

/// n! for small n (FMM orders).
AccReal factorial(int n) {
    AccReal f = 1.0;
    for (int i = 2; i <= n; ++i)
        f *= static_cast<AccReal>(i);
    return f;
}

/// Associated Legendre P_n^m(x), m ≥ 0, Ferrers (no Condon–Shortley (−1)^m).
/// P_m^m = (2m−1)!! (1−x²)^{m/2} ≥ 0.
AccReal associated_legendre(int n, int m, AccReal x) {
    if (m < 0 || m > n)
        return 0.0;
    AccReal pmm = 1.0;
    if (m > 0) {
        const AccReal somx2 = std::sqrt(std::max(AccReal(0), 1.0 - x * x));
        AccReal odd = 1.0;
        for (int i = 1; i <= m; ++i) {
            pmm *= odd * somx2;
            odd += 2.0;
        }
    }
    if (n == m)
        return pmm;
    AccReal pmmp1 = x * (2 * m + 1) * pmm;
    if (n == m + 1)
        return pmmp1;
    AccReal p_nm1 = pmm;
    AccReal p_n = pmmp1;
    for (int ell = m + 2; ell <= n; ++ell) {
        AccReal p_np1 = (x * (2 * ell - 1) * p_n - (ell + m - 1) * p_nm1)
                        / static_cast<AccReal>(ell - m);
        p_nm1 = p_n;
        p_n = p_np1;
    }
    return p_n;
}

/// e^{i m φ} from (x,y), m ≥ 0. On the z-axis returns 1 (P_n^m(±1)=0 for m>0).
void exp_imphi(int m, AccReal x, AccReal y, AccReal& c, AccReal& s) {
    c = 1.0;
    s = 0.0;
    if (m <= 0)
        return;
    const AccReal rho_xy = std::sqrt(x * x + y * y);
    if (rho_xy == 0.0)
        return;
    const AccReal base_c = x / rho_xy;
    const AccReal base_s = y / rho_xy;
    AccReal cr = 1.0, sr = 0.0;
    for (int k = 0; k < m; ++k) {
        const AccReal nr = cr * base_c - sr * base_s;
        const AccReal ni = cr * base_s + sr * base_c;
        cr = nr;
        sr = ni;
    }
    c = cr;
    s = sr;
}

} // namespace

std::complex<AccReal> regular_solid(int n, int m, const Vec3& r) {
    // R_n^m = r^n P_n^m(cosθ) e^{imφ} / (n+m)!
    // with P_n^{-m} = (−1)^m (n−m)!/(n+m)! P_n^m, so that
    // R_n^{-m} = (−1)^m conj(R_n^m) and the addition theorem is a plain product.
    if (n == 0)
        return {1.0, 0.0};

    const int absm = std::abs(m);
    const AccReal rho = r.norm();
    AccReal cos_th = (rho > 0.0) ? (r.z() / rho) : 1.0;
    if (cos_th > 1.0)
        cos_th = 1.0;
    if (cos_th < -1.0)
        cos_th = -1.0;

    const AccReal p_pos = associated_legendre(n, absm, cos_th);
    AccReal cos_ph = 1.0, sin_ph = 0.0;
    exp_imphi(absm, r.x(), r.y(), cos_ph, sin_ph);
    const AccReal rn = std::pow(rho, n);

    if (m >= 0) {
        const AccReal scale = rn * p_pos / factorial(n + m);
        return {scale * cos_ph, scale * sin_ph};
    }
    // m = −absm: R = r^n P_n^{-m} e^{-imφ} / (n−m)!
    // P_n^{-m} = (−1)^m (n−m)!/(n+m)! P_n^m
    // ⇒ R = r^n (−1)^m /(n+m)! P e^{-i|m|φ}
    const AccReal sign = (absm & 1) ? AccReal(-1) : AccReal(1);
    const AccReal scale = rn * p_pos / factorial(n + absm);
    return {sign * scale * cos_ph, -sign * scale * sin_ph};
}

std::complex<AccReal> irregular_solid(int n, int m, const Vec3& r) {
    // I_n^m = (n−m)! r^{-(n+1)} P_n^m e^{imφ}
    // with the same negative-m extension of P. I_0^0 = 1/r.
    const AccReal rho = r.norm();
    if (rho == 0.0)
        throw std::invalid_argument("irregular_solid: r = 0");

    const int absm = std::abs(m);
    AccReal cos_th = r.z() / rho;
    if (cos_th > 1.0)
        cos_th = 1.0;
    if (cos_th < -1.0)
        cos_th = -1.0;

    const AccReal p_pos = associated_legendre(n, absm, cos_th);
    AccReal cos_ph = 1.0, sin_ph = 0.0;
    exp_imphi(absm, r.x(), r.y(), cos_ph, sin_ph);
    const AccReal inv_rn1 = 1.0 / std::pow(rho, n + 1);

    if (m >= 0) {
        const AccReal scale = factorial(n - m) * p_pos * inv_rn1;
        return {scale * cos_ph, scale * sin_ph};
    }
    // m = −absm: I = (n−m)! r^{-n-1} P_n^{-m} e^{-i|m|φ}
    //           = (n+|m|)! r^{-n-1} (−1)^m (n−|m|)!/(n+|m|)! P e^{-iφ}
    //           = (−1)^m (n−|m|)! r^{-n-1} P e^{-iφ}
    const AccReal sign = (absm & 1) ? AccReal(-1) : AccReal(1);
    const AccReal scale = factorial(n - absm) * p_pos * inv_rn1;
    return {sign * scale * cos_ph, -sign * scale * sin_ph};
}

// ── MultipoleExpansion ──────────────────────────────────────────────────────

MultipoleExpansion::MultipoleExpansion(int order, Vec3 center)
    : order_(order), center_(std::move(center)),
      c_(static_cast<std::size_t>(solid_ncoef(order)), {0.0, 0.0}) {
    if (order < 0)
        throw std::invalid_argument("MultipoleExpansion: order must be >= 0");
}

void MultipoleExpansion::clear() {
    for (auto& z : c_)
        z = {0.0, 0.0};
}

void MultipoleExpansion::accumulate(AccReal q, const Vec3& pos) {
    // P2M: M_n^m += q R_n^{-m}(pos − centre).
    const Vec3 d = pos - center_;
    for (int n = 0; n <= order_; ++n) {
        for (int m = -n; m <= n; ++m) {
            c_[static_cast<std::size_t>(solid_index(n, m))] +=
                q * regular_solid(n, -m, d);
        }
    }
}

AccReal MultipoleExpansion::evaluate(const Vec3& point) const {
    const Vec3 d = point - center_;
    if (d.norm_squared() == 0.0)
        throw std::invalid_argument("MultipoleExpansion::evaluate: at centre");
    // 1/|r−r′| = Σ_{n,m} (−1)^m R_n^{−m}(r′) I_n^m(r)  (|r′|<|r|), so with
    // M_n^m = Σ q R_n^{−m} the potential is Σ (−1)^m M_n^m I_n^m.
    std::complex<AccReal> sum{0.0, 0.0};
    for (int n = 0; n <= order_; ++n) {
        for (int m = -n; m <= n; ++m) {
            const AccReal sign = (m & 1) ? AccReal(-1) : AccReal(1);
            sum += sign * c_[static_cast<std::size_t>(solid_index(n, m))]
                   * irregular_solid(n, m, d);
        }
    }
    // Kernel is real; residual imag is roundoff.
    return sum.real();
}

// ── LocalExpansion ──────────────────────────────────────────────────────────

LocalExpansion::LocalExpansion(int order, Vec3 center)
    : order_(order), center_(std::move(center)),
      c_(static_cast<std::size_t>(solid_ncoef(order)), {0.0, 0.0}) {
    if (order < 0)
        throw std::invalid_argument("LocalExpansion: order must be >= 0");
}

void LocalExpansion::clear() {
    for (auto& z : c_)
        z = {0.0, 0.0};
}

AccReal LocalExpansion::evaluate(const Vec3& point) const {
    const Vec3 d = point - center_;
    std::complex<AccReal> sum{0.0, 0.0};
    for (int n = 0; n <= order_; ++n) {
        for (int m = -n; m <= n; ++m) {
            sum += c_[static_cast<std::size_t>(solid_index(n, m))]
                   * regular_solid(n, m, d);
        }
    }
    return sum.real();
}

} // namespace longrange
} // namespace ATV

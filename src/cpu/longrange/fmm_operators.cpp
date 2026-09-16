/// @file cpu/longrange/fmm_operators.cpp
/// @brief M2M / M2L / L2L solid-harmonic translation operators.

#include "atomiverse/cpu/longrange/fmm_operators.h"

#include <cmath>
#include <string>

namespace ATV {
namespace longrange {
namespace {

void require_same_order(int a, int b, const char* op) {
    if (a != b)
        throw std::invalid_argument(std::string(op) + ": order mismatch");
}

/// (−1)^n as AccReal.
AccReal minus_one_pow(int n) { return (n & 1) ? AccReal(-1) : AccReal(1); }

} // namespace

// ── M2M ─────────────────────────────────────────────────────────────────────

M2M::M2M(int order) : order_(order) {
    if (order < 0)
        throw std::invalid_argument("M2M: order must be >= 0");
}

void M2M::apply(const MultipoleExpansion& src, MultipoleExpansion& dst) const {
    require_same_order(src.order(), order_, "M2M");
    require_same_order(dst.order(), order_, "M2M");

    // t = child − parent: source points relative to parent = s_child + t.
    // M_n^m(parent) += Σ_{k,l} R_k^l(t) M_{n−k}^{m+l}(child)
    //   with |l|≤k, |m+l|≤n−k, 0≤k≤n.
    const Vec3 t = src.center() - dst.center();
    auto& Mdst = dst.coeffs();
    const auto& Msrc = src.coeffs();

    for (int n = 0; n <= order_; ++n) {
        for (int m = -n; m <= n; ++m) {
            std::complex<AccReal> acc{0.0, 0.0};
            for (int k = 0; k <= n; ++k) {
                const int j = n - k; // child degree
                for (int l = -k; l <= k; ++l) {
                    const int mu = m + l;
                    if (mu < -j || mu > j)
                        continue;
                    acc += regular_solid(k, l, t)
                           * Msrc[static_cast<std::size_t>(solid_index(j, mu))];
                }
            }
            Mdst[static_cast<std::size_t>(solid_index(n, m))] += acc;
        }
    }
}

// ── L2L ─────────────────────────────────────────────────────────────────────

L2L::L2L(int order) : order_(order) {
    if (order < 0)
        throw std::invalid_argument("L2L: order must be >= 0");
}

void L2L::apply(const LocalExpansion& src, LocalExpansion& dst) const {
    require_same_order(src.order(), order_, "L2L");
    require_same_order(dst.order(), order_, "L2L");

    // t = child − parent.
    // L_j^k(child) += Σ_{n≥j} Σ_m L_n^m(parent) R_{n−j}^{m−k}(t)
    const Vec3 t = dst.center() - src.center();
    auto& Ldst = dst.coeffs();
    const auto& Lsrc = src.coeffs();

    for (int j = 0; j <= order_; ++j) {
        for (int k = -j; k <= j; ++k) {
            std::complex<AccReal> acc{0.0, 0.0};
            for (int n = j; n <= order_; ++n) {
                const int nk = n - j;
                for (int m = -n; m <= n; ++m) {
                    const int lm = m - k;
                    if (lm < -nk || lm > nk)
                        continue;
                    acc += Lsrc[static_cast<std::size_t>(solid_index(n, m))]
                           * regular_solid(nk, lm, t);
                }
            }
            Ldst[static_cast<std::size_t>(solid_index(j, k))] += acc;
        }
    }
}

// ── M2L ─────────────────────────────────────────────────────────────────────

M2L::M2L(int order) : order_(order) {
    if (order < 0)
        throw std::invalid_argument("M2L: order must be >= 0");
}

void M2L::apply(const MultipoleExpansion& src, LocalExpansion& dst) const {
    require_same_order(src.order(), order_, "M2L");
    require_same_order(dst.order(), order_, "M2L");

    // t = multipole centre − local centre (vector from local to multipole).
    // With M_n^m = Σ q R_n^{−m} and Φ = Σ (−1)^m M_n^m I_n^m(r − src),
    // the M2L that matches L2P = Σ L_j^k R_j^k on general 3-D geometries is
    //   L_j^k += Σ_{n,m} M_n^m · I_{j+n}^{m−k}(t) · (−1)^{n+m+k}
    // (validated against direct 1/r for axis-aligned and fully 3-D offsets).
    const Vec3 t = src.center() - dst.center();
    if (t.norm_squared() == 0.0)
        throw std::invalid_argument("M2L: coincident centres");

    auto& L = dst.coeffs();
    const auto& M = src.coeffs();

    for (int j = 0; j <= order_; ++j) {
        for (int k = -j; k <= j; ++k) {
            std::complex<AccReal> acc{0.0, 0.0};
            for (int n = 0; n <= order_; ++n) {
                const int jn = j + n;
                for (int m = -n; m <= n; ++m) {
                    const int mk = m - k;
                    if (mk < -jn || mk > jn)
                        continue;
                    acc += M[static_cast<std::size_t>(solid_index(n, m))]
                           * irregular_solid(jn, mk, t)
                           * minus_one_pow(n + m + k);
                }
            }
            L[static_cast<std::size_t>(solid_index(j, k))] += acc;
        }
    }
}

} // namespace longrange
} // namespace ATV

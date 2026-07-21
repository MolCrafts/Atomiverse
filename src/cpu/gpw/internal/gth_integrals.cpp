/// @file src/cpu/gpw/internal/gth_integrals.cpp
/// @brief Implementation of analytic GTH matrix elements (see header).
///
/// Core idea (separable Gaussian-product / moment method):
///   Every integrand here is a product of Cartesian Gaussians (centered at the
///   orbital centers and at the pseudopotential center D) times Cartesian
///   monomials. By the Gaussian product theorem the product of N 1D Gaussians
///   exp(-a_k (x-c_k)^2) is a single Gaussian exp(-gamma (x-Q)^2) times a constant
///   prefactor, with gamma = sum a_k and Q = (sum a_k c_k)/gamma. After shifting
///   t = x - Q, the remaining Cartesian monomials become a polynomial in t whose
///   even moments integrate to (m-1)!!/(2 gamma)^{m/2} sqrt(pi/gamma). Because the
///   3D integrand factorizes across x, y, z (after expanding |r-D|^{2k} with the
///   multinomial theorem), the 3D integral is a product of three such 1D moment
///   integrals. This is exact for any angular momentum.

#include "atomiverse/cpu/gpw/internal/gth_integrals.h"

#include <cmath>
#include <vector>

#include "atomiverse/core/real_harmonics.h"

namespace ATV {
namespace cpu {
namespace gpw {

namespace {

constexpr double kPi = 3.14159265358979323846;

/// One 1D Gaussian factor (x-c)^l exp(-alpha (x-c)^2).
struct G1 {
    int l;
    double alpha;
    double c;
};

double factorial(int n) {
    double f = 1.0;
    for (int i = 2; i <= n; ++i)
        f *= i;
    return f;
}

/// Double factorial (m)!! with (-1)!! = 0!! = 1.
double dfact(int m) {
    double f = 1.0;
    for (int k = m; k > 1; k -= 2)
        f *= k;
    return f;
}

double binom(int n, int k) {
    return factorial(n) / (factorial(k) * factorial(n - k));
}

/// Coefficients (ascending in t) of (t + d)^l: coeff[k] = C(l,k) d^{l-k}.
std::vector<double> binom_pow(double d, int l) {
    std::vector<double> c(l + 1);
    for (int k = 0; k <= l; ++k)
        c[k] = binom(l, k) * std::pow(d, l - k);
    return c;
}

std::vector<double> polymul(const std::vector<double>& a,
                            const std::vector<double>& b) {
    std::vector<double> r(a.size() + b.size() - 1, 0.0);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j)
            r[i + j] += a[i] * b[j];
    return r;
}

/// int_{-inf}^{inf} t^m exp(-gamma t^2) dt; zero for odd m.
double gauss_moment(int m, double gamma) {
    if (m % 2 != 0)
        return 0.0;
    return dfact(m - 1) / std::pow(2.0 * gamma, m / 2) * std::sqrt(kPi / gamma);
}

/// int_{-inf}^{inf} prod_k (x-c_k)^{l_k} exp(-a_k (x-c_k)^2) dx, exact.
double integral_1d(const std::vector<G1>& gs) {
    double gamma = 0.0, wc = 0.0;
    for (const auto& g : gs) {
        gamma += g.alpha;
        wc += g.alpha * g.c;
    }
    const double Q = wc / gamma;

    double s = 0.0;
    for (std::size_t j = 0; j < gs.size(); ++j)
        for (std::size_t k = j + 1; k < gs.size(); ++k)
            s += gs[j].alpha * gs[k].alpha * (gs[j].c - gs[k].c)
                 * (gs[j].c - gs[k].c);
    const double pref = std::exp(-s / gamma);

    std::vector<double> poly{1.0};
    for (const auto& g : gs)
        poly = polymul(poly, binom_pow(Q - g.c, g.l));

    double res = 0.0;
    for (std::size_t m = 0; m < poly.size(); ++m)
        res += poly[m] * gauss_moment(static_cast<int>(m), gamma);
    return pref * res;
}

/// 3D integral of g_a * g_b * (extra D-centered moment |r-D|^{2k}) * exp(-beta|r-D|^2).
/// `extra_dim` adds an extra power to the D-centered factor in that dimension
/// (used for l=1 solid-harmonic component); pass {0,0,0} for pure local/s.
double moment3d(const Prim& a, const Prim* b, const double D[3], double beta,
                int k, const int extra_dim[3]) {
    // |r-D|^{2k} = sum_{p+q+s=k} k!/(p!q!s!) (x-Dx)^{2p}(y-Dy)^{2q}(z-Dz)^{2s}.
    double total = 0.0;
    for (int p = 0; p <= k; ++p) {
        for (int q = 0; q <= k - p; ++q) {
            const int r = k - p - q;
            const double mult =
                factorial(k) / (factorial(p) * factorial(q) * factorial(r));
            const int dpow[3] = {2 * p, 2 * q, 2 * r};
            double prod = 1.0;
            for (int dim = 0; dim < 3; ++dim) {
                std::vector<G1> gs;
                gs.push_back({a.l[dim], a.alpha, a.center[dim]});
                if (b)
                    gs.push_back({b->l[dim], b->alpha, b->center[dim]});
                gs.push_back({dpow[dim] + extra_dim[dim], beta, D[dim]});
                prod *= integral_1d(gs);
            }
            total += mult * prod;
        }
    }
    return total;
}

/// GTH radial normalization N_i^l = sqrt(2) / (r_l^{l+(4i-1)/2} sqrt(Gamma(l+(4i-1)/2))).
double gth_norm(int l, int iproj, double r_l) {
    const double e = l + (4.0 * iproj - 1.0) / 2.0;
    return std::sqrt(2.0) / (std::pow(r_l, e) * std::sqrt(std::tgamma(e)));
}

/// Cached L2(S^2)-normalized real spherical harmonics (r^l Y_lm) for any l.
/// Indexed [l][m]; m in [0, 2l]. Built once (thread-safe static init).
constexpr int kProjLMax = 6;
const std::vector<std::vector<RealHarmonic>>& normalized_harmonics() {
    static const std::vector<std::vector<RealHarmonic>> H = [] {
        std::vector<std::vector<RealHarmonic>> t;
        t.reserve(kProjLMax + 1);
        for (int l = 0; l <= kProjLMax; ++l)
            t.push_back(real_spherical_harmonics_normalized(l));
        return t;
    }();
    return H;
}

} // namespace

double gth_local_sr(const Prim& a, const Prim& b, const double pp_center[3],
                    double rloc, const std::vector<double>& c) {
    const double beta = 1.0 / (2.0 * rloc * rloc);
    const int no_extra[3] = {0, 0, 0};
    double total = 0.0;
    for (std::size_t i = 0; i < c.size(); ++i) {
        const int k = static_cast<int>(
            i); // term i (0-based) carries moment 2k, k=i-? -> here power 2*i
        const double coeff = c[i] / std::pow(rloc, 2 * k);
        total += coeff * moment3d(a, &b, pp_center, beta, k, no_extra);
    }
    return total;
}

double gth_projector_overlap(const Prim& g, const double pp_center[3], int l,
                             int iproj, double r_l, int m) {
    const double beta = 1.0 / (2.0 * r_l * r_l);
    const int k =
        iproj - 1; // radial power r^{2(i-1)} (times r^l from angular part)
    const double N = gth_norm(l, iproj, r_l);
    // r^l Y_lm as a normalized Cartesian polynomial (Use B); the degree-l monomials
    // are the "extra" D-centered powers, the radial r^{2(i-1)} comes from k. Sum
    // moment3d over the polynomial terms — exact for any l.
    const auto& harm = normalized_harmonics()[static_cast<std::size_t>(l)]
                                             [static_cast<std::size_t>(m)];
    double total = 0.0;
    for (const auto& t : harm) {
        const int extra[3] = {t.lx, t.ly, t.lz};
        total += t.coeff * moment3d(g, nullptr, pp_center, beta, k, extra);
    }
    return N * total;
}

double gth_nonlocal(const Prim& a, const Prim& b, const double pp_center[3],
                    const GthNonlocalChannel& ch) {
    if (ch.nproj == 0)
        return 0.0;
    const int n_m = 2 * ch.l + 1; // all m-components of the l-channel (any l).

    double total = 0.0;
    for (int m = 0; m < n_m; ++m) {
        std::vector<double> oa(ch.nproj), ob(ch.nproj);
        for (int i = 0; i < ch.nproj; ++i) {
            oa[i] = gth_projector_overlap(a, pp_center, ch.l, i + 1, ch.r, m);
            ob[i] = gth_projector_overlap(b, pp_center, ch.l, i + 1, ch.r, m);
        }
        for (int i = 0; i < ch.nproj; ++i)
            for (int j = 0; j < ch.nproj; ++j)
                total += oa[i] * ch.hij(i, j) * ob[j];
    }
    return total;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV

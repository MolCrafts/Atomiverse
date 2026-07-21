/// @file src/cpu/scf/ecp_integrals.cpp
/// @brief Semi-local ECP integrals ⟨μ|U|ν⟩ — McMurchie–Davidson (1981)
///        type-1 / type-2 with scaled modified spherical Bessel radial
///        integrands and per-term Gauss–Legendre radial quadrature.
///
/// Numerical scheme (validated to ~3e-13 absolute against PySCF 2.14.0
/// `mol.intor('ECPscalar')`, see tests/cpu/scf/test_ecp.cpp):
///   * All Bessel factors enter as M_l(x) = e^{−x} i_l(x), so every radial
///     integrand is r^p · M · M · exp(−α(r−|A'|)² − β(r−|B'|)² − ζr²) with a
///     non-positive exponent — no overflow, no cancellation at high l.
///   * M_l: all-positive-term power series for x < 20; normalized downward
///     recursion (Miller) anchored at M_0(x) = −expm1(−2x)/(2x) for x ≥ 20.
///   * Radial quadrature: per Gaussian term, 128-point Gauss–Legendre on
///     [max(0, r* − 9σ), r* + 9σ], r* = (k_A + k_B)/(2γ_tot),
///     σ = 1/√γ_tot — the integrand is an entire function times a Gaussian
///     of width σ, for which this converges far below the 1e-10 target
///     (PySCF itself uses a 99-point radial grid for the same integrals).
///   * Angular integrals reuse ATV::real_spherical_harmonics_normalized and
///     the exact sphere_monomial_integral; sums run over complete m/μ sets,
///     so the m-ordering convention of the harmonics is immaterial.

#include "atomiverse/cpu/scf/internal/ecp_integrals.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/cpu/scf/cart.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

namespace {

constexpr int kNRad = 128;            ///< Radial Gauss–Legendre points.
constexpr double kWidthSigmas = 9.0;  ///< Half-width of the radial window.
constexpr double kScreenLnEps = -60.0; ///< ln cutoff for a primitive term.
constexpr double kFourPi = 4.0 * M_PI;

/// Gauss–Legendre nodes/weights on [-1, 1], computed once by Newton
/// iteration on P_n (standard textbook scheme, ~1e-15 node accuracy).
struct GaussLegendre {
    double x[kNRad];
    double w[kNRad];
    GaussLegendre() {
        const int n = kNRad;
        for (int i = 0; i < (n + 1) / 2; ++i) {
            double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
            double pp = 0.0;
            for (int it = 0; it < 100; ++it) {
                double p0 = 1.0, p1 = 0.0;
                for (int j = 0; j < n; ++j) {
                    const double p2 = p1;
                    p1 = p0;
                    p0 = ((2.0 * j + 1.0) * z * p1 - j * p2) / (j + 1.0);
                }
                pp = n * (z * p0 - p1) / (z * z - 1.0);
                const double dz = p0 / pp;
                z -= dz;
                if (std::abs(dz) < 1e-15)
                    break;
            }
            x[i] = -z;
            x[n - 1 - i] = z;
            w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
            w[n - 1 - i] = w[i];
        }
    }
};

const GaussLegendre& gl_rule() {
    static const GaussLegendre rule;
    return rule;
}

/// (2l+1)!! for the Bessel series.
double odd_double_factorial(int l) {
    double v = 1.0;
    for (int k = 3; k <= 2 * l + 1; k += 2)
        v *= k;
    return v;
}

/// Scaled modified spherical Bessel M_l(x) = e^{−x} i_l(x) for l = 0..lmax.
void scaled_bessel_m(double x, int lmax, double* out) {
    if (x < 1e-14) {
        out[0] = 1.0;
        for (int l = 1; l <= lmax; ++l)
            out[l] = 0.0;
        return;
    }
    if (x < 20.0) {
        // All-positive-term series i_l(x) = Σ_k x^{l+2k} / (2^k k! (2l+2k+1)!!)
        // scaled by e^{-x}: no cancellation, ≤ ~40 terms at x = 20.
        const double ex = std::exp(-x);
        for (int l = 0; l <= lmax; ++l) {
            double term = std::pow(x, l) / odd_double_factorial(l);
            double sum = term;
            for (int k = 1; k < 400; ++k) {
                term *= x * x / (2.0 * k * (2.0 * l + 2.0 * k + 1.0));
                sum += term;
                if (term < sum * 1e-17)
                    break;
            }
            out[l] = ex * sum;
        }
        return;
    }
    // Miller downward recursion M_{l-1} = M_{l+1} + (2l+1)/x M_l, anchored
    // at the closed form M_0(x) = −expm1(−2x)/(2x) (expm1 avoids the
    // small-x cancellation; here x ≥ 20 so it is exact anyway).
    double buf[64];
    const int lstart = lmax + 30;
    buf[lstart + 1] = 0.0;
    buf[lstart] = 1.0;
    for (int l = lstart; l >= 1; --l)
        buf[l - 1] = buf[l + 1] + (2.0 * l + 1.0) / x * buf[l];
    const double scale = (-std::expm1(-2.0 * x) / (2.0 * x)) / buf[0];
    for (int l = 0; l <= lmax; ++l)
        out[l] = buf[l] * scale;
}

/// One Cartesian monomial x^i y^j z^k with coefficient c.
struct Mono {
    int i, j, k;
    double c;
};

/// Binomial table up to the largest AO angular momentum.
double binom(int n, int k) {
    double v = 1.0;
    for (int t = 0; t < k; ++t)
        v = v * (n - t) / (t + 1);
    return v;
}

/// Expand (x−Ax)^ax (y−Ay)^ay (z−Az)^az into monomials about the origin.
std::vector<Mono> expand_shifted(const int ang[3], const double A[3]) {
    std::vector<Mono> out;
    out.reserve(static_cast<std::size_t>((ang[0] + 1) * (ang[1] + 1)
                                         * (ang[2] + 1)));
    for (int i = 0; i <= ang[0]; ++i) {
        const double ci = binom(ang[0], i) * std::pow(-A[0], ang[0] - i);
        for (int j = 0; j <= ang[1]; ++j) {
            const double cj = binom(ang[1], j) * std::pow(-A[1], ang[1] - j);
            for (int k = 0; k <= ang[2]; ++k) {
                const double ck =
                    binom(ang[2], k) * std::pow(-A[2], ang[2] - k);
                out.push_back(Mono{i, j, k, ci * cj * ck});
            }
        }
    }
    return out;
}

/// Product of two monomial expansions.
std::vector<Mono> product_monos(const std::vector<Mono>& a,
                                const std::vector<Mono>& b) {
    std::vector<Mono> out;
    out.reserve(a.size() * b.size());
    for (const auto& ma : a)
        for (const auto& mb : b)
            out.push_back(
                Mono{ma.i + mb.i, ma.j + mb.j, ma.k + mb.k, ma.c * mb.c});
    return out;
}

/// Evaluate a real-harmonic Cartesian polynomial at a unit vector.
double eval_harm(const RealHarmonic& h, const double v[3]) {
    double s = 0.0;
    for (const auto& t : h) {
        double p = t.coeff;
        for (int q = 0; q < t.lx; ++q)
            p *= v[0];
        for (int q = 0; q < t.ly; ++q)
            p *= v[1];
        for (int q = 0; q < t.lz; ++q)
            p *= v[2];
        s += p;
    }
    return s;
}

/// Pack an angular-cache key (all fields < 32).
std::int64_t pack_key(int a, int b, int c, int d, int e) {
    return ((((static_cast<std::int64_t>(a) * 32 + b) * 32 + c) * 32 + d) * 32
            + e);
}

} // namespace

EcpIntegrals::EcpIntegrals(const std::vector<native::Shell>& shells,
                           std::vector<EcpCenter> centers)
    : shells_(shells), centers_(std::move(centers)) {
    int lmax_ao = 0;
    for (const auto& s : shells_)
        lmax_ao = std::max(lmax_ao, s.l);
    int lmax_proj = 0;
    for (const auto& c : centers_) {
        if (c.ecp == nullptr || c.ecp->channels.empty())
            throw std::invalid_argument(
                "EcpIntegrals: every EcpCenter must carry a non-empty "
                "AtomEcp (all-electron atoms are not ECP centers)");
        const auto& ch = c.ecp->channels;
        for (std::size_t k = 0; k + 1 < ch.size(); ++k)
            lmax_proj = std::max(lmax_proj, ch[k].l);
    }
    lmax_harm_ = std::max(2 * lmax_ao, lmax_proj + lmax_ao);
    harm_.reserve(static_cast<std::size_t>(lmax_harm_) + 1);
    for (int l = 0; l <= lmax_harm_; ++l)
        harm_.push_back(real_spherical_harmonics_normalized(l));
}

const std::vector<double>& EcpIntegrals::omega1(int lam, int i, int j, int k) {
    const auto key = pack_key(0, lam, i, j, k);
    auto it = omega1_.find(key);
    if (it != omega1_.end())
        return it->second;
    std::vector<double> vals(static_cast<std::size_t>(2 * lam + 1), 0.0);
    for (int mu = 0; mu < 2 * lam + 1; ++mu) {
        double s = 0.0;
        for (const auto& t : harm_[static_cast<std::size_t>(lam)]
                                  [static_cast<std::size_t>(mu)])
            s += t.coeff * sphere_monomial_integral(t.lx + i, t.ly + j,
                                                    t.lz + k);
        vals[static_cast<std::size_t>(mu)] = s;
    }
    return omega1_.emplace(key, std::move(vals)).first->second;
}

const std::vector<double>& EcpIntegrals::omega2(int l, int lam, int i, int j,
                                                int k) {
    const auto key = pack_key(l + 1, lam, i, j, k);
    auto it = omega2_.find(key);
    if (it != omega2_.end())
        return it->second;
    const int nm = 2 * l + 1, nmu = 2 * lam + 1;
    std::vector<double> vals(static_cast<std::size_t>(nm * nmu), 0.0);
    for (int m = 0; m < nm; ++m) {
        for (int mu = 0; mu < nmu; ++mu) {
            double s = 0.0;
            for (const auto& ta : harm_[static_cast<std::size_t>(l)]
                                       [static_cast<std::size_t>(m)])
                for (const auto& tb : harm_[static_cast<std::size_t>(lam)]
                                           [static_cast<std::size_t>(mu)])
                    s += ta.coeff * tb.coeff
                         * sphere_monomial_integral(ta.lx + tb.lx + i,
                                                    ta.ly + tb.ly + j,
                                                    ta.lz + tb.lz + k);
            vals[static_cast<std::size_t>(m * nmu + mu)] = s;
        }
    }
    return omega2_.emplace(key, std::move(vals)).first->second;
}

double EcpIntegrals::pair_center(const native::Shell& a,
                                 const native::Shell& b,
                                 const EcpCenter& center) {
    const auto& gl = gl_rule();

    double Ap[3], Bp[3];
    for (int d = 0; d < 3; ++d) {
        Ap[d] = a.center[d] - center.pos[d];
        Bp[d] = b.center[d] - center.pos[d];
    }
    const double aA =
        std::sqrt(Ap[0] * Ap[0] + Ap[1] * Ap[1] + Ap[2] * Ap[2]);
    const double aB =
        std::sqrt(Bp[0] * Bp[0] + Bp[1] * Bp[1] + Bp[2] * Bp[2]);
    double Ah[3] = {0.0, 0.0, 1.0}, Bh[3] = {0.0, 0.0, 1.0};
    if (aA > 1e-14)
        for (int d = 0; d < 3; ++d)
            Ah[d] = Ap[d] / aA;
    if (aB > 1e-14)
        for (int d = 0; d < 3; ++d)
            Bh[d] = Bp[d] / aB;

    int comps[kMaxNCart][3];
    fill_cart_comps(a.l, comps);
    int angA[3] = {comps[a.cart][0], comps[a.cart][1], comps[a.cart][2]};
    fill_cart_comps(b.l, comps);
    int angB[3] = {comps[b.cart][0], comps[b.cart][1], comps[b.cart][2]};

    const auto polyA = expand_shifted(angA, Ap);
    const auto polyB = expand_shifted(angB, Bp);

    // Real harmonics evaluated at the two center-to-shell unit vectors,
    // shared across the projector channels of this pair.
    std::vector<std::vector<double>> YA(
        static_cast<std::size_t>(lmax_harm_) + 1),
        YB(static_cast<std::size_t>(lmax_harm_) + 1);
    for (int lam = 0; lam <= lmax_harm_; ++lam) {
        YA[static_cast<std::size_t>(lam)].resize(
            static_cast<std::size_t>(2 * lam + 1));
        YB[static_cast<std::size_t>(lam)].resize(
            static_cast<std::size_t>(2 * lam + 1));
        for (int mu = 0; mu < 2 * lam + 1; ++mu) {
            const auto& h = harm_[static_cast<std::size_t>(lam)]
                                 [static_cast<std::size_t>(mu)];
            YA[static_cast<std::size_t>(lam)][static_cast<std::size_t>(mu)] =
                eval_harm(h, Ah);
            YB[static_cast<std::size_t>(lam)][static_cast<std::size_t>(mu)] =
                eval_harm(h, Bh);
        }
    }

    const auto& channels = center.ecp->channels;
    double val = 0.0;

    // ── Type-2: projector channels (all but the local last) ──────────────
    // Per-side angular table F[m][λ][N] = Σ_{monomials of degree N}
    //   c · Σ_μ Y_λμ(Â') ∫ Y_lm Y_λμ x̂^i ŷ^j ẑ^k dΩ — primitive-independent.
    for (std::size_t ch = 0; ch + 1 < channels.size(); ++ch) {
        const int l = channels[ch].l;
        const auto& terms = channels[ch].terms;
        const int nm = 2 * l + 1;
        const int lamA_max = l + a.l, lamB_max = l + b.l;

        auto angular_table = [&](const std::vector<Mono>& poly, int l_ao,
                           int lam_max,
                           const std::vector<std::vector<double>>& Yh) {
            std::vector<double> F(
                static_cast<std::size_t>(nm * (lam_max + 1) * (l_ao + 1)),
                0.0);
            for (const auto& mo : poly) {
                const int N = mo.i + mo.j + mo.k;
                for (int lam = 0; lam <= l + N; ++lam) {
                    if ((l + lam + N) % 2)
                        continue;
                    const auto& O2 = omega2(l, lam, mo.i, mo.j, mo.k);
                    const int nmu = 2 * lam + 1;
                    for (int m = 0; m < nm; ++m) {
                        double s = 0.0;
                        for (int mu = 0; mu < nmu; ++mu)
                            s += Yh[static_cast<std::size_t>(lam)]
                                   [static_cast<std::size_t>(mu)]
                                 * O2[static_cast<std::size_t>(m * nmu + mu)];
                        F[static_cast<std::size_t>(
                            (m * (lam_max + 1) + lam) * (l_ao + 1) + N)] +=
                            mo.c * s;
                    }
                }
            }
            return F;
        };
        const auto FA = angular_table(polyA, a.l, lamA_max, YA);
        const auto FB = angular_table(polyB, b.l, lamB_max, YB);

        // Primitive-independent contraction G(λA, λB, N_A+N_B).
        struct Trip {
            int lamA, lamB, NN;
            double g;
        };
        std::vector<Trip> trips;
        for (int lamA = 0; lamA <= lamA_max; ++lamA) {
            for (int lamB = 0; lamB <= lamB_max; ++lamB) {
                for (int NN = 0; NN <= a.l + b.l; ++NN) {
                    double g = 0.0;
                    for (int m = 0; m < nm; ++m) {
                        const int N0 = std::max(0, NN - b.l);
                        const int N1 = std::min(a.l, NN);
                        for (int N = N0; N <= N1; ++N)
                            g += FA[static_cast<std::size_t>(
                                     (m * (lamA_max + 1) + lamA) * (a.l + 1)
                                     + N)]
                                 * FB[static_cast<std::size_t>(
                                     (m * (lamB_max + 1) + lamB) * (b.l + 1)
                                     + (NN - N))];
                    }
                    if (std::abs(g) > 1e-16)
                        trips.push_back(Trip{lamA, lamB, NN, g});
                }
            }
        }
        if (trips.empty())
            continue;

        std::vector<double> acc(trips.size());
        double MA[16], MB[16], rpow[16];
        for (int p = 0; p < a.n_prim; ++p) {
            for (int q = 0; q < b.n_prim; ++q) {
                const double alpha = a.alpha[p], ca = a.coeff[p];
                const double beta = b.alpha[q], cb = b.coeff[q];
                const double kA = 2.0 * alpha * aA, kB = 2.0 * beta * aB;
                const double base = -alpha * aA * aA - beta * aB * aB;
                for (const auto& term : terms) {
                    const double zeta = term.alpha, dcf = term.coeff;
                    const double gt = alpha + beta + zeta;
                    const double rstar = (kA + kB) / (2.0 * gt);
                    if (base + (kA + kB) * rstar - gt * rstar * rstar
                        < kScreenLnEps)
                        continue;
                    const double w = kWidthSigmas / std::sqrt(gt);
                    const double lo = std::max(0.0, rstar - w);
                    const double hi = rstar + w;
                    const double half = 0.5 * (hi - lo);
                    const double mid = 0.5 * (hi + lo);
                    std::fill(acc.begin(), acc.end(), 0.0);
                    for (int pt = 0; pt < kNRad; ++pt) {
                        const double r = mid + half * gl.x[pt];
                        const double wq = half * gl.w[pt];
                        const double pref = wq
                                            * std::exp(base + (kA + kB) * r
                                                       - gt * r * r);
                        scaled_bessel_m(kA * r, lamA_max, MA);
                        scaled_bessel_m(kB * r, lamB_max, MB);
                        rpow[0] = std::pow(r, term.n);
                        for (int qq = 1; qq <= a.l + b.l; ++qq)
                            rpow[qq] = rpow[qq - 1] * r;
                        for (std::size_t t = 0; t < trips.size(); ++t)
                            acc[t] += pref * rpow[trips[t].NN]
                                      * MA[trips[t].lamA] * MB[trips[t].lamB];
                    }
                    double s = 0.0;
                    for (std::size_t t = 0; t < trips.size(); ++t)
                        s += trips[t].g * acc[t];
                    val += ca * cb * dcf * kFourPi * kFourPi * s;
                }
            }
        }
    }

    // ── Type-1: the local channel U_L, applied to all angular momenta ────
    const auto& local = channels.back().terms;
    const auto polyAB = product_monos(polyA, polyB);
    const int NABmax = a.l + b.l;
    double RadL[8][8]; // [λ][N], λ, N ≤ 2*kMaxL
    double M1[16], rpow1[16], Yk[8][17];
    for (int p = 0; p < a.n_prim; ++p) {
        for (int q = 0; q < b.n_prim; ++q) {
            const double alpha = a.alpha[p], ca = a.coeff[p];
            const double beta = b.alpha[q], cb = b.coeff[q];
            double kvec[3];
            for (int d = 0; d < 3; ++d)
                kvec[d] = 2.0 * (alpha * Ap[d] + beta * Bp[d]);
            const double kn = std::sqrt(kvec[0] * kvec[0] + kvec[1] * kvec[1]
                                        + kvec[2] * kvec[2]);
            double khat[3] = {0.0, 0.0, 1.0};
            if (kn > 1e-14)
                for (int d = 0; d < 3; ++d)
                    khat[d] = kvec[d] / kn;
            const double base = -alpha * aA * aA - beta * aB * aB;

            for (int lam = 0; lam <= NABmax; ++lam)
                for (int N = 0; N <= NABmax; ++N)
                    RadL[lam][N] = 0.0;
            bool any = false;
            for (const auto& term : local) {
                const double zeta = term.alpha, dcf = term.coeff;
                const double gt = alpha + beta + zeta;
                const double rstar = kn / (2.0 * gt);
                if (base + kn * rstar - gt * rstar * rstar < kScreenLnEps)
                    continue;
                any = true;
                const double w = kWidthSigmas / std::sqrt(gt);
                const double lo = std::max(0.0, rstar - w);
                const double hi = rstar + w;
                const double half = 0.5 * (hi - lo);
                const double mid = 0.5 * (hi + lo);
                for (int pt = 0; pt < kNRad; ++pt) {
                    const double r = mid + half * gl.x[pt];
                    const double wq = half * gl.w[pt];
                    const double pref =
                        wq * dcf * std::exp(base + kn * r - gt * r * r);
                    scaled_bessel_m(kn * r, NABmax, M1);
                    rpow1[0] = std::pow(r, term.n);
                    for (int qq = 1; qq <= NABmax; ++qq)
                        rpow1[qq] = rpow1[qq - 1] * r;
                    for (int lam = 0; lam <= NABmax; ++lam) {
                        const double ml = pref * M1[lam];
                        for (int N = lam; N <= NABmax; N += 2)
                            RadL[lam][N] += ml * rpow1[N];
                    }
                }
            }
            if (!any)
                continue;

            for (int lam = 0; lam <= NABmax; ++lam)
                for (int mu = 0; mu < 2 * lam + 1; ++mu)
                    Yk[lam][mu] =
                        eval_harm(harm_[static_cast<std::size_t>(lam)]
                                       [static_cast<std::size_t>(mu)],
                                  khat);

            double s = 0.0;
            for (const auto& mo : polyAB) {
                const int N = mo.i + mo.j + mo.k;
                for (int lam = N; lam >= 0; lam -= 2) {
                    const auto& w1 = omega1(lam, mo.i, mo.j, mo.k);
                    double a1 = 0.0;
                    for (int mu = 0; mu < 2 * lam + 1; ++mu)
                        a1 += Yk[lam][mu] * w1[static_cast<std::size_t>(mu)];
                    s += mo.c * a1 * RadL[lam][N];
                }
            }
            val += ca * cb * kFourPi * s;
        }
    }

    return val;
}

xt::xtensor<double, 2> EcpIntegrals::matrix() {
    const int n = static_cast<int>(shells_.size());
    xt::xtensor<double, 2> U = xt::zeros<double>({n, n});
    for (const auto& center : centers_) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                U(i, j) += pair_center(shells_[static_cast<std::size_t>(i)],
                                       shells_[static_cast<std::size_t>(j)],
                                       center);
            }
        }
    }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j)
            U(j, i) = U(i, j);
    return U;
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

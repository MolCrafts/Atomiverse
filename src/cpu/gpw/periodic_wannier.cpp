/// @file cpu/gpw/periodic_wannier.cpp
/// @brief Multi-k Marzari–Vanderbilt MLWF on frozen GPW multi-k RKS orbitals.

#include "atomiverse/cpu/gpw/periodic_wannier.h"

#include "atomiverse/core/log.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/one_electron.h"
#include "atomiverse/cpu/gpw/internal/periodic.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ATV {
namespace cpu {
namespace gpw {

namespace {

using Cdouble = std::complex<double>;
constexpr Cdouble I{0.0, 1.0};
constexpr double kTwoPi = 6.283185307179586476925286766559;

/// Neighbor of a k-point along one reciprocal axis by ±1 mesh step.
struct Neighbor {
    int ik_nb = 0;
    double wb = 0.0;                 ///< SMV weight [Å²]
    std::array<double, 3> b_cart{};  ///< b Cartesian [1/Å]
    std::array<double, 3> b_bohr{};  ///< b Cartesian [1/Bohr] for phase ops
};

/// SMV axis weights so 2 Σ_i w_i (b_i ⊗ b_i) ≈ I (same as PW path).
std::array<double, 3>
smv_axis_weights(const std::array<std::array<double, 3>, 3>& b_shell) {
    double A[6][3]{};
    double rhs[6] = {1, 1, 1, 0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        const auto& b = b_shell[static_cast<std::size_t>(i)];
        A[0][i] = 2.0 * b[0] * b[0];
        A[1][i] = 2.0 * b[1] * b[1];
        A[2][i] = 2.0 * b[2] * b[2];
        A[3][i] = 2.0 * b[0] * b[1];
        A[4][i] = 2.0 * b[0] * b[2];
        A[5][i] = 2.0 * b[1] * b[2];
    }
    double N[3][3]{};
    double v[3]{};
    for (int r = 0; r < 6; ++r) {
        for (int i = 0; i < 3; ++i) {
            v[i] += A[r][i] * rhs[r];
            for (int j = 0; j < 3; ++j)
                N[i][j] += A[r][i] * A[r][j];
        }
    }
    double M[3][4] = {{N[0][0], N[0][1], N[0][2], v[0]},
                      {N[1][0], N[1][1], N[1][2], v[1]},
                      {N[2][0], N[2][1], N[2][2], v[2]}};
    for (int col = 0; col < 3; ++col) {
        int piv = col;
        for (int r = col + 1; r < 3; ++r)
            if (std::fabs(M[r][col]) > std::fabs(M[piv][col]))
                piv = r;
        if (std::fabs(M[piv][col]) < 1e-18) {
            std::array<double, 3> w{};
            for (int i = 0; i < 3; ++i) {
                const auto& b = b_shell[static_cast<std::size_t>(i)];
                const double bl2 = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
                w[static_cast<std::size_t>(i)] =
                    (bl2 > 0.0) ? 1.0 / (2.0 * bl2) : 0.0;
            }
            return w;
        }
        if (piv != col)
            for (int c = 0; c < 4; ++c)
                std::swap(M[col][c], M[piv][c]);
        const double inv = 1.0 / M[col][col];
        for (int c = col; c < 4; ++c)
            M[col][c] *= inv;
        for (int r = 0; r < 3; ++r) {
            if (r == col)
                continue;
            const double f = M[r][col];
            for (int c = col; c < 4; ++c)
                M[r][c] -= f * M[col][c];
        }
    }
    return {M[0][3], M[1][3], M[2][3]};
}

/// Canonicalize fractional κ to (−1/2, 1/2].
double canon_frac(double f) {
    f -= std::floor(f + 0.5);
    if (f <= -0.5)
        f += 1.0;
    return f;
}

/// Find index of k-point matching fractional coords (mod 1).
int find_k(const std::vector<std::array<double, 3>>& k_frac,
           const double target[3]) {
    const double tol = 1e-8;
    for (std::size_t i = 0; i < k_frac.size(); ++i) {
        bool ok = true;
        for (int d = 0; d < 3; ++d) {
            double diff = k_frac[i][static_cast<std::size_t>(d)] - target[d];
            diff -= std::nearbyint(diff);
            if (std::abs(diff) > tol) {
                ok = false;
                break;
            }
        }
        if (ok)
            return static_cast<int>(i);
    }
    return -1;
}

/// Build ±e_i neighbors on the sampled mesh (step 1/N in fractional κ).
std::vector<std::vector<Neighbor>>
build_neighbors(const std::vector<std::array<double, 3>>& k_frac, int N,
                const double L_ang[3]) {
    const int nk = static_cast<int>(k_frac.size());
    std::vector<std::vector<Neighbor>> nb(static_cast<std::size_t>(nk));
    // Shell b_i = (2π / L_i) / N  along Cartesian axis i (orthorhombic).
    std::array<std::array<double, 3>, 3> b_shell{};
    for (int d = 0; d < 3; ++d) {
        b_shell[static_cast<std::size_t>(d)] = {0, 0, 0};
        b_shell[static_cast<std::size_t>(d)][static_cast<std::size_t>(d)] =
            kTwoPi / (L_ang[d] * static_cast<double>(N));
    }
    const auto w_axis = smv_axis_weights(b_shell);
    const double step = 1.0 / static_cast<double>(N);

    for (int ik = 0; ik < nk; ++ik) {
        const auto& kf = k_frac[static_cast<std::size_t>(ik)];
        for (int d = 0; d < 3; ++d) {
            for (int s : {+1, -1}) {
                double tgt[3] = {kf[0], kf[1], kf[2]};
                tgt[d] = canon_frac(tgt[d] + static_cast<double>(s) * step);
                const int j = find_k(k_frac, tgt);
                if (j < 0)
                    throw std::runtime_error(
                        "PeriodicWannier: neighbor k not found on mesh "
                        "(require full N³ mesh with k_time_reversal=false)");
                Neighbor e;
                e.ik_nb = j;
                e.wb = w_axis[static_cast<std::size_t>(d)];
                e.b_cart = {0, 0, 0};
                e.b_cart[static_cast<std::size_t>(d)] =
                    static_cast<double>(s)
                    * b_shell[static_cast<std::size_t>(d)]
                             [static_cast<std::size_t>(d)];
                // b in 1/Bohr: b_Å * a0  (since 1/Å * Å/Bohr = 1/Bohr)
                const double a0 = units::kBohrInAngstrom;
                e.b_bohr = {e.b_cart[0] * a0, e.b_cart[1] * a0,
                            e.b_cart[2] * a0};
                nb[static_cast<std::size_t>(ik)].push_back(e);
            }
        }
    }
    return nb;
}

/// ⟨a|e^{-i b·r}|b⟩ for Cartesian Gaussian primitives (product-center phase;
/// exact for s; product-phase × form-factor for higher ℓ).
Cdouble prim_overlap_phase(const GridPrimitive& a, const GridPrimitive& b,
                           const double bvec[3]) {
    const double s = prim_overlap(a, b);
    if (s == 0.0)
        return Cdouble{0.0, 0.0};
    const double p = a.alpha + b.alpha;
    if (!(p > 0.0))
        return Cdouble{s, 0.0};
    const double Rp[3] = {(a.alpha * a.center[0] + b.alpha * b.center[0]) / p,
                          (a.alpha * a.center[1] + b.alpha * b.center[1]) / p,
                          (a.alpha * a.center[2] + b.alpha * b.center[2]) / p};
    const double b2 =
        bvec[0] * bvec[0] + bvec[1] * bvec[1] + bvec[2] * bvec[2];
    const double phase =
        bvec[0] * Rp[0] + bvec[1] * Rp[1] + bvec[2] * Rp[2];
    const double damp = std::exp(-b2 / (4.0 * p));
    return Cdouble{s, 0.0} * damp * std::exp(-I * phase);
}

/// Cross-k AO overlap S^{k, k+b}_{μν} = Σ_T e^{i q·T} ⟨φ_μ|e^{-ib·r}|φ_ν^T⟩
/// with q = k+b (phase on ket image) and b = q−k.
void build_S_cross(const std::vector<GridBasisFunction>& basis,
                   const std::vector<double>& radii, const double L[3],
                   const int m[3], double screen,
                   const std::array<double, 3>& k_frac_nb,
                   const double b_bohr[3], int nb,
                   std::vector<Cdouble>& S /* nb*nb row-major */) {
    S.assign(static_cast<std::size_t>(nb * nb), Cdouble{0, 0});
    for (int i = 0; i < nb; ++i) {
        const auto& a = basis[static_cast<std::size_t>(i)];
        if (a.empty())
            continue;
        for (int j = 0; j < nb; ++j) {
            const auto& b = basis[static_cast<std::size_t>(j)];
            if (b.empty())
                continue;
            // Pair screen (same idea as kpoints.cpp pair_bound).
            double best = 0.0;
            for (const auto& p : a)
                for (const auto& q : b) {
                    double d2 = 0.0;
                    for (int d = 0; d < 3; ++d) {
                        double e = p.center[d] - q.center[d];
                        if (L[d] > 0.0)
                            e -= L[d] * std::nearbyint(e / L[d]);
                        d2 += e * e;
                    }
                    best = std::max(
                        best, std::abs(p.coeff * q.coeff)
                                  * std::exp(-p.alpha * q.alpha
                                             / (p.alpha + q.alpha) * d2));
                }
            if (best < screen)
                continue;
            const double reach =
                radii[static_cast<std::size_t>(i)]
                + radii[static_cast<std::size_t>(j)];
            const double reach2 = reach * reach;
            const double* ac = a[0].center;
            const double* bc = b[0].center;
            Cdouble s{0, 0};
            for (int tx = -m[0]; tx <= m[0]; ++tx)
                for (int ty = -m[1]; ty <= m[1]; ++ty)
                    for (int tz = -m[2]; tz <= m[2]; ++tz) {
                        const double T[3] = {tx * L[0], ty * L[1], tz * L[2]};
                        const double dx = ac[0] - bc[0] - T[0];
                        const double dy = ac[1] - bc[1] - T[1];
                        const double dz = ac[2] - bc[2] - T[2];
                        if (dx * dx + dy * dy + dz * dz > reach2)
                            continue;
                        // Bloch phase e^{i q · T} with q fractional → 2π κ·t
                        const double arg =
                            kTwoPi
                            * (k_frac_nb[0] * static_cast<double>(tx)
                               + k_frac_nb[1] * static_cast<double>(ty)
                               + k_frac_nb[2] * static_cast<double>(tz));
                        const Cdouble phase(std::cos(arg), std::sin(arg));
                        Cdouble local{0, 0};
                        for (const auto& pa : a)
                            for (const auto& pb : b) {
                                GridPrimitive q = pb;
                                q.center[0] += T[0];
                                q.center[1] += T[1];
                                q.center[2] += T[2];
                                local += prim_overlap_phase(pa, q, b_bohr);
                            }
                        s += phase * local;
                    }
            S[static_cast<std::size_t>(i * nb + j)] = s;
        }
    }
}

/// M = C†(k) S_cross C(k+b); C is nb × n_wann (columns = occupied MOs).
void mo_overlap(const std::vector<Cdouble>& Ck, const std::vector<Cdouble>& Ckb,
                const std::vector<Cdouble>& S, int nbas, int n_wann,
                std::vector<Cdouble>& M) {
    // T = S @ Ckb  (nbas × n_wann)
    std::vector<Cdouble> T(static_cast<std::size_t>(nbas * n_wann),
                           Cdouble{0, 0});
    for (int mu = 0; mu < nbas; ++mu)
        for (int n = 0; n < n_wann; ++n) {
            Cdouble s{0, 0};
            for (int nu = 0; nu < nbas; ++nu)
                s += S[static_cast<std::size_t>(mu * nbas + nu)]
                     * Ckb[static_cast<std::size_t>(nu * n_wann + n)];
            T[static_cast<std::size_t>(mu * n_wann + n)] = s;
        }
    // M = Ck† @ T  (n_wann × n_wann)
    M.assign(static_cast<std::size_t>(n_wann * n_wann), Cdouble{0, 0});
    for (int m = 0; m < n_wann; ++m)
        for (int n = 0; n < n_wann; ++n) {
            Cdouble s{0, 0};
            for (int mu = 0; mu < nbas; ++mu)
                s += std::conj(Ck[static_cast<std::size_t>(mu * n_wann + m)])
                     * T[static_cast<std::size_t>(mu * n_wann + n)];
            M[static_cast<std::size_t>(m * n_wann + n)] = s;
        }
}

void rotate_M(const std::vector<Cdouble>& M0, const std::vector<Cdouble>& Uk,
              const std::vector<Cdouble>& Ukb, int n,
              std::vector<Cdouble>& M) {
    std::vector<Cdouble> T(static_cast<std::size_t>(n * n), Cdouble{0, 0});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            Cdouble s{0, 0};
            for (int k = 0; k < n; ++k)
                s += M0[static_cast<std::size_t>(i * n + k)]
                     * Ukb[static_cast<std::size_t>(k * n + j)];
            T[static_cast<std::size_t>(i * n + j)] = s;
        }
    M.assign(static_cast<std::size_t>(n * n), Cdouble{0, 0});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            Cdouble s{0, 0};
            for (int k = 0; k < n; ++k)
                s += std::conj(Uk[static_cast<std::size_t>(k * n + i)])
                     * T[static_cast<std::size_t>(k * n + j)];
            M[static_cast<std::size_t>(i * n + j)] = s;
        }
}

double safe_arg(Cdouble z) {
    if (std::norm(z) < 1e-24)
        return 0.0;
    return std::arg(z);
}

double total_spread_from_M(
    const std::vector<std::vector<std::vector<Cdouble>>>& M_kb,
    const std::vector<std::vector<Neighbor>>& nb, int n_wann) {
    const int nk = static_cast<int>(M_kb.size());
    if (nk == 0)
        return 0.0;
    std::vector<std::array<double, 3>> r(static_cast<std::size_t>(n_wann),
                                         std::array<double, 3>{0, 0, 0});
    std::vector<double> r2(static_cast<std::size_t>(n_wann), 0.0);
    const double inv_nk = 1.0 / static_cast<double>(nk);
    for (int ik = 0; ik < nk; ++ik) {
        const auto& neigh = nb[static_cast<std::size_t>(ik)];
        for (std::size_t ib = 0; ib < neigh.size(); ++ib) {
            const auto& e = neigh[ib];
            const auto& M = M_kb[static_cast<std::size_t>(ik)][ib];
            for (int n = 0; n < n_wann; ++n) {
                const Cdouble Mnn = M[static_cast<std::size_t>(n * n_wann + n)];
                const double mag2 = std::min(1.0, std::norm(Mnn));
                const double phi = safe_arg(Mnn);
                r2[static_cast<std::size_t>(n)] +=
                    inv_nk * e.wb * (phi * phi + 1.0 - mag2);
                for (int a = 0; a < 3; ++a)
                    r[static_cast<std::size_t>(n)][static_cast<std::size_t>(a)] -=
                        inv_nk * e.wb * e.b_cart[static_cast<std::size_t>(a)]
                        * phi;
            }
        }
    }
    double tot = 0.0;
    for (int n = 0; n < n_wann; ++n) {
        const auto& rn = r[static_cast<std::size_t>(n)];
        const double rsq = rn[0] * rn[0] + rn[1] * rn[1] + rn[2] * rn[2];
        tot += r2[static_cast<std::size_t>(n)] - rsq;
    }
    return tot;
}

void fill_centers_spreads(
    const std::vector<std::vector<std::vector<Cdouble>>>& M_kb,
    const std::vector<std::vector<Neighbor>>& nb, int n_wann,
    PeriodicWannierResult& out) {
    const int nk = static_cast<int>(M_kb.size());
    const double inv_nk = 1.0 / static_cast<double>(nk);
    out.centers.assign(static_cast<std::size_t>(n_wann), {0, 0, 0});
    out.spreads.assign(static_cast<std::size_t>(n_wann), 0.0);
    std::vector<double> r2(static_cast<std::size_t>(n_wann), 0.0);

    for (int ik = 0; ik < nk; ++ik) {
        const auto& neigh = nb[static_cast<std::size_t>(ik)];
        for (std::size_t ib = 0; ib < neigh.size(); ++ib) {
            const auto& e = neigh[ib];
            const auto& M = M_kb[static_cast<std::size_t>(ik)][ib];
            for (int n = 0; n < n_wann; ++n) {
                const Cdouble Mnn = M[static_cast<std::size_t>(n * n_wann + n)];
                const double mag2 = std::min(1.0, std::norm(Mnn));
                const double phi = safe_arg(Mnn);
                r2[static_cast<std::size_t>(n)] +=
                    inv_nk * e.wb * (phi * phi + 1.0 - mag2);
                for (int a = 0; a < 3; ++a)
                    out.centers[static_cast<std::size_t>(n)]
                               [static_cast<std::size_t>(a)] -=
                        inv_nk * e.wb * e.b_cart[static_cast<std::size_t>(a)]
                        * phi;
            }
        }
    }
    out.total_spread = 0.0;
    for (int n = 0; n < n_wann; ++n) {
        const auto& c = out.centers[static_cast<std::size_t>(n)];
        const double rsq = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
        double om = r2[static_cast<std::size_t>(n)] - rsq;
        if (om < 0.0 && om > -1e-6)
            om = 0.0;
        out.spreads[static_cast<std::size_t>(n)] = om;
        out.total_spread += om;
    }
}

void expm_antihermitian(const std::vector<Cdouble>& G, int n, double alpha,
                        std::vector<Cdouble>& expG) {
    std::vector<Cdouble> A(static_cast<std::size_t>(n * n));
    for (std::size_t i = 0; i < A.size(); ++i)
        A[i] = -alpha * G[i];
    expG.assign(static_cast<std::size_t>(n * n), Cdouble{0, 0});
    for (int i = 0; i < n; ++i)
        expG[static_cast<std::size_t>(i * n + i)] = Cdouble{1, 0};
    std::vector<Cdouble> term = expG;
    for (int p = 1; p <= 12; ++p) {
        std::vector<Cdouble> next(static_cast<std::size_t>(n * n),
                                  Cdouble{0, 0});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                Cdouble s{0, 0};
                for (int k = 0; k < n; ++k)
                    s += term[static_cast<std::size_t>(i * n + k)]
                         * A[static_cast<std::size_t>(k * n + j)];
                next[static_cast<std::size_t>(i * n + j)] =
                    s / static_cast<double>(p);
            }
        term.swap(next);
        for (std::size_t i = 0; i < expG.size(); ++i)
            expG[i] += term[i];
    }
}

void right_mul(std::vector<Cdouble>& U, const std::vector<Cdouble>& expG,
               int n) {
    std::vector<Cdouble> out(static_cast<std::size_t>(n * n), Cdouble{0, 0});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            Cdouble s{0, 0};
            for (int k = 0; k < n; ++k)
                s += U[static_cast<std::size_t>(i * n + k)]
                     * expG[static_cast<std::size_t>(k * n + j)];
            out[static_cast<std::size_t>(i * n + j)] = s;
        }
    U.swap(out);
}

void mv_gradient(const std::vector<std::vector<std::vector<Cdouble>>>& M_kb,
                 const std::vector<std::vector<Neighbor>>& nb, int n_wann,
                 std::vector<std::vector<Cdouble>>& G_k) {
    const int nk = static_cast<int>(M_kb.size());
    G_k.assign(static_cast<std::size_t>(nk),
               std::vector<Cdouble>(static_cast<std::size_t>(n_wann * n_wann),
                                    Cdouble{0, 0}));
    for (int ik = 0; ik < nk; ++ik) {
        const auto& neigh = nb[static_cast<std::size_t>(ik)];
        auto& G = G_k[static_cast<std::size_t>(ik)];
        for (std::size_t ib = 0; ib < neigh.size(); ++ib) {
            const double wb = neigh[ib].wb;
            const auto& M = M_kb[static_cast<std::size_t>(ik)][ib];
            std::vector<double> q(static_cast<std::size_t>(n_wann));
            for (int n = 0; n < n_wann; ++n)
                q[static_cast<std::size_t>(n)] =
                    safe_arg(M[static_cast<std::size_t>(n * n_wann + n)]);
            for (int m = 0; m < n_wann; ++m) {
                for (int n = 0; n < n_wann; ++n) {
                    const Cdouble Rmn =
                        M[static_cast<std::size_t>(m * n_wann + n)]
                        * std::exp(-I * q[static_cast<std::size_t>(n)]);
                    const Cdouble Rnm_c =
                        std::conj(M[static_cast<std::size_t>(n * n_wann + m)]
                                  * std::exp(-I * q[static_cast<std::size_t>(m)]));
                    const Cdouble A = (Rmn - Rnm_c) / (2.0 * I);
                    const Cdouble S = 0.5 * (Rmn + Rnm_c);
                    const Cdouble H = A + S * q[static_cast<std::size_t>(n)];
                    G[static_cast<std::size_t>(m * n_wann + n)] +=
                        4.0 * wb * (I * H);
                }
            }
        }
        std::vector<Cdouble> Ga = G;
        for (int m = 0; m < n_wann; ++m)
            for (int n = 0; n < n_wann; ++n) {
                const Cdouble gmn = Ga[static_cast<std::size_t>(m * n_wann + n)];
                const Cdouble gnm = Ga[static_cast<std::size_t>(n * n_wann + m)];
                G[static_cast<std::size_t>(m * n_wann + n)] =
                    0.5 * (gmn - std::conj(gnm));
            }
    }
}

double max_abs_G(const std::vector<std::vector<Cdouble>>& G_k) {
    double m = 0.0;
    for (const auto& G : G_k)
        for (const auto& z : G)
            m = std::max(m, std::abs(z));
    return m;
}

void projection_U(const std::vector<Cdouble>& A, int n,
                  std::vector<Cdouble>& U) {
    std::vector<Cdouble> H(static_cast<std::size_t>(n * n), Cdouble{0, 0});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            Cdouble s{0, 0};
            for (int k = 0; k < n; ++k)
                s += std::conj(A[static_cast<std::size_t>(k * n + i)])
                     * A[static_cast<std::size_t>(k * n + j)];
            H[static_cast<std::size_t>(i * n + j)] = s;
        }
    std::vector<Cdouble> V(static_cast<std::size_t>(n * n), Cdouble{0, 0});
    for (int i = 0; i < n; ++i)
        V[static_cast<std::size_t>(i * n + i)] = Cdouble{1, 0};
    std::vector<double> eval(static_cast<std::size_t>(n), 0.0);
    for (int sweep = 0; sweep < 40; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const Cdouble apq = H[static_cast<std::size_t>(p * n + q)];
                off += std::abs(apq);
                if (std::abs(apq) < 1e-14)
                    continue;
                const double app =
                    H[static_cast<std::size_t>(p * n + p)].real();
                const double aqq =
                    H[static_cast<std::size_t>(q * n + q)].real();
                const double thr = 0.5 * (aqq - app);
                const double phi = std::arg(apq);
                const double apq_r = std::abs(apq);
                const double t =
                    (thr == 0.0)
                        ? 1.0
                        : ((thr > 0 ? 1.0 : -1.0)
                           / (std::abs(thr) / apq_r
                              + std::sqrt(1.0 + (thr / apq_r) * (thr / apq_r))));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = t * c;
                const Cdouble sp = s * std::exp(-I * phi);
                const Cdouble sm = s * std::exp(+I * phi);
                for (int j = 0; j < n; ++j) {
                    const Cdouble hip = H[static_cast<std::size_t>(j * n + p)];
                    const Cdouble hiq = H[static_cast<std::size_t>(j * n + q)];
                    H[static_cast<std::size_t>(j * n + p)] = c * hip + sm * hiq;
                    H[static_cast<std::size_t>(j * n + q)] =
                        -sp * hip + c * hiq;
                }
                for (int j = 0; j < n; ++j) {
                    const Cdouble hpj = H[static_cast<std::size_t>(p * n + j)];
                    const Cdouble hqj = H[static_cast<std::size_t>(q * n + j)];
                    H[static_cast<std::size_t>(p * n + j)] = c * hpj + sp * hqj;
                    H[static_cast<std::size_t>(q * n + j)] =
                        -sm * hpj + c * hqj;
                }
                for (int j = 0; j < n; ++j) {
                    const Cdouble vjp = V[static_cast<std::size_t>(j * n + p)];
                    const Cdouble vjq = V[static_cast<std::size_t>(j * n + q)];
                    V[static_cast<std::size_t>(j * n + p)] = c * vjp + sm * vjq;
                    V[static_cast<std::size_t>(j * n + q)] =
                        -sp * vjp + c * vjq;
                }
            }
        }
        if (off < 1e-14)
            break;
    }
    for (int i = 0; i < n; ++i)
        eval[static_cast<std::size_t>(i)] =
            H[static_cast<std::size_t>(i * n + i)].real();
    std::vector<Cdouble> WSinv(static_cast<std::size_t>(n * n), Cdouble{0, 0});
    for (int j = 0; j < n; ++j) {
        const double s =
            (eval[static_cast<std::size_t>(j)] > 1e-20)
                ? 1.0 / std::sqrt(eval[static_cast<std::size_t>(j)])
                : 0.0;
        for (int i = 0; i < n; ++i)
            WSinv[static_cast<std::size_t>(i * n + j)] =
                V[static_cast<std::size_t>(i * n + j)] * s;
    }
    std::vector<Cdouble> Tm(static_cast<std::size_t>(n * n), Cdouble{0, 0});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            Cdouble s{0, 0};
            for (int k = 0; k < n; ++k)
                s += WSinv[static_cast<std::size_t>(i * n + k)]
                     * std::conj(V[static_cast<std::size_t>(j * n + k)]);
            Tm[static_cast<std::size_t>(i * n + j)] = s;
        }
    U.assign(static_cast<std::size_t>(n * n), Cdouble{0, 0});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            Cdouble s{0, 0};
            for (int k = 0; k < n; ++k)
                s += A[static_cast<std::size_t>(i * n + k)]
                     * Tm[static_cast<std::size_t>(k * n + j)];
            U[static_cast<std::size_t>(i * n + j)] = s;
        }
}

double cubic_a_ang_from_bct(const double cell_bohr[3]);
Cdouble gaussian_ao_overlap_k(const GridBasisFunction& bf,
                              const double tau_bohr[3], double alpha_B,
                              const double L[3], const int m[3],
                              const std::array<double, 3>& k_frac);

/// Flatten occupied C_k (nb × n_wann) from GpwResult payload.
/// @param spin 0 = RKS / UKS α (`C_occ_k` / `C_k`); 1 = UKS β
///             (`C_occ_k_beta` / `C_k_beta`).
std::vector<Cdouble> flatten_C_occ(const GpwResult& res, int ik, int nbas,
                                   int n_wann, int spin = 0) {
    std::vector<Cdouble> C(static_cast<std::size_t>(nbas * n_wann),
                           Cdouble{0, 0});
    const auto& occ_stack = (spin == 0) ? res.C_occ_k : res.C_occ_k_beta;
    const auto& full_stack = (spin == 0) ? res.C_k : res.C_k_beta;
    if (!occ_stack.empty() && static_cast<int>(occ_stack.size()) > ik
        && !occ_stack[static_cast<std::size_t>(ik)].empty()) {
        const auto& raw = occ_stack[static_cast<std::size_t>(ik)];
        const int ncol =
            static_cast<int>(raw.size()) / std::max(nbas, 1);
        const int nuse = std::min(n_wann, ncol);
        for (int mu = 0; mu < nbas; ++mu)
            for (int o = 0; o < nuse; ++o)
                C[static_cast<std::size_t>(mu * n_wann + o)] =
                    raw[static_cast<std::size_t>(mu * ncol + o)];
        return C;
    }
    // Fallback: full C_k / C_k_beta columns 0..n_wann-1
    if (static_cast<int>(full_stack.size()) <= ik)
        throw std::runtime_error(
            spin == 0 ? "PeriodicWannier: missing C_k at k-point"
                      : "PeriodicWannier: missing C_k_beta at k-point");
    const auto& Ck = full_stack[static_cast<std::size_t>(ik)];
    const int nb0 = static_cast<int>(Ck.shape(0));
    const int nmo = static_cast<int>(Ck.shape(1));
    if (nb0 != nbas)
        throw std::runtime_error("PeriodicWannier: C_k row count mismatch");
    const int nuse = std::min(n_wann, nmo);
    for (int mu = 0; mu < nbas; ++mu)
        for (int o = 0; o < nuse; ++o)
            C[static_cast<std::size_t>(mu * n_wann + o)] = Ck(mu, o);
    return C;
}

/// Per-channel localization result (one spin / RKS manifold).
struct ChannelLoc {
    std::vector<std::array<double, 3>> centers;
    std::vector<double> spreads;
    double total_spread = 0.0;
    double total_spread_initial = 0.0;
    int iterations = 0;
    bool converged = false;
};

/// Marzari–Vanderbilt SD on one occupied manifold (shared geometry, own C).
ChannelLoc localize_channel(
    const std::vector<std::vector<Cdouble>>& C, int n_wann, int nk,
    const std::vector<std::array<double, 3>>& k_frac,
    const std::vector<std::vector<Neighbor>>& nb,
    const std::vector<GridBasisFunction>& basis,
    const std::vector<double>& radii, const double cell[3], const int mimg[3],
    double screen, const std::vector<std::array<double, 3>>& trials,
    bool use_projection, double proj_alpha_ang, double a0, int max_iter,
    double tol, double step) {
    const int nbas = static_cast<int>(basis.size());
    ChannelLoc ch;
    if (n_wann < 1)
        return ch;

    // Bare M0(k,b) = C†(k) S_cross C(k+b)
    std::vector<std::vector<std::vector<Cdouble>>> M0(
        static_cast<std::size_t>(nk));
    for (int ik = 0; ik < nk; ++ik) {
        const auto& neigh = nb[static_cast<std::size_t>(ik)];
        M0[static_cast<std::size_t>(ik)].resize(neigh.size());
        for (std::size_t ib = 0; ib < neigh.size(); ++ib) {
            const auto& e = neigh[ib];
            const auto& kfn = k_frac[static_cast<std::size_t>(e.ik_nb)];
            std::vector<Cdouble> S;
            const double bb[3] = {e.b_bohr[0], e.b_bohr[1], e.b_bohr[2]};
            build_S_cross(basis, radii, cell, mimg, screen, kfn, bb, nbas, S);
            mo_overlap(C[static_cast<std::size_t>(ik)],
                       C[static_cast<std::size_t>(e.ik_nb)], S, nbas, n_wann,
                       M0[static_cast<std::size_t>(ik)][ib]);
        }
    }

    // Unitary gauge U(k)
    std::vector<std::vector<Cdouble>> U(static_cast<std::size_t>(nk));
    for (int ik = 0; ik < nk; ++ik) {
        U[static_cast<std::size_t>(ik)].assign(
            static_cast<std::size_t>(n_wann * n_wann), Cdouble{0, 0});
        for (int i = 0; i < n_wann; ++i)
            U[static_cast<std::size_t>(ik)]
             [static_cast<std::size_t>(i * n_wann + i)] = Cdouble{1, 0};
    }

    if (static_cast<int>(trials.size()) != n_wann)
        throw std::invalid_argument(
            "PeriodicWannier: trial_centers size must equal n_wann per channel");

    if (use_projection) {
        const double alpha_B =
            proj_alpha_ang * a0 * a0; // α_Å |r_Å|² = α_B |r_B|²
        int mproj[3];
        pair_image_counts(cell, true, 10.0, mproj);
        for (int ik = 0; ik < nk; ++ik) {
            std::vector<Cdouble> A(static_cast<std::size_t>(n_wann * n_wann),
                                   Cdouble{0, 0});
            const auto& Ck = C[static_cast<std::size_t>(ik)];
            const auto& kf = k_frac[static_cast<std::size_t>(ik)];
            for (int n = 0; n < n_wann; ++n) {
                const double tau_B[3] = {
                    trials[static_cast<std::size_t>(n)][0] / a0,
                    trials[static_cast<std::size_t>(n)][1] / a0,
                    trials[static_cast<std::size_t>(n)][2] / a0};
                std::vector<Cdouble> gproj(static_cast<std::size_t>(nbas));
                for (int mu = 0; mu < nbas; ++mu)
                    gproj[static_cast<std::size_t>(mu)] = gaussian_ao_overlap_k(
                        basis[static_cast<std::size_t>(mu)], tau_B, alpha_B,
                        cell, mproj, kf);
                for (int j = 0; j < n_wann; ++j) {
                    Cdouble s{0, 0};
                    for (int mu = 0; mu < nbas; ++mu)
                        s += std::conj(
                                 Ck[static_cast<std::size_t>(mu * n_wann + j)])
                             * gproj[static_cast<std::size_t>(mu)];
                    A[static_cast<std::size_t>(j * n_wann + n)] = s;
                }
            }
            projection_U(A, n_wann, U[static_cast<std::size_t>(ik)]);
        }
    }

    auto rebuild_M =
        [&](std::vector<std::vector<std::vector<Cdouble>>>& M_kb) {
            M_kb.assign(static_cast<std::size_t>(nk), {});
            for (int ik = 0; ik < nk; ++ik) {
                const auto& neigh = nb[static_cast<std::size_t>(ik)];
                M_kb[static_cast<std::size_t>(ik)].resize(neigh.size());
                for (std::size_t ib = 0; ib < neigh.size(); ++ib) {
                    const int ikb = neigh[ib].ik_nb;
                    rotate_M(M0[static_cast<std::size_t>(ik)][ib],
                             U[static_cast<std::size_t>(ik)],
                             U[static_cast<std::size_t>(ikb)], n_wann,
                             M_kb[static_cast<std::size_t>(ik)][ib]);
                }
            }
        };

    std::vector<std::vector<std::vector<Cdouble>>> M_kb;
    {
        auto U_save = U;
        for (int ik = 0; ik < nk; ++ik) {
            U[static_cast<std::size_t>(ik)].assign(
                static_cast<std::size_t>(n_wann * n_wann), Cdouble{0, 0});
            for (int i = 0; i < n_wann; ++i)
                U[static_cast<std::size_t>(ik)]
                 [static_cast<std::size_t>(i * n_wann + i)] = Cdouble{1, 0};
        }
        rebuild_M(M_kb);
        ch.total_spread_initial = total_spread_from_M(M_kb, nb, n_wann);
        U = std::move(U_save);
    }
    rebuild_M(M_kb);

    bool conv = (max_iter == 0);
    int iters = 0;
    double omega_prev = total_spread_from_M(M_kb, nb, n_wann);
    auto U_best = U;
    auto M_best = M_kb;
    double omega_best = omega_prev;

    for (int it = 0; it < max_iter; ++it) {
        ++iters;
        std::vector<std::vector<Cdouble>> G_k;
        mv_gradient(M_kb, nb, n_wann, G_k);
        const double gmax = max_abs_G(G_k);
        ATV_DEBUG_CAT("gpw", "mlwf iter={} |G|_max={:.3e} Omega={:.6f}", it,
                      gmax, omega_prev);
        if (it > 0 && (!(gmax > 0.0) || gmax < tol || !std::isfinite(gmax))) {
            conv = std::isfinite(gmax) && gmax < tol;
            break;
        }
        double alpha = step;
        bool accepted = false;
        for (int attempt = 0; attempt < 12; ++attempt) {
            auto U_try = U;
            for (int ik = 0; ik < nk; ++ik) {
                std::vector<Cdouble> expG;
                expm_antihermitian(G_k[static_cast<std::size_t>(ik)], n_wann,
                                   alpha, expG);
                right_mul(U_try[static_cast<std::size_t>(ik)], expG, n_wann);
            }
            auto U_save = U;
            U = U_try;
            rebuild_M(M_kb);
            const double omega_new = total_spread_from_M(M_kb, nb, n_wann);
            if (std::isfinite(omega_new) && omega_new <= omega_prev + 1e-6) {
                accepted = true;
                omega_prev = omega_new;
                if (omega_new < omega_best) {
                    omega_best = omega_new;
                    U_best = U;
                    M_best = M_kb;
                }
                break;
            }
            U = std::move(U_save);
            rebuild_M(M_kb);
            alpha *= 0.5;
        }
        if (!accepted)
            break;
    }
    if (max_iter == 0)
        conv = true;

    PeriodicWannierResult tmp;
    fill_centers_spreads(M_best, nb, n_wann, tmp);
    ch.centers = std::move(tmp.centers);
    ch.spreads = std::move(tmp.spreads);
    ch.total_spread = tmp.total_spread;
    ch.iterations = iters;
    ch.converged = conv && std::isfinite(ch.total_spread);
    return ch;
}

/// Default trial centers for one channel of size n_wann.
std::vector<std::array<double, 3>>
default_trials(int n_wann, const std::vector<GpwAtom>& atoms,
               const double cell[3],
               const std::vector<std::array<double, 3>>& trial_centers) {
    if (!trial_centers.empty()) {
        if (static_cast<int>(trial_centers.size()) != n_wann)
            throw std::invalid_argument(
                "PeriodicWannier::compute: trial_centers size must equal "
                "n_wann per spin channel");
        return trial_centers;
    }
    if (n_wann == 8 && static_cast<int>(atoms.size()) == 4) {
        const double a_ang = cubic_a_ang_from_bct(cell);
        const auto mids = diamond_bct_bond_midpoints(a_ang);
        return std::vector<std::array<double, 3>>(mids.begin(), mids.end());
    }
    return std::vector<std::array<double, 3>>(
        static_cast<std::size_t>(n_wann), std::array<double, 3>{0, 0, 0});
}

/// Bloch-phased AO–Gaussian overlap
/// ⟨φ_μ^k | g_τ⟩ = Σ_T e^{i k·T} ⟨χ_μ | g_τ(·−T)⟩ [dimensionless].
///
/// g is an s-type normalized Gaussian at τ [Bohr]. Images of the trial
/// orbital carry the usual lattice phase so projection is k-dependent
/// (required for multi-k MV; phase-free real overlaps only match Γ).
Cdouble gaussian_ao_overlap_k(const GridBasisFunction& bf,
                              const double tau_bohr[3], double alpha_B,
                              const double L[3], const int m[3],
                              const std::array<double, 3>& k_frac) {
    // g0(r) = (2α/π)^{3/4} exp(-α |r-τ|²)
    GridPrimitive g0;
    g0.alpha = alpha_B;
    g0.center[0] = tau_bohr[0];
    g0.center[1] = tau_bohr[1];
    g0.center[2] = tau_bohr[2];
    g0.l[0] = g0.l[1] = g0.l[2] = 0;
    g0.coeff = std::pow(2.0 * alpha_B / units::pi, 0.75);

    if (bf.empty())
        return Cdouble{0, 0};

    const double* ac = bf[0].center;
    // Compact projector: image shell limited by m[]; also skip far images.
    const double reach = 10.0;
    const double reach2 = reach * reach;
    Cdouble s{0, 0};
    for (int tx = -m[0]; tx <= m[0]; ++tx)
        for (int ty = -m[1]; ty <= m[1]; ++ty)
            for (int tz = -m[2]; tz <= m[2]; ++tz) {
                const double T[3] = {tx * L[0], ty * L[1], tz * L[2]};
                const double dx = ac[0] - (tau_bohr[0] + T[0]);
                const double dy = ac[1] - (tau_bohr[1] + T[1]);
                const double dz = ac[2] - (tau_bohr[2] + T[2]);
                if (dx * dx + dy * dy + dz * dz > reach2)
                    continue;
                GridPrimitive g = g0;
                g.center[0] += T[0];
                g.center[1] += T[1];
                g.center[2] += T[2];
                double local = 0.0;
                for (const auto& pa : bf)
                    local += prim_overlap(pa, g);
                if (local == 0.0)
                    continue;
                const double arg =
                    kTwoPi
                    * (k_frac[0] * static_cast<double>(tx)
                       + k_frac[1] * static_cast<double>(ty)
                       + k_frac[2] * static_cast<double>(tz));
                s += Cdouble{std::cos(arg), std::sin(arg)} * local;
            }
    return s;
}

std::vector<GridBasisFunction>
flatten_basis(const std::vector<GpwAtom>& atoms) {
    std::vector<GridBasisFunction> basis;
    for (const auto& a : atoms)
        for (const auto& bf : a.basis)
            basis.push_back(bf);
    return basis;
}

double cubic_a_ang_from_bct(const double cell_bohr[3]) {
    // cell = (a/√2, a/√2, a) [Bohr] → a [Å]
    const double a0 = units::kBohrInAngstrom;
    return cell_bohr[2] * a0;
}

} // namespace

// ── Public pure helpers ───────────────────────────────────────────────────

std::array<std::array<double, 3>, 8> diamond_bct_bond_midpoints(double a_ang) {
    if (!(a_ang > 0.0))
        throw std::invalid_argument(
            "diamond_bct_bond_midpoints: a_ang must be > 0");
    const double at = a_ang / std::sqrt(2.0);
    // Atoms [Å]
    const std::array<std::array<double, 3>, 4> atoms = {{
        {{0.0, 0.0, 0.0}},
        {{0.5 * at, 0.5 * at, 0.5 * a_ang}},
        {{0.5 * at, 0.0, 0.25 * a_ang}},
        {{0.0, 0.5 * at, 0.75 * a_ang}},
    }};
    const double L[3] = {at, at, a_ang};
    const double bond = a_ang * std::sqrt(3.0) / 4.0;
    const double bond2_lo = (bond * 0.98) * (bond * 0.98);
    const double bond2_hi = (bond * 1.02) * (bond * 1.02);

    std::vector<std::array<double, 3>> mids;
    mids.reserve(8);
    auto min_image = [&](double& dx, double& dy, double& dz) {
        dx -= L[0] * std::nearbyint(dx / L[0]);
        dy -= L[1] * std::nearbyint(dy / L[1]);
        dz -= L[2] * std::nearbyint(dz / L[2]);
    };
    // Enumerate unique undirected bonds (i < j with images of j).
    for (int i = 0; i < 4; ++i) {
        for (int j = i; j < 4; ++j) {
            for (int tx = -1; tx <= 1; ++tx)
                for (int ty = -1; ty <= 1; ++ty)
                    for (int tz = -1; tz <= 1; ++tz) {
                        if (i == j && tx == 0 && ty == 0 && tz == 0)
                            continue;
                        // Avoid double-counting: for i==j require ordered
                        // lattice index; for i<j allow all images.
                        if (i == j) {
                            const int key = (tx + 1) * 9 + (ty + 1) * 3 + (tz + 1);
                            if (key > 13) // only half of self-image shell
                                continue;
                        }
                        double dx = atoms[static_cast<std::size_t>(j)][0]
                                    + tx * L[0]
                                    - atoms[static_cast<std::size_t>(i)][0];
                        double dy = atoms[static_cast<std::size_t>(j)][1]
                                    + ty * L[1]
                                    - atoms[static_cast<std::size_t>(i)][1];
                        double dz = atoms[static_cast<std::size_t>(j)][2]
                                    + tz * L[2]
                                    - atoms[static_cast<std::size_t>(i)][2];
                        // Use unwrapped distance for bond check (already imaged).
                        const double d2 = dx * dx + dy * dy + dz * dz;
                        if (d2 < bond2_lo || d2 > bond2_hi)
                            continue;
                        std::array<double, 3> mid = {
                            atoms[static_cast<std::size_t>(i)][0] + 0.5 * dx,
                            atoms[static_cast<std::size_t>(i)][1] + 0.5 * dy,
                            atoms[static_cast<std::size_t>(i)][2] + 0.5 * dz};
                        // Wrap midpoint into [0,L)
                        for (int d = 0; d < 3; ++d) {
                            double& c = mid[static_cast<std::size_t>(d)];
                            c -= L[d] * std::floor(c / L[d]);
                            if (c < 0.0)
                                c += L[d];
                        }
                        // Dedup
                        bool dupe = false;
                        for (const auto& m : mids) {
                            double ddx = m[0] - mid[0];
                            double ddy = m[1] - mid[1];
                            double ddz = m[2] - mid[2];
                            min_image(ddx, ddy, ddz);
                            if (ddx * ddx + ddy * ddy + ddz * ddz < 1e-8) {
                                dupe = true;
                                break;
                            }
                        }
                        if (!dupe)
                            mids.push_back(mid);
                    }
        }
    }
    if (mids.size() != 8)
        throw std::runtime_error(
            "diamond_bct_bond_midpoints: expected 8 unique bonds, got "
            + std::to_string(mids.size()));
    std::array<std::array<double, 3>, 8> out{};
    for (int i = 0; i < 8; ++i)
        out[static_cast<std::size_t>(i)] = mids[static_cast<std::size_t>(i)];
    return out;
}

double mv_pair_weight_cubic(double b_len) {
    if (!(b_len > 0.0))
        throw std::invalid_argument("mv_pair_weight_cubic: b_len must be > 0");
    return 1.0 / (2.0 * b_len * b_len);
}

std::array<double, 3> center_from_diagonal_phases(
    const std::vector<std::array<double, 3>>& b_cart,
    const std::vector<double>& w_b,
    const std::vector<std::complex<double>>& M_diag) {
    if (b_cart.empty() || b_cart.size() != w_b.size()
        || b_cart.size() != M_diag.size())
        throw std::invalid_argument(
            "center_from_diagonal_phases: size mismatch or empty");
    std::array<double, 3> r{0, 0, 0};
    for (std::size_t i = 0; i < b_cart.size(); ++i) {
        const double phi = std::arg(M_diag[i]);
        for (int a = 0; a < 3; ++a)
            r[static_cast<std::size_t>(a)] -=
                w_b[i] * b_cart[i][static_cast<std::size_t>(a)] * phi;
    }
    return r;
}

// ── Main compute ──────────────────────────────────────────────────────────

PeriodicWannierResult
PeriodicWannier::compute(GpwDft& method,
                         const std::vector<GpwAtom>& atoms) const {
    if (method.smearing != Smearing::none)
        throw std::invalid_argument(
            "PeriodicWannier::compute: requires an insulator (no metallic "
            "smearing)");
    const bool uks = method.is_uks();
    if (!uks && method.n_occ < 1)
        throw std::invalid_argument(
            "PeriodicWannier::compute: n_occ must be >= 1");
    if (uks && method.n_alpha + method.n_beta < 1)
        throw std::invalid_argument(
            "PeriodicWannier::compute: n_alpha + n_beta must be >= 1");
    if (k_mesh < 1)
        throw std::invalid_argument(
            "PeriodicWannier::compute: k_mesh must be >= 1");
    if (max_iter < 0)
        throw std::invalid_argument(
            "PeriodicWannier::compute: max_iter must be >= 0");
    if (!(tol > 0.0))
        throw std::invalid_argument(
            "PeriodicWannier::compute: tol must be > 0");
    if (!(step > 0.0))
        throw std::invalid_argument(
            "PeriodicWannier::compute: step must be > 0");
    if (!(proj_alpha_ang > 0.0))
        throw std::invalid_argument(
            "PeriodicWannier::compute: proj_alpha_ang must be > 0");
    if (method.nonperiodic)
        throw std::invalid_argument(
            "PeriodicWannier::compute: requires a periodic cell");

    const int N = k_mesh;
    method.k_mesh[0] = N;
    method.k_mesh[1] = N;
    method.k_mesh[2] = N;
    method.k_time_reversal = false;
    method.k_points.clear();
    method.k_point_weights.clear();

    const GpwResult& res = method.run(atoms);
    PeriodicWannierResult out;
    out.scf_converged = res.converged;
    out.energy_ha = res.e_total;
    if (!res.converged) {
        ATV_WARN("PeriodicWannier: SCF did not converge; continuing with "
                 "last orbitals");
    }
    if (res.n_kpts < 1 || res.k_frac.empty())
        throw std::runtime_error(
            "PeriodicWannier::compute: multi-k payload missing after run "
            "(need k_mesh > 1 or explicit k list)");
    if (res.n_kpts != N * N * N)
        throw std::runtime_error(
            "PeriodicWannier::compute: expected N³ k-points with TR off, got "
            + std::to_string(res.n_kpts));

    const int n_up = uks ? method.n_alpha : method.n_occ;
    const int n_dn = uks ? method.n_beta : 0;
    const int n_total = n_up + n_dn;
    if (n_total < 1)
        throw std::invalid_argument(
            "PeriodicWannier::compute: empty occupied manifold");

    const auto basis = flatten_basis(atoms);
    const int nbas = static_cast<int>(basis.size());
    const int n_max_ch = std::max(n_up, std::max(n_dn, 1));
    if (nbas < n_max_ch)
        throw std::runtime_error(
            "PeriodicWannier::compute: basis smaller than occupied count");

    std::vector<double> radii(static_cast<std::size_t>(nbas));
    double rmax = 0.0;
    for (int i = 0; i < nbas; ++i) {
        radii[static_cast<std::size_t>(i)] =
            basis_function_radius(basis[static_cast<std::size_t>(i)],
                                  method.eps_default);
        rmax = std::max(rmax, radii[static_cast<std::size_t>(i)]);
    }
    int mimg[3];
    pair_image_counts(method.cell, /*periodic=*/true, 2.0 * rmax, mimg);
    const double screen = method.eps_default;

    const double a0 = units::kBohrInAngstrom;
    const double L_ang[3] = {method.cell[0] * a0, method.cell[1] * a0,
                             method.cell[2] * a0};

    const auto nb = build_neighbors(res.k_frac, N, L_ang);
    const int nk = res.n_kpts;

    out.centers.clear();
    out.spreads.clear();
    out.spin.clear();
    out.total_spread = 0.0;
    out.total_spread_initial = 0.0;
    out.iterations = 0;
    out.converged = true;
    out.n_wann_up = uks ? n_up : 0;
    out.n_wann_dn = uks ? n_dn : 0;
    out.n_wann = n_total;

    const int n_spin = uks ? 2 : 1;
    for (int s = 0; s < n_spin; ++s) {
        const int n_ch = (s == 0) ? n_up : n_dn;
        if (n_ch < 1)
            continue;

        std::vector<std::vector<Cdouble>> C(static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik)
            C[static_cast<std::size_t>(ik)] =
                flatten_C_occ(res, ik, nbas, n_ch, /*spin=*/s);

        const auto trials =
            default_trials(n_ch, atoms, method.cell, trial_centers);
        ChannelLoc ch = localize_channel(
            C, n_ch, nk, res.k_frac, nb, basis, radii, method.cell, mimg,
            screen, trials, use_projection, proj_alpha_ang, a0, max_iter, tol,
            step);

        for (std::size_t i = 0; i < ch.centers.size(); ++i) {
            out.centers.push_back(ch.centers[i]);
            out.spreads.push_back(ch.spreads[i]);
            if (uks)
                out.spin.push_back(s);
        }
        out.total_spread += ch.total_spread;
        out.total_spread_initial += ch.total_spread_initial;
        out.iterations = std::max(out.iterations, ch.iterations);
        out.converged = out.converged && ch.converged;
    }

    if (static_cast<int>(out.centers.size()) != n_total)
        throw std::runtime_error(
            "PeriodicWannier::compute: center count mismatch after localization");
    return out;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV

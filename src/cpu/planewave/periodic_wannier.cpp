/// @file cpu/planewave/periodic_wannier.cpp
/// @brief Multi-k Marzari–Vanderbilt MLWF on frozen PW \(V_\mathrm{eff}\).

#include "atomiverse/cpu/planewave/periodic_wannier.h"

#include "atomiverse/core/log.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/planewave/internal/convergence.h"
#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/density_builder.h"
#include "atomiverse/cpu/planewave/internal/kpoint.h"
#include "atomiverse/cpu/planewave/internal/poisson.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/restricted_scf.h"
#include "atomiverse/cpu/planewave/internal/veff_builder.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ATV {
namespace cpu {
namespace pw {

namespace {

using Cdouble = std::complex<double>;
constexpr Cdouble I{0.0, 1.0};

long long pack_hkl(int h, int k, int l) {
    const long long off = 1 << 20;
    return ((static_cast<long long>(h) + off) << 42)
           | ((static_cast<long long>(k) + off) << 21)
           | (static_cast<long long>(l) + off);
}

/// Full occupied overlap \(M_{mn}=\sum_G c^*_m(G)\,c_n(G+\Delta)\).
void overlap_matrix(const internal::GVectorList& bra,
                    const std::vector<std::complex<AccReal>>& cb,
                    const internal::GVectorList& ket,
                    const std::vector<std::complex<AccReal>>& ck, int n_occ,
                    int shift_h, int shift_k, int shift_l,
                    std::vector<Cdouble>& M /* n_occ*n_occ row-major */) {
    const int ng_bra = bra.count();
    const int ng_ket = ket.count();
    std::unordered_map<long long, int> ket_idx;
    ket_idx.reserve(static_cast<std::size_t>(2 * ng_ket));
    for (int jg = 0; jg < ng_ket; ++jg)
        ket_idx.emplace(pack_hkl(ket.h[static_cast<std::size_t>(jg)],
                                 ket.k[static_cast<std::size_t>(jg)],
                                 ket.l[static_cast<std::size_t>(jg)]),
                        jg);

    M.assign(static_cast<std::size_t>(n_occ * n_occ), Cdouble{0.0, 0.0});
    for (int ig = 0; ig < ng_bra; ++ig) {
        const auto it = ket_idx.find(
            pack_hkl(bra.h[static_cast<std::size_t>(ig)] + shift_h,
                     bra.k[static_cast<std::size_t>(ig)] + shift_k,
                     bra.l[static_cast<std::size_t>(ig)] + shift_l));
        if (it == ket_idx.end())
            continue;
        const int jg = it->second;
        for (int m = 0; m < n_occ; ++m) {
            const auto bm =
                std::conj(cb[static_cast<std::size_t>(m * ng_bra + ig)]);
            for (int n = 0; n < n_occ; ++n)
                M[static_cast<std::size_t>(m * n_occ + n)] +=
                    static_cast<Cdouble>(
                        bm * ck[static_cast<std::size_t>(n * ng_ket + jg)]);
        }
    }
}

/// Neighbor of MP index along lattice direction `dir` by ±1 step.
struct Neighbor {
    int ik_nb;   ///< neighbor k-index
    int dir;     ///< 0,1,2 lattice direction
    int sign;    ///< +1 or -1
    int wrap;    ///< +1 if crossed +BZ, -1 if -BZ, 0 else
    double wb;   ///< SMV weight [Å²]
    std::array<double, 3> b_cart; ///< b in Cartesian [1/Å]
};

/// SMV shell weights for ±b_i (i=0,1,2) so that
/// \(\sum_b w_b b_\alpha b_\beta \approx \delta_{\alpha\beta}\).
///
/// With three independent pair-weights \(w_i\) for \(\pm\mathbf b_i\),
/// \(2\sum_i w_i (b_i\otimes b_i) = I\). Solved by 3×3 normal equations on the
/// six independent components of the symmetric tensor (least squares). Falls
/// back to the cubic formula when the system is ill-conditioned.
std::array<double, 3>
smv_axis_weights(const std::array<std::array<double, 3>, 3>& b_shell) {
    // Columns of A (6 x 3): 2 * vech(b_i ⊗ b_i)
    // Order: xx, yy, zz, xy, xz, yz
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
    // Normal equations N = A^T A (3x3), v = A^T rhs
    double N[3][3]{};
    double v[3]{};
    for (int r = 0; r < 6; ++r) {
        for (int i = 0; i < 3; ++i) {
            v[i] += A[r][i] * rhs[r];
            for (int j = 0; j < 3; ++j)
                N[i][j] += A[r][i] * A[r][j];
        }
    }
    // Solve N w = v by Gaussian elimination with partial pivot.
    double M[3][4] = {{N[0][0], N[0][1], N[0][2], v[0]},
                      {N[1][0], N[1][1], N[1][2], v[1]},
                      {N[2][0], N[2][1], N[2][2], v[2]}};
    for (int col = 0; col < 3; ++col) {
        int piv = col;
        for (int r = col + 1; r < 3; ++r)
            if (std::fabs(M[r][col]) > std::fabs(M[piv][col]))
                piv = r;
        if (std::fabs(M[piv][col]) < 1e-18) {
            // Ill-conditioned: cubic fallback per axis.
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

/// Build ±e_i neighbors for an N³ Γ-centered MP mesh.
std::vector<std::vector<Neighbor>>
build_neighbors(int N, const Mat3& recip /* rows b_i [1/Å], incl. 2π */) {
    const int nk = N * N * N;
    std::vector<std::vector<Neighbor>> nb(static_cast<std::size_t>(nk));
    auto idx = [N](int i, int j, int k) {
        return (i * N + j) * N + k;
    };
    auto ijk = [N](int ik) {
        const int k = ik % N;
        const int j = (ik / N) % N;
        const int i = ik / (N * N);
        return std::array<int, 3>{i, j, k};
    };

    // Shell vectors b_i = B_i / N [1/Å]; SMV weights from Σ w b⊗b = I.
    std::array<std::array<double, 3>, 3> b_shell{};
    for (int d = 0; d < 3; ++d) {
        b_shell[static_cast<std::size_t>(d)] = {
            static_cast<double>(recip(d, 0)) / static_cast<double>(N),
            static_cast<double>(recip(d, 1)) / static_cast<double>(N),
            static_cast<double>(recip(d, 2)) / static_cast<double>(N)};
    }
    const auto w_axis = smv_axis_weights(b_shell);

    for (int ik = 0; ik < nk; ++ik) {
        const auto c = ijk(ik);
        for (int d = 0; d < 3; ++d) {
            for (int s : {+1, -1}) {
                std::array<int, 3> n = c;
                n[static_cast<std::size_t>(d)] += s;
                int wrap = 0;
                if (n[static_cast<std::size_t>(d)] >= N) {
                    n[static_cast<std::size_t>(d)] -= N;
                    wrap = +1;
                } else if (n[static_cast<std::size_t>(d)] < 0) {
                    n[static_cast<std::size_t>(d)] += N;
                    wrap = -1;
                }
                Neighbor e;
                e.ik_nb = idx(n[0], n[1], n[2]);
                e.dir = d;
                e.sign = s;
                e.wrap = wrap;
                e.wb = w_axis[static_cast<std::size_t>(d)];
                e.b_cart = {s * b_shell[static_cast<std::size_t>(d)][0],
                            s * b_shell[static_cast<std::size_t>(d)][1],
                            s * b_shell[static_cast<std::size_t>(d)][2]};
                nb[static_cast<std::size_t>(ik)].push_back(e);
            }
        }
    }
    return nb;
}

/// Rotate M0 by U†(k) M0 U(k+b).
void rotate_M(const std::vector<Cdouble>& M0, const std::vector<Cdouble>& Uk,
              const std::vector<Cdouble>& Ukb, int n,
              std::vector<Cdouble>& M) {
    // T = M0 @ Ukb; M = Uk† @ T
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

/// Safe Im ln of a complex number (0 if nearly vanishing).
double safe_arg(Cdouble z) {
    if (std::norm(z) < 1e-24)
        return 0.0;
    return std::arg(z);
}

/// Total SMV spread Σ_n Ω_n [Å²] from rotated diagonal M.
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
                // r = −Σ w b Im ln M  (Marzari–Vanderbilt 1997 eq. 31)
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
        // Numerical noise can make Ω slightly negative for well-localized
        // orbitals; clamp for reporting.
        if (om < 0.0 && om > -1e-6)
            om = 0.0;
        out.spreads[static_cast<std::size_t>(n)] = om;
        out.total_spread += om;
    }
}

/// Matrix exponential of a small anti-Hermitian matrix via scaled Taylor.
void expm_antihermitian(const std::vector<Cdouble>& G, int n, double alpha,
                        std::vector<Cdouble>& expG) {
    // exp(-alpha G) with G antihermitian → unitary.
    std::vector<Cdouble> A(static_cast<std::size_t>(n * n));
    for (std::size_t i = 0; i < A.size(); ++i)
        A[i] = -alpha * G[i];

    expG.assign(static_cast<std::size_t>(n * n), Cdouble{0, 0});
    for (int i = 0; i < n; ++i)
        expG[static_cast<std::size_t>(i * n + i)] = Cdouble{1, 0};

    std::vector<Cdouble> term = expG; // I
    // 12th-order Taylor is plenty for small steps.
    for (int p = 1; p <= 12; ++p) {
        std::vector<Cdouble> next(static_cast<std::size_t>(n * n), Cdouble{0, 0});
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

/// Right-multiply U ← U @ expG (n x n).
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

/// MV gradient G(k) (antihermitian) — Mostofi CPC / Wannier90 form on R̃.
///
/// With R̃_mn = M_mn exp(-i arg M_nn):
///   G += 4 w_b [ (R̃ − R̃†)/(2i)  +  i (R̃ + R̃†)/2  ⊙ q  ]  then antihermitize.
/// The factor of i on the second term yields an antihermitian matrix so that
/// U ← U exp(−α G) stays unitary for small α.
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
                    // Off-diagonal localization force: antihermitian part of R
                    // plus phase (diagonal-spread) force.
                    // T_mn = 4 wb * [ Im(R_mn) * i?  standard:
                    //   A = (R - R†)/(2i) Hermitian
                    //   S = (R + R†)/2 Hermitian
                    //   dΩ ~ 4 wb (A + S q_n) which is Hermitian; convert via i*
                    //   so G is antihermitian: G = i * Hermitian.
                    const Cdouble Rnm_c =
                        std::conj(M[static_cast<std::size_t>(n * n_wann + m)]
                                  * std::exp(-I * q[static_cast<std::size_t>(m)]));
                    const Cdouble A = (Rmn - Rnm_c) / (2.0 * I);
                    const Cdouble S = 0.5 * (Rmn + Rnm_c);
                    const Cdouble H =
                        A + S * q[static_cast<std::size_t>(n)]; // Hermitian-ish
                    G[static_cast<std::size_t>(m * n_wann + n)] +=
                        4.0 * wb * (I * H);
                }
            }
        }
        // Exact antihermitization (kill residual Herm parts / noise).
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

/// Fourier transform of a normalized Gaussian at τ [Å] on a G-sphere.
/// PW coefficients live in Bohr^{-1} G-space; convert consistently.
void gaussian_proj_coeffs(const internal::GVectorList& gv, double vol_bohr,
                          const std::array<double, 3>& tau_ang, double alpha_ang,
                          std::vector<Cdouble>& g) {
    // α_Bohr from α_Å: exp(-α_Å |r_Å|²) = exp(-α_Å a0² |r_B|²)
    // so α_B = α_Å * a0² with a0 = kBohrInAngstrom.
    const double a0 = units::kBohrInAngstrom;
    const double alpha_B = alpha_ang * a0 * a0;
    const double tau_B[3] = {tau_ang[0] / a0, tau_ang[1] / a0, tau_ang[2] / a0};
    // ⟨k+G|g⟩ ∝ Ω^{-1/2} (π/α)^{3/4} exp(-i q·τ) exp(-|q|²/(4α))
    const double pref =
        std::pow(units::pi / alpha_B, 0.75) / std::sqrt(vol_bohr);
    const int ng = gv.count();
    g.resize(static_cast<std::size_t>(ng));
    for (int ig = 0; ig < ng; ++ig) {
        // gv.g is G only; k is in gv.kvec; q = k+G [1/Bohr]
        const double qx = static_cast<double>(
            gv.kvec.x() + gv.g[static_cast<std::size_t>(ig)].x());
        const double qy = static_cast<double>(
            gv.kvec.y() + gv.g[static_cast<std::size_t>(ig)].y());
        const double qz = static_cast<double>(
            gv.kvec.z() + gv.g[static_cast<std::size_t>(ig)].z());
        const double q2 = qx * qx + qy * qy + qz * qz;
        const double phase = qx * tau_B[0] + qy * tau_B[1] + qz * tau_B[2];
        g[static_cast<std::size_t>(ig)] =
            pref * std::exp(-q2 / (4.0 * alpha_B)) * std::exp(-I * phase);
    }
}

/// Thin SVD for n x n complex A → U = Z W† via one-sided Jacobi (n small).
void projection_U(const std::vector<Cdouble>& A /* n x n, rows=bands */,
                  int n, std::vector<Cdouble>& U) {
    // SVD A = Z Σ W† via eigen-decomposition of A†A (Hermitian n x n).
    // A†A_ij = sum_k conj(A_ki)*A_kj
    std::vector<Cdouble> H(static_cast<std::size_t>(n * n), Cdouble{0, 0});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            Cdouble s{0, 0};
            for (int k = 0; k < n; ++k)
                s += std::conj(A[static_cast<std::size_t>(k * n + i)])
                     * A[static_cast<std::size_t>(k * n + j)];
            H[static_cast<std::size_t>(i * n + j)] = s;
        }

    // Jacobi Hermitian eigensolve (in-place on H, vectors in V).
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
                // Complex Jacobi: phase to make apq real after rotation.
                const double phi = std::arg(apq);
                const Cdouble phase = std::exp(-I * phi); // so phase*apq real+
                const double apq_r = std::abs(apq);
                const double t =
                    (thr == 0.0)
                        ? 1.0
                        : ((thr > 0 ? 1.0 : -1.0)
                           / (std::abs(thr) / apq_r
                              + std::sqrt(1.0 + (thr / apq_r) * (thr / apq_r))));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = t * c;
                // Rotate H and V with complex 2x2.
                // For Hermitian Jacobi with complex off-diag: use
                //   [c, s*e^{-iφ}; -s*e^{iφ}, c]
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

    // W = V (eigenvectors of A†A). Σ_i = sqrt(eval_i).
    // Z = A W Σ^{-1}; U_proj = Z W† = A W Σ^{-1} W†.
    // For well-conditioned projection, U = A (A†A)^{-1/2}.
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
    // T = W * Σ^{-1} * W†  (= (A†A)^{-1/2})
    std::vector<Cdouble> T(static_cast<std::size_t>(n * n), Cdouble{0, 0});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            Cdouble s{0, 0};
            for (int k = 0; k < n; ++k)
                s += WSinv[static_cast<std::size_t>(i * n + k)]
                     * std::conj(V[static_cast<std::size_t>(j * n + k)]);
            T[static_cast<std::size_t>(i * n + j)] = s;
        }
    // U = A @ T
    U.assign(static_cast<std::size_t>(n * n), Cdouble{0, 0});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            Cdouble s{0, 0};
            for (int k = 0; k < n; ++k)
                s += A[static_cast<std::size_t>(i * n + k)]
                     * T[static_cast<std::size_t>(k * n + j)];
            U[static_cast<std::size_t>(i * n + j)] = s;
        }
}

/// Infer cubic a from fcc primitive cell (row length * √2).
double cubic_a_from_cell(const Mat3& cell) {
    const double ax = static_cast<double>(cell(0, 0));
    const double ay = static_cast<double>(cell(0, 1));
    const double az = static_cast<double>(cell(0, 2));
    const double len = std::sqrt(ax * ax + ay * ay + az * az);
    return len * std::sqrt(2.0);
}

/// Per-channel MV localization result (restricted or one spin).
struct ChannelLoc {
    std::vector<std::array<double, 3>> centers;
    std::vector<double> spreads;
    double total_spread = 0.0;
    double total_spread_initial = 0.0;
    int iterations = 0;
    bool converged = false;
};

/// Marzari–Vanderbilt localization on precomputed NSCF coefficients.
ChannelLoc localize_channel(
    const std::vector<std::vector<std::complex<AccReal>>>& coef,
    const std::vector<internal::GVectorList>& gv_k, int n_wann, int N,
    const Mat3& recip, double vol_bohr,
    const std::vector<std::array<double, 3>>& trials, bool use_projection,
    double proj_alpha_ang, int max_iter, double tol, double step) {
    ChannelLoc out;
    if (n_wann < 1)
        return out;
    const int nk = N * N * N;
    const auto nb = build_neighbors(N, recip);

    std::vector<std::vector<std::vector<Cdouble>>> M0(
        static_cast<std::size_t>(nk));
    for (int ik = 0; ik < nk; ++ik) {
        const auto& neigh = nb[static_cast<std::size_t>(ik)];
        M0[static_cast<std::size_t>(ik)].resize(neigh.size());
        for (std::size_t ib = 0; ib < neigh.size(); ++ib) {
            const auto& e = neigh[ib];
            const int sh = (e.wrap != 0 && e.dir == 0) ? e.wrap : 0;
            const int sk = (e.wrap != 0 && e.dir == 1) ? e.wrap : 0;
            const int sl = (e.wrap != 0 && e.dir == 2) ? e.wrap : 0;
            overlap_matrix(gv_k[static_cast<std::size_t>(ik)],
                           coef[static_cast<std::size_t>(ik)],
                           gv_k[static_cast<std::size_t>(e.ik_nb)],
                           coef[static_cast<std::size_t>(e.ik_nb)], n_wann, sh,
                           sk, sl, M0[static_cast<std::size_t>(ik)][ib]);
        }
    }

    std::vector<std::vector<Cdouble>> U(static_cast<std::size_t>(nk));
    for (int ik = 0; ik < nk; ++ik) {
        U[static_cast<std::size_t>(ik)].assign(
            static_cast<std::size_t>(n_wann * n_wann), Cdouble{0, 0});
        for (int i = 0; i < n_wann; ++i)
            U[static_cast<std::size_t>(ik)]
             [static_cast<std::size_t>(i * n_wann + i)] = Cdouble{1, 0};
    }

    if (use_projection) {
        for (int ik = 0; ik < nk; ++ik) {
            auto& gvk = gv_k[static_cast<std::size_t>(ik)];
            const int ng = gvk.count();
            std::vector<Cdouble> A(static_cast<std::size_t>(n_wann * n_wann),
                                   Cdouble{0, 0});
            for (int n = 0; n < n_wann; ++n) {
                std::vector<Cdouble> gcoef;
                gaussian_proj_coeffs(gvk, vol_bohr,
                                     trials[static_cast<std::size_t>(n)],
                                     proj_alpha_ang, gcoef);
                const auto& c = coef[static_cast<std::size_t>(ik)];
                for (int j = 0; j < n_wann; ++j) {
                    Cdouble s{0, 0};
                    for (int ig = 0; ig < ng; ++ig)
                        s += std::conj(static_cast<Cdouble>(
                                 c[static_cast<std::size_t>(j * ng + ig)]))
                             * gcoef[static_cast<std::size_t>(ig)];
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
        std::vector<std::vector<Cdouble>> Uid(static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik) {
            Uid[static_cast<std::size_t>(ik)].assign(
                static_cast<std::size_t>(n_wann * n_wann), Cdouble{0, 0});
            for (int i = 0; i < n_wann; ++i)
                Uid[static_cast<std::size_t>(ik)]
                   [static_cast<std::size_t>(i * n_wann + i)] = Cdouble{1, 0};
        }
        auto U_save = U;
        U = Uid;
        rebuild_M(M_kb);
        out.total_spread_initial = total_spread_from_M(M_kb, nb, n_wann);
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
        ATV_DEBUG_CAT("pw", "mlwf iter={} |G|_max={:.3e} Omega={:.6f}", it, gmax,
                      omega_prev);
        if (it > 0 && (!(gmax > 0.0) || gmax < tol || !std::isfinite(gmax))) {
            conv = std::isfinite(gmax) && gmax < tol;
            break;
        }

        double alpha = step;
        bool accepted = false;
        auto U_try = U;
        for (int attempt = 0; attempt < 6; ++attempt) {
            U_try = U;
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
        if (!accepted) {
            conv = true;
            break;
        }
    }
    if (max_iter == 0)
        conv = true;

    U = std::move(U_best);
    M_kb = std::move(M_best);
    PeriodicWannierResult tmp;
    fill_centers_spreads(M_kb, nb, n_wann, tmp);
    out.centers = std::move(tmp.centers);
    out.spreads = std::move(tmp.spreads);
    out.total_spread = tmp.total_spread;
    out.iterations = iters;
    out.converged = conv && std::isfinite(out.total_spread);
    return out;
}

/// Resolve projection trial centers for a manifold of size n_wann.
std::vector<std::array<double, 3>>
resolve_trials(const std::vector<std::array<double, 3>>& trial_centers,
               int n_wann, const cpu::Frame& frame) {
    std::vector<std::array<double, 3>> trials = trial_centers;
    if (trials.empty()) {
        if (n_wann == 4 && frame.nlocal() == 2) {
            const double a = cubic_a_from_cell(frame.box.cell());
            const auto mids = diamond_bond_midpoints(a);
            trials.assign(mids.begin(), mids.end());
        } else {
            trials.assign(static_cast<std::size_t>(n_wann),
                          std::array<double, 3>{0, 0, 0});
        }
    }
    if (static_cast<int>(trials.size()) != n_wann)
        throw std::invalid_argument(
            "PeriodicWannier::compute: trial_centers size must equal n_wann "
            "per spin channel");
    return trials;
}

} // namespace

// ── Public pure helpers ───────────────────────────────────────────────────

std::array<std::array<double, 3>, 4> diamond_bond_midpoints(double a_ang) {
    if (!(a_ang > 0.0))
        throw std::invalid_argument(
            "diamond_bond_midpoints: a_ang must be > 0");
    const double s = a_ang / 8.0;
    return {{{{s, s, s}}, {{s, -s, -s}}, {{-s, s, -s}}, {{-s, -s, s}}}};
}

double mv_pair_weight_cubic(double b_len) {
    if (!(b_len > 0.0))
        throw std::invalid_argument("mv_pair_weight_cubic: b_len must be > 0");
    // N_b = 6 → w = 3/(6 b²) = 1/(2 b²)
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

PeriodicWannierResult PeriodicWannier::compute(const PW& method,
                                               const cpu::Frame& frame) const {
    using internal::DavidsonSolver;

    if (method.smearing != Smearing::none)
        throw std::invalid_argument(
            "PeriodicWannier::compute: requires an insulator (no metallic "
            "smearing)");
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

    const int N = k_mesh;
    const int nk = N * N * N;
    const Mat3 recip = frame.box.recip_cell();

    // ── Collinear spin-polarized path ───────────────────────────────────
    if (method.spin_polarized) {
        if (method.multiplicity < 1)
            throw std::invalid_argument(
                "PeriodicWannier::compute: multiplicity must be >= 1 when "
                "spin_polarized");

        internal::PWProblem prob;
        internal::setup_pw_problem(frame, method.upf_path, method.ecut_ry,
                                   method.k_mesh, prob);
        internal::PWWorkspace& ws = prob.ws;
        auto& gv0 = prob.gv0;
        const int n_kpts_scf = static_cast<int>(prob.kg.kpoints.size());
        const int ng0 = gv0.count();
        const int n_elec = prob.n_elec;
        const int mag = method.multiplicity - 1;
        if ((n_elec + mag) % 2 != 0 || mag > n_elec)
            throw std::invalid_argument(
                "PeriodicWannier::compute: n_elec and multiplicity have "
                "incompatible parity");
        const int n_alpha = (n_elec + mag) / 2;
        const int n_beta = (n_elec - mag) / 2;
        const int n_occ[2] = {n_alpha, n_beta};
        const int n_bands = std::max(std::max(n_alpha, n_beta), 1);
        if (n_alpha + n_beta < 1)
            throw std::invalid_argument(
                "PeriodicWannier::compute: empty occupied manifold");

        ws.mix_beta = AccReal(method.mix_beta);
        ws.pulay_depth = method.pulay_depth;
        ws.nspin = 2;
        const std::size_t ngs = static_cast<std::size_t>(ng0);
        ws.rho_spin_G.resize(2);
        ws.rho_spin_G_old.resize(2);
        ws.V_xc_spin_G.resize(2);
        for (int s = 0; s < 2; ++s) {
            ws.rho_spin_G[s].resize({ngs});
            ws.rho_spin_G_old[s].resize({ngs});
            ws.V_xc_spin_G[s].resize({ngs});
        }
        const AccReal fa =
            (n_elec > 0) ? AccReal(n_alpha) / AccReal(n_elec) : AccReal(0.5);
        const AccReal fb =
            (n_elec > 0) ? AccReal(n_beta) / AccReal(n_elec) : AccReal(0.5);
        for (int ig = 0; ig < ng0; ++ig) {
            ws.rho_spin_G[0][ig] = fa * ws.rho_G[ig];
            ws.rho_spin_G[1][ig] = fb * ws.rho_G[ig];
            ws.rho_spin_G_old[0][ig] = ws.rho_spin_G[0][ig];
            ws.rho_spin_G_old[1][ig] = ws.rho_spin_G[1][ig];
        }

        internal::PoissonSolver poisson;
        internal::XCBuilder xc;
        internal::VeffBuilder veff;
        internal::DensityBuilder density;
        DavidsonSolver davidson(n_bands, AccReal(method.tol) * AccReal(0.1),
                                40);
        xc.prepare(ws);
        veff.prepare(ws);
        density.prepare(ws);
        density.reset(ws);
        davidson.prepare(ws);

        bool scf_ok = false;
        AccReal e_band_last = AccReal(0);
        for (int iter = 0; iter < method.max_iter; ++iter) {
            for (int ig = 0; ig < ng0; ++ig)
                ws.rho_G[ig] = ws.rho_spin_G[0][ig] + ws.rho_spin_G[1][ig];
            poisson.step(ws);
            xc.step_spin(ws, method.xc);

            xt::xarray<std::complex<AccReal>> rho_out[2];
            e_band_last = AccReal(0);
            auto* saved_gvec = ws.gvec;
            for (int s = 0; s < 2; ++s) {
                for (int ig = 0; ig < ng0; ++ig)
                    ws.V_xc_G[ig] = ws.V_xc_spin_G[s][ig];
                veff.step(ws);
                density.begin(ws);
                for (int ik = 0; ik < n_kpts_scf; ++ik) {
                    auto& gvk = prob.gv_k[static_cast<std::size_t>(ik)];
                    ws.gvec = &gvk;
                    ws.n_g = gvk.count();
                    ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
                    for (int ig = 0; ig < ws.n_g; ++ig)
                        ws.T_G[ig] = gvk.ekin[ig];
                    if (!prob.kb_proj_k.empty())
                        ws.kb_proj =
                            prob.kb_proj_k[static_cast<std::size_t>(ik)];
                    if (!prob.atom_wfc_k.empty())
                        ws.atom_wfc =
                            prob.atom_wfc_k[static_cast<std::size_t>(ik)];
                    davidson.step(ws);
                    const AccReal weight =
                        prob.kg.weights[static_cast<std::size_t>(ik)];
                    for (int b = 0; b < n_occ[s]; ++b)
                        e_band_last += weight * ws.eigenvalues[b];
                    density.accumulate_kpoint(ws, weight, n_occ[s], AccReal(1));
                    ws.gvec = saved_gvec;
                    ws.n_g = ng0;
                }
                density.extract(ws, rho_out[s]);
            }
            for (int s = 0; s < 2; ++s) {
                for (int ig = 0; ig < ng0; ++ig) {
                    ws.rho_spin_G_old[s][ig] = ws.rho_spin_G[s][ig];
                    ws.rho_spin_G[s][ig] =
                        ws.rho_spin_G_old[s][ig]
                        + AccReal(method.mix_beta)
                              * (rho_out[s][ig] - ws.rho_spin_G_old[s][ig]);
                }
            }
            AccReal dr = 0;
            for (int ig = 0; ig < ng0; ++ig) {
                const auto old = ws.rho_G[ig];
                ws.rho_G[ig] = ws.rho_spin_G[0][ig] + ws.rho_spin_G[1][ig];
                const auto d = ws.rho_G[ig] - old;
                dr += d.real() * d.real() + d.imag() * d.imag();
            }
            if (std::sqrt(static_cast<double>(dr)) < method.tol) {
                scf_ok = true;
                break;
            }
        }

        // Freeze V_eff,σ from the mixed density.
        for (int ig = 0; ig < ng0; ++ig)
            ws.rho_G[ig] = ws.rho_spin_G[0][ig] + ws.rho_spin_G[1][ig];
        poisson.step(ws);
        xc.step_spin(ws, method.xc);
        std::vector<std::vector<AccReal>> veff_r(2);
        for (int s = 0; s < 2; ++s) {
            for (int ig = 0; ig < ng0; ++ig)
                ws.V_xc_G[ig] = ws.V_xc_spin_G[s][ig];
            veff.step(ws);
            veff_r[static_cast<std::size_t>(s)].assign(
                ws.V_eff_r.begin(), ws.V_eff_r.begin() + ws.nr);
        }

        PeriodicWannierResult out;
        out.scf_converged = scf_ok;
        // Density residual is the SCF gate; full E_tot assembly is not required
        // for MLWF centers (kept 0 rather than a partial band sum).
        (void)e_band_last;
        out.energy_ry = 0.0;
        out.n_wann_up = n_alpha;
        out.n_wann_dn = n_beta;
        out.n_wann = n_alpha + n_beta;
        out.converged = true;

        // MLWF k-mesh tables (shared geometry; V_eff swapped per spin).
        auto kg = internal::generate_mp_mesh(N, N, N);
        std::vector<Vec3> kvec;
        kvec.reserve(static_cast<std::size_t>(nk));
        for (const auto& kf : kg.kpoints)
            kvec.push_back(kf);
        internal::rebuild_k_tables(prob, frame, method.upf_path, method.ecut_ry,
                                   kvec);

        for (int s = 0; s < 2; ++s) {
            if (n_occ[s] < 1)
                continue;
            // Bind frozen V_eff for this spin on the density grid.
            for (int i = 0; i < ws.nr; ++i)
                ws.V_eff_r[i] =
                    veff_r[static_cast<std::size_t>(s)][static_cast<std::size_t>(
                        i)];

            DavidsonSolver dav_s(n_occ[s], AccReal(method.tol) * AccReal(0.1),
                                 50);
            dav_s.prepare(ws);
            std::vector<std::vector<std::complex<AccReal>>> coef(
                static_cast<std::size_t>(nk));
            auto* saved_gvec = ws.gvec;
            for (int ik = 0; ik < nk; ++ik) {
                auto& gvk = prob.gv_k[static_cast<std::size_t>(ik)];
                ws.gvec = &gvk;
                ws.n_g = gvk.count();
                ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
                for (int ig = 0; ig < ws.n_g; ++ig)
                    ws.T_G[ig] = gvk.ekin[ig];
                if (!prob.kb_proj_k.empty())
                    ws.kb_proj = prob.kb_proj_k[static_cast<std::size_t>(ik)];
                if (!prob.atom_wfc_k.empty())
                    ws.atom_wfc = prob.atom_wfc_k[static_cast<std::size_t>(ik)];
                dav_s.step(ws);
                auto& c = coef[static_cast<std::size_t>(ik)];
                c.resize(static_cast<std::size_t>(n_occ[s] * ws.n_g));
                for (int b = 0; b < n_occ[s]; ++b)
                    for (int ig = 0; ig < ws.n_g; ++ig)
                        c[static_cast<std::size_t>(b * ws.n_g + ig)] =
                            ws.wfc(b, ig);
            }
            ws.gvec = saved_gvec;
            ws.n_g = ng0;

            const auto trials =
                resolve_trials(trial_centers, n_occ[s], frame);
            const ChannelLoc ch = localize_channel(
                coef, prob.gv_k, n_occ[s], N, recip,
                static_cast<double>(prob.vol_bohr), trials, use_projection,
                proj_alpha_ang, max_iter, tol, step);

            for (std::size_t i = 0; i < ch.centers.size(); ++i) {
                out.centers.push_back(ch.centers[i]);
                out.spreads.push_back(ch.spreads[i]);
                out.spin.push_back(s);
            }
            out.total_spread += ch.total_spread;
            out.total_spread_initial += ch.total_spread_initial;
            out.iterations = std::max(out.iterations, ch.iterations);
            out.converged = out.converged && ch.converged;
            ATV_DEBUG_CAT("pw",
                          "mlwf spin={} n_occ={} Omega={:.6f} (init {:.6f}) "
                          "conv={} iters={}",
                          s, n_occ[s], ch.total_spread, ch.total_spread_initial,
                          ch.converged, ch.iterations);
        }
        return out;
    }

    // ── Restricted closed-shell path ────────────────────────────────────
    internal::PWProblem prob;
    const PWResult scf = internal::run_restricted_scf(frame, method, prob);
    const int n_wann = prob.n_bands;
    if (n_wann < 1)
        throw std::invalid_argument(
            "PeriodicWannier::compute: empty occupied manifold");

    PeriodicWannierResult out;
    out.scf_converged = scf.converged;
    out.energy_ry = scf.energy;
    out.n_wann = n_wann;
    out.n_wann_up = 0;
    out.n_wann_dn = 0;

    const auto trials = resolve_trials(trial_centers, n_wann, frame);

    auto kg = internal::generate_mp_mesh(N, N, N);
    std::vector<Vec3> kvec;
    kvec.reserve(static_cast<std::size_t>(nk));
    for (const auto& kf : kg.kpoints)
        kvec.push_back(kf);

    internal::rebuild_k_tables(prob, frame, method.upf_path, method.ecut_ry,
                               kvec);

    internal::PWWorkspace& ws = prob.ws;
    std::vector<std::vector<std::complex<AccReal>>> coef(
        static_cast<std::size_t>(nk));
    DavidsonSolver davidson(n_wann, AccReal(method.tol) * AccReal(0.1), 50);
    davidson.prepare(ws);
    auto* saved_gvec = ws.gvec;
    const int n_g0 = ws.n_g;
    for (int ik = 0; ik < nk; ++ik) {
        auto& gvk = prob.gv_k[static_cast<std::size_t>(ik)];
        ws.gvec = &gvk;
        ws.n_g = gvk.count();
        ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
        for (int ig = 0; ig < ws.n_g; ++ig)
            ws.T_G[ig] = gvk.ekin[ig];
        if (!prob.kb_proj_k.empty())
            ws.kb_proj = prob.kb_proj_k[static_cast<std::size_t>(ik)];
        if (!prob.atom_wfc_k.empty())
            ws.atom_wfc = prob.atom_wfc_k[static_cast<std::size_t>(ik)];

        davidson.step(ws);

        auto& c = coef[static_cast<std::size_t>(ik)];
        c.resize(static_cast<std::size_t>(n_wann * ws.n_g));
        for (int b = 0; b < n_wann; ++b)
            for (int ig = 0; ig < ws.n_g; ++ig)
                c[static_cast<std::size_t>(b * ws.n_g + ig)] = ws.wfc(b, ig);
    }
    ws.gvec = saved_gvec;
    ws.n_g = n_g0;

    const ChannelLoc ch = localize_channel(
        coef, prob.gv_k, n_wann, N, recip, static_cast<double>(prob.vol_bohr),
        trials, use_projection, proj_alpha_ang, max_iter, tol, step);
    out.centers = std::move(ch.centers);
    out.spreads = std::move(ch.spreads);
    out.total_spread = ch.total_spread;
    out.total_spread_initial = ch.total_spread_initial;
    out.iterations = ch.iterations;
    out.converged = ch.converged;

    ATV_DEBUG_CAT("pw",
                  "mlwf done: Omega={:.6f} (init {:.6f}) conv={} iters={}",
                  out.total_spread, out.total_spread_initial, out.converged,
                  out.iterations);
    return out;
}

} // namespace pw
} // namespace cpu
} // namespace ATV

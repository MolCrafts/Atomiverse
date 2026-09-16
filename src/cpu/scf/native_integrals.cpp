/// @file cpu/scf/native_integrals.cpp
/// @brief Native C++ molecular integral implementations for s+p type Gaussians.
///
/// Drop-in replacements for libcint functions. All formulas from:
///   Szabo & Ostlund, "Modern Quantum Chemistry" (1996), Ch. 3 (Appendix A).
///   Obara-Saika recurrence for general case; direct formulas for s-type.
///
/// Validated element-by-element against libcint in test_native_integrals.cu.

#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/core/gto.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/cart.h"
#include "atomiverse/cpu/scf/rys_eri.h"

#include <cmath>
#include <xsimd/xsimd.hpp>

namespace ATV {
namespace cpu {
namespace scf {
namespace native {

// ── Helpers ───────────────────────────────────────────────────────────

/// Squared distance between two 3D points.
static inline double dist2(const double a[3], const double b[3]) {
    double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

// ── Phase 1: GTO normalization ────────────────────────────────────────

double gto_norm(int l, double alpha) { return ATV::gto_norm(l, alpha); }
double phys_norm(double alpha) { return ATV::phys_norm_s(alpha); }
double phys_norm_p(double alpha) { return ATV::phys_norm_p(alpha); }
double phys_norm_d(double alpha) { return ATV::phys_norm_d(alpha); }
double phys_norm_f(double alpha) { return ATV::phys_norm_f(alpha); }
double phys_norm_g(double alpha) { return ATV::phys_norm_g(alpha); }

// ── Phase 4 prerequisite: Boys function ───────────────────────────────

double boys_f0(double x) { return ATV::boys_f0(x); }
double boys_fn(int n, double x) { return ATV::boys_fn(n, x); }

// ── Phase 2: Overlap integral (s|s) ───────────────────────────────────

/// Primitive overlap for s-type Gaussians.
/// <a|b> = (pi/gamma)^{3/2} * exp(-mu * R^2)
/// where gamma = alpha_a + alpha_b, mu = alpha_a * alpha_b / gamma
static inline double overlap_primitive(double alpha_a, const double A[3],
                                       double alpha_b, const double B[3]) {
    double gamma = alpha_a + alpha_b;
    double mu = alpha_a * alpha_b / gamma;
    double R2 = dist2(A, B);
    return std::pow(units::pi / gamma, 1.5) * std::exp(-mu * R2);
}

double overlap_ss(const Shell& a, const Shell& b) {
    // S = sum_pq c_p * c_q * <prim_p | prim_q>
    // PERF: unchecked() candidate — 9 iterations for STO-3G
    double S = 0;
    for (int p = 0; p < a.n_prim; ++p)
        for (int q = 0; q < b.n_prim; ++q)
            S +=
                a.coeff[p] * b.coeff[q]
                * overlap_primitive(a.alpha[p], a.center, b.alpha[q], b.center);
    return S;
}

// ── Phase 3: Kinetic energy integral (s|T|s) ──────────────────────────

double kinetic_ss(const Shell& a, const Shell& b) {
    // T_pq = mu * (3 - 2*mu*R^2) * S_pq  for each primitive pair
    // PERF: unchecked() candidate — 9 iterations for STO-3G
    double T = 0;
    double R2 = dist2(a.center, b.center);
    for (int p = 0; p < a.n_prim; ++p)
        for (int q = 0; q < b.n_prim; ++q) {
            double gamma = a.alpha[p] + b.alpha[q];
            double mu = a.alpha[p] * b.alpha[q] / gamma;
            double S_pq = std::pow(units::pi / gamma, 1.5) * std::exp(-mu * R2);
            T += a.coeff[p] * b.coeff[q] * mu * (3.0 - 2.0 * mu * R2) * S_pq;
        }
    return T;
}

// ── Phase 4: Nuclear attraction integral (s|V|s) ──────────────────────

double nuclear_ss(const Shell& a, const Shell& b, const Atom* atoms, int natm) {
    // V = sum_C (-Z_C) * sum_pq c_p c_q * (2*pi/gamma) * exp(-mu*R^2) * F_0(gamma*|P-C|^2)
    // P = Gaussian product center = (alpha_a * A + alpha_b * B) / gamma
    // PERF: unchecked() candidate — 9 * natm iterations
    double V = 0;
    double R2_AB = dist2(a.center, b.center);

    for (int p = 0; p < a.n_prim; ++p)
        for (int q = 0; q < b.n_prim; ++q) {
            double alpha_a = a.alpha[p];
            double alpha_b = b.alpha[q];
            double gamma = alpha_a + alpha_b;
            double mu = alpha_a * alpha_b / gamma;

            // Gaussian product center P
            double P[3];
            for (int d = 0; d < 3; ++d)
                P[d] = (alpha_a * a.center[d] + alpha_b * b.center[d]) / gamma;

            double prefactor = a.coeff[p] * b.coeff[q]
                               * (2.0 * units::pi / gamma)
                               * std::exp(-mu * R2_AB);

            // Sum over all nuclei
            for (int c = 0; c < natm; ++c) {
                double PC2 = dist2(P, atoms[c].pos);
                V += prefactor * (-atoms[c].Z) * boys_f0(gamma * PC2);
            }
        }
    return V;
}

// ── Phase 5: Two-electron ERI (ss|ss) ─────────────────────────────────

double eri_ssss(const Shell& a, const Shell& b, const Shell& c,
                const Shell& d) {
    // (ab|cd) = sum_pqrs c_p c_q c_r c_s
    //   * 2*pi^{5/2} / (gamma1 * gamma2 * sqrt(gamma1 + gamma2))
    //   * exp(-mu1*R_AB^2 - mu2*R_CD^2)
    //   * F_0(rho * |PQ|^2)
    //
    // gamma1 = alpha_p + alpha_q, gamma2 = alpha_r + alpha_s
    // mu1 = alpha_p*alpha_q/gamma1, mu2 = alpha_r*alpha_s/gamma2
    // rho = gamma1 * gamma2 / (gamma1 + gamma2)
    // P = (alpha_p*A + alpha_q*B)/gamma1, Q = (alpha_r*C + alpha_s*D)/gamma2
    //
    // PERF: unchecked() candidate — 81 iterations for STO-3G
    double result = 0;
    double R2_AB = dist2(a.center, b.center);
    double R2_CD = dist2(c.center, d.center);

    for (int p = 0; p < a.n_prim; ++p)
        for (int q = 0; q < b.n_prim; ++q) {
            double gamma1 = a.alpha[p] + b.alpha[q];
            double mu1 = a.alpha[p] * b.alpha[q] / gamma1;
            double K_AB = std::exp(-mu1 * R2_AB);

            double P[3];
            for (int dd = 0; dd < 3; ++dd)
                P[dd] = (a.alpha[p] * a.center[dd] + b.alpha[q] * b.center[dd])
                        / gamma1;

            for (int r = 0; r < c.n_prim; ++r)
                for (int s = 0; s < d.n_prim; ++s) {
                    double gamma2 = c.alpha[r] + d.alpha[s];
                    double mu2 = c.alpha[r] * d.alpha[s] / gamma2;
                    double K_CD = std::exp(-mu2 * R2_CD);

                    double Q[3];
                    for (int dd = 0; dd < 3; ++dd)
                        Q[dd] = (c.alpha[r] * c.center[dd]
                                 + d.alpha[s] * d.center[dd])
                                / gamma2;

                    double rho = gamma1 * gamma2 / (gamma1 + gamma2);
                    double PQ2 = dist2(P, Q);

                    result += a.coeff[p] * b.coeff[q] * c.coeff[r] * d.coeff[s]
                              * 2.0 * std::pow(units::pi, 2.5)
                              / (gamma1 * gamma2 * std::sqrt(gamma1 + gamma2))
                              * K_AB * K_CD * boys_f0(rho * PQ2);
                }
        }
    return result;
}

// ═══════════════════════════════════════════════════════════════════════
// General integrals (s+p type, l_max=1) — Obara-Saika recurrence
// ═══════════════════════════════════════════════════════════════════════

// ── Angular momentum helpers ─────────────────────────────────────────

/// @brief Get the Cartesian `(l_x, l_y, l_z)` tuple for one AO in a shell.
///
/// Looks up the `sh.cart`-th tuple produced by `fill_cart_comps(sh.l, …)`.
/// Cheaper than filling the whole table when only one AO is needed.
static inline void cart_exponents(const Shell& sh, int a_out[3]) {
    int table[kMaxNCart][3];
    fill_cart_comps(sh.l, table);
    a_out[0] = table[sh.cart][0];
    a_out[1] = table[sh.cart][1];
    a_out[2] = table[sh.cart][2];
}

// ── 1D Obara-Saika overlap table ────────────────────────────────────

/// @brief 1D overlap integrals S_{i,j} for i in [0, imax], j in [0, jmax].
/// @details Uses the Obara-Saika recurrence:
///   S_{i+1,j} = (P-A) * S_{ij} + (1/(2*gamma)) * (i*S_{i-1,j} + j*S_{i,j-1})
///   S_{0,j+1} = (P-B) * S_{0,j} + (1/(2*gamma)) * j * S_{0,j-1}
///   S_{0,0}   = sqrt(pi/gamma) * exp(-mu * d^2)  [1D factor]
///
///   The table is stored as S[i*(jmax+1)+j].
///   Maximum needed: imax=1, jmax=3 for kinetic; imax=1, jmax=1 for overlap.
/// @param PA     P_d - A_d for this direction.
/// @param PB     P_d - B_d for this direction.
/// @param inv2g  0.5 / gamma.
/// @param S00    S_{0,0} = sqrt(pi/gamma) * exp(-mu * d^2).
/// @param imax   Maximum first index.
/// @param jmax   Maximum second index.
/// @param S      Output array of size (imax+1)*(jmax+1).
static void os_overlap_1d(double PA, double PB, double inv2g, double S00,
                          int imax, int jmax, double* S) {
    const int jstride = jmax + 1;

    // Accessor lambda (row = i, col = j)
    auto at = [&](int i, int j) -> double& { return S[i * jstride + j]; };

    // Base: S_{0,0}
    at(0, 0) = S00;

    // Build row i=0: S_{0, j+1} = PB * S_{0,j} + inv2g * j * S_{0, j-1}
    for (int j = 0; j < jmax; ++j) {
        double prev = (j > 0) ? at(0, j - 1) : 0.0;
        at(0, j + 1) = PB * at(0, j) + inv2g * j * prev;
    }

    // Build rows i=1..imax using:
    //   S_{i+1, j} = PA * S_{i,j} + inv2g * (i * S_{i-1,j} + j * S_{i,j-1})
    for (int i = 0; i < imax; ++i) {
        for (int j = 0; j <= jmax; ++j) {
            double s_im1_j = (i > 0) ? at(i - 1, j) : 0.0;
            double s_i_jm1 = (j > 0) ? at(i, j - 1) : 0.0;
            at(i + 1, j) = PA * at(i, j) + inv2g * (i * s_im1_j + j * s_i_jm1);
        }
    }
}

// ── Phase 6: General overlap integral ────────────────────────────────

double overlap(const Shell& a, const Shell& b) {
    int a_ang[3], b_ang[3];
    cart_exponents(a, a_ang);
    cart_exponents(b, b_ang);

    double result = 0.0;
    for (int p = 0; p < a.n_prim; ++p) {
        for (int q = 0; q < b.n_prim; ++q) {
            const double prim = overlap_prim_3d(a_ang, b_ang, a.alpha[p],
                                                a.center, b.alpha[q], b.center);
            result += a.coeff[p] * b.coeff[q] * prim;
        }
    }
    return result;
}

// ── Phase 7: General kinetic energy integral ─────────────────────────

/// @brief 1D kinetic integral T_1d(i,j) from 1D overlaps and beta.
/// @details Uses the direct second-derivative formula:
///   d^2/dx^2 [x^j exp(-b x^2)] =
///     j(j-1) x^{j-2} exp - 2b(2j+1) x^j exp + 4b^2 x^{j+2} exp
///
///   So T_1d(i,j) = -(1/2) * [j(j-1)*S(i,j-2) - 2b(2j+1)*S(i,j) + 4b^2*S(i,j+2)]
///
///   For j=0: T_1d(i,0) = b * S(i,0) - 2b^2 * S(i,2)
///   For j=1: T_1d(i,1) = 3b * S(i,1) - 2b^2 * S(i,3)
///
/// @param i    Angular momentum on first center (0 or 1).
/// @param j    Angular momentum on second center (0 or 1).
/// @param beta Exponent of second Gaussian (alpha_b).
/// @param S    1D overlap table (imax=1, jmax=3), stored as S[i*4+j].
/// @return 1D kinetic integral.
static inline double kinetic_1d(int i, int j, double beta, const double* S) {
    const int jstride = 4; // jmax+1 = 3+1

    // T_1d(i,j) = -(1/2) * [j(j-1)*S(i,j-2) - 2*beta*(2j+1)*S(i,j) + 4*beta^2*S(i,j+2)]
    double s_ij = S[i * jstride + j];
    double s_ijp2 = S[i * jstride + j + 2];
    double s_ijm2 = (j >= 2) ? S[i * jstride + j - 2] : 0.0;

    return -0.5
           * (j * (j - 1) * s_ijm2 - 2.0 * beta * (2 * j + 1) * s_ij
              + 4.0 * beta * beta * s_ijp2);
}

double kinetic(const Shell& a, const Shell& b) {
    int a_ang[3], b_ang[3];
    cart_exponents(a, a_ang);
    cart_exponents(b, b_ang);

    double result = 0.0;
    for (int p = 0; p < a.n_prim; ++p) {
        for (int q = 0; q < b.n_prim; ++q) {
            const double prim = kinetic_prim_3d(a_ang, b_ang, a.alpha[p],
                                                a.center, b.alpha[q], b.center);
            result += a.coeff[p] * b.coeff[q] * prim;
        }
    }
    return result;
}

// ── Phase 8: General nuclear attraction integral ─────────────────────

/// @brief Compute primitive nuclear attraction integral for one nucleus C
///   using the McMurchie-Davidson scheme with Hermite Coulomb integrals.
/// @details Expands the Cartesian overlap distribution in Hermite Gaussians:
///   V(a,b) = (2*pi/gamma) * K_AB * sum_{tuv} E_x[t] E_y[u] E_z[v] R^0_{tuv}
///
///   E coefficients: E^{i,j}_t from Obara-Saika Hermite expansion.
///   R integrals:    R^m_{t+1,u,v} = t*R^{m+1}_{t-1,u,v} + PC_x*R^{m+1}_{t,u,v}
///                   R^m_{0,0,0}   = (-2*gamma)^m * F_m(gamma*|PC|^2)
double nuclear(const Shell& a, const Shell& b, const Atom* atoms, int natm) {
    int a_ang[3], b_ang[3];
    cart_exponents(a, a_ang);
    cart_exponents(b, b_ang);

    double V_total = 0.0;

    for (int p = 0; p < a.n_prim; ++p) {
        for (int q = 0; q < b.n_prim; ++q) {
            const double aa = a.alpha[p];
            const double ab = b.alpha[q];
            const double gamma = aa + ab;
            const double mu = aa * ab / gamma;
            const double R2_AB = dist2(a.center, b.center);
            const double K_AB = std::exp(-mu * R2_AB);

            // Gaussian product center
            double P[3], PA[3], PB[3];
            for (int d = 0; d < 3; ++d) {
                P[d] = (aa * a.center[d] + ab * b.center[d]) / gamma;
                PA[d] = P[d] - a.center[d];
                PB[d] = P[d] - b.center[d];
            }

            // Sum over all nuclei
            for (int c = 0; c < natm; ++c) {
                double PC[3];
                for (int d = 0; d < 3; ++d) {
                    PC[d] = P[d] - atoms[c].pos[d];
                }

                double V_prim =
                    nuclear_prim(a_ang, b_ang, gamma, PA, PB, PC, K_AB);
                V_total += a.coeff[p] * b.coeff[q] * (-atoms[c].Z) * V_prim;
            }
        }
    }
    return V_total;
}

// ── Phase 9: General two-electron repulsion integral ─────────────────

/// @brief Compute a primitive ERI using McMurchie-Davidson approach.
/// @details For four primitive Gaussians with angular momenta a_ang, b_ang,
///   c_ang, d_ang (each component 0 or 1), computes the two-electron integral
///   (ab|cd) in chemist's notation.
///
///   Uses McMurchie-Davidson expansion with Hermite Coulomb integrals R_{tuv}.
///   For l_max=1, total angular momentum Ltot <= 4, requiring Boys functions
///   up to F_4.
///
/// @return Primitive ERI value (without contraction coefficients).
static double eri_prim_md(const int a_ang[3], const int b_ang[3],
                          const int c_ang[3], const int d_ang[3],
                          double alpha_a, const double A[3], double alpha_b,
                          const double B[3], double alpha_c, const double C[3],
                          double alpha_d, const double D[3]) {
    const double gamma1 = alpha_a + alpha_b;
    const double gamma2 = alpha_c + alpha_d;
    const double delta = gamma1 + gamma2;
    const double mu1 = alpha_a * alpha_b / gamma1;
    const double mu2 = alpha_c * alpha_d / gamma2;
    const double rho = gamma1 * gamma2 / delta;
    const double inv2g1 = 0.5 / gamma1;
    const double inv2g2 = 0.5 / gamma2;

    // Gaussian product centers
    double P[3], Q[3], PA[3], PB[3], QC[3], QD[3], PQ[3];
    double R2_AB = 0.0, R2_CD = 0.0, PQ2 = 0.0;
    for (int d = 0; d < 3; ++d) {
        P[d] = (alpha_a * A[d] + alpha_b * B[d]) / gamma1;
        Q[d] = (alpha_c * C[d] + alpha_d * D[d]) / gamma2;
        PA[d] = P[d] - A[d];
        PB[d] = P[d] - B[d];
        QC[d] = Q[d] - C[d];
        QD[d] = Q[d] - D[d];
        PQ[d] = P[d] - Q[d];
        double dAB = A[d] - B[d];
        double dCD = C[d] - D[d];
        R2_AB += dAB * dAB;
        R2_CD += dCD * dCD;
        PQ2 += PQ[d] * PQ[d];
    }

    const double K_AB = std::exp(-mu1 * R2_AB);
    const double K_CD = std::exp(-mu2 * R2_CD);
    const double T = rho * PQ2;

    // Prefactor: 2*pi^{5/2} / (gamma1 * gamma2 * sqrt(delta))
    const double prefactor = 2.0 * std::pow(units::pi, 2.5)
                             / (gamma1 * gamma2 * std::sqrt(delta)) * K_AB
                             * K_CD;

    // Angular momentum totals per direction
    const int Lx1 = a_ang[0] + b_ang[0]; // bra x angular momentum
    const int Ly1 = a_ang[1] + b_ang[1];
    const int Lz1 = a_ang[2] + b_ang[2];
    const int Lx2 = c_ang[0] + d_ang[0]; // ket x angular momentum
    const int Ly2 = c_ang[1] + d_ang[1];
    const int Lz2 = c_ang[2] + d_ang[2];
    const int Ltot = Lx1 + Ly1 + Lz1 + Lx2 + Ly2 + Lz2;

    // Boys functions F_m(T) for m = 0..Ltot (max 4)
    double Fm[5]; // max Ltot = 4
    for (int m = 0; m <= Ltot; ++m) {
        Fm[m] = boys_fn(m, T);
    }

    // E coefficients for bra pair (a,b) in each direction
    // E1[d][t] for t = 0..Lx1 (or Ly1, Lz1)
    double E1[3][3]; // E^{a_d, b_d}_t for bra, max t = 2
    for (int d = 0; d < 3; ++d) {
        E1[d][0] = E1[d][1] = E1[d][2] = 0.0;
        const int ai = a_ang[d];
        const int bj = b_ang[d];

        if (ai == 0 && bj == 0) {
            E1[d][0] = 1.0;
        } else if (ai == 1 && bj == 0) {
            E1[d][0] = PA[d];
            E1[d][1] = inv2g1;
        } else if (ai == 0 && bj == 1) {
            E1[d][0] = PB[d];
            E1[d][1] = inv2g1;
        } else { // ai == 1, bj == 1
            E1[d][0] = PA[d] * PB[d] + inv2g1;
            E1[d][1] = inv2g1 * (PA[d] + PB[d]);
            E1[d][2] = inv2g1 * inv2g1;
        }
    }

    // E coefficients for ket pair (c,d)
    double E2[3][3]; // E^{c_d, d_d}_t for ket, max t = 2
    for (int d = 0; d < 3; ++d) {
        E2[d][0] = E2[d][1] = E2[d][2] = 0.0;
        const int ci = c_ang[d];
        const int dj = d_ang[d];

        if (ci == 0 && dj == 0) {
            E2[d][0] = 1.0;
        } else if (ci == 1 && dj == 0) {
            E2[d][0] = QC[d];
            E2[d][1] = inv2g2;
        } else if (ci == 0 && dj == 1) {
            E2[d][0] = QD[d];
            E2[d][1] = inv2g2;
        } else { // ci == 1, dj == 1
            E2[d][0] = QC[d] * QD[d] + inv2g2;
            E2[d][1] = inv2g2 * (QC[d] + QD[d]);
            E2[d][2] = inv2g2 * inv2g2;
        }
    }

    // Hermite Coulomb integrals R^0_{t,u,v}.
    // ERI formula: (ab|cd) = prefactor * sum_{t,u,v} sum_{tau,nu,phi}
    //   E1[t] E1[u] E1[v] * E2[tau] E2[nu] E2[phi] *
    //   (-1)^{tau+nu+phi} * R^0_{t+tau, u+nu, v+phi}
    //
    // R recursion: R^m_{t+1,u,v} = t*R^{m+1}_{t-1,u,v} + PQ_x*R^{m+1}_{t,u,v}
    // Starting:    R^m_{0,0,0} = (-2*rho)^m * F_m(T)

    const int Rx_max = Lx1 + Lx2;
    const int Ry_max = Ly1 + Ly2;
    const int Rz_max = Lz1 + Lz2;

    // R table: R_tab[m][t][u][v]. Build from m=Ltot down to m=0.
    // Valid entries satisfy m + t + u + v <= Ltot.
    // For l_max=1: Ltot <= 4, so 5^4 = 625 doubles on stack.
    double R_tab[5][5][5][5]; // [m][t][u][v]
    // Zero-initialize
    for (int m = 0; m < 5; ++m)
        for (int t = 0; t < 5; ++t)
            for (int u = 0; u < 5; ++u)
                for (int v = 0; v < 5; ++v)
                    R_tab[m][t][u][v] = 0.0;

    // Initialize: R^m_{0,0,0} for m = 0..Ltot
    double neg2rho = -2.0 * rho;
    double neg2rho_pow = 1.0;
    for (int m = 0; m <= Ltot; ++m) {
        R_tab[m][0][0][0] = neg2rho_pow * Fm[m];
        neg2rho_pow *= neg2rho;
    }

    // Build R^m for m = Ltot-1 down to 0.
    // At each m, compute all R^m_{t,u,v} with t+u+v <= Ltot-m.
    // Three passes per level: x-pass (u=v=0), y-pass (v=0), z-pass.
    for (int m = Ltot - 1; m >= 0; --m) {
        const int max_tuv = Ltot - m;

        // x-pass: R^m_{t,0,0} = (t-1)*R^{m+1}_{t-2,0,0} + PQ_x*R^{m+1}_{t-1,0,0}
        for (int t = 1; t <= max_tuv && t <= Rx_max; ++t) {
            R_tab[m][t][0][0] = PQ[0] * R_tab[m + 1][t - 1][0][0];
            if (t >= 2) {
                R_tab[m][t][0][0] += (t - 1) * R_tab[m + 1][t - 2][0][0];
            }
        }

        // y-pass: R^m_{t,u,0} for u >= 1
        for (int t = 0; t <= Rx_max && t <= max_tuv; ++t) {
            for (int u = 1; u <= Ry_max && t + u <= max_tuv; ++u) {
                R_tab[m][t][u][0] = PQ[1] * R_tab[m + 1][t][u - 1][0];
                if (u >= 2) {
                    R_tab[m][t][u][0] += (u - 1) * R_tab[m + 1][t][u - 2][0];
                }
            }
        }

        // z-pass: build R^m_{t, u, v} for v >= 1
        for (int t = 0; t <= Rx_max && t <= max_tuv; ++t) {
            for (int u = 0; u <= Ry_max && t + u <= max_tuv; ++u) {
                for (int v = 1; v <= Rz_max && t + u + v <= max_tuv; ++v) {
                    R_tab[m][t][u][v] = PQ[2] * R_tab[m + 1][t][u][v - 1];
                    if (v >= 2) {
                        R_tab[m][t][u][v] +=
                            (v - 1) * R_tab[m + 1][t][u][v - 2];
                    }
                }
            }
        }
    }

    // Now compute the ERI:
    // (ab|cd) = prefactor * sum_{t,u,v} sum_{tau,nu,phi}
    //           E1_x[t] * E1_y[u] * E1_z[v] *
    //           E2_x[tau] * E2_y[nu] * E2_z[phi] *
    //           (-1)^{tau+nu+phi} * R^0_{t+tau, u+nu, v+phi}

    double eri_val = 0.0;
    for (int t = 0; t <= Lx1; ++t) {
        for (int u = 0; u <= Ly1; ++u) {
            for (int v = 0; v <= Lz1; ++v) {
                const double e1_tuv = E1[0][t] * E1[1][u] * E1[2][v];
                if (e1_tuv == 0.0)
                    continue;
                for (int tau = 0; tau <= Lx2; ++tau) {
                    for (int nu = 0; nu <= Ly2; ++nu) {
                        for (int phi = 0; phi <= Lz2; ++phi) {
                            const double e2_tnp =
                                E2[0][tau] * E2[1][nu] * E2[2][phi];
                            if (e2_tnp == 0.0)
                                continue;
                            int sign = ((tau + nu + phi) & 1) ? -1 : 1;
                            eri_val += e1_tuv * e2_tnp * sign
                                       * R_tab[0][t + tau][u + nu][v + phi];
                        }
                    }
                }
            }
        }
    }

    return prefactor * eri_val;
}

double eri(const Shell& a, const Shell& b, const Shell& c, const Shell& d) {
    int a_ang[3], b_ang[3], c_ang[3], d_ang[3];
    cart_exponents(a, a_ang);
    cart_exponents(b, b_ang);
    cart_exponents(c, c_ang);
    cart_exponents(d, d_ang);

    // `eri_prim_md` has hand-unrolled E-coefficient tables for l ≤ 1
    // (Lx1/Lx2 ≤ 2, Ltot ≤ 4). For any d-shell in the quartet (Ltot up
    // to 8) the unrolled path would write past its E / F_m buffers; fall
    // through to the general-l `eri_prim` instead, which uses the same
    // `md_e_coeffs` recursion the derivative path already relies on.
    const int Lmax = std::max({a.l, b.l, c.l, d.l});
    auto prim = (Lmax <= 1) ? eri_prim_md : eri_prim;

    double result = 0.0;
    for (int p = 0; p < a.n_prim; ++p) {
        for (int q = 0; q < b.n_prim; ++q) {
            for (int r = 0; r < c.n_prim; ++r) {
                for (int s = 0; s < d.n_prim; ++s) {
                    result += a.coeff[p] * b.coeff[q] * c.coeff[r] * d.coeff[s]
                              * prim(a_ang, b_ang, c_ang, d_ang, a.alpha[p],
                                     a.center, b.alpha[q], b.center, c.alpha[r],
                                     c.center, d.alpha[s], d.center);
                }
            }
        }
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════════════
// General primitive functions (l up to 2 per component for derivatives)
// ═══════════════════════════════════════════════════════════════════════

double overlap_prim_3d(const int a_ang[3], const int b_ang[3], double alpha_a,
                       const double A[3], double alpha_b, const double B[3]) {
    const double gamma = alpha_a + alpha_b;
    const double mu = alpha_a * alpha_b / gamma;
    const double inv2g = 0.5 / gamma;
    const double sqrt_pi_g = std::sqrt(units::pi / gamma);

    double result = 1.0;
    for (int d = 0; d < 3; ++d) {
        const double dd = A[d] - B[d];
        const double Pd = (alpha_a * A[d] + alpha_b * B[d]) / gamma;
        const double PA = Pd - A[d];
        const double PB = Pd - B[d];
        const double S00 = sqrt_pi_g * std::exp(-mu * dd * dd);

        // Build 1D overlap table up to (imax, jmax). Sized for
        // derivative callers that pass a_ang or b_ang bumped by 1:
        // max entry is (kMaxLOrbital+1)*(kMaxLOrbital+1) = 4*4 = 16 for kMaxLOrbital=2.
        const int imax = a_ang[d];
        const int jmax = b_ang[d];
        constexpr int kMaxOverlapCells =
            (kMaxLOrbital + 2) * (kMaxLOrbital + 2);
        double S[kMaxOverlapCells];
        os_overlap_1d(PA, PB, inv2g, S00, imax, jmax, S);

        result *= S[imax * (jmax + 1) + jmax];
    }
    return result;
}

double kinetic_prim_3d(const int a_ang[3], const int b_ang[3], double alpha_a,
                       const double A[3], double alpha_b, const double B[3]) {
    const double gamma = alpha_a + alpha_b;
    const double mu = alpha_a * alpha_b / gamma;
    const double inv2g = 0.5 / gamma;
    const double sqrt_pi_g = std::sqrt(units::pi / gamma);

    // Determine uniform table size across all directions.
    // For kinetic, need S(i, j+2) in each direction.
    // imax = max(a_ang[d]) across all d, jmax = max(b_ang[d]) + 2.
    int imax_all = a_ang[0];
    int jmax_base = b_ang[0];
    for (int d = 1; d < 3; ++d) {
        if (a_ang[d] > imax_all)
            imax_all = a_ang[d];
        if (b_ang[d] > jmax_base)
            jmax_base = b_ang[d];
    }
    const int im = imax_all; // max kMaxLOrbital+2 = 5 (2nd-derivative bump)
    const int jm = jmax_base + 2; // max kMaxLOrbital+2 = 5
    const int jstride = jm + 1;   // max 6

    // Build 1D overlap tables in each direction with uniform size.
    // Sized for the SECOND-derivative path (hf-capability-gaps-13-cphf),
    // which raises the bra angular momentum by up to +2 on one axis:
    // im ≤ kMaxLOrbital+2 and jm ≤ kMaxLOrbital+2, so the table needs
    // (kMaxLOrbital+3)² = 36 cells per direction for f-shells.
    constexpr int kMaxKineticCells = (kMaxLOrbital + 3) * (kMaxLOrbital + 3);
    double Sd[3][kMaxKineticCells];
    double S1d[3];

    for (int d = 0; d < 3; ++d) {
        const double dd = A[d] - B[d];
        const double Pd = (alpha_a * A[d] + alpha_b * B[d]) / gamma;
        const double PA = Pd - A[d];
        const double PB = Pd - B[d];
        const double S00 = sqrt_pi_g * std::exp(-mu * dd * dd);

        os_overlap_1d(PA, PB, inv2g, S00, im, jm, Sd[d]);
        S1d[d] = Sd[d][a_ang[d] * jstride + b_ang[d]];
    }

    // T_3D = sum_d { T_1d(a_d, b_d) * product_{e!=d} S_1d(a_e, b_e) }
    double prim_kin = 0.0;
    for (int d = 0; d < 3; ++d) {
        const int i = a_ang[d];
        const int j = b_ang[d];
        const double beta = alpha_b;

        double s_ij = Sd[d][i * jstride + j];
        double s_ijp2 = Sd[d][i * jstride + j + 2];
        double s_ijm2 = (j >= 2) ? Sd[d][i * jstride + j - 2] : 0.0;

        double T1d_val =
            -0.5
            * (j * (j - 1) * s_ijm2 - 2.0 * beta * (2 * j + 1) * s_ij
               + 4.0 * beta * beta * s_ijp2);

        double other = 1.0;
        for (int e = 0; e < 3; ++e) {
            if (e != d)
                other *= S1d[e];
        }
        prim_kin += T1d_val * other;
    }
    return prim_kin;
}

// ── MD-recurrence buffer sizing ──────────────────────────────────────
//
// Nuclear- and ERI-integral paths reuse the same McMurchie-Davidson
// Hermite coefficients E^{i,j}_t. Gradient paths raise one Gaussian
// center's angular momentum by +1 on the fly (ai+1 or bj+1), so every
// table must be sized for ai or bj reaching kMaxLOrbital+1 — not just kMaxLOrbital.

/// Per-center i/j upper bound after a +1 derivative raise.
inline constexpr int kDerivL = kMaxLOrbital + 1;
/// t-index dim in E^{i,j}_t (t ∈ [0, ai+bj]; +1 slack lets the recurrence
/// read E[t+1] without a bounds branch when t hits the real max). The
/// second-derivative path (hf-capability-gaps-13-cphf) raises AT MOST +2
/// in total across the two centers of one integral, so t stays ≤
/// kMaxLOrbital + (kMaxLOrbital + 2) = 2·kDerivL — within this dim.
inline constexpr int kMDEDim = 2 * kDerivL + 1; // 9 for f-shells
/// i-index dim in the E_buf[i][t] intermediate. Sized for the
/// second-derivative +2 raise on ONE center (ai ≤ kMaxLOrbital + 2).
inline constexpr int kMDEIDim = kMaxLOrbital + 3; // 6 for f-shells

/// @brief McMurchie-Davidson E coefficients E^{ai,bj}_t.
/// @details Supports ai, bj up to kDerivL = kMaxLOrbital + 1 to accommodate the
///   gradient path which raises one center's angular by +1. Output is
///   populated for t = 0..ai+bj; entries past that are zero.
static void md_e_coeffs(int ai, int bj, double PA, double PB, double inv2g,
                        double E[kMDEDim]) {
    for (int t = 0; t < kMDEDim; ++t)
        E[t] = 0.0;

    // Step 1: build E^{i,0}_t for i = 0..ai via the upward recurrence
    //   E^{i+1,0}_t = inv2g * E^{i,0}_{t-1} + PA * E^{i,0}_t + (t+1)*E^{i,0}_{t+1}
    double E_buf[kMDEIDim][kMDEDim] = {};
    E_buf[0][0] = 1.0;

    for (int i = 0; i < ai; ++i) {
        const int tmax = i + 1; // max t reachable at level i+1
        for (int t = 0; t <= tmax; ++t) {
            const double etm1 = (t > 0) ? E_buf[i][t - 1] : 0.0;
            const double et = E_buf[i][t];
            const double etp1 = (t + 1 < kMDEDim) ? E_buf[i][t + 1] : 0.0;
            E_buf[i + 1][t] = inv2g * etm1 + PA * et + (t + 1) * etp1;
        }
    }

    // Step 2: increment j from 0 to bj via
    //   E^{i,j+1}_t = inv2g * E^{i,j}_{t-1} + PB * E^{i,j}_t + (t+1)*E^{i,j}_{t+1}
    double W[2][kMDEDim] = {};
    int cur = 0;
    for (int t = 0; t < kMDEDim; ++t)
        W[cur][t] = E_buf[ai][t];

    for (int jj = 0; jj < bj; ++jj) {
        const int nxt = 1 - cur;
        for (int t = 0; t < kMDEDim; ++t)
            W[nxt][t] = 0.0;
        const int tmax = ai + jj + 1;
        for (int t = 0; t <= tmax && t < kMDEDim; ++t) {
            const double etm1 = (t > 0) ? W[cur][t - 1] : 0.0;
            const double et = W[cur][t];
            const double etp1 = (t + 1 < kMDEDim) ? W[cur][t + 1] : 0.0;
            W[nxt][t] = inv2g * etm1 + PB * et + (t + 1) * etp1;
        }
        cur = nxt;
    }

    for (int t = 0; t <= ai + bj && t < kMDEDim; ++t)
        E[t] = W[cur][t];
}

double nuclear_prim(const int a_ang[3], const int b_ang[3], double gamma,
                    const double PA[3], const double PB[3], const double PC[3],
                    double K_AB) {
    const double inv2g = 0.5 / gamma;
    const double prefactor = 2.0 * units::pi / gamma * K_AB;
    const double PC2 = PC[0] * PC[0] + PC[1] * PC[1] + PC[2] * PC[2];
    const double T = gamma * PC2;

    // Angular momentum totals per direction. Derivative raising bumps
    // one component by +1, so sum(a_ang) ≤ kDerivL and sum(b_ang) ≤ kDerivL.
    // Per-axis Lx = a+b ≤ 2*kDerivL, Ltot ≤ 2*kDerivL (only one side raised
    // at a time in 1st derivatives, but size conservatively for safety).
    const int Lx = a_ang[0] + b_ang[0];
    const int Ly = a_ang[1] + b_ang[1];
    const int Lz = a_ang[2] + b_ang[2];
    const int Ltot = Lx + Ly + Lz;

    constexpr int kRDim = 2 * kDerivL + 1; // 7 for d-shells; m and per-axis dim

    // Boys functions F_m for m = 0..Ltot
    double Fm[kRDim];
    for (int m = 0; m <= Ltot; ++m) {
        Fm[m] = boys_fn(m, T);
    }

    // E coefficients for each direction
    double E[3][kMDEDim];
    for (int d = 0; d < 3; ++d) {
        md_e_coeffs(a_ang[d], b_ang[d], PA[d], PB[d], inv2g, E[d]);
    }

    // R^m_{0,0,0} = (-2*gamma)^m * F_m(T)
    double R000[kRDim];
    double neg2g = -2.0 * gamma;
    double neg2g_pow = 1.0;
    for (int m = 0; m <= Ltot; ++m) {
        R000[m] = neg2g_pow * Fm[m];
        neg2g_pow *= neg2g;
    }

    // Build R^0_{t,u,v} by recurrence. Table up to (Lx, Ly, Lz).
    //   R^m_{t+1,u,v} = t*R^{m+1}_{t-1,u,v} + PC_x*R^{m+1}_{t,u,v}
    double R_tab[kRDim][kRDim][kRDim][kRDim] = {};

    // Initialize: R^m_{0,0,0}
    for (int m = 0; m <= Ltot; ++m)
        R_tab[m][0][0][0] = R000[m];

    // Build R^m for m = Ltot-1 down to 0.
    for (int m = Ltot - 1; m >= 0; --m) {
        const int max_tuv = Ltot - m;

        // x-pass
        for (int t = 1; t <= Lx && t <= max_tuv; ++t) {
            R_tab[m][t][0][0] = PC[0] * R_tab[m + 1][t - 1][0][0];
            if (t >= 2)
                R_tab[m][t][0][0] += (t - 1) * R_tab[m + 1][t - 2][0][0];
        }

        // y-pass
        for (int t = 0; t <= Lx && t <= max_tuv; ++t) {
            for (int u = 1; u <= Ly && t + u <= max_tuv; ++u) {
                R_tab[m][t][u][0] = PC[1] * R_tab[m + 1][t][u - 1][0];
                if (u >= 2)
                    R_tab[m][t][u][0] += (u - 1) * R_tab[m + 1][t][u - 2][0];
            }
        }

        // z-pass
        for (int t = 0; t <= Lx && t <= max_tuv; ++t) {
            for (int u = 0; u <= Ly && t + u <= max_tuv; ++u) {
                for (int v = 1; v <= Lz && t + u + v <= max_tuv; ++v) {
                    R_tab[m][t][u][v] = PC[2] * R_tab[m + 1][t][u][v - 1];
                    if (v >= 2)
                        R_tab[m][t][u][v] +=
                            (v - 1) * R_tab[m + 1][t][u][v - 2];
                }
            }
        }
    }

    // Compute V = prefactor * sum_{t,u,v} E_x[t] * E_y[u] * E_z[v] * R^0[t][u][v]
    double V = 0.0;
    for (int t = 0; t <= Lx; ++t) {
        for (int u = 0; u <= Ly; ++u) {
            for (int v = 0; v <= Lz; ++v) {
                V += E[0][t] * E[1][u] * E[2][v] * R_tab[0][t][u][v];
            }
        }
    }

    return prefactor * V;
}

double eri_prim(const int a_ang[3], const int b_ang[3], const int c_ang[3],
                const int d_ang[3], double alpha_a, const double A[3],
                double alpha_b, const double B[3], double alpha_c,
                const double C[3], double alpha_d, const double D[3]) {
    const double gamma1 = alpha_a + alpha_b;
    const double gamma2 = alpha_c + alpha_d;
    const double delta = gamma1 + gamma2;
    const double mu1 = alpha_a * alpha_b / gamma1;
    const double mu2 = alpha_c * alpha_d / gamma2;
    const double rho = gamma1 * gamma2 / delta;
    const double inv2g1 = 0.5 / gamma1;
    const double inv2g2 = 0.5 / gamma2;

    // Gaussian product centers
    double P[3], Q[3], PA[3], PB[3], QC[3], QD[3], PQ[3];
    double R2_AB = 0.0, R2_CD = 0.0, PQ2 = 0.0;
    for (int d = 0; d < 3; ++d) {
        P[d] = (alpha_a * A[d] + alpha_b * B[d]) / gamma1;
        Q[d] = (alpha_c * C[d] + alpha_d * D[d]) / gamma2;
        PA[d] = P[d] - A[d];
        PB[d] = P[d] - B[d];
        QC[d] = Q[d] - C[d];
        QD[d] = Q[d] - D[d];
        PQ[d] = P[d] - Q[d];
        double dAB = A[d] - B[d];
        double dCD = C[d] - D[d];
        R2_AB += dAB * dAB;
        R2_CD += dCD * dCD;
        PQ2 += PQ[d] * PQ[d];
    }

    const double K_AB = std::exp(-mu1 * R2_AB);
    const double K_CD = std::exp(-mu2 * R2_CD);
    const double T = rho * PQ2;

    // Prefactor: 2*pi^{5/2} / (gamma1 * gamma2 * sqrt(delta))
    const double prefactor = 2.0 * std::pow(units::pi, 2.5)
                             / (gamma1 * gamma2 * std::sqrt(delta)) * K_AB
                             * K_CD;

    // Angular momentum totals per direction. Derivative path raises one
    // of a/b/c/d on one axis by +1, so per-pair Lxi may reach kDerivL+kMaxLOrbital
    // and the grand Ltot may reach 4*kMaxLOrbital+1 (all angular on one axis).
    const int Lx1 = a_ang[0] + b_ang[0];
    const int Ly1 = a_ang[1] + b_ang[1];
    const int Lz1 = a_ang[2] + b_ang[2];
    const int Lx2 = c_ang[0] + d_ang[0];
    const int Ly2 = c_ang[1] + d_ang[1];
    const int Lz2 = c_ang[2] + d_ang[2];
    const int Ltot = Lx1 + Ly1 + Lz1 + Lx2 + Ly2 + Lz2;

    // R-table sizing: m ∈ [0, Ltot] and per-axis index ∈ [0, Lxi+Lx2_i].
    // Max = 4*kMaxLOrbital+1 for 1st derivatives of a (dd|dd) quartet.
    constexpr int kERIRDim = 4 * kMaxLOrbital + 2; // 10 for d-shells

    // Boys functions F_m(T) for m = 0..Ltot
    double Fm[kERIRDim];
    for (int m = 0; m <= Ltot; ++m) {
        Fm[m] = boys_fn(m, T);
    }

    // E coefficients for bra pair (a,b)
    double E1[3][kMDEDim];
    for (int d = 0; d < 3; ++d) {
        md_e_coeffs(a_ang[d], b_ang[d], PA[d], PB[d], inv2g1, E1[d]);
    }

    // E coefficients for ket pair (c,d)
    double E2[3][kMDEDim];
    for (int d = 0; d < 3; ++d) {
        md_e_coeffs(c_ang[d], d_ang[d], QC[d], QD[d], inv2g2, E2[d]);
    }

    // R table — max Ltot = 4*kMaxLOrbital+1, max per-dir = 4*kMaxLOrbital+1 (all ang on
    // one axis in the derivative limit). Stack footprint: 10^4 doubles
    // ≈ 80 KB per call; only used in gradient paths, not hot SCF loop.
    const int Rx_max = Lx1 + Lx2;
    const int Ry_max = Ly1 + Ly2;
    const int Rz_max = Lz1 + Lz2;
    double R_tab[kERIRDim][kERIRDim][kERIRDim][kERIRDim] = {};

    // Initialize: R^m_{0,0,0}
    double neg2rho = -2.0 * rho;
    double neg2rho_pow = 1.0;
    for (int m = 0; m <= Ltot; ++m) {
        R_tab[m][0][0][0] = neg2rho_pow * Fm[m];
        neg2rho_pow *= neg2rho;
    }

    // Build R^m for m = Ltot-1 down to 0
    for (int m = Ltot - 1; m >= 0; --m) {
        const int max_tuv = Ltot - m;

        for (int t = 1; t <= Rx_max && t <= max_tuv; ++t) {
            R_tab[m][t][0][0] = PQ[0] * R_tab[m + 1][t - 1][0][0];
            if (t >= 2)
                R_tab[m][t][0][0] += (t - 1) * R_tab[m + 1][t - 2][0][0];
        }

        for (int t = 0; t <= Rx_max && t <= max_tuv; ++t) {
            for (int u = 1; u <= Ry_max && t + u <= max_tuv; ++u) {
                R_tab[m][t][u][0] = PQ[1] * R_tab[m + 1][t][u - 1][0];
                if (u >= 2)
                    R_tab[m][t][u][0] += (u - 1) * R_tab[m + 1][t][u - 2][0];
            }
        }

        for (int t = 0; t <= Rx_max && t <= max_tuv; ++t) {
            for (int u = 0; u <= Ry_max && t + u <= max_tuv; ++u) {
                for (int v = 1; v <= Rz_max && t + u + v <= max_tuv; ++v) {
                    R_tab[m][t][u][v] = PQ[2] * R_tab[m + 1][t][u][v - 1];
                    if (v >= 2)
                        R_tab[m][t][u][v] +=
                            (v - 1) * R_tab[m + 1][t][u][v - 2];
                }
            }
        }
    }

    // Compute ERI
    double eri_val = 0.0;
    for (int t = 0; t <= Lx1; ++t) {
        for (int u = 0; u <= Ly1; ++u) {
            for (int v = 0; v <= Lz1; ++v) {
                const double e1_tuv = E1[0][t] * E1[1][u] * E1[2][v];
                if (e1_tuv == 0.0)
                    continue;
                for (int tau = 0; tau <= Lx2; ++tau) {
                    for (int nu = 0; nu <= Ly2; ++nu) {
                        for (int phi = 0; phi <= Lz2; ++phi) {
                            const double e2_tnp =
                                E2[0][tau] * E2[1][nu] * E2[2][phi];
                            if (e2_tnp == 0.0)
                                continue;
                            int sign = ((tau + nu + phi) & 1) ? -1 : 1;
                            eri_val += e1_tuv * e2_tnp * sign
                                       * R_tab[0][t + tau][u + nu][v + phi];
                        }
                    }
                }
            }
        }
    }

    return prefactor * eri_val;
}

// ═══════════════════════════════════════════════════════════════════════
// Shell-block ERI — shell-type dispatch (libcint/PySCF strategy)
//
// Public entry: eri_block() dispatches by (A.l<<3)|(B.l<<2)|(C.l<<1)|D.l.
// Each specialization has compile-time Ltot → the Boys recurrence and
// R-table loops are fixed-bound and compiler-unrollable.
// eri_blk_ss_ss uses xsimd to vectorize the ket primitive loop.
// ═══════════════════════════════════════════════════════════════════════

// Forward declaration — defined later in this translation unit.
static inline void boys_fn_vec(double T, int n_max, double Fm[]);

// ── Shell-block one-electron integrals ───────────────────────────────
//
// Each block function amortizes the per-primitive-pair setup (Gaussian
// product center, 1D tables, Hermite coefficients, R-table ...) across
// all A.n_cart × B.n_cart angular combinations in the block. For d-d
// that factor is 36; for p-p it's 9. The output layout is
//   out[ia * B.n_cart + ib] += <A_ia|op|B_ib>
// Caller must pre-zero the output buffer.

void overlap_block(const ShellBlock& A, const ShellBlock& B, double* out) {
    int a_ang[kMaxNCart][3], b_ang[kMaxNCart][3];
    fill_cart_comps(A.l, a_ang);
    fill_cart_comps(B.l, b_ang);

    const int imax = A.l;
    const int jmax = B.l;
    const int jstride = jmax + 1;

    for (int p = 0; p < A.n_prim; ++p) {
        for (int q = 0; q < B.n_prim; ++q) {
            const double aa = A.alpha[p];
            const double ab = B.alpha[q];
            const double gamma = aa + ab;
            const double mu = aa * ab / gamma;
            const double inv2g = 0.5 / gamma;
            const double sqrt_pi_g = std::sqrt(units::pi / gamma);
            const double cpq = A.coeff[p] * B.coeff[q];

            // Build 1D overlap tables once per primitive pair, per axis.
            // Sized for the largest shell pair: (imax+1)(jmax+1) with
            // imax, jmax <= kMaxLOrbital — 16 for f-shells.
            double S1d[3][(kMaxLOrbital + 1) * (kMaxLOrbital + 1)];
            for (int d = 0; d < 3; ++d) {
                const double dd = A.center[d] - B.center[d];
                const double Pd = (aa * A.center[d] + ab * B.center[d]) / gamma;
                const double PA = Pd - A.center[d];
                const double PB = Pd - B.center[d];
                const double S00 = sqrt_pi_g * std::exp(-mu * dd * dd);
                os_overlap_1d(PA, PB, inv2g, S00, imax, jmax, S1d[d]);
            }

            // Scatter across angular combinations — shared 1D tables.
            for (int ia = 0; ia < A.n_cart; ++ia) {
                const int ax = a_ang[ia][0], ay = a_ang[ia][1],
                          az = a_ang[ia][2];
                for (int ib = 0; ib < B.n_cart; ++ib) {
                    const int bx = b_ang[ib][0], by = b_ang[ib][1],
                              bz = b_ang[ib][2];
                    const double v = S1d[0][ax * jstride + bx]
                                     * S1d[1][ay * jstride + by]
                                     * S1d[2][az * jstride + bz];
                    out[ia * B.n_cart + ib] += cpq * v;
                }
            }
        }
    }
}

void kinetic_block(const ShellBlock& A, const ShellBlock& B, double* out) {
    int a_ang[kMaxNCart][3], b_ang[kMaxNCart][3];
    fill_cart_comps(A.l, a_ang);
    fill_cart_comps(B.l, b_ang);

    const int imax = A.l;
    const int jmax_k = B.l + 2;     // kinetic needs S(i, j+2)
    const int jstride = jmax_k + 1; // table column stride

    for (int p = 0; p < A.n_prim; ++p) {
        for (int q = 0; q < B.n_prim; ++q) {
            const double aa = A.alpha[p];
            const double ab = B.alpha[q];
            const double gamma = aa + ab;
            const double mu = aa * ab / gamma;
            const double inv2g = 0.5 / gamma;
            const double sqrt_pi_g = std::sqrt(units::pi / gamma);
            const double cpq = A.coeff[p] * B.coeff[q];

            // 1D overlap tables with the (jmax+2) extension needed for T.
            // (imax+1)(jmax_k+1) with imax <= kMaxLOrbital, jmax_k = B.l+2 <=
            // kMaxLOrbital+2 — 4*6 = 24 for f-shells.
            double S1d[3][(kMaxLOrbital + 1) * (kMaxLOrbital + 2 + 1)];
            for (int d = 0; d < 3; ++d) {
                const double dd = A.center[d] - B.center[d];
                const double Pd = (aa * A.center[d] + ab * B.center[d]) / gamma;
                const double PA = Pd - A.center[d];
                const double PB = Pd - B.center[d];
                const double S00 = sqrt_pi_g * std::exp(-mu * dd * dd);
                os_overlap_1d(PA, PB, inv2g, S00, imax, jmax_k, S1d[d]);
            }

            for (int ia = 0; ia < A.n_cart; ++ia) {
                const int* aa_ang = a_ang[ia];
                for (int ib = 0; ib < B.n_cart; ++ib) {
                    const int* bb_ang = b_ang[ib];
                    // T_3D = Σ_d T_1d(a_d, b_d) Π_{e≠d} S_1d(a_e, b_e)
                    double s_prod[3];
                    for (int d = 0; d < 3; ++d)
                        s_prod[d] = S1d[d][aa_ang[d] * jstride + bb_ang[d]];

                    double t_sum = 0.0;
                    for (int d = 0; d < 3; ++d) {
                        const int i = aa_ang[d];
                        const int j = bb_ang[d];
                        // T_1d(i,j) = -(1/2) [j(j-1) S(i,j-2) - 2β(2j+1) S(i,j) + 4β² S(i,j+2)]
                        const double s_ij = S1d[d][i * jstride + j];
                        const double s_ijp2 = S1d[d][i * jstride + j + 2];
                        const double s_ijm2 =
                            (j >= 2) ? S1d[d][i * jstride + j - 2] : 0.0;
                        const double t1d = -0.5
                                           * (j * (j - 1) * s_ijm2
                                              - 2.0 * ab * (2 * j + 1) * s_ij
                                              + 4.0 * ab * ab * s_ijp2);
                        double other = 1.0;
                        for (int e = 0; e < 3; ++e)
                            if (e != d)
                                other *= s_prod[e];
                        t_sum += t1d * other;
                    }
                    out[ia * B.n_cart + ib] += cpq * t_sum;
                }
            }
        }
    }
}

void nuclear_block(const ShellBlock& A, const ShellBlock& B, const Atom* atoms,
                   int natm, double* out) {
    int a_ang[kMaxNCart][3], b_ang[kMaxNCart][3];
    fill_cart_comps(A.l, a_ang);
    fill_cart_comps(B.l, b_ang);

    const int imax = A.l;
    const int jmax = B.l;
    const int Lmax_dir = imax + jmax; // max Hermite index per axis

    for (int p = 0; p < A.n_prim; ++p) {
        for (int q = 0; q < B.n_prim; ++q) {
            const double aa = A.alpha[p];
            const double ab = B.alpha[q];
            const double gamma = aa + ab;
            const double mu = aa * ab / gamma;
            const double inv2g = 0.5 / gamma;
            const double cpq = A.coeff[p] * B.coeff[q];
            const double R2_AB = dist2(A.center, B.center);
            const double K_AB = std::exp(-mu * R2_AB);
            const double prefactor_pq = 2.0 * units::pi / gamma * K_AB;

            double P_c[3], PA[3], PB[3];
            for (int d = 0; d < 3; ++d) {
                P_c[d] = (aa * A.center[d] + ab * B.center[d]) / gamma;
                PA[d] = P_c[d] - A.center[d];
                PB[d] = P_c[d] - B.center[d];
            }

            // E_full[d][ai][bj][t] — Hermite expansion of xA^ai * xB^bj.
            // Dimensions: 3 axes × (imax+1) × (jmax+1) × (Lmax_dir+1)
            // with imax, jmax <= kMaxLOrbital. t runs to ai+bj <= 2*kMaxLOrbital, so
            // the Hermite-index dimension is 2*kMaxLOrbital+1 (7 for f-shells).
            // No zero-init: every entry read is explicitly written below.
            constexpr int kAng = kMaxLOrbital + 1;   // 4 for f
            constexpr int kL = 2 * kMaxLOrbital + 1; // 7 for f
            double E_full[3][kAng][kAng][kL];
            for (int d = 0; d < 3; ++d) {
                for (int ai = 0; ai <= imax; ++ai) {
                    for (int bj = 0; bj <= jmax; ++bj) {
                        // md_e_coeffs always writes kMDEDim entries
                        // (it zero-fills the whole buffer first), so Ev
                        // must be sized kMDEDim, not just ai+bj+1.
                        double Ev[kMDEDim];
                        md_e_coeffs(ai, bj, PA[d], PB[d], inv2g, Ev);
                        for (int t = 0; t <= ai + bj; ++t)
                            E_full[d][ai][bj][t] = Ev[t];
                    }
                }
            }

            for (int c = 0; c < natm; ++c) {
                double PC[3];
                for (int d = 0; d < 3; ++d)
                    PC[d] = P_c[d] - atoms[c].pos[d];
                const double PC2 =
                    PC[0] * PC[0] + PC[1] * PC[1] + PC[2] * PC[2];
                const double T = gamma * PC2;

                // Boys functions F_0..F_{Ltot_max} — Ltot_max = 3 * Lmax_dir.
                // Lmax_dir = imax+jmax <= 2*kMaxLOrbital, so Ltot_max <= 6*kMaxLOrbital
                // (18 for f-shells) and Fm needs 6*kMaxLOrbital+1 = 19 entries.
                const int Ltot_max = 3 * Lmax_dir;
                constexpr int kBoysN = 6 * kMaxLOrbital + 1; // 19 for f
                double Fm[kBoysN];
                boys_fn_vec(T, Ltot_max, Fm);

                // R_tab[m][t][u][v] — McMurchie–Davidson recurrence.
                // m ∈ [0, Ltot_max] ≤ 6*kMaxLOrbital; t, u, v ∈ [0, Lmax_dir]
                // ≤ 2*kMaxLOrbital. No zero-init needed: the recurrence only
                // reads entries it has already written.
                constexpr int kRm = 6 * kMaxLOrbital + 1; // 19 for f
                constexpr int kRd = 2 * kMaxLOrbital + 1; // 7 for f
                double R_tab[kRm][kRd][kRd][kRd];
                double pow_neg2g = 1.0;
                const double neg2g = -2.0 * gamma;
                for (int m = 0; m <= Ltot_max; ++m) {
                    R_tab[m][0][0][0] = pow_neg2g * Fm[m];
                    pow_neg2g *= neg2g;
                }
                for (int m = Ltot_max - 1; m >= 0; --m) {
                    const int max_tuv = Ltot_max - m;
                    for (int t = 1; t <= Lmax_dir && t <= max_tuv; ++t) {
                        R_tab[m][t][0][0] = PC[0] * R_tab[m + 1][t - 1][0][0];
                        if (t >= 2)
                            R_tab[m][t][0][0] +=
                                (t - 1) * R_tab[m + 1][t - 2][0][0];
                    }
                    for (int t = 0; t <= Lmax_dir && t <= max_tuv; ++t) {
                        for (int u = 1; u <= Lmax_dir && t + u <= max_tuv;
                             ++u) {
                            R_tab[m][t][u][0] =
                                PC[1] * R_tab[m + 1][t][u - 1][0];
                            if (u >= 2)
                                R_tab[m][t][u][0] +=
                                    (u - 1) * R_tab[m + 1][t][u - 2][0];
                        }
                    }
                    for (int t = 0; t <= Lmax_dir && t <= max_tuv; ++t) {
                        for (int u = 0; u <= Lmax_dir && t + u <= max_tuv;
                             ++u) {
                            for (int v = 1;
                                 v <= Lmax_dir && t + u + v <= max_tuv; ++v) {
                                R_tab[m][t][u][v] =
                                    PC[2] * R_tab[m + 1][t][u][v - 1];
                                if (v >= 2)
                                    R_tab[m][t][u][v] +=
                                        (v - 1) * R_tab[m + 1][t][u][v - 2];
                            }
                        }
                    }
                }

                // Scatter across angular combinations.
                const double prefactor_c =
                    prefactor_pq * (-static_cast<double>(atoms[c].Z));
                for (int ia = 0; ia < A.n_cart; ++ia) {
                    const int ax = a_ang[ia][0], ay = a_ang[ia][1],
                              az = a_ang[ia][2];
                    for (int ib = 0; ib < B.n_cart; ++ib) {
                        const int bx = b_ang[ib][0], by = b_ang[ib][1],
                                  bz = b_ang[ib][2];
                        const int Lx = ax + bx, Ly = ay + by, Lz = az + bz;
                        double V = 0.0;
                        for (int t = 0; t <= Lx; ++t) {
                            const double Ex = E_full[0][ax][bx][t];
                            for (int u = 0; u <= Ly; ++u) {
                                const double Ey = E_full[1][ay][by][u];
                                for (int v = 0; v <= Lz; ++v) {
                                    V += Ex * Ey * E_full[2][az][bz][v]
                                         * R_tab[0][t][u][v];
                                }
                            }
                        }
                        out[ia * B.n_cart + ib] += cpq * prefactor_c * V;
                    }
                }
            }
        }
    }
}

/// @brief Compute Boys functions F_0 .. F_{n_max} in one call.
/// Uses upward recurrence from F_0 (one erf + one exp total).
static inline void boys_fn_vec(double T, int n_max, double Fm[]) {
    Fm[0] = boys_f0(T);
    if (n_max == 0)
        return;
    if (T < 1e-12) {
        for (int m = 1; m <= n_max; ++m)
            Fm[m] = 1.0 / (2 * m + 1);
    } else {
        const double ex = std::exp(-T);
        const double inv2T = 0.5 / T;
        for (int m = 1; m <= n_max; ++m)
            Fm[m] = ((2 * m - 1) * Fm[m - 1] - ex) * inv2T;
    }
}

// 2 * π^2.5 — precomputed constant shared by all eri_blk_* functions.
static constexpr double TWO_PI_2_5 = 34.98683665524972497; // 2 * pi^2.5

// ── ss|ss specialization: Ltot=0, 1 ERI, xsimd ket-loop vectorization ──

/// @brief (ss|ss) ERI block. Ltot=0: only F_0, no R-table recurrence.
/// The r×s ket primitive loop is vectorized with xsimd::batch<double>.
static void eri_blk_ss_ss(const ShellBlock& A, const ShellBlock& B,
                          const ShellBlock& C, const ShellBlock& D,
                          double* out) {
    using batch_t = xsimd::batch<double>;
    constexpr int W = static_cast<int>(batch_t::size);

    // ── Precompute ket geometry (invariant across bra primitives) ──────
    // Max ket primitive pairs: 8×8=64 covers cc-pVTZ; 16×16 for very large basis.
    // STO-3G: ≤ 3×3=9.  cc-pVDZ: ≤ 6×6=36.
    constexpr int KMAX = 64;
    const int nket = C.n_prim * D.n_prim;
    double gamma2_k[KMAX], Q0_k[KMAX], Q1_k[KMAX], Q2_k[KMAX];
    double kcd_coeff_k[KMAX]; // K_CD * C.coeff[r] * D.coeff[s]

    int k = 0;
    for (int r = 0; r < C.n_prim; ++r) {
        const double ac = C.alpha[r];
        for (int s = 0; s < D.n_prim; ++s) {
            const double ad = D.alpha[s];
            const double g2 = ac + ad;
            gamma2_k[k] = g2;
            Q0_k[k] = (ac * C.center[0] + ad * D.center[0]) / g2;
            Q1_k[k] = (ac * C.center[1] + ad * D.center[1]) / g2;
            Q2_k[k] = (ac * C.center[2] + ad * D.center[2]) / g2;
            const double dcd0 = C.center[0] - D.center[0];
            const double dcd1 = C.center[1] - D.center[1];
            const double dcd2 = C.center[2] - D.center[2];
            const double R2_CD = dcd0 * dcd0 + dcd1 * dcd1 + dcd2 * dcd2;
            kcd_coeff_k[k] =
                std::exp(-(ac * ad / g2) * R2_CD) * C.coeff[r] * D.coeff[s];
            ++k;
        }
    }

    // ── Bra primitive loop ─────────────────────────────────────────────
    for (int p = 0; p < A.n_prim; ++p) {
        const double aa = A.alpha[p];
        for (int q = 0; q < B.n_prim; ++q) {
            const double ab = B.alpha[q];
            const double gamma1 = aa + ab;
            const double P0 = (aa * A.center[0] + ab * B.center[0]) / gamma1;
            const double P1 = (aa * A.center[1] + ab * B.center[1]) / gamma1;
            const double P2 = (aa * A.center[2] + ab * B.center[2]) / gamma1;
            const double dab0 = A.center[0] - B.center[0];
            const double dab1 = A.center[1] - B.center[1];
            const double dab2 = A.center[2] - B.center[2];
            const double R2_AB = dab0 * dab0 + dab1 * dab1 + dab2 * dab2;
            const double K_AB = std::exp(-(aa * ab / gamma1) * R2_AB);
            const double bra_fac =
                A.coeff[p] * B.coeff[q] * K_AB * TWO_PI_2_5 / gamma1;
            if (bra_fac == 0.0)
                continue;

            // Build T_k and pref_k for all ket pairs
            double T_k[KMAX], pref_k[KMAX];
            for (int kk = 0; kk < nket; ++kk) {
                const double g2 = gamma2_k[kk];
                const double delta = gamma1 + g2;
                const double rho = gamma1 * g2 / delta;
                const double dPQ0 = P0 - Q0_k[kk];
                const double dPQ1 = P1 - Q1_k[kk];
                const double dPQ2 = P2 - Q2_k[kk];
                T_k[kk] = rho * (dPQ0 * dPQ0 + dPQ1 * dPQ1 + dPQ2 * dPQ2);
                pref_k[kk] =
                    bra_fac * kcd_coeff_k[kk] / (g2 * std::sqrt(delta));
            }

            // xsimd-vectorized: acc += pref * boys_f0(T) for each ket pair
            // boys_f0(T) = 0.5 * sqrt(units::pi/T) * erf(sqrt(T));  = 1.0 for T < 1e-12
            batch_t acc_v(0.0);
            int kk = 0;
            for (; kk + W <= nket; kk += W) {
                batch_t T_v = batch_t::load_unaligned(T_k + kk);
                batch_t pref_v = batch_t::load_unaligned(pref_k + kk);
                auto small = T_v < batch_t(1e-12);
                auto safe_T = xsimd::select(small, batch_t(1.0), T_v);
                auto sqrtT = xsimd::sqrt(safe_T);
                auto F0_v = xsimd::select(
                    small, batch_t(1.0),
                    batch_t(0.5) * xsimd::sqrt(batch_t(units::pi) / safe_T)
                        * xsimd::erf(sqrtT));
                acc_v += pref_v * F0_v;
            }
            // Horizontal reduce: store to array and sum (portable across xsimd versions)
            double acc_arr[16]; // W ≤ 16 (AVX-512 = 8 doubles)
            acc_v.store_unaligned(acc_arr);
            double acc = 0.0;
            for (int i = 0; i < W; ++i)
                acc += acc_arr[i];
            // Scalar tail
            for (; kk < nket; ++kk)
                acc += pref_k[kk] * boys_f0(T_k[kk]);
            out[0] += acc;
        }
    }
}

// ── General-Ltot template: compile-time Ltot for Boys and R-table loops ─

/// @brief Shell-block ERI with compile-time maximum Ltot.
/// Identical algorithm to the generic case but Boys recurrence and R-table
/// build loops have fixed bounds → the compiler can unroll them.
template <int LTOT>
static void eri_blk_ltot(const ShellBlock& A, const ShellBlock& B,
                         const ShellBlock& C, const ShellBlock& D,
                         double* out) {
    constexpr int Ltot_max = LTOT;
    constexpr int RS = LTOT + 1; // R-table dimension per axis
    const int na = A.n_cart, nb = B.n_cart;
    const int nc = C.n_cart, nd = D.n_cart;

    int a_ang_tbl[kMaxNCart][3], b_ang_tbl[kMaxNCart][3];
    int c_ang_tbl[kMaxNCart][3], d_ang_tbl[kMaxNCart][3];
    fill_cart_comps(A.l, a_ang_tbl);
    fill_cart_comps(B.l, b_ang_tbl);
    fill_cart_comps(C.l, c_ang_tbl);
    fill_cart_comps(D.l, d_ang_tbl);

    for (int p = 0; p < A.n_prim; ++p) {
        for (int q = 0; q < B.n_prim; ++q) {
            const double aa = A.alpha[p], ab = B.alpha[q];
            const double gamma1 = aa + ab;
            const double inv2g1 = 0.5 / gamma1;

            double P[3], PA[3], PB[3];
            double R2_AB = 0.0;
            for (int d = 0; d < 3; ++d) {
                P[d] = (aa * A.center[d] + ab * B.center[d]) / gamma1;
                PA[d] = P[d] - A.center[d];
                PB[d] = P[d] - B.center[d];
                const double dAB = A.center[d] - B.center[d];
                R2_AB += dAB * dAB;
            }
            const double K_AB = std::exp(-(aa * ab / gamma1) * R2_AB);

            // Bra prefactor: skip entire ket loop when bra contribution is negligible
            const double bra_fac =
                A.coeff[p] * B.coeff[q] * K_AB * TWO_PI_2_5 / gamma1;
            if (std::abs(bra_fac) < 1e-15)
                continue;

            double E_bra[3][2][2][3];
            for (int d = 0; d < 3; ++d) {
                E_bra[d][0][0][0] = 1.0;
                E_bra[d][0][0][1] = 0.0;
                E_bra[d][0][0][2] = 0.0;
                E_bra[d][1][0][0] = PA[d];
                E_bra[d][1][0][1] = inv2g1;
                E_bra[d][1][0][2] = 0.0;
                E_bra[d][0][1][0] = PB[d];
                E_bra[d][0][1][1] = inv2g1;
                E_bra[d][0][1][2] = 0.0;
                E_bra[d][1][1][0] = PA[d] * PB[d] + inv2g1;
                E_bra[d][1][1][1] = inv2g1 * (PA[d] + PB[d]);
                E_bra[d][1][1][2] = inv2g1 * inv2g1;
            }

            for (int r = 0; r < C.n_prim; ++r) {
                for (int s = 0; s < D.n_prim; ++s) {
                    const double ac = C.alpha[r], ad = D.alpha[s];
                    const double gamma2 = ac + ad;
                    const double inv2g2 = 0.5 / gamma2;
                    const double delta = gamma1 + gamma2;
                    const double rho = gamma1 * gamma2 / delta;

                    double Q[3], QC[3], QD[3], PQ[3];
                    double R2_CD = 0.0, PQ2 = 0.0;
                    for (int d = 0; d < 3; ++d) {
                        Q[d] = (ac * C.center[d] + ad * D.center[d]) / gamma2;
                        QC[d] = Q[d] - C.center[d];
                        QD[d] = Q[d] - D.center[d];
                        PQ[d] = P[d] - Q[d];
                        const double dCD = C.center[d] - D.center[d];
                        R2_CD += dCD * dCD;
                        PQ2 += PQ[d] * PQ[d];
                    }
                    const double K_CD = std::exp(-(ac * ad / gamma2) * R2_CD);

                    const double prefactor = bra_fac * C.coeff[r] * D.coeff[s]
                                             * K_CD
                                             / (gamma2 * std::sqrt(delta));
                    if (std::abs(prefactor) < 1e-15)
                        continue;

                    const double T_pq = rho * PQ2;

                    // Boys functions F_0..F_LTOT (fixed-size array, loop unrollable)
                    double Fm[LTOT + 1];
                    boys_fn_vec(T_pq, Ltot_max, Fm);

                    double E_ket[3][2][2][3];
                    for (int d = 0; d < 3; ++d) {
                        E_ket[d][0][0][0] = 1.0;
                        E_ket[d][0][0][1] = 0.0;
                        E_ket[d][0][0][2] = 0.0;
                        E_ket[d][1][0][0] = QC[d];
                        E_ket[d][1][0][1] = inv2g2;
                        E_ket[d][1][0][2] = 0.0;
                        E_ket[d][0][1][0] = QD[d];
                        E_ket[d][0][1][1] = inv2g2;
                        E_ket[d][0][1][2] = 0.0;
                        E_ket[d][1][1][0] = QC[d] * QD[d] + inv2g2;
                        E_ket[d][1][1][1] = inv2g2 * (QC[d] + QD[d]);
                        E_ket[d][1][1][2] = inv2g2 * inv2g2;
                    }

                    // R-table: sized exactly to [LTOT+1]^4 — avoids wasted zero-init.
                    // LTOT=1: 16 doubles (128 B) vs 625 (5000 B) previously.
                    double R_tab[RS][RS][RS][RS] = {};

                    double pow_neg2rho = 1.0;
                    const double neg2rho = -2.0 * rho;
                    for (int m = 0; m <= Ltot_max; ++m) {
                        R_tab[m][0][0][0] = pow_neg2rho * Fm[m];
                        pow_neg2rho *= neg2rho;
                    }

                    for (int m = Ltot_max - 1; m >= 0; --m) {
                        const int max_tuv = Ltot_max - m;
                        for (int t = 1; t <= Ltot_max && t <= max_tuv; ++t) {
                            R_tab[m][t][0][0] =
                                PQ[0] * R_tab[m + 1][t - 1][0][0];
                            if (t >= 2)
                                R_tab[m][t][0][0] +=
                                    (t - 1) * R_tab[m + 1][t - 2][0][0];
                        }
                        for (int t = 0; t <= Ltot_max && t <= max_tuv; ++t)
                            for (int u = 1; u <= Ltot_max && t + u <= max_tuv;
                                 ++u) {
                                R_tab[m][t][u][0] =
                                    PQ[1] * R_tab[m + 1][t][u - 1][0];
                                if (u >= 2)
                                    R_tab[m][t][u][0] +=
                                        (u - 1) * R_tab[m + 1][t][u - 2][0];
                            }
                        for (int t = 0; t <= Ltot_max && t <= max_tuv; ++t)
                            for (int u = 0; u <= Ltot_max && t + u <= max_tuv;
                                 ++u)
                                for (int v = 1;
                                     v <= Ltot_max && t + u + v <= max_tuv;
                                     ++v) {
                                    R_tab[m][t][u][v] =
                                        PQ[2] * R_tab[m + 1][t][u][v - 1];
                                    if (v >= 2)
                                        R_tab[m][t][u][v] +=
                                            (v - 1) * R_tab[m + 1][t][u][v - 2];
                                }
                    }

                    for (int ia = 0; ia < na; ++ia) {
                        const int* a_ang = a_ang_tbl[ia];
                        for (int ib = 0; ib < nb; ++ib) {
                            const int* b_ang = b_ang_tbl[ib];
                            const int Lx1 = a_ang[0] + b_ang[0];
                            const int Ly1 = a_ang[1] + b_ang[1];
                            const int Lz1 = a_ang[2] + b_ang[2];
                            double* out_ab = out + (ia * nb + ib) * (nc * nd);

                            for (int ic = 0; ic < nc; ++ic) {
                                const int* c_ang = c_ang_tbl[ic];
                                for (int id = 0; id < nd; ++id) {
                                    const int* d_ang = d_ang_tbl[id];
                                    const int Lx2 = c_ang[0] + d_ang[0];
                                    const int Ly2 = c_ang[1] + d_ang[1];
                                    const int Lz2 = c_ang[2] + d_ang[2];

                                    double val = 0.0;
                                    for (int t = 0; t <= Lx1; ++t) {
                                        const double Ex1 =
                                            E_bra[0][a_ang[0]][b_ang[0]][t];
                                        for (int u = 0; u <= Ly1; ++u) {
                                            const double Exy1 =
                                                Ex1
                                                * E_bra[1][a_ang[1]][b_ang[1]]
                                                       [u];
                                            for (int v = 0; v <= Lz1; ++v) {
                                                const double e1 =
                                                    Exy1
                                                    * E_bra[2][a_ang[2]]
                                                           [b_ang[2]][v];
                                                for (int tau = 0; tau <= Lx2;
                                                     ++tau) {
                                                    const double Ex2 =
                                                        e1
                                                        * E_ket[0][c_ang[0]]
                                                               [d_ang[0]][tau];
                                                    for (int nu = 0; nu <= Ly2;
                                                         ++nu) {
                                                        const double Exy2 =
                                                            Ex2
                                                            * E_ket[1][c_ang[1]]
                                                                   [d_ang[1]]
                                                                   [nu];
                                                        for (int phi = 0;
                                                             phi <= Lz2;
                                                             ++phi) {
                                                            const double e2 =
                                                                Exy2
                                                                * E_ket
                                                                    [2]
                                                                    [c_ang[2]]
                                                                    [d_ang[2]]
                                                                    [phi];
                                                            const int sign =
                                                                ((tau + nu
                                                                  + phi)
                                                                 & 1)
                                                                    ? -1
                                                                    : 1;
                                                            val +=
                                                                e2 * sign
                                                                * R_tab[0]
                                                                       [t + tau]
                                                                       [u + nu]
                                                                       [v
                                                                        + phi];
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    out_ab[ic * nd + id] += prefactor * val;
                                }
                            }
                        }
                    }
                } // s
            } // r
        } // q
    } // p
}

// ── Fallback for l ≥ 2 shells ───────────────────────────────────────────

/// @brief Generic shell-block ERI for shells with l ≥ 2 (d, f, …).
/// Identical to the original eri_block() implementation; runtime Ltot.
static void eri_blk_general(const ShellBlock& A, const ShellBlock& B,
                            const ShellBlock& C, const ShellBlock& D,
                            double* out) {
    const int Ltot_max = A.l + B.l + C.l + D.l;
    const int na = A.n_cart, nb = B.n_cart;
    const int nc = C.n_cart, nd = D.n_cart;

    int a_ang_tbl[kMaxNCart][3], b_ang_tbl[kMaxNCart][3];
    int c_ang_tbl[kMaxNCart][3], d_ang_tbl[kMaxNCart][3];
    fill_cart_comps(A.l, a_ang_tbl);
    fill_cart_comps(B.l, b_ang_tbl);
    fill_cart_comps(C.l, c_ang_tbl);
    fill_cart_comps(D.l, d_ang_tbl);

    for (int p = 0; p < A.n_prim; ++p) {
        for (int q = 0; q < B.n_prim; ++q) {
            const double aa = A.alpha[p], ab = B.alpha[q];
            const double gamma1 = aa + ab;
            const double inv2g1 = 0.5 / gamma1;

            double P[3], PA[3], PB[3];
            double R2_AB = 0.0;
            for (int d = 0; d < 3; ++d) {
                P[d] = (aa * A.center[d] + ab * B.center[d]) / gamma1;
                PA[d] = P[d] - A.center[d];
                PB[d] = P[d] - B.center[d];
                const double dAB = A.center[d] - B.center[d];
                R2_AB += dAB * dAB;
            }
            const double K_AB = std::exp(-(aa * ab / gamma1) * R2_AB);

            const double bra_fac_gen =
                A.coeff[p] * B.coeff[q] * K_AB * TWO_PI_2_5 / gamma1;
            if (std::abs(bra_fac_gen) < 1e-15)
                continue;

            double E_bra[3][2][2][3];
            for (int d = 0; d < 3; ++d) {
                E_bra[d][0][0][0] = 1.0;
                E_bra[d][0][0][1] = 0.0;
                E_bra[d][0][0][2] = 0.0;
                E_bra[d][1][0][0] = PA[d];
                E_bra[d][1][0][1] = inv2g1;
                E_bra[d][1][0][2] = 0.0;
                E_bra[d][0][1][0] = PB[d];
                E_bra[d][0][1][1] = inv2g1;
                E_bra[d][0][1][2] = 0.0;
                E_bra[d][1][1][0] = PA[d] * PB[d] + inv2g1;
                E_bra[d][1][1][1] = inv2g1 * (PA[d] + PB[d]);
                E_bra[d][1][1][2] = inv2g1 * inv2g1;
            }

            for (int r = 0; r < C.n_prim; ++r) {
                for (int s = 0; s < D.n_prim; ++s) {
                    const double ac = C.alpha[r], ad = D.alpha[s];
                    const double gamma2 = ac + ad;
                    const double inv2g2 = 0.5 / gamma2;
                    const double delta = gamma1 + gamma2;
                    const double rho = gamma1 * gamma2 / delta;

                    double Q[3], QC[3], QD[3], PQ[3];
                    double R2_CD = 0.0, PQ2 = 0.0;
                    for (int d = 0; d < 3; ++d) {
                        Q[d] = (ac * C.center[d] + ad * D.center[d]) / gamma2;
                        QC[d] = Q[d] - C.center[d];
                        QD[d] = Q[d] - D.center[d];
                        PQ[d] = P[d] - Q[d];
                        const double dCD = C.center[d] - D.center[d];
                        R2_CD += dCD * dCD;
                        PQ2 += PQ[d] * PQ[d];
                    }
                    const double K_CD = std::exp(-(ac * ad / gamma2) * R2_CD);
                    const double T_pq = rho * PQ2;

                    const double prefactor = bra_fac_gen * C.coeff[r]
                                             * D.coeff[s] * K_CD
                                             / (gamma2 * std::sqrt(delta));
                    if (std::abs(prefactor) < 1e-15)
                        continue;

                    double Fm[9]; // generous upper bound for high-l shells
                    boys_fn_vec(T_pq, Ltot_max, Fm);

                    double E_ket[3][2][2][3];
                    for (int d = 0; d < 3; ++d) {
                        E_ket[d][0][0][0] = 1.0;
                        E_ket[d][0][0][1] = 0.0;
                        E_ket[d][0][0][2] = 0.0;
                        E_ket[d][1][0][0] = QC[d];
                        E_ket[d][1][0][1] = inv2g2;
                        E_ket[d][1][0][2] = 0.0;
                        E_ket[d][0][1][0] = QD[d];
                        E_ket[d][0][1][1] = inv2g2;
                        E_ket[d][0][1][2] = 0.0;
                        E_ket[d][1][1][0] = QC[d] * QD[d] + inv2g2;
                        E_ket[d][1][1][1] = inv2g2 * (QC[d] + QD[d]);
                        E_ket[d][1][1][2] = inv2g2 * inv2g2;
                    }

                    const int Rmax = Ltot_max;
                    double R_tab[9][9][9][9] = {};
                    double pow_neg2rho = 1.0;
                    const double neg2rho = -2.0 * rho;
                    for (int m = 0; m <= Ltot_max; ++m) {
                        R_tab[m][0][0][0] = pow_neg2rho * Fm[m];
                        pow_neg2rho *= neg2rho;
                    }
                    for (int m = Ltot_max - 1; m >= 0; --m) {
                        const int max_tuv = Ltot_max - m;
                        for (int t = 1; t <= Rmax && t <= max_tuv; ++t) {
                            R_tab[m][t][0][0] =
                                PQ[0] * R_tab[m + 1][t - 1][0][0];
                            if (t >= 2)
                                R_tab[m][t][0][0] +=
                                    (t - 1) * R_tab[m + 1][t - 2][0][0];
                        }
                        for (int t = 0; t <= Rmax && t <= max_tuv; ++t)
                            for (int u = 1; u <= Rmax && t + u <= max_tuv;
                                 ++u) {
                                R_tab[m][t][u][0] =
                                    PQ[1] * R_tab[m + 1][t][u - 1][0];
                                if (u >= 2)
                                    R_tab[m][t][u][0] +=
                                        (u - 1) * R_tab[m + 1][t][u - 2][0];
                            }
                        for (int t = 0; t <= Rmax && t <= max_tuv; ++t)
                            for (int u = 0; u <= Rmax && t + u <= max_tuv; ++u)
                                for (int v = 1;
                                     v <= Rmax && t + u + v <= max_tuv; ++v) {
                                    R_tab[m][t][u][v] =
                                        PQ[2] * R_tab[m + 1][t][u][v - 1];
                                    if (v >= 2)
                                        R_tab[m][t][u][v] +=
                                            (v - 1) * R_tab[m + 1][t][u][v - 2];
                                }
                    }

                    for (int ia = 0; ia < na; ++ia) {
                        const int* a_ang = a_ang_tbl[ia];
                        for (int ib = 0; ib < nb; ++ib) {
                            const int* b_ang = b_ang_tbl[ib];
                            const int Lx1 = a_ang[0] + b_ang[0];
                            const int Ly1 = a_ang[1] + b_ang[1];
                            const int Lz1 = a_ang[2] + b_ang[2];
                            double* out_ab = out + (ia * nb + ib) * (nc * nd);

                            for (int ic = 0; ic < nc; ++ic) {
                                const int* c_ang = c_ang_tbl[ic];
                                for (int id = 0; id < nd; ++id) {
                                    const int* d_ang = d_ang_tbl[id];
                                    const int Lx2 = c_ang[0] + d_ang[0];
                                    const int Ly2 = c_ang[1] + d_ang[1];
                                    const int Lz2 = c_ang[2] + d_ang[2];

                                    double val = 0.0;
                                    for (int t = 0; t <= Lx1; ++t) {
                                        const double Ex1 =
                                            E_bra[0][a_ang[0]][b_ang[0]][t];
                                        for (int u = 0; u <= Ly1; ++u) {
                                            const double Exy1 =
                                                Ex1
                                                * E_bra[1][a_ang[1]][b_ang[1]]
                                                       [u];
                                            for (int v = 0; v <= Lz1; ++v) {
                                                const double e1 =
                                                    Exy1
                                                    * E_bra[2][a_ang[2]]
                                                           [b_ang[2]][v];
                                                for (int tau = 0; tau <= Lx2;
                                                     ++tau) {
                                                    const double Ex2 =
                                                        e1
                                                        * E_ket[0][c_ang[0]]
                                                               [d_ang[0]][tau];
                                                    for (int nu = 0; nu <= Ly2;
                                                         ++nu) {
                                                        const double Exy2 =
                                                            Ex2
                                                            * E_ket[1][c_ang[1]]
                                                                   [d_ang[1]]
                                                                   [nu];
                                                        for (int phi = 0;
                                                             phi <= Lz2;
                                                             ++phi) {
                                                            const double e2 =
                                                                Exy2
                                                                * E_ket
                                                                    [2]
                                                                    [c_ang[2]]
                                                                    [d_ang[2]]
                                                                    [phi];
                                                            const int sign =
                                                                ((tau + nu
                                                                  + phi)
                                                                 & 1)
                                                                    ? -1
                                                                    : 1;
                                                            val +=
                                                                e2 * sign
                                                                * R_tab[0]
                                                                       [t + tau]
                                                                       [u + nu]
                                                                       [v
                                                                        + phi];
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    out_ab[ic * nd + id] += prefactor * val;
                                }
                            }
                        }
                    }
                } // s
            } // r
        } // q
    } // p
}

// ── Dispatcher ──────────────────────────────────────────────────────────

/// @brief Compute all ERIs for a shell quartet (A B | C D) into `out`.
///
/// Dispatches to the Rys-quadrature engine for all non-ssss cases.
/// The ssss case (Ltot = 0) stays on the dedicated xsimd-vectorized
/// `eri_blk_ss_ss`: that kernel precomputes ket geometry, vectorizes
/// the Boys sum over the ket-primitive loop, and avoids all of Rys's
/// general infrastructure (heap buffer, 2D-VRR, HRR). In benchmarks
/// it beats Rys by ~1.5× for pure s-shells because a 1-root Rys
/// quadrature with allocation overhead is slower than a direct
/// F_0(T)-sum in that corner of the parameter space.
void eri_block(const ShellBlock& A, const ShellBlock& B, const ShellBlock& C,
               const ShellBlock& D, double* out) {
    if (A.l == 0 && B.l == 0 && C.l == 0 && D.l == 0) {
        return eri_blk_ss_ss(A, B, C, D, out);
    }
    rys_eri_block(A, B, C, D, out);
}

} // namespace native
} // namespace scf
} // namespace cpu
} // namespace ATV

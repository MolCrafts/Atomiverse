/// @file cuda/scf/device_integrals.cu
/// @brief GPU molecular integral kernels — pure-s (W0–3) and s+p MD/OS (W4).
///
/// Pure-s: 1 thread / shell pair|quartet, F0 Boys, fused 1e.
/// s+p (l≤1): 1 thread / AO pair|quartet; OS overlap/kinetic; MD nuclear + ERI.
///
/// References:
///   Szabo & Ostlund, Modern Quantum Chemistry, Appendix A
///   Helgaker, Jørgensen, Olsen — MD recurrence
///   CPU twin: cpu/scf/native_integrals.cpp (overlap_prim_3d, eri_prim_md)

#include "atomiverse/core/gto.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/scf/device_integrals.cuh"

#include "atomiverse/cuda/frame/types.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace ATV {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace scf {

// Scalar and batched Boys share the stable law in core/gto.h.

__device__ __forceinline__ work_t dist2_device(work_t ax, work_t ay, work_t az,
                                               work_t bx, work_t by,
                                               work_t bz) {
    work_t dx = ax - bx, dy = ay - by, dz = az - bz;
    return dx * dx + dy * dy + dz * dz;
}

/// Cartesian (lx,ly,lz) for shell angular L and cart index (px=0,py=1,pz=2).
__device__ __forceinline__ void cart_ang_device(int L, int cart, int ang[3]) {
    ang[0] = ang[1] = ang[2] = 0;
    if (L == 1) {
        if (cart == 0)
            ang[0] = 1;
        else if (cart == 1)
            ang[1] = 1;
        else
            ang[2] = 1;
    } else if (L == 2) {
        // libcint / fill_cart_comps: xx, xy, xz, yy, yz, zz.
        if (cart == 0)
            ang[0] = 2;
        else if (cart == 1) {
            ang[0] = 1;
            ang[1] = 1;
        } else if (cart == 2) {
            ang[0] = 1;
            ang[2] = 1;
        } else if (cart == 3)
            ang[1] = 2;
        else if (cart == 4) {
            ang[1] = 1;
            ang[2] = 1;
        } else
            ang[2] = 2;
    } else if (L == 3) {
        // fill_cart_comps(l=3): xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz.
        if (cart == 0)
            ang[0] = 3;
        else if (cart == 1) {
            ang[0] = 2;
            ang[1] = 1;
        } else if (cart == 2) {
            ang[0] = 2;
            ang[2] = 1;
        } else if (cart == 3) {
            ang[0] = 1;
            ang[1] = 2;
        } else if (cart == 4) {
            ang[0] = 1;
            ang[1] = 1;
            ang[2] = 1;
        } else if (cart == 5) {
            ang[0] = 1;
            ang[2] = 2;
        } else if (cart == 6)
            ang[1] = 3;
        else if (cart == 7) {
            ang[1] = 2;
            ang[2] = 1;
        } else if (cart == 8) {
            ang[1] = 1;
            ang[2] = 2;
        } else
            ang[2] = 3;
    }
}

// ── OS 1D overlap (imax,jmax ≤ 3 for kinetic on p) ───────────────────

__device__ void os_overlap_1d_device(work_t PA, work_t PB, work_t inv2g,
                                     work_t S00, int imax, int jmax,
                                     work_t* S) {
    const int jstride = jmax + 1;
    S[0] = S00;
    for (int j = 0; j < jmax; ++j) {
        work_t prev = (j > 0) ? S[j - 1] : work_t(0);
        S[j + 1] = PB * S[j] + inv2g * work_t(j) * prev;
    }
    for (int i = 0; i < imax; ++i) {
        for (int j = 0; j <= jmax; ++j) {
            work_t s_im1 = (i > 0) ? S[(i - 1) * jstride + j] : work_t(0);
            work_t s_jm1 = (j > 0) ? S[i * jstride + j - 1] : work_t(0);
            S[(i + 1) * jstride + j] =
                PA * S[i * jstride + j]
                + inv2g * (work_t(i) * s_im1 + work_t(j) * s_jm1);
        }
    }
}

__device__ work_t overlap_prim_3d_device(const int a_ang[3], const int b_ang[3],
                                         work_t alpha_a, const work_t A[3],
                                         work_t alpha_b, const work_t B[3]) {
    const work_t gamma = alpha_a + alpha_b;
    const work_t mu = alpha_a * alpha_b / gamma;
    const work_t inv2g = work_t(0.5) / gamma;
    const work_t sqrt_pi_g = sqrt(work_t(ATV::units::pi) / gamma);
    work_t result = work_t(1);
    for (int d = 0; d < 3; ++d) {
        const work_t dd = A[d] - B[d];
        const work_t Pd = (alpha_a * A[d] + alpha_b * B[d]) / gamma;
        const work_t PA = Pd - A[d];
        const work_t PB = Pd - B[d];
        const work_t S00 = sqrt_pi_g * exp(-mu * dd * dd);
        const int imax = a_ang[d];
        const int jmax = b_ang[d];
        // Force raise: imax,jmax ≤2 → (3)*(3)=9 cells
        work_t S[16];
        for (int k = 0; k < 16; ++k)
            S[k] = work_t(0);
        os_overlap_1d_device(PA, PB, inv2g, S00, imax, jmax, S);
        result *= S[imax * (jmax + 1) + jmax];
    }
    return result;
}

__device__ work_t kinetic_prim_3d_device(const int a_ang[3], const int b_ang[3],
                                         work_t alpha_a, const work_t A[3],
                                         work_t alpha_b, const work_t B[3]) {
    const work_t gamma = alpha_a + alpha_b;
    const work_t mu = alpha_a * alpha_b / gamma;
    const work_t inv2g = work_t(0.5) / gamma;
    const work_t sqrt_pi_g = sqrt(work_t(ATV::units::pi) / gamma);

    int imax_all = a_ang[0];
    int jmax_base = b_ang[0];
    for (int d = 1; d < 3; ++d) {
        if (a_ang[d] > imax_all)
            imax_all = a_ang[d];
        if (b_ang[d] > jmax_base)
            jmax_base = b_ang[d];
    }
    const int im = imax_all;      // ≤ 1 for l≤1
    const int jm = jmax_base + 2; // ≤ 3
    const int jstride = jm + 1;   // ≤ 4
    // Force raise of p → d: im≤2, jm≤ jmax+2 ≤4 → (3)*(5)=15 cells
    work_t Sd[3][32];
    work_t S1d[3];
    for (int d = 0; d < 3; ++d) {
        const work_t dd = A[d] - B[d];
        const work_t Pd = (alpha_a * A[d] + alpha_b * B[d]) / gamma;
        const work_t PA = Pd - A[d];
        const work_t PB = Pd - B[d];
        const work_t S00 = sqrt_pi_g * exp(-mu * dd * dd);
        for (int k = 0; k < 32; ++k)
            Sd[d][k] = work_t(0);
        os_overlap_1d_device(PA, PB, inv2g, S00, im, jm, Sd[d]);
        S1d[d] = Sd[d][a_ang[d] * jstride + b_ang[d]];
    }

    work_t prim_kin = work_t(0);
    for (int d = 0; d < 3; ++d) {
        const int i = a_ang[d];
        const int j = b_ang[d];
        const work_t beta = alpha_b;
        work_t s_ij = Sd[d][i * jstride + j];
        work_t s_ijp2 = Sd[d][i * jstride + j + 2];
        work_t s_ijm2 = (j >= 2) ? Sd[d][i * jstride + j - 2] : work_t(0);
        work_t T1d = -work_t(0.5)
                     * (work_t(j * (j - 1)) * s_ijm2
                        - work_t(2) * beta * work_t(2 * j + 1) * s_ij
                        + work_t(4) * beta * beta * s_ijp2);
        work_t other = work_t(1);
        for (int e = 0; e < 3; ++e) {
            if (e != d)
                other *= S1d[e];
        }
        prim_kin += T1d * other;
    }
    return prim_kin;
}

// Force raise of p (L=1) needs ai/bj ≤ 2. Sized like CPU kDerivL for Lorb=1.
static constexpr int kDeviceDerivL = 2; // max L after +1 raise
static constexpr int kDeviceMDEDim = 2 * kDeviceDerivL + 1; // 5
static constexpr int kDeviceMDEIDim = kDeviceDerivL + 2;    // 4
static constexpr int kDeviceRNuc = 2 * kDeviceDerivL + 1; // 5 (nuclear Ltot≤3)
static constexpr int kDeviceREri = 2 * kDeviceDerivL + 2; // 6 (ERI Ltot≤5)

/// MD E coeffs — CPU twin md_e_coeffs; ai,bj ≤ kDeviceDerivL.
__device__ void md_e_coeffs_device(int ai, int bj, work_t PA, work_t PB,
                                   work_t inv2g, work_t E[kDeviceMDEDim]) {
    for (int t = 0; t < kDeviceMDEDim; ++t)
        E[t] = work_t(0);
    work_t E_buf[kDeviceMDEIDim][kDeviceMDEDim];
    for (int i = 0; i < kDeviceMDEIDim; ++i)
        for (int t = 0; t < kDeviceMDEDim; ++t)
            E_buf[i][t] = work_t(0);
    E_buf[0][0] = work_t(1);
    for (int i = 0; i < ai; ++i) {
        const int tmax = i + 1;
        for (int t = 0; t <= tmax && t < kDeviceMDEDim; ++t) {
            const work_t etm1 = (t > 0) ? E_buf[i][t - 1] : work_t(0);
            const work_t et = E_buf[i][t];
            const work_t etp1 =
                (t + 1 < kDeviceMDEDim) ? E_buf[i][t + 1] : work_t(0);
            E_buf[i + 1][t] = inv2g * etm1 + PA * et + work_t(t + 1) * etp1;
        }
    }
    work_t W[2][kDeviceMDEDim];
    int cur = 0;
    for (int t = 0; t < kDeviceMDEDim; ++t)
        W[cur][t] = E_buf[ai][t];
    for (int jj = 0; jj < bj; ++jj) {
        const int nxt = 1 - cur;
        for (int t = 0; t < kDeviceMDEDim; ++t)
            W[nxt][t] = work_t(0);
        const int tmax = ai + jj + 1;
        for (int t = 0; t <= tmax && t < kDeviceMDEDim; ++t) {
            const work_t etm1 = (t > 0) ? W[cur][t - 1] : work_t(0);
            const work_t et = W[cur][t];
            const work_t etp1 =
                (t + 1 < kDeviceMDEDim) ? W[cur][t + 1] : work_t(0);
            W[nxt][t] = inv2g * etm1 + PB * et + work_t(t + 1) * etp1;
        }
        cur = nxt;
    }
    for (int t = 0; t <= ai + bj && t < kDeviceMDEDim; ++t)
        E[t] = W[cur][t];
}

/// Thin l≤1 wrapper used by energy kernels (E[0..2] only).
__device__ __forceinline__ void md_e_l1(int ai, int bj, work_t PA, work_t PB,
                                        work_t inv2g, work_t E[3]) {
    work_t full[kDeviceMDEDim];
    md_e_coeffs_device(ai, bj, PA, PB, inv2g, full);
    E[0] = full[0];
    E[1] = full[1];
    E[2] = full[2];
}

__device__ work_t nuclear_prim_device(const int a_ang[3], const int b_ang[3],
                                      work_t gamma, const work_t PA[3],
                                      const work_t PB[3], const work_t PC[3],
                                      work_t K_AB) {
    const work_t inv2g = work_t(0.5) / gamma;
    const work_t prefactor = work_t(2) * work_t(ATV::units::pi) / gamma * K_AB;
    const work_t PC2 = PC[0] * PC[0] + PC[1] * PC[1] + PC[2] * PC[2];
    const work_t T = gamma * PC2;
    const int Lx = a_ang[0] + b_ang[0];
    const int Ly = a_ang[1] + b_ang[1];
    const int Lz = a_ang[2] + b_ang[2];
    const int Ltot = Lx + Ly + Lz; // ≤3 with force raise of p

    work_t Fm[kDeviceRNuc];
    ATV::boys_sequence(Ltot, T, Fm);

    work_t E[3][kDeviceMDEDim];
    for (int d = 0; d < 3; ++d)
        md_e_coeffs_device(a_ang[d], b_ang[d], PA[d], PB[d], inv2g, E[d]);

    // R_tab sized for force raise (Ltot≤3 → indices 0..3)
    work_t R_tab[kDeviceRNuc][kDeviceRNuc][kDeviceRNuc][kDeviceRNuc];
    for (int m = 0; m < kDeviceRNuc; ++m)
        for (int t = 0; t < kDeviceRNuc; ++t)
            for (int u = 0; u < kDeviceRNuc; ++u)
                for (int v = 0; v < kDeviceRNuc; ++v)
                    R_tab[m][t][u][v] = work_t(0);

    work_t neg2g = -work_t(2) * gamma;
    work_t neg2g_pow = work_t(1);
    for (int m = 0; m <= Ltot; ++m) {
        R_tab[m][0][0][0] = neg2g_pow * Fm[m];
        neg2g_pow *= neg2g;
    }
    for (int m = Ltot - 1; m >= 0; --m) {
        const int max_tuv = Ltot - m;
        for (int t = 1; t <= Lx && t <= max_tuv; ++t) {
            R_tab[m][t][0][0] = PC[0] * R_tab[m + 1][t - 1][0][0];
            if (t >= 2)
                R_tab[m][t][0][0] += work_t(t - 1) * R_tab[m + 1][t - 2][0][0];
        }
        for (int t = 0; t <= Lx && t <= max_tuv; ++t) {
            for (int u = 1; u <= Ly && t + u <= max_tuv; ++u) {
                R_tab[m][t][u][0] = PC[1] * R_tab[m + 1][t][u - 1][0];
                if (u >= 2)
                    R_tab[m][t][u][0] +=
                        work_t(u - 1) * R_tab[m + 1][t][u - 2][0];
            }
        }
        for (int t = 0; t <= Lx && t <= max_tuv; ++t) {
            for (int u = 0; u <= Ly && t + u <= max_tuv; ++u) {
                for (int v = 1; v <= Lz && t + u + v <= max_tuv; ++v) {
                    R_tab[m][t][u][v] = PC[2] * R_tab[m + 1][t][u][v - 1];
                    if (v >= 2)
                        R_tab[m][t][u][v] +=
                            work_t(v - 1) * R_tab[m + 1][t][u][v - 2];
                }
            }
        }
    }

    work_t V = work_t(0);
    for (int t = 0; t <= Lx; ++t)
        for (int u = 0; u <= Ly; ++u)
            for (int v = 0; v <= Lz; ++v)
                V += E[0][t] * E[1][u] * E[2][v] * R_tab[0][t][u][v];
    return prefactor * V;
}

/// Primitive ERI (ab|cd) for l≤1 — port of CPU eri_prim_md.
__device__ work_t eri_prim_md_device(const int a_ang[3], const int b_ang[3],
                                     const int c_ang[3], const int d_ang[3],
                                     work_t alpha_a, const work_t A[3],
                                     work_t alpha_b, const work_t B[3],
                                     work_t alpha_c, const work_t C[3],
                                     work_t alpha_d, const work_t D[3]) {
    const work_t gamma1 = alpha_a + alpha_b;
    const work_t gamma2 = alpha_c + alpha_d;
    const work_t delta = gamma1 + gamma2;
    const work_t mu1 = alpha_a * alpha_b / gamma1;
    const work_t mu2 = alpha_c * alpha_d / gamma2;
    const work_t rho = gamma1 * gamma2 / delta;
    const work_t inv2g1 = work_t(0.5) / gamma1;
    const work_t inv2g2 = work_t(0.5) / gamma2;

    work_t P[3], Q[3], PA[3], PB[3], QC[3], QD[3], PQ[3];
    work_t R2_AB = work_t(0), R2_CD = work_t(0), PQ2 = work_t(0);
    for (int d = 0; d < 3; ++d) {
        P[d] = (alpha_a * A[d] + alpha_b * B[d]) / gamma1;
        Q[d] = (alpha_c * C[d] + alpha_d * D[d]) / gamma2;
        PA[d] = P[d] - A[d];
        PB[d] = P[d] - B[d];
        QC[d] = Q[d] - C[d];
        QD[d] = Q[d] - D[d];
        PQ[d] = P[d] - Q[d];
        work_t dAB = A[d] - B[d];
        work_t dCD = C[d] - D[d];
        R2_AB += dAB * dAB;
        R2_CD += dCD * dCD;
        PQ2 += PQ[d] * PQ[d];
    }

    const work_t K_AB = exp(-mu1 * R2_AB);
    const work_t K_CD = exp(-mu2 * R2_CD);
    const work_t Tval = rho * PQ2;
    const work_t pi = work_t(ATV::units::pi);
    // 2 * pi^{5/2} / (g1 g2 sqrt(delta)) * K_AB * K_CD
    const work_t prefactor = work_t(2) * pi * pi * sqrt(pi)
                             / (gamma1 * gamma2 * sqrt(delta)) * K_AB * K_CD;

    if (prefactor == work_t(0)) {
        // This is an exact-zero shortcut, not near-zero screening. In this
        // bounded domain the Ltot<=5 E/R polynomial is finite; outside it,
        // preserve the original arithmetic (including 0*Inf/NaN failures).
        constexpr work_t bound = work_t(1e16);
        bool finite_domain = Tval >= work_t(0) && isfinite(Tval)
                             && rho > work_t(0) && inv2g1 > work_t(0)
                             && inv2g2 > work_t(0)
                             && fabs(work_t(2) * rho) <= bound
                             && fabs(inv2g1) <= bound && fabs(inv2g2) <= bound;
        for (int d = 0; d < 3; ++d) {
            // Each comparison rejects NaN; fmax could silently discard it.
            finite_domain = finite_domain && fabs(PA[d]) <= bound
                            && fabs(PB[d]) <= bound && fabs(QC[d]) <= bound
                            && fabs(QD[d]) <= bound && fabs(PQ[d]) <= bound;
        }
        if (finite_domain)
            return work_t(0);
    }

    const int Lx1 = a_ang[0] + b_ang[0];
    const int Ly1 = a_ang[1] + b_ang[1];
    const int Lz1 = a_ang[2] + b_ang[2];
    const int Lx2 = c_ang[0] + d_ang[0];
    const int Ly2 = c_ang[1] + d_ang[1];
    const int Lz2 = c_ang[2] + d_ang[2];
    const int Ltot = Lx1 + Ly1 + Lz1 + Lx2 + Ly2 + Lz2; // ≤5 with force raise

    work_t Fm[kDeviceREri];
    ATV::boys_sequence(Ltot, Tval, Fm);

    work_t E1[3][kDeviceMDEDim], E2[3][kDeviceMDEDim];
    for (int d = 0; d < 3; ++d) {
        md_e_coeffs_device(a_ang[d], b_ang[d], PA[d], PB[d], inv2g1, E1[d]);
        md_e_coeffs_device(c_ang[d], d_ang[d], QC[d], QD[d], inv2g2, E2[d]);
    }

    const int Rx_max = Lx1 + Lx2;
    const int Ry_max = Ly1 + Ly2;
    const int Rz_max = Lz1 + Lz2;

    // Rx+Ry+Rz=Ltot<=5, so the axis-bounded box needs at most 18 cells
    // (2+1)*(2+1)*(1+1). Pack that box within each existing m-parity layer.
    // All m+1 sources were written by the previous layer; padding is unread.
    const int u_stride = Rz_max + 1;
    const int t_stride = (Ry_max + 1) * u_stride;
    work_t R_tab[2][18];
    work_t neg2rho = -work_t(2) * rho;
    work_t neg2rho_pow = work_t(1);
    for (int m = 0; m <= Ltot; ++m) {
        Fm[m] = neg2rho_pow * Fm[m];
        neg2rho_pow *= neg2rho;
    }
    R_tab[Ltot & 1][0] = Fm[Ltot];
    for (int m = Ltot - 1; m >= 0; --m) {
        const int current = m & 1, previous = (m + 1) & 1;
        const int max_tuv = Ltot - m;
        R_tab[current][0] = Fm[m];
        for (int t = 1; t <= max_tuv && t <= Rx_max; ++t) {
            const int offset = t * t_stride;
            R_tab[current][offset] = PQ[0] * R_tab[previous][offset - t_stride];
            if (t >= 2)
                R_tab[current][offset] +=
                    work_t(t - 1) * R_tab[previous][offset - 2 * t_stride];
        }
        for (int t = 0; t <= Rx_max && t <= max_tuv; ++t) {
            for (int u = 1; u <= Ry_max && t + u <= max_tuv; ++u) {
                const int offset = t * t_stride + u * u_stride;
                R_tab[current][offset] =
                    PQ[1] * R_tab[previous][offset - u_stride];
                if (u >= 2)
                    R_tab[current][offset] +=
                        work_t(u - 1) * R_tab[previous][offset - 2 * u_stride];
            }
        }
        for (int t = 0; t <= Rx_max && t <= max_tuv; ++t) {
            for (int u = 0; u <= Ry_max && t + u <= max_tuv; ++u) {
                for (int v = 1; v <= Rz_max && t + u + v <= max_tuv; ++v) {
                    const int offset = t * t_stride + u * u_stride + v;
                    R_tab[current][offset] =
                        PQ[2] * R_tab[previous][offset - 1];
                    if (v >= 2)
                        R_tab[current][offset] +=
                            work_t(v - 1) * R_tab[previous][offset - 2];
                }
            }
        }
    }

    work_t eri_val = work_t(0);
    for (int t = 0; t <= Lx1; ++t) {
        for (int u = 0; u <= Ly1; ++u) {
            for (int v = 0; v <= Lz1; ++v) {
                const work_t e1 = E1[0][t] * E1[1][u] * E1[2][v];
                if (e1 == work_t(0))
                    continue;
                for (int tau = 0; tau <= Lx2; ++tau) {
                    for (int nu = 0; nu <= Ly2; ++nu) {
                        for (int phi = 0; phi <= Lz2; ++phi) {
                            const work_t e2 =
                                E2[0][tau] * E2[1][nu] * E2[2][phi];
                            if (e2 == work_t(0))
                                continue;
                            work_t sign =
                                ((tau + nu + phi) & 1) ? work_t(-1) : work_t(1);
                            eri_val +=
                                e1 * e2 * sign
                                * R_tab[0][(t + tau) * t_stride
                                           + (u + nu) * u_stride + v + phi];
                        }
                    }
                }
            }
        }
    }
    return prefactor * eri_val;
}

// ── Boys F0 test ──────────────────────────────────────────────────────

__global__ void boys_f0_test_kernel(const prec_t* x, prec_t* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
        out[i] = ATV::boys_f0(x[i]);
}

void launch_boys_f0_test(const prec_t* d_x, prec_t* d_out, int n,
                         void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    int threads = cuda::kDefaultBlock;
    int blocks = cuda::grid_size(n, threads);
    boys_f0_test_kernel<<<blocks, threads, 0, s>>>(d_x, d_out, n);
}

// ── Pure-s kernels (Wave 0–3, kept for unit tests) ────────────────────

__global__ void compute_1e_kernel(prec_t* S, prec_t* T, prec_t* V,
                                  const prec_t* shell_alpha,
                                  const prec_t* shell_coeff,
                                  const prec_t* shell_centers,
                                  const int* atom_Z, const prec_t* atom_pos,
                                  int n, int n_prim, int natm) {
    int n_pairs = n * (n + 1) / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int pair = idx; pair < n_pairs; pair += blockDim.x * gridDim.x) {
        int si = (int)((-1.0 + sqrt(1.0 + 8.0 * pair)) * 0.5);
        int sj = pair - si * (si + 1) / 2;
        if (sj > si) {
            si++;
            sj = pair - si * (si + 1) / 2;
        }

        work_t Ax = static_cast<work_t>(shell_centers[si * 3 + 0]);
        work_t Ay = static_cast<work_t>(shell_centers[si * 3 + 1]);
        work_t Az = static_cast<work_t>(shell_centers[si * 3 + 2]);
        work_t Bx = static_cast<work_t>(shell_centers[sj * 3 + 0]);
        work_t By = static_cast<work_t>(shell_centers[sj * 3 + 1]);
        work_t Bz = static_cast<work_t>(shell_centers[sj * 3 + 2]);
        work_t R2 = dist2_device(Ax, Ay, Az, Bx, By, Bz);

        work_t acc_S = 0, acc_T = 0, acc_V = 0;
        for (int p = 0; p < n_prim; ++p) {
            work_t alpha_p = static_cast<work_t>(shell_alpha[si * n_prim + p]);
            work_t c_p = static_cast<work_t>(shell_coeff[si * n_prim + p]);
            for (int q = 0; q < n_prim; ++q) {
                work_t alpha_q =
                    static_cast<work_t>(shell_alpha[sj * n_prim + q]);
                work_t c_q = static_cast<work_t>(shell_coeff[sj * n_prim + q]);
                work_t c_pq = c_p * c_q;
                work_t gamma = alpha_p + alpha_q;
                work_t mu = alpha_p * alpha_q / gamma;
                work_t K_AB = exp(-mu * R2);
                work_t pig = work_t(ATV::units::pi) / gamma;
                work_t S_pq = pig * sqrt(pig) * K_AB;
                acc_S += c_pq * S_pq;
                acc_T += c_pq * mu * (work_t(3) - work_t(2) * mu * R2) * S_pq;
                work_t inv_gamma = work_t(1) / gamma;
                work_t Px = (alpha_p * Ax + alpha_q * Bx) * inv_gamma;
                work_t Py = (alpha_p * Ay + alpha_q * By) * inv_gamma;
                work_t Pz = (alpha_p * Az + alpha_q * Bz) * inv_gamma;
                work_t v_prefactor = c_pq * work_t(2) * work_t(ATV::units::pi)
                                     * inv_gamma * K_AB;
                for (int a = 0; a < natm; ++a) {
                    work_t Cx = static_cast<work_t>(atom_pos[a * 3 + 0]);
                    work_t Cy = static_cast<work_t>(atom_pos[a * 3 + 1]);
                    work_t Cz = static_cast<work_t>(atom_pos[a * 3 + 2]);
                    work_t PC2 = dist2_device(Px, Py, Pz, Cx, Cy, Cz);
                    acc_V += v_prefactor * work_t(-atom_Z[a])
                             * ATV::boys_f0(gamma * PC2);
                }
            }
        }
        int ij = si + sj * n;
        int ji = sj + si * n;
        S[ij] = acc_S;
        T[ij] = acc_T;
        V[ij] = acc_V;
        if (si != sj) {
            S[ji] = acc_S;
            T[ji] = acc_T;
            V[ji] = acc_V;
        }
    }
}

void launch_1e_integrals(prec_t* S, prec_t* T, prec_t* V,
                         const prec_t* shell_alpha, const prec_t* shell_coeff,
                         const prec_t* shell_centers, const int* atom_Z,
                         const prec_t* atom_pos, int n_shells, int n_prim,
                         int natm, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    int n_pairs = n_shells * (n_shells + 1) / 2;
    int threads = cuda::kDefaultBlock;
    int blocks = cuda::grid_size(n_pairs, threads);
    compute_1e_kernel<<<blocks, threads, 0, s>>>(
        S, T, V, shell_alpha, shell_coeff, shell_centers, atom_Z, atom_pos,
        n_shells, n_prim, natm);
}

// Wave 7: register-heavy MD/Boys ERI — launch_bounds asks for ≥2 resident
// blocks/SM (choose-launch-configuration). Grid-stride loop below keeps
// work covered when occupancy caps the active grid.
__global__ void __launch_bounds__(256, 2)
    compute_eri_kernel(prec_t* ERI, const prec_t* shell_alpha,
                       const prec_t* shell_coeff, const prec_t* shell_centers,
                       int n, int n_prim) {
    int n4 = n * n * n * n;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int q_idx = idx; q_idx < n4; q_idx += blockDim.x * gridDim.x) {
        int sl = q_idx / (n * n * n);
        int rem = q_idx % (n * n * n);
        int sk = rem / (n * n);
        rem = rem % (n * n);
        int sj = rem / n;
        int si = rem % n;

        work_t Ax = static_cast<work_t>(shell_centers[si * 3]),
               Ay = static_cast<work_t>(shell_centers[si * 3 + 1]),
               Az = static_cast<work_t>(shell_centers[si * 3 + 2]);
        work_t Bx = static_cast<work_t>(shell_centers[sj * 3]),
               By = static_cast<work_t>(shell_centers[sj * 3 + 1]),
               Bz = static_cast<work_t>(shell_centers[sj * 3 + 2]);
        work_t Cx = static_cast<work_t>(shell_centers[sk * 3]),
               Cy = static_cast<work_t>(shell_centers[sk * 3 + 1]),
               Cz = static_cast<work_t>(shell_centers[sk * 3 + 2]);
        work_t Dx = static_cast<work_t>(shell_centers[sl * 3]),
               Dy = static_cast<work_t>(shell_centers[sl * 3 + 1]),
               Dz = static_cast<work_t>(shell_centers[sl * 3 + 2]);
        work_t R2_AB = dist2_device(Ax, Ay, Az, Bx, By, Bz);
        work_t R2_CD = dist2_device(Cx, Cy, Cz, Dx, Dy, Dz);
        work_t result = 0;
        for (int p = 0; p < n_prim; ++p) {
            work_t a_p = static_cast<work_t>(shell_alpha[si * n_prim + p]);
            work_t c_p = static_cast<work_t>(shell_coeff[si * n_prim + p]);
            for (int q = 0; q < n_prim; ++q) {
                work_t a_q = static_cast<work_t>(shell_alpha[sj * n_prim + q]);
                work_t c_q = static_cast<work_t>(shell_coeff[sj * n_prim + q]);
                work_t gamma1 = a_p + a_q;
                work_t mu1 = a_p * a_q / gamma1;
                work_t K_AB = exp(-mu1 * R2_AB);
                work_t inv_g1 = work_t(1) / gamma1;
                work_t Px = (a_p * Ax + a_q * Bx) * inv_g1;
                work_t Py = (a_p * Ay + a_q * By) * inv_g1;
                work_t Pz = (a_p * Az + a_q * Bz) * inv_g1;
                for (int r = 0; r < n_prim; ++r) {
                    work_t a_r =
                        static_cast<work_t>(shell_alpha[sk * n_prim + r]);
                    work_t c_r =
                        static_cast<work_t>(shell_coeff[sk * n_prim + r]);
                    for (int s = 0; s < n_prim; ++s) {
                        work_t a_s =
                            static_cast<work_t>(shell_alpha[sl * n_prim + s]);
                        work_t c_s =
                            static_cast<work_t>(shell_coeff[sl * n_prim + s]);
                        work_t gamma2 = a_r + a_s;
                        work_t mu2 = a_r * a_s / gamma2;
                        work_t K_CD = exp(-mu2 * R2_CD);
                        work_t inv_g2 = work_t(1) / gamma2;
                        work_t Qx = (a_r * Cx + a_s * Dx) * inv_g2;
                        work_t Qy = (a_r * Cy + a_s * Dy) * inv_g2;
                        work_t Qz = (a_r * Cz + a_s * Dz) * inv_g2;
                        work_t rho = gamma1 * gamma2 / (gamma1 + gamma2);
                        work_t PQ2 = dist2_device(Px, Py, Pz, Qx, Qy, Qz);
                        work_t pi_val = work_t(ATV::units::pi);
                        result += c_p * c_q * c_r * c_s * work_t(2) * pi_val
                                  * pi_val * sqrt(pi_val)
                                  / (gamma1 * gamma2 * sqrt(gamma1 + gamma2))
                                  * K_AB * K_CD * ATV::boys_f0(rho * PQ2);
                    }
                }
            }
        }
        ERI[q_idx] = result;
    }
}

void launch_eri(prec_t* ERI, const prec_t* shell_alpha,
                const prec_t* shell_coeff, const prec_t* shell_centers,
                int n_shells, int n_prim, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    const int n4 = n_shells * n_shells * n_shells * n_shells;
    constexpr int threads = cuda::kDefaultBlock;
    // Occupancy-capped grid + grid-stride in the kernel (skill: choose-launch).
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, compute_eri_kernel, threads, /*dynSmem=*/0);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    const int cover = cuda::grid_size(n4, threads);
    const int occ_cap =
        (blocks_per_sm > 0 && sm_count > 0) ? blocks_per_sm * sm_count : cover;
    // Prefer full cover for small tensors; cap to a few waves over occupancy
    // so large n^4 does not flood the launch queue.
    const int blocks = (cover <= occ_cap)
                           ? cover
                           : ((occ_cap * 4 < cover) ? occ_cap * 4 : cover);
    compute_eri_kernel<<<blocks, threads, 0, s>>>(
        ERI, shell_alpha, shell_coeff, shell_centers, n_shells, n_prim);
}

// ── Wave 4: s+p AO-indexed kernels ────────────────────────────────────

__global__ void compute_1e_sp_kernel_full(
    prec_t* S, prec_t* T, prec_t* V, const prec_t* shell_alpha,
    const prec_t* shell_coeff, const prec_t* shell_centers, const int* shell_L,
    const int* ao_shell, const int* ao_cart, const int* atom_Z,
    const prec_t* atom_pos, int n_ao, int n_prim, int natm) {
    const int n_pairs = n_ao * (n_ao + 1) / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int pair = idx; pair < n_pairs; pair += blockDim.x * gridDim.x) {
        int mu = (int)((-1.0 + sqrt(1.0 + 8.0 * pair)) * 0.5);
        int nu = pair - mu * (mu + 1) / 2;
        if (nu > mu) {
            ++mu;
            nu = pair - mu * (mu + 1) / 2;
        }

        const int si = ao_shell[mu];
        const int sj = ao_shell[nu];
        int a_ang[3], b_ang[3];
        cart_ang_device(shell_L[si], ao_cart[mu], a_ang);
        cart_ang_device(shell_L[sj], ao_cart[nu], b_ang);

        work_t A[3] = {static_cast<work_t>(shell_centers[si * 3 + 0]),
                       static_cast<work_t>(shell_centers[si * 3 + 1]),
                       static_cast<work_t>(shell_centers[si * 3 + 2])};
        work_t B[3] = {static_cast<work_t>(shell_centers[sj * 3 + 0]),
                       static_cast<work_t>(shell_centers[sj * 3 + 1]),
                       static_cast<work_t>(shell_centers[sj * 3 + 2])};

        work_t acc_S = 0, acc_T = 0, acc_V = 0;
        for (int p = 0; p < n_prim; ++p) {
            work_t alpha_p = static_cast<work_t>(shell_alpha[si * n_prim + p]);
            work_t c_p = static_cast<work_t>(shell_coeff[si * n_prim + p]);
            for (int q = 0; q < n_prim; ++q) {
                work_t alpha_q =
                    static_cast<work_t>(shell_alpha[sj * n_prim + q]);
                work_t c_q = static_cast<work_t>(shell_coeff[sj * n_prim + q]);
                work_t c_pq = c_p * c_q;

                acc_S += c_pq
                         * overlap_prim_3d_device(a_ang, b_ang, alpha_p, A,
                                                  alpha_q, B);
                acc_T += c_pq
                         * kinetic_prim_3d_device(a_ang, b_ang, alpha_p, A,
                                                  alpha_q, B);

                const work_t gamma = alpha_p + alpha_q;
                const work_t mu_ab = alpha_p * alpha_q / gamma;
                work_t R2 = dist2_device(A[0], A[1], A[2], B[0], B[1], B[2]);
                work_t K_AB = exp(-mu_ab * R2);
                work_t inv_g = work_t(1) / gamma;
                work_t PA[3], PB[3], P[3];
                for (int d = 0; d < 3; ++d) {
                    P[d] = (alpha_p * A[d] + alpha_q * B[d]) * inv_g;
                    PA[d] = P[d] - A[d];
                    PB[d] = P[d] - B[d];
                }
                for (int a = 0; a < natm; ++a) {
                    work_t Cpos[3] = {static_cast<work_t>(atom_pos[a * 3 + 0]),
                                      static_cast<work_t>(atom_pos[a * 3 + 1]),
                                      static_cast<work_t>(atom_pos[a * 3 + 2])};
                    work_t PC[3] = {P[0] - Cpos[0], P[1] - Cpos[1],
                                    P[2] - Cpos[2]};
                    acc_V += c_pq * work_t(-atom_Z[a])
                             * nuclear_prim_device(a_ang, b_ang, gamma, PA, PB,
                                                   PC, K_AB);
                }
            }
        }

        int ij = mu + nu * n_ao;
        int ji = nu + mu * n_ao;
        S[ij] = acc_S;
        T[ij] = acc_T;
        V[ij] = acc_V;
        if (mu != nu) {
            S[ji] = acc_S;
            T[ji] = acc_T;
            V[ji] = acc_V;
        }
    }
}

void launch_1e_integrals_sp(double* S, double* T, double* V,
                            const double* shell_alpha,
                            const double* shell_coeff,
                            const double* shell_centers, const int* shell_L,
                            const int* /*shell_ao_start*/, const int* ao_shell,
                            const int* ao_cart, const int* atom_Z,
                            const double* atom_pos, int /*n_shells*/, int n_ao,
                            int n_prim, int natm, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    int n_pairs = n_ao * (n_ao + 1) / 2;
    int threads = cuda::kDefaultBlock;
    int blocks = cuda::grid_size(n_pairs, threads);
    compute_1e_sp_kernel_full<<<blocks, threads, 0, s>>>(
        S, T, V, shell_alpha, shell_coeff, shell_centers, shell_L, ao_shell,
        ao_cart, atom_Z, atom_pos, n_ao, n_prim, natm);
}

// MINAO setup: one warp owns a variable-length radial contraction. Its
// primitive-pair self-overlap reduction shares the existing OS primitive.
// Tables contain s/p only; cart=0 matches the CPU normalization convention.
__global__ void normalize_minao_sp_kernel(const double* alpha,
                                          double* coefficient,
                                          const int* offset, const int* angular,
                                          int n_contractions, int* bad) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int shell =
        static_cast<int>(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    if (shell >= n_contractions)
        return;
    const int begin = offset[shell], np = offset[shell + 1] - begin;
    for (int p = lane; p < np; p += 32) {
        const double a = alpha[begin + p];
        const double norm =
            angular[shell] == 0 ? ATV::phys_norm_s(a) : ATV::phys_norm_p(a);
        coefficient[begin + p] *= norm;
    }
    __syncwarp();
    int ang[3];
    cart_ang_device(angular[shell], 0, ang);
    const work_t origin[3] = {0, 0, 0};
    double self = 0;
    for (int pq = lane; pq < np * np; pq += 32) {
        const int p = begin + pq / np, q = begin + pq % np;
        self += coefficient[p] * coefficient[q]
                * overlap_prim_3d_device(ang, ang, alpha[p], origin, alpha[q],
                                         origin);
    }
    for (int delta = 16; delta > 0; delta /= 2)
        self += __shfl_down_sync(0xffffffffu, self, delta);
    self = __shfl_sync(0xffffffffu, self, 0);
    if (!(self > 0) || !isfinite(self)) {
        if (lane == 0)
            atomicExch(bad, 1);
        return;
    }
    const double inverse_norm = 1.0 / sqrt(self);
    for (int p = lane; p < np; p += 32)
        coefficient[begin + p] *= inverse_norm;
}

void launch_normalize_minao_sp(const double* alpha, double* coefficient,
                               const int* offset, const int* angular,
                               int n_contractions, int* bad, void* stream) {
    constexpr int threads = 128;
    normalize_minao_sp_kernel<<<cuda::grid_size(n_contractions, threads / 32),
                                threads, 0,
                                static_cast<cudaStream_t>(stream)>>>(
        alpha, coefficient, offset, angular, n_contractions, bad);
}

// Each thread contracts one target--MINAO AO pair (at most 3*14 primitives
// for current tables). Consecutive threads publish consecutive target rows.
__global__ void overlap_minao_sp_kernel(
    double* B, const double* target_alpha, const double* target_coefficient,
    const double* target_centers, const int* target_angular,
    const int* target_ao_shell, const int* target_ao_cart, int n_ao,
    int target_n_prim, const double* minao_alpha,
    const double* minao_coefficient, const int* minao_offset,
    const int* minao_angular, const int* minao_atom, const int* minao_ao_shell,
    const int* minao_ao_cart, int n_minao, const double* atom_pos) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n_ao * n_minao)
        return;
    const int mu = idx % n_ao, j = idx / n_ao;
    const int si = target_ao_shell[mu], sj = minao_ao_shell[j];
    int aa[3], bb[3];
    cart_ang_device(target_angular[si], target_ao_cart[mu], aa);
    cart_ang_device(minao_angular[sj], minao_ao_cart[j], bb);
    const int atom = minao_atom[sj];
    const work_t A[3] = {target_centers[si * 3], target_centers[si * 3 + 1],
                         target_centers[si * 3 + 2]};
    const work_t C[3] = {atom_pos[atom * 3], atom_pos[atom * 3 + 1],
                         atom_pos[atom * 3 + 2]};
    work_t value = 0;
    for (int p = 0; p < target_n_prim; ++p) {
        const int k = si * target_n_prim + p;
        for (int q = minao_offset[sj]; q < minao_offset[sj + 1]; ++q)
            value += target_coefficient[k] * minao_coefficient[q]
                     * overlap_prim_3d_device(aa, bb, target_alpha[k], A,
                                              minao_alpha[q], C);
    }
    B[idx] = value;
}

void launch_overlap_minao_sp(
    double* B, const double* target_alpha, const double* target_coefficient,
    const double* target_centers, const int* target_angular,
    const int* target_ao_shell, const int* target_ao_cart, int n_ao,
    int target_n_prim, const double* minao_alpha,
    const double* minao_coefficient, const int* minao_offset,
    const int* minao_angular, const int* minao_atom, const int* minao_ao_shell,
    const int* minao_ao_cart, int n_minao, const double* atom_pos,
    void* stream) {
    constexpr int threads = 128;
    overlap_minao_sp_kernel<<<cuda::grid_size(n_ao * n_minao, threads), threads,
                              0, static_cast<cudaStream_t>(stream)>>>(
        B, target_alpha, target_coefficient, target_centers, target_angular,
        target_ao_shell, target_ao_cart, n_ao, target_n_prim, minao_alpha,
        minao_coefficient, minao_offset, minao_angular, minao_atom,
        minao_ao_shell, minao_ao_cart, n_minao, atom_pos);
}

// Wave 7: s/p ERI is register-heavy (MD raise to d, nested 4-prim loops).
// __launch_bounds__(256, 2) targets ≥2 blocks/SM occupancy.
__global__ void __launch_bounds__(256, 2)
    compute_eri_sp_kernel(prec_t* ERI, const prec_t* shell_alpha,
                          const prec_t* shell_coeff,
                          const prec_t* shell_centers, const int* shell_L,
                          const int* ao_shell, const int* ao_cart, int n_ao,
                          int n_prim) {
    const int n4 = n_ao * n_ao * n_ao * n_ao;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int q = idx; q < n4; q += blockDim.x * gridDim.x) {
        int sig = q / (n_ao * n_ao * n_ao);
        int rem = q % (n_ao * n_ao * n_ao);
        int lam = rem / (n_ao * n_ao);
        rem = rem % (n_ao * n_ao);
        int nu = rem / n_ao;
        int mu = rem % n_ao;

        // One owner per real-AO Coulomb symmetry orbit. All eight writes
        // below belong to this owner, including coincident-index cases.
        if (mu > nu || lam > sig
            || nu * (nu + 1) / 2 + mu > sig * (sig + 1) / 2 + lam)
            continue;
        const int sa = ao_shell[mu], sb = ao_shell[nu];
        const int sc = ao_shell[lam], sd = ao_shell[sig];
        int a_ang[3], b_ang[3], c_ang[3], d_ang[3];
        cart_ang_device(shell_L[sa], ao_cart[mu], a_ang);
        cart_ang_device(shell_L[sb], ao_cart[nu], b_ang);
        cart_ang_device(shell_L[sc], ao_cart[lam], c_ang);
        cart_ang_device(shell_L[sd], ao_cart[sig], d_ang);

        work_t A[3] = {static_cast<work_t>(shell_centers[sa * 3 + 0]),
                       static_cast<work_t>(shell_centers[sa * 3 + 1]),
                       static_cast<work_t>(shell_centers[sa * 3 + 2])};
        work_t B[3] = {static_cast<work_t>(shell_centers[sb * 3 + 0]),
                       static_cast<work_t>(shell_centers[sb * 3 + 1]),
                       static_cast<work_t>(shell_centers[sb * 3 + 2])};
        work_t C[3] = {static_cast<work_t>(shell_centers[sc * 3 + 0]),
                       static_cast<work_t>(shell_centers[sc * 3 + 1]),
                       static_cast<work_t>(shell_centers[sc * 3 + 2])};
        work_t D[3] = {static_cast<work_t>(shell_centers[sd * 3 + 0]),
                       static_cast<work_t>(shell_centers[sd * 3 + 1]),
                       static_cast<work_t>(shell_centers[sd * 3 + 2])};

        work_t result = 0;
        for (int p = 0; p < n_prim; ++p) {
            work_t ap = static_cast<work_t>(shell_alpha[sa * n_prim + p]);
            work_t cp = static_cast<work_t>(shell_coeff[sa * n_prim + p]);
            for (int qq = 0; qq < n_prim; ++qq) {
                work_t aq = static_cast<work_t>(shell_alpha[sb * n_prim + qq]);
                work_t cq = static_cast<work_t>(shell_coeff[sb * n_prim + qq]);
                for (int r = 0; r < n_prim; ++r) {
                    work_t ar =
                        static_cast<work_t>(shell_alpha[sc * n_prim + r]);
                    work_t cr =
                        static_cast<work_t>(shell_coeff[sc * n_prim + r]);
                    for (int s = 0; s < n_prim; ++s) {
                        work_t as =
                            static_cast<work_t>(shell_alpha[sd * n_prim + s]);
                        work_t cs =
                            static_cast<work_t>(shell_coeff[sd * n_prim + s]);
                        result +=
                            cp * cq * cr * cs
                            * eri_prim_md_device(a_ang, b_ang, c_ang, d_ang, ap,
                                                 A, aq, B, ar, C, as, D);
                    }
                }
            }
        }
        const int n2 = n_ao * n_ao;
        const int n3 = n2 * n_ao;
        ERI[mu + nu * n_ao + lam * n2 + sig * n3] = result;
        ERI[nu + mu * n_ao + lam * n2 + sig * n3] = result;
        ERI[mu + nu * n_ao + sig * n2 + lam * n3] = result;
        ERI[nu + mu * n_ao + sig * n2 + lam * n3] = result;
        ERI[lam + sig * n_ao + mu * n2 + nu * n3] = result;
        ERI[sig + lam * n_ao + mu * n2 + nu * n3] = result;
        ERI[lam + sig * n_ao + nu * n2 + mu * n3] = result;
        ERI[sig + lam * n_ao + nu * n2 + mu * n3] = result;
    }
}

void launch_eri_sp(double* ERI, const double* shell_alpha,
                   const double* shell_coeff, const double* shell_centers,
                   const int* shell_L, const int* /*shell_ao_start*/,
                   const int* ao_shell, const int* ao_cart, int /*n_shells*/,
                   int n_ao, int n_prim, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    const int n4 = n_ao * n_ao * n_ao * n_ao;
    constexpr int threads = cuda::kDefaultBlock;
    // Occupancy-capped grid + grid-stride (skill: choose-launch-configuration).
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, compute_eri_sp_kernel, threads, /*dynSmem=*/0);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    const int cover = cuda::grid_size(n4, threads);
    const int occ_cap =
        (blocks_per_sm > 0 && sm_count > 0) ? blocks_per_sm * sm_count : cover;
    const int blocks = (cover <= occ_cap)
                           ? cover
                           : ((occ_cap * 4 < cover) ? occ_cap * 4 : cover);
    compute_eri_sp_kernel<<<blocks, threads, 0, s>>>(
        ERI, shell_alpha, shell_coeff, shell_centers, shell_L, ao_shell,
        ao_cart, n_ao, n_prim);
}

// ── Wave 5: pure-s RHF forces + W build ───────────────────────────────

/// W_μν = 2 Σ_{i<n_occ} ε_i C_μi C_νi  (col-major C, W).
__global__ void build_W_rhf_kernel(prec_t* __restrict__ W,
                                   const prec_t* __restrict__ C,
                                   const prec_t* __restrict__ eig, int n,
                                   int n_occ) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int nn = n * n;
    if (idx >= nn)
        return;
    const int mu = idx % n;
    const int nu = idx / n;
    work_t acc = work_t(0);
    for (int i = 0; i < n_occ; ++i) {
        const work_t ci_mu = static_cast<work_t>(C[mu + i * n]);
        const work_t ci_nu = static_cast<work_t>(C[nu + i * n]);
        acc += static_cast<work_t>(eig[i]) * ci_mu * ci_nu;
    }
    W[idx] = static_cast<prec_t>(work_t(2) * acc);
}

void launch_build_W_rhf(double* W, const double* C, const double* eig, int n_ao,
                        int n_occ, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    const int nn = n_ao * n_ao;
    int threads = cuda::kDefaultBlock;
    int blocks = cuda::grid_size(nn, threads);
    build_W_rhf_kernel<<<blocks, threads, 0, s>>>(W, C, eig, n_ao, n_occ);
}

// Contracted pure-s 1e derivatives w.r.t. Gaussian center of shell `which`
// (0 = bra center A, 1 = ket center B), all 3 axes. Uses raising on s → p.
__device__ void
pure_s_1e_deriv_center(int which, // 0 = d/dA, 1 = d/dB
                       const prec_t* alpha_i, const prec_t* coeff_i,
                       const work_t Ai[3], const prec_t* alpha_j,
                       const prec_t* coeff_j, const work_t Bj[3], int n_prim,
                       const int* atom_Z, const prec_t* atom_pos, int natm,
                       work_t dS[3], work_t dT[3], work_t dV[3]) {
    for (int d = 0; d < 3; ++d) {
        dS[d] = dT[d] = dV[d] = work_t(0);
    }
    int zero[3] = {0, 0, 0};
    for (int p = 0; p < n_prim; ++p) {
        work_t ap = static_cast<work_t>(alpha_i[p]);
        work_t cp = static_cast<work_t>(coeff_i[p]);
        for (int q = 0; q < n_prim; ++q) {
            work_t aq = static_cast<work_t>(alpha_j[q]);
            work_t cq = static_cast<work_t>(coeff_j[q]);
            work_t cpq = cp * cq;
            const work_t alpha_raise = (which == 0) ? ap : aq;
            for (int d = 0; d < 3; ++d) {
                int a_ang[3] = {0, 0, 0};
                int b_ang[3] = {0, 0, 0};
                if (which == 0)
                    a_ang[d] = 1;
                else
                    b_ang[d] = 1;
                // Raising: 2 α * I(raised); pure-s has no lowering.
                work_t Sp =
                    overlap_prim_3d_device(a_ang, b_ang, ap, Ai, aq, Bj);
                work_t Tp =
                    kinetic_prim_3d_device(a_ang, b_ang, ap, Ai, aq, Bj);
                dS[d] += cpq * work_t(2) * alpha_raise * Sp;
                dT[d] += cpq * work_t(2) * alpha_raise * Tp;

                // Nuclear attraction: raise Gaussian center only.
                work_t gamma = ap + aq;
                work_t mu_ab = ap * aq / gamma;
                work_t R2 =
                    dist2_device(Ai[0], Ai[1], Ai[2], Bj[0], Bj[1], Bj[2]);
                work_t K_AB = exp(-mu_ab * R2);
                work_t inv_g = work_t(1) / gamma;
                work_t PA[3], PB[3], P[3];
                for (int e = 0; e < 3; ++e) {
                    P[e] = (ap * Ai[e] + aq * Bj[e]) * inv_g;
                    PA[e] = P[e] - Ai[e];
                    PB[e] = P[e] - Bj[e];
                }
                work_t Vsum = work_t(0);
                for (int a = 0; a < natm; ++a) {
                    work_t Cpos[3] = {static_cast<work_t>(atom_pos[a * 3 + 0]),
                                      static_cast<work_t>(atom_pos[a * 3 + 1]),
                                      static_cast<work_t>(atom_pos[a * 3 + 2])};
                    work_t PC[3] = {P[0] - Cpos[0], P[1] - Cpos[1],
                                    P[2] - Cpos[2]};
                    // Raised nuclear primitive (no −Z yet).
                    work_t Vprim = nuclear_prim_device(a_ang, b_ang, gamma, PA,
                                                       PB, PC, K_AB);
                    Vsum +=
                        work_t(-atom_Z[a]) * work_t(2) * alpha_raise * Vprim;
                }
                dV[d] += cpq * Vsum;
            }
        }
    }
}

// Extra nuclear-attraction term: ∂/∂R_C of −Z_C V when the operator center moves.
// For pure-s: dV/dC_d = (2π/γ) K_AB * R^0 with raised Hermite index on axis d
// times −Z_C — use nuclear_prim with a zero ang and differentiate via PC.
// Equivalent form via raising on the operator (Helgaker): for pure-s
//   d/dC_d V_ss = −Z * (2π/γ) K * R_{e_d}^0
// which equals nuclear_prim with one side raised? Use finite formula:
// d/dC_d of F0(γ|P−C|²) chain rule.
__device__ void pure_s_nuclear_operator_deriv(
    const prec_t* alpha_i, const prec_t* coeff_i, const work_t Ai[3],
    const prec_t* alpha_j, const prec_t* coeff_j, const work_t Bj[3],
    int n_prim, int C_atom, const prec_t* atom_pos, int Zc, work_t dV_dC[3]) {
    for (int d = 0; d < 3; ++d)
        dV_dC[d] = work_t(0);
    if (Zc == 0)
        return;
    work_t Cpos[3] = {static_cast<work_t>(atom_pos[C_atom * 3 + 0]),
                      static_cast<work_t>(atom_pos[C_atom * 3 + 1]),
                      static_cast<work_t>(atom_pos[C_atom * 3 + 2])};
    int zero[3] = {0, 0, 0};
    for (int p = 0; p < n_prim; ++p) {
        work_t ap = static_cast<work_t>(alpha_i[p]);
        work_t cp = static_cast<work_t>(coeff_i[p]);
        for (int q = 0; q < n_prim; ++q) {
            work_t aq = static_cast<work_t>(alpha_j[q]);
            work_t cq = static_cast<work_t>(coeff_j[q]);
            work_t cpq = cp * cq;
            work_t gamma = ap + aq;
            work_t mu_ab = ap * aq / gamma;
            work_t R2 = dist2_device(Ai[0], Ai[1], Ai[2], Bj[0], Bj[1], Bj[2]);
            work_t K_AB = exp(-mu_ab * R2);
            work_t inv_g = work_t(1) / gamma;
            work_t PA[3], PB[3], P[3];
            for (int e = 0; e < 3; ++e) {
                P[e] = (ap * Ai[e] + aq * Bj[e]) * inv_g;
                PA[e] = P[e] - Ai[e];
                PB[e] = P[e] - Bj[e];
            }
            work_t PC[3] = {P[0] - Cpos[0], P[1] - Cpos[1], P[2] - Cpos[2]};
            // V = −Z * nuclear_prim(ss); d/dC of nuclear_prim via MD R-table.
            // For ss: nuclear_prim = (2π/γ) K F0(γ|PC|²)
            // d/dC_d of F0(T), T=γ|PC|²: F0' = −F1, dT/dC_d = −2γ PC_d
            // ⇒ dV/dC_d = −Z * (2π/γ) K * (−F1) * (−2γ PC_d) wait:
            // dF0/dC_d = F0'(T) * dT/dC_d = (−F1) * (−2 γ PC_d) = 2γ PC_d F1
            // V_ss = −Z * (2π/γ) K * F0  ⇒ dV/dC_d = −Z*(2π/γ)*K*2γ PC_d F1
            //       = −Z * 4π K * PC_d * F1
            work_t PC2 = PC[0] * PC[0] + PC[1] * PC[1] + PC[2] * PC[2];
            work_t T = gamma * PC2;
            work_t F1 = ATV::boys_fn(1, T);
            work_t pref = cpq * work_t(-Zc) * work_t(4) * work_t(ATV::units::pi)
                          * K_AB * F1;
            for (int d = 0; d < 3; ++d)
                dV_dC[d] += pref * PC[d];
            (void)zero;
            (void)PA;
            (void)PB;
        }
    }
}

__device__ void pure_s_eri_deriv_center(int which, // 0..3 = A,B,C,D
                                        const prec_t* aA, const prec_t* cA,
                                        const work_t A[3], const prec_t* aB,
                                        const prec_t* cB, const work_t B[3],
                                        const prec_t* aC, const prec_t* cC,
                                        const work_t C[3], const prec_t* aD,
                                        const prec_t* cD, const work_t D[3],
                                        int n_prim, work_t dERI[3]) {
    for (int d = 0; d < 3; ++d)
        dERI[d] = work_t(0);
    for (int p = 0; p < n_prim; ++p) {
        work_t ap = static_cast<work_t>(aA[p]);
        work_t cp = static_cast<work_t>(cA[p]);
        for (int q = 0; q < n_prim; ++q) {
            work_t aq = static_cast<work_t>(aB[q]);
            work_t cq = static_cast<work_t>(cB[q]);
            for (int r = 0; r < n_prim; ++r) {
                work_t ar = static_cast<work_t>(aC[r]);
                work_t cr = static_cast<work_t>(cC[r]);
                for (int s = 0; s < n_prim; ++s) {
                    work_t as = static_cast<work_t>(aD[s]);
                    work_t cs = static_cast<work_t>(cD[s]);
                    work_t c4 = cp * cq * cr * cs;
                    work_t alpha_raise = ap;
                    if (which == 1)
                        alpha_raise = aq;
                    else if (which == 2)
                        alpha_raise = ar;
                    else if (which == 3)
                        alpha_raise = as;
                    for (int d = 0; d < 3; ++d) {
                        int aa[3] = {0, 0, 0}, bb[3] = {0, 0, 0},
                            cc[3] = {0, 0, 0}, dd[3] = {0, 0, 0};
                        if (which == 0)
                            aa[d] = 1;
                        else if (which == 1)
                            bb[d] = 1;
                        else if (which == 2)
                            cc[d] = 1;
                        else
                            dd[d] = 1;
                        work_t ep = eri_prim_md_device(aa, bb, cc, dd, ap, A,
                                                       aq, B, ar, C, as, D);
                        dERI[d] += c4 * work_t(2) * alpha_raise * ep;
                    }
                }
            }
        }
    }
}

// Ha/Bohr → eV/Å: F_eV/Å = F_Ha/Bohr * kHartreeInEv * kAngstromInBohr
// (kAngstromInBohr = Bohr per Å ≈ 1.8897; scale ≈ 51.42 — same as
// units::kHartreePerBohrInEvPerAngstrom / GradientResult::FromHartreeBohr).
__device__ __forceinline__ work_t ha_bohr_to_ev_ang(work_t g_ha_bohr) {
    return g_ha_bohr
           * static_cast<work_t>(ATV::units::kHartreePerBohrInEvPerAngstrom);
}

// Force accumulation uses double atomics (sm_60+; GH200 sm_90). Gradients
// land in Ha/Bohr; a final scale kernel converts to −eV/Å.

__device__ __forceinline__ void atomic_add_grad3(real_t* __restrict__ fx,
                                                 real_t* __restrict__ fy,
                                                 real_t* __restrict__ fz,
                                                 int atom, work_t gx, work_t gy,
                                                 work_t gz) {
    if (atom < 0)
        return;
    atomicAdd(&fx[atom], static_cast<real_t>(gx));
    atomicAdd(&fy[atom], static_cast<real_t>(gy));
    atomicAdd(&fz[atom], static_cast<real_t>(gz));
}

__global__ void zero_forces_kernel(real_t* __restrict__ fx,
                                   real_t* __restrict__ fy,
                                   real_t* __restrict__ fz, int natm) {
    const int A = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (A < natm) {
        fx[A] = real_t(0);
        fy[A] = real_t(0);
        fz[A] = real_t(0);
    }
}

/// Nuclear repulsion ∂V_nuc/∂R [Ha/Bohr] — one thread per ordered pair (A,B).
__global__ void
rhf_forces_nuc_kernel(real_t* __restrict__ fx, real_t* __restrict__ fy,
                      real_t* __restrict__ fz, const int* __restrict__ atom_Z,
                      const prec_t* __restrict__ atom_pos, int natm) {
    const int npair = natm * (natm - 1) / 2;
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= npair)
        return;
    // Decode upper triangle: A < B
    int A = 0, B = 1, rem = tid;
    for (int a = 0; a < natm; ++a) {
        const int nright = natm - a - 1;
        if (rem < nright) {
            A = a;
            B = a + 1 + rem;
            break;
        }
        rem -= nright;
    }
    work_t Ra[3] = {static_cast<work_t>(atom_pos[A * 3 + 0]),
                    static_cast<work_t>(atom_pos[A * 3 + 1]),
                    static_cast<work_t>(atom_pos[A * 3 + 2])};
    work_t Rb[3] = {static_cast<work_t>(atom_pos[B * 3 + 0]),
                    static_cast<work_t>(atom_pos[B * 3 + 1]),
                    static_cast<work_t>(atom_pos[B * 3 + 2])};
    work_t d[3] = {Ra[0] - Rb[0], Ra[1] - Rb[1], Ra[2] - Rb[2]};
    work_t r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    if (r2 < work_t(1e-30))
        return;
    work_t r = sqrt(r2);
    work_t pref = static_cast<work_t>(atom_Z[A] * atom_Z[B]) / (r2 * r);
    // ∂(ZaZb/R)/∂R_A = −ZaZb (R_A−R_B)/R³ ; ∂/∂R_B = −∂/∂R_A
    work_t gA[3] = {-pref * d[0], -pref * d[1], -pref * d[2]};
    atomic_add_grad3(fx, fy, fz, A, gA[0], gA[1], gA[2]);
    atomic_add_grad3(fx, fy, fz, B, -gA[0], -gA[1], -gA[2]);
}

/// One-electron + Pulay + nuclear-operator deriv — one thread per (μ,ν).
__global__ void rhf_forces_1e_kernel(
    real_t* __restrict__ fx, real_t* __restrict__ fy, real_t* __restrict__ fz,
    const prec_t* __restrict__ P, const prec_t* __restrict__ W,
    const prec_t* __restrict__ shell_alpha,
    const prec_t* __restrict__ shell_coeff,
    const prec_t* __restrict__ shell_centers,
    const int* __restrict__ shell_atom, const int* __restrict__ atom_Z,
    const prec_t* __restrict__ atom_pos, int n, int n_prim, int natm) {
    const int nn = n * n;
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= nn)
        return;
    const int mu = idx % n;
    const int nu = idx / n;
    const work_t Pmn = static_cast<work_t>(P[mu + nu * n]);
    const work_t Wmn = static_cast<work_t>(W[mu + nu * n]);
    work_t Ai[3] = {static_cast<work_t>(shell_centers[mu * 3 + 0]),
                    static_cast<work_t>(shell_centers[mu * 3 + 1]),
                    static_cast<work_t>(shell_centers[mu * 3 + 2])};
    work_t Bj[3] = {static_cast<work_t>(shell_centers[nu * 3 + 0]),
                    static_cast<work_t>(shell_centers[nu * 3 + 1]),
                    static_cast<work_t>(shell_centers[nu * 3 + 2])};
    const prec_t* a_i = shell_alpha + mu * n_prim;
    const prec_t* c_i = shell_coeff + mu * n_prim;
    const prec_t* a_j = shell_alpha + nu * n_prim;
    const prec_t* c_j = shell_coeff + nu * n_prim;
    const int Amu = shell_atom[mu];
    const int Anu = shell_atom[nu];

    {
        work_t dS[3], dT[3], dV[3];
        pure_s_1e_deriv_center(0, a_i, c_i, Ai, a_j, c_j, Bj, n_prim, atom_Z,
                               atom_pos, natm, dS, dT, dV);
        work_t gx = Pmn * (dT[0] + dV[0]) - Wmn * dS[0];
        work_t gy = Pmn * (dT[1] + dV[1]) - Wmn * dS[1];
        work_t gz = Pmn * (dT[2] + dV[2]) - Wmn * dS[2];
        atomic_add_grad3(fx, fy, fz, Amu, gx, gy, gz);
    }
    {
        work_t dS[3], dT[3], dV[3];
        pure_s_1e_deriv_center(1, a_i, c_i, Ai, a_j, c_j, Bj, n_prim, atom_Z,
                               atom_pos, natm, dS, dT, dV);
        work_t gx = Pmn * (dT[0] + dV[0]) - Wmn * dS[0];
        work_t gy = Pmn * (dT[1] + dV[1]) - Wmn * dS[1];
        work_t gz = Pmn * (dT[2] + dV[2]) - Wmn * dS[2];
        atomic_add_grad3(fx, fy, fz, Anu, gx, gy, gz);
    }
    // Nuclear operator ∂/∂R_C for every atom C (cheap pure-s form).
    for (int C = 0; C < natm; ++C) {
        work_t dVc[3];
        pure_s_nuclear_operator_deriv(a_i, c_i, Ai, a_j, c_j, Bj, n_prim, C,
                                      atom_pos, atom_Z[C], dVc);
        atomic_add_grad3(fx, fy, fz, C, Pmn * dVc[0], Pmn * dVc[1],
                         Pmn * dVc[2]);
    }
}

/// Two-electron RHF gradient — one thread per ordered quartet (i,j,k,l).
__global__ void
rhf_forces_2e_kernel(real_t* __restrict__ fx, real_t* __restrict__ fy,
                     real_t* __restrict__ fz, const prec_t* __restrict__ P,
                     const prec_t* __restrict__ shell_alpha,
                     const prec_t* __restrict__ shell_coeff,
                     const prec_t* __restrict__ shell_centers,
                     const int* __restrict__ shell_atom, int n, int n_prim) {
    const int n2 = n * n;
    const int n3 = n2 * n;
    const int n4 = n3 * n;
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(blockDim.x * gridDim.x);
    for (int t = tid; t < n4; t += stride) {
        const int i = t % n;
        const int j = (t / n) % n;
        const int k = (t / n2) % n;
        const int l = t / n3;
        const work_t Pij = static_cast<work_t>(P[i + j * n]);
        const work_t Pkl = static_cast<work_t>(P[k + l * n]);
        const work_t Pik = static_cast<work_t>(P[i + k * n]);
        const work_t Pjl = static_cast<work_t>(P[j + l * n]);
        const work_t factor =
            work_t(0.5) * Pij * Pkl - work_t(0.25) * Pik * Pjl;
        if (fabs(factor) < work_t(1e-18))
            continue;

        work_t Apos[3] = {static_cast<work_t>(shell_centers[i * 3 + 0]),
                          static_cast<work_t>(shell_centers[i * 3 + 1]),
                          static_cast<work_t>(shell_centers[i * 3 + 2])};
        work_t Bpos[3] = {static_cast<work_t>(shell_centers[j * 3 + 0]),
                          static_cast<work_t>(shell_centers[j * 3 + 1]),
                          static_cast<work_t>(shell_centers[j * 3 + 2])};
        work_t Cpos[3] = {static_cast<work_t>(shell_centers[k * 3 + 0]),
                          static_cast<work_t>(shell_centers[k * 3 + 1]),
                          static_cast<work_t>(shell_centers[k * 3 + 2])};
        work_t Dpos[3] = {static_cast<work_t>(shell_centers[l * 3 + 0]),
                          static_cast<work_t>(shell_centers[l * 3 + 1]),
                          static_cast<work_t>(shell_centers[l * 3 + 2])};
        const prec_t* aA = shell_alpha + i * n_prim;
        const prec_t* cA = shell_coeff + i * n_prim;
        const prec_t* aB = shell_alpha + j * n_prim;
        const prec_t* cB = shell_coeff + j * n_prim;
        const prec_t* aC = shell_alpha + k * n_prim;
        const prec_t* cC = shell_coeff + k * n_prim;
        const prec_t* aD = shell_alpha + l * n_prim;
        const prec_t* cD = shell_coeff + l * n_prim;

        // Four center derivatives (may hit the same atom → separate atomics).
#pragma unroll
        for (int which = 0; which < 4; ++which) {
            work_t dE[3];
            pure_s_eri_deriv_center(which, aA, cA, Apos, aB, cB, Bpos, aC, cC,
                                    Cpos, aD, cD, Dpos, n_prim, dE);
            const int atom = (which == 0)   ? shell_atom[i]
                             : (which == 1) ? shell_atom[j]
                             : (which == 2) ? shell_atom[k]
                                            : shell_atom[l];
            atomic_add_grad3(fx, fy, fz, atom, factor * dE[0], factor * dE[1],
                             factor * dE[2]);
        }
    }
}

/// F = −∂E/∂R with Ha/Bohr → eV/Å.
__global__ void rhf_forces_scale_kernel(real_t* __restrict__ fx,
                                        real_t* __restrict__ fy,
                                        real_t* __restrict__ fz, int natm) {
    const int A = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (A >= natm)
        return;
    fx[A] = static_cast<real_t>(-ha_bohr_to_ev_ang(static_cast<work_t>(fx[A])));
    fy[A] = static_cast<real_t>(-ha_bohr_to_ev_ang(static_cast<work_t>(fy[A])));
    fz[A] = static_cast<real_t>(-ha_bohr_to_ev_ang(static_cast<work_t>(fz[A])));
}

void launch_rhf_forces_pure_s(
    double* fx, double* fy, double* fz, const double* P, const double* W,
    const double* shell_alpha, const double* shell_coeff,
    const double* shell_centers, const int* shell_atom, const int* atom_Z,
    const double* atom_pos, int n_ao, int n_prim, int natm, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    const int threads = cuda::kDefaultBlock;
    // 1) zero
    zero_forces_kernel<<<cuda::grid_size(natm, threads), threads, 0, s>>>(
        fx, fy, fz, natm);
    // 2) nuclear pairs
    if (natm >= 2) {
        const int npair = natm * (natm - 1) / 2;
        rhf_forces_nuc_kernel<<<cuda::grid_size(npair, threads), threads, 0,
                                s>>>(fx, fy, fz, atom_Z, atom_pos, natm);
    }
    // 3) 1e + Pulay over AO pairs
    {
        const int nn = n_ao * n_ao;
        rhf_forces_1e_kernel<<<cuda::grid_size(nn, threads), threads, 0, s>>>(
            fx, fy, fz, P, W, shell_alpha, shell_coeff, shell_centers,
            shell_atom, atom_Z, atom_pos, n_ao, n_prim, natm);
    }
    // 4) 2e over full ordered quartets (grid-stride inside)
    {
        const int n4 = n_ao * n_ao * n_ao * n_ao;
        // Cap grid so each block gets several quartets (H2 n4=16 still fine).
        int blocks = cuda::grid_size(n4, threads);
        if (blocks > 1024)
            blocks = 1024;
        rhf_forces_2e_kernel<<<blocks, threads, 0, s>>>(
            fx, fy, fz, P, shell_alpha, shell_coeff, shell_centers, shell_atom,
            n_ao, n_prim);
    }
    // 5) unit conversion
    rhf_forces_scale_kernel<<<cuda::grid_size(natm, threads), threads, 0, s>>>(
        fx, fy, fz, natm);
}

// ── Wave 6: s+p (l≤1) analytic RHF forces ─────────────────────────────
// Angular-momentum raise/lower (Obara–Saika / Helgaker):
//   d/dA_d I(a,b) = 2 α_a I(a+e_d,b) − a_d I(a−e_d,b)
// Pure-s is the a_d=0 special case. Nuclear-operator ∂/∂R_C via TI:
//   dV_C/dC = −(dV_C/dA + dV_C/dB) on the primitive without −Z.

__device__ void sp_1e_deriv_center(int which, // 0 = d/dA, 1 = d/dB
                                   const int a_ang[3], const int b_ang[3],
                                   const prec_t* alpha_i, const prec_t* coeff_i,
                                   const work_t Ai[3], const prec_t* alpha_j,
                                   const prec_t* coeff_j, const work_t Bj[3],
                                   int n_prim, const int* atom_Z,
                                   const prec_t* atom_pos, int natm,
                                   work_t dS[3], work_t dT[3], work_t dV[3]) {
    for (int d = 0; d < 3; ++d)
        dS[d] = dT[d] = dV[d] = work_t(0);
    for (int p = 0; p < n_prim; ++p) {
        work_t ap = static_cast<work_t>(alpha_i[p]);
        work_t cp = static_cast<work_t>(coeff_i[p]);
        for (int q = 0; q < n_prim; ++q) {
            work_t aq = static_cast<work_t>(alpha_j[q]);
            work_t cq = static_cast<work_t>(coeff_j[q]);
            work_t cpq = cp * cq;
            const work_t alpha_raise = (which == 0) ? ap : aq;
            const int* ang_ref = (which == 0) ? a_ang : b_ang;
            for (int d = 0; d < 3; ++d) {
                int aa[3] = {a_ang[0], a_ang[1], a_ang[2]};
                int bb[3] = {b_ang[0], b_ang[1], b_ang[2]};
                if (which == 0)
                    aa[d] += 1;
                else
                    bb[d] += 1;
                work_t Sp = overlap_prim_3d_device(aa, bb, ap, Ai, aq, Bj);
                work_t Tp = kinetic_prim_3d_device(aa, bb, ap, Ai, aq, Bj);
                work_t s_val = work_t(2) * alpha_raise * Sp;
                work_t t_val = work_t(2) * alpha_raise * Tp;
                if (ang_ref[d] > 0) {
                    int aa_m[3] = {a_ang[0], a_ang[1], a_ang[2]};
                    int bb_m[3] = {b_ang[0], b_ang[1], b_ang[2]};
                    if (which == 0)
                        aa_m[d] -= 1;
                    else
                        bb_m[d] -= 1;
                    s_val -=
                        work_t(ang_ref[d])
                        * overlap_prim_3d_device(aa_m, bb_m, ap, Ai, aq, Bj);
                    t_val -=
                        work_t(ang_ref[d])
                        * kinetic_prim_3d_device(aa_m, bb_m, ap, Ai, aq, Bj);
                }
                dS[d] += cpq * s_val;
                dT[d] += cpq * t_val;

                work_t gamma = ap + aq;
                work_t mu_ab = ap * aq / gamma;
                work_t R2 =
                    dist2_device(Ai[0], Ai[1], Ai[2], Bj[0], Bj[1], Bj[2]);
                work_t K_AB = exp(-mu_ab * R2);
                work_t inv_g = work_t(1) / gamma;
                work_t PA[3], PB[3], P[3];
                for (int e = 0; e < 3; ++e) {
                    P[e] = (ap * Ai[e] + aq * Bj[e]) * inv_g;
                    PA[e] = P[e] - Ai[e];
                    PB[e] = P[e] - Bj[e];
                }
                work_t Vsum = work_t(0);
                for (int a = 0; a < natm; ++a) {
                    work_t Cpos[3] = {static_cast<work_t>(atom_pos[a * 3 + 0]),
                                      static_cast<work_t>(atom_pos[a * 3 + 1]),
                                      static_cast<work_t>(atom_pos[a * 3 + 2])};
                    work_t PC[3] = {P[0] - Cpos[0], P[1] - Cpos[1],
                                    P[2] - Cpos[2]};
                    work_t Vprim =
                        nuclear_prim_device(aa, bb, gamma, PA, PB, PC, K_AB);
                    work_t v_val = work_t(2) * alpha_raise * Vprim;
                    if (ang_ref[d] > 0) {
                        int aa_m[3] = {a_ang[0], a_ang[1], a_ang[2]};
                        int bb_m[3] = {b_ang[0], b_ang[1], b_ang[2]};
                        if (which == 0)
                            aa_m[d] -= 1;
                        else
                            bb_m[d] -= 1;
                        v_val -= work_t(ang_ref[d])
                                 * nuclear_prim_device(aa_m, bb_m, gamma, PA,
                                                       PB, PC, K_AB);
                    }
                    Vsum += work_t(-atom_Z[a]) * v_val;
                }
                dV[d] += cpq * Vsum;
            }
        }
    }
}

// dV/dC for one nucleus (includes −Z) via translational invariance on V_C.
__device__ void sp_nuclear_operator_deriv(
    const int a_ang[3], const int b_ang[3], const prec_t* alpha_i,
    const prec_t* coeff_i, const work_t Ai[3], const prec_t* alpha_j,
    const prec_t* coeff_j, const work_t Bj[3], int n_prim, int C_atom,
    const prec_t* atom_pos, int Zc, work_t dV_dC[3]) {
    for (int d = 0; d < 3; ++d)
        dV_dC[d] = work_t(0);
    if (Zc == 0)
        return;
    work_t Cpos[3] = {static_cast<work_t>(atom_pos[C_atom * 3 + 0]),
                      static_cast<work_t>(atom_pos[C_atom * 3 + 1]),
                      static_cast<work_t>(atom_pos[C_atom * 3 + 2])};
    for (int p = 0; p < n_prim; ++p) {
        work_t ap = static_cast<work_t>(alpha_i[p]);
        work_t cp = static_cast<work_t>(coeff_i[p]);
        for (int q = 0; q < n_prim; ++q) {
            work_t aq = static_cast<work_t>(alpha_j[q]);
            work_t cq = static_cast<work_t>(coeff_j[q]);
            work_t cpq = cp * cq;
            work_t gamma = ap + aq;
            work_t mu_ab = ap * aq / gamma;
            work_t R2 = dist2_device(Ai[0], Ai[1], Ai[2], Bj[0], Bj[1], Bj[2]);
            work_t K_AB = exp(-mu_ab * R2);
            work_t inv_g = work_t(1) / gamma;
            work_t PA[3], PB[3], P[3];
            for (int e = 0; e < 3; ++e) {
                P[e] = (ap * Ai[e] + aq * Bj[e]) * inv_g;
                PA[e] = P[e] - Ai[e];
                PB[e] = P[e] - Bj[e];
            }
            work_t PC[3] = {P[0] - Cpos[0], P[1] - Cpos[1], P[2] - Cpos[2]};
            // d(np)/dA and d(np)/dB (np = nuclear_prim, no −Z)
            work_t dA[3], dB[3];
            for (int d = 0; d < 3; ++d) {
                int aa[3] = {a_ang[0], a_ang[1], a_ang[2]};
                aa[d] += 1;
                work_t va =
                    work_t(2) * ap
                    * nuclear_prim_device(aa, b_ang, gamma, PA, PB, PC, K_AB);
                if (a_ang[d] > 0) {
                    int aa_m[3] = {a_ang[0], a_ang[1], a_ang[2]};
                    aa_m[d] -= 1;
                    va -= work_t(a_ang[d])
                          * nuclear_prim_device(aa_m, b_ang, gamma, PA, PB, PC,
                                                K_AB);
                }
                dA[d] = va;

                int bb[3] = {b_ang[0], b_ang[1], b_ang[2]};
                bb[d] += 1;
                work_t vb =
                    work_t(2) * aq
                    * nuclear_prim_device(a_ang, bb, gamma, PA, PB, PC, K_AB);
                if (b_ang[d] > 0) {
                    int bb_m[3] = {b_ang[0], b_ang[1], b_ang[2]};
                    bb_m[d] -= 1;
                    vb -= work_t(b_ang[d])
                          * nuclear_prim_device(a_ang, bb_m, gamma, PA, PB, PC,
                                                K_AB);
                }
                dB[d] = vb;
            }
            // V = −Z * np; dV/dC = −Z * d(np)/dC = −Z * (−(dA+dB))
            for (int d = 0; d < 3; ++d)
                dV_dC[d] += cpq * work_t(-Zc) * (-(dA[d] + dB[d]));
        }
    }
}

__device__ void
sp_eri_deriv_center(int which, // 0..3 = A,B,C,D
                    const int a_ang[3], const int b_ang[3], const int c_ang[3],
                    const int d_ang[3], const prec_t* aA, const prec_t* cA,
                    const work_t A[3], const prec_t* aB, const prec_t* cB,
                    const work_t B[3], const prec_t* aC, const prec_t* cC,
                    const work_t C[3], const prec_t* aD, const prec_t* cD,
                    const work_t D[3], int n_prim, work_t dERI[3]) {
    for (int d = 0; d < 3; ++d)
        dERI[d] = work_t(0);
    for (int p = 0; p < n_prim; ++p) {
        work_t ap = static_cast<work_t>(aA[p]);
        work_t cp = static_cast<work_t>(cA[p]);
        for (int q = 0; q < n_prim; ++q) {
            work_t aq = static_cast<work_t>(aB[q]);
            work_t cq = static_cast<work_t>(cB[q]);
            for (int r = 0; r < n_prim; ++r) {
                work_t ar = static_cast<work_t>(aC[r]);
                work_t cr = static_cast<work_t>(cC[r]);
                for (int s = 0; s < n_prim; ++s) {
                    work_t as = static_cast<work_t>(aD[s]);
                    work_t cs = static_cast<work_t>(cD[s]);
                    work_t c4 = cp * cq * cr * cs;
                    work_t alpha_raise = ap;
                    const int* ang_ref = a_ang;
                    if (which == 1) {
                        alpha_raise = aq;
                        ang_ref = b_ang;
                    } else if (which == 2) {
                        alpha_raise = ar;
                        ang_ref = c_ang;
                    } else if (which == 3) {
                        alpha_raise = as;
                        ang_ref = d_ang;
                    }
                    for (int d = 0; d < 3; ++d) {
                        int aa[3] = {a_ang[0], a_ang[1], a_ang[2]};
                        int bb[3] = {b_ang[0], b_ang[1], b_ang[2]};
                        int cc[3] = {c_ang[0], c_ang[1], c_ang[2]};
                        int dd[3] = {d_ang[0], d_ang[1], d_ang[2]};
                        if (which == 0)
                            aa[d] += 1;
                        else if (which == 1)
                            bb[d] += 1;
                        else if (which == 2)
                            cc[d] += 1;
                        else
                            dd[d] += 1;
                        work_t ep = eri_prim_md_device(aa, bb, cc, dd, ap, A,
                                                       aq, B, ar, C, as, D);
                        work_t val = work_t(2) * alpha_raise * ep;
                        if (ang_ref[d] > 0) {
                            int aa_m[3] = {a_ang[0], a_ang[1], a_ang[2]};
                            int bb_m[3] = {b_ang[0], b_ang[1], b_ang[2]};
                            int cc_m[3] = {c_ang[0], c_ang[1], c_ang[2]};
                            int dd_m[3] = {d_ang[0], d_ang[1], d_ang[2]};
                            if (which == 0)
                                aa_m[d] -= 1;
                            else if (which == 1)
                                bb_m[d] -= 1;
                            else if (which == 2)
                                cc_m[d] -= 1;
                            else
                                dd_m[d] -= 1;
                            val -=
                                work_t(ang_ref[d])
                                * eri_prim_md_device(aa_m, bb_m, cc_m, dd_m, ap,
                                                     A, aq, B, ar, C, as, D);
                        }
                        dERI[d] += c4 * val;
                    }
                }
            }
        }
    }
}

__global__ void rhf_forces_1e_sp_kernel(
    real_t* __restrict__ fx, real_t* __restrict__ fy, real_t* __restrict__ fz,
    const prec_t* __restrict__ P, const prec_t* __restrict__ W,
    const prec_t* __restrict__ shell_alpha,
    const prec_t* __restrict__ shell_coeff,
    const prec_t* __restrict__ shell_centers,
    const int* __restrict__ shell_atom, const int* __restrict__ shell_L,
    const int* __restrict__ ao_shell, const int* __restrict__ ao_cart,
    const int* __restrict__ atom_Z, const prec_t* __restrict__ atom_pos, int n,
    int n_prim, int natm) {
    const int nn = n * n;
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= nn)
        return;
    const int mu = idx % n;
    const int nu = idx / n;
    const work_t Pmn = static_cast<work_t>(P[mu + nu * n]);
    const work_t Wmn = static_cast<work_t>(W[mu + nu * n]);
    const int si = ao_shell[mu];
    const int sj = ao_shell[nu];
    int a_ang[3], b_ang[3];
    cart_ang_device(shell_L[si], ao_cart[mu], a_ang);
    cart_ang_device(shell_L[sj], ao_cart[nu], b_ang);
    work_t Ai[3] = {static_cast<work_t>(shell_centers[si * 3 + 0]),
                    static_cast<work_t>(shell_centers[si * 3 + 1]),
                    static_cast<work_t>(shell_centers[si * 3 + 2])};
    work_t Bj[3] = {static_cast<work_t>(shell_centers[sj * 3 + 0]),
                    static_cast<work_t>(shell_centers[sj * 3 + 1]),
                    static_cast<work_t>(shell_centers[sj * 3 + 2])};
    const prec_t* a_i = shell_alpha + si * n_prim;
    const prec_t* c_i = shell_coeff + si * n_prim;
    const prec_t* a_j = shell_alpha + sj * n_prim;
    const prec_t* c_j = shell_coeff + sj * n_prim;
    const int Amu = shell_atom[si];
    const int Anu = shell_atom[sj];

    {
        work_t dS[3], dT[3], dV[3];
        sp_1e_deriv_center(0, a_ang, b_ang, a_i, c_i, Ai, a_j, c_j, Bj, n_prim,
                           atom_Z, atom_pos, natm, dS, dT, dV);
        atomic_add_grad3(fx, fy, fz, Amu, Pmn * (dT[0] + dV[0]) - Wmn * dS[0],
                         Pmn * (dT[1] + dV[1]) - Wmn * dS[1],
                         Pmn * (dT[2] + dV[2]) - Wmn * dS[2]);
    }
    {
        work_t dS[3], dT[3], dV[3];
        sp_1e_deriv_center(1, a_ang, b_ang, a_i, c_i, Ai, a_j, c_j, Bj, n_prim,
                           atom_Z, atom_pos, natm, dS, dT, dV);
        atomic_add_grad3(fx, fy, fz, Anu, Pmn * (dT[0] + dV[0]) - Wmn * dS[0],
                         Pmn * (dT[1] + dV[1]) - Wmn * dS[1],
                         Pmn * (dT[2] + dV[2]) - Wmn * dS[2]);
    }
    for (int C = 0; C < natm; ++C) {
        work_t dVc[3];
        sp_nuclear_operator_deriv(a_ang, b_ang, a_i, c_i, Ai, a_j, c_j, Bj,
                                  n_prim, C, atom_pos, atom_Z[C], dVc);
        atomic_add_grad3(fx, fy, fz, C, Pmn * dVc[0], Pmn * dVc[1],
                         Pmn * dVc[2]);
    }
}

__global__ void rhf_forces_2e_sp_kernel(
    real_t* __restrict__ fx, real_t* __restrict__ fy, real_t* __restrict__ fz,
    const prec_t* __restrict__ P, const prec_t* __restrict__ shell_alpha,
    const prec_t* __restrict__ shell_coeff,
    const prec_t* __restrict__ shell_centers,
    const int* __restrict__ shell_atom, const int* __restrict__ shell_L,
    const int* __restrict__ ao_shell, const int* __restrict__ ao_cart, int n,
    int n_prim) {
    const int n2 = n * n;
    const int n3 = n2 * n;
    const int n4 = n3 * n;
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(blockDim.x * gridDim.x);
    for (int t = tid; t < n4; t += stride) {
        const int i = t % n;
        const int j = (t / n) % n;
        const int k = (t / n2) % n;
        const int l = t / n3;
        const work_t Pij = static_cast<work_t>(P[i + j * n]);
        const work_t Pkl = static_cast<work_t>(P[k + l * n]);
        const work_t Pik = static_cast<work_t>(P[i + k * n]);
        const work_t Pjl = static_cast<work_t>(P[j + l * n]);
        const work_t factor =
            work_t(0.5) * Pij * Pkl - work_t(0.25) * Pik * Pjl;
        if (fabs(factor) < work_t(1e-18))
            continue;

        const int sa = ao_shell[i], sb = ao_shell[j];
        const int sc = ao_shell[k], sd = ao_shell[l];
        int aa[3], bb[3], cc[3], dd[3];
        cart_ang_device(shell_L[sa], ao_cart[i], aa);
        cart_ang_device(shell_L[sb], ao_cart[j], bb);
        cart_ang_device(shell_L[sc], ao_cart[k], cc);
        cart_ang_device(shell_L[sd], ao_cart[l], dd);

        work_t Apos[3] = {static_cast<work_t>(shell_centers[sa * 3 + 0]),
                          static_cast<work_t>(shell_centers[sa * 3 + 1]),
                          static_cast<work_t>(shell_centers[sa * 3 + 2])};
        work_t Bpos[3] = {static_cast<work_t>(shell_centers[sb * 3 + 0]),
                          static_cast<work_t>(shell_centers[sb * 3 + 1]),
                          static_cast<work_t>(shell_centers[sb * 3 + 2])};
        work_t Cpos[3] = {static_cast<work_t>(shell_centers[sc * 3 + 0]),
                          static_cast<work_t>(shell_centers[sc * 3 + 1]),
                          static_cast<work_t>(shell_centers[sc * 3 + 2])};
        work_t Dpos[3] = {static_cast<work_t>(shell_centers[sd * 3 + 0]),
                          static_cast<work_t>(shell_centers[sd * 3 + 1]),
                          static_cast<work_t>(shell_centers[sd * 3 + 2])};
        const prec_t* aA = shell_alpha + sa * n_prim;
        const prec_t* cA = shell_coeff + sa * n_prim;
        const prec_t* aB = shell_alpha + sb * n_prim;
        const prec_t* cB = shell_coeff + sb * n_prim;
        const prec_t* aC = shell_alpha + sc * n_prim;
        const prec_t* cC = shell_coeff + sc * n_prim;
        const prec_t* aD = shell_alpha + sd * n_prim;
        const prec_t* cD = shell_coeff + sd * n_prim;

#pragma unroll
        for (int which = 0; which < 4; ++which) {
            work_t dE[3];
            sp_eri_deriv_center(which, aa, bb, cc, dd, aA, cA, Apos, aB, cB,
                                Bpos, aC, cC, Cpos, aD, cD, Dpos, n_prim, dE);
            const int atom = (which == 0)   ? shell_atom[sa]
                             : (which == 1) ? shell_atom[sb]
                             : (which == 2) ? shell_atom[sc]
                                            : shell_atom[sd];
            atomic_add_grad3(fx, fy, fz, atom, factor * dE[0], factor * dE[1],
                             factor * dE[2]);
        }
    }
}

void launch_rhf_forces_sp(double* fx, double* fy, double* fz, const double* P,
                          const double* W, const double* shell_alpha,
                          const double* shell_coeff,
                          const double* shell_centers, const int* shell_atom,
                          const int* shell_L, const int* ao_shell,
                          const int* ao_cart, const int* atom_Z,
                          const double* atom_pos, int n_ao, int n_prim,
                          int natm, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    const int threads = cuda::kDefaultBlock;
    zero_forces_kernel<<<cuda::grid_size(natm, threads), threads, 0, s>>>(
        fx, fy, fz, natm);
    if (natm >= 2) {
        const int npair = natm * (natm - 1) / 2;
        rhf_forces_nuc_kernel<<<cuda::grid_size(npair, threads), threads, 0,
                                s>>>(fx, fy, fz, atom_Z, atom_pos, natm);
    }
    {
        const int nn = n_ao * n_ao;
        rhf_forces_1e_sp_kernel<<<cuda::grid_size(nn, threads), threads, 0,
                                  s>>>(fx, fy, fz, P, W, shell_alpha,
                                       shell_coeff, shell_centers, shell_atom,
                                       shell_L, ao_shell, ao_cart, atom_Z,
                                       atom_pos, n_ao, n_prim, natm);
    }
    {
        const int n4 = n_ao * n_ao * n_ao * n_ao;
        int blocks = cuda::grid_size(n4, threads);
        if (blocks > 1024)
            blocks = 1024;
        rhf_forces_2e_sp_kernel<<<blocks, threads, 0, s>>>(
            fx, fy, fz, P, shell_alpha, shell_coeff, shell_centers, shell_atom,
            shell_L, ao_shell, ao_cart, n_ao, n_prim);
    }
    rhf_forces_scale_kernel<<<cuda::grid_size(natm, threads), threads, 0, s>>>(
        fx, fy, fz, natm);
}

/// W_μν = Σ_{i<n_occ} ε_i C_μi C_νi (occupation 1).
__global__ void build_W_uhf_kernel(prec_t* __restrict__ W,
                                   const prec_t* __restrict__ C,
                                   const prec_t* __restrict__ eig, int n,
                                   int n_occ) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int nn = n * n;
    if (idx >= nn)
        return;
    const int mu = idx % n;
    const int nu = idx / n;
    work_t acc = work_t(0);
    for (int i = 0; i < n_occ; ++i) {
        acc += static_cast<work_t>(eig[i]) * static_cast<work_t>(C[mu + i * n])
               * static_cast<work_t>(C[nu + i * n]);
    }
    W[idx] = static_cast<prec_t>(acc);
}

void launch_build_W_uhf(double* W, const double* C, const double* eig, int n_ao,
                        int n_occ, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    const int nn = n_ao * n_ao;
    const int threads = cuda::kDefaultBlock;
    build_W_uhf_kernel<<<cuda::grid_size(nn, threads), threads, 0, s>>>(
        W, C, eig, n_ao, n_occ);
}

/// Two-electron UHF gradient. Closed shell (Pα=Pβ=P/2) recovers the RHF factor.
__global__ void uhf_forces_2e_sp_kernel(
    real_t* __restrict__ fx, real_t* __restrict__ fy, real_t* __restrict__ fz,
    const prec_t* __restrict__ Pa, const prec_t* __restrict__ Pb,
    const prec_t* __restrict__ shell_alpha,
    const prec_t* __restrict__ shell_coeff,
    const prec_t* __restrict__ shell_centers,
    const int* __restrict__ shell_atom, const int* __restrict__ shell_L,
    const int* __restrict__ ao_shell, const int* __restrict__ ao_cart, int n,
    int n_prim) {
    const int n2 = n * n;
    const int n3 = n2 * n;
    const int n4 = n3 * n;
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(blockDim.x * gridDim.x);
    for (int t = tid; t < n4; t += stride) {
        const int i = t % n;
        const int j = (t / n) % n;
        const int k = (t / n2) % n;
        const int l = t / n3;
        const work_t Pa_ij = static_cast<work_t>(Pa[i + j * n]);
        const work_t Pb_ij = static_cast<work_t>(Pb[i + j * n]);
        const work_t Pa_kl = static_cast<work_t>(Pa[k + l * n]);
        const work_t Pb_kl = static_cast<work_t>(Pb[k + l * n]);
        const work_t Pa_ik = static_cast<work_t>(Pa[i + k * n]);
        const work_t Pa_jl = static_cast<work_t>(Pa[j + l * n]);
        const work_t Pb_ik = static_cast<work_t>(Pb[i + k * n]);
        const work_t Pb_jl = static_cast<work_t>(Pb[j + l * n]);
        const work_t factor = work_t(0.5) * (Pa_ij + Pb_ij) * (Pa_kl + Pb_kl)
                              - work_t(0.5) * (Pa_ik * Pa_jl + Pb_ik * Pb_jl);
        if (fabs(factor) < work_t(1e-18))
            continue;

        const int sa = ao_shell[i], sb = ao_shell[j];
        const int sc = ao_shell[k], sd = ao_shell[l];
        int aa[3], bb[3], cc[3], dd[3];
        cart_ang_device(shell_L[sa], ao_cart[i], aa);
        cart_ang_device(shell_L[sb], ao_cart[j], bb);
        cart_ang_device(shell_L[sc], ao_cart[k], cc);
        cart_ang_device(shell_L[sd], ao_cart[l], dd);

        work_t Apos[3] = {static_cast<work_t>(shell_centers[sa * 3 + 0]),
                          static_cast<work_t>(shell_centers[sa * 3 + 1]),
                          static_cast<work_t>(shell_centers[sa * 3 + 2])};
        work_t Bpos[3] = {static_cast<work_t>(shell_centers[sb * 3 + 0]),
                          static_cast<work_t>(shell_centers[sb * 3 + 1]),
                          static_cast<work_t>(shell_centers[sb * 3 + 2])};
        work_t Cpos[3] = {static_cast<work_t>(shell_centers[sc * 3 + 0]),
                          static_cast<work_t>(shell_centers[sc * 3 + 1]),
                          static_cast<work_t>(shell_centers[sc * 3 + 2])};
        work_t Dpos[3] = {static_cast<work_t>(shell_centers[sd * 3 + 0]),
                          static_cast<work_t>(shell_centers[sd * 3 + 1]),
                          static_cast<work_t>(shell_centers[sd * 3 + 2])};
        const prec_t* aA = shell_alpha + sa * n_prim;
        const prec_t* cA = shell_coeff + sa * n_prim;
        const prec_t* aB = shell_alpha + sb * n_prim;
        const prec_t* cB = shell_coeff + sb * n_prim;
        const prec_t* aC = shell_alpha + sc * n_prim;
        const prec_t* cC = shell_coeff + sc * n_prim;
        const prec_t* aD = shell_alpha + sd * n_prim;
        const prec_t* cD = shell_coeff + sd * n_prim;

#pragma unroll
        for (int which = 0; which < 4; ++which) {
            work_t dE[3];
            sp_eri_deriv_center(which, aa, bb, cc, dd, aA, cA, Apos, aB, cB,
                                Bpos, aC, cC, Cpos, aD, cD, Dpos, n_prim, dE);
            const int atom = (which == 0)   ? shell_atom[sa]
                             : (which == 1) ? shell_atom[sb]
                             : (which == 2) ? shell_atom[sc]
                                            : shell_atom[sd];
            atomic_add_grad3(fx, fy, fz, atom, factor * dE[0], factor * dE[1],
                             factor * dE[2]);
        }
    }
}

void launch_uhf_forces_sp(double* fx, double* fy, double* fz,
                          const double* Ptot, const double* Wtot,
                          const double* Pa, const double* Pb,
                          const double* shell_alpha, const double* shell_coeff,
                          const double* shell_centers, const int* shell_atom,
                          const int* shell_L, const int* ao_shell,
                          const int* ao_cart, const int* atom_Z,
                          const double* atom_pos, int n_ao, int n_prim,
                          int natm, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    const int threads = cuda::kDefaultBlock;
    zero_forces_kernel<<<cuda::grid_size(natm, threads), threads, 0, s>>>(
        fx, fy, fz, natm);
    if (natm >= 2) {
        const int npair = natm * (natm - 1) / 2;
        rhf_forces_nuc_kernel<<<cuda::grid_size(npair, threads), threads, 0,
                                s>>>(fx, fy, fz, atom_Z, atom_pos, natm);
    }
    {
        const int nn = n_ao * n_ao;
        rhf_forces_1e_sp_kernel<<<cuda::grid_size(nn, threads), threads, 0,
                                  s>>>(fx, fy, fz, Ptot, Wtot, shell_alpha,
                                       shell_coeff, shell_centers, shell_atom,
                                       shell_L, ao_shell, ao_cart, atom_Z,
                                       atom_pos, n_ao, n_prim, natm);
    }
    {
        const int n4 = n_ao * n_ao * n_ao * n_ao;
        int blocks = cuda::grid_size(n4, threads);
        if (blocks > 1024)
            blocks = 1024;
        uhf_forces_2e_sp_kernel<<<blocks, threads, 0, s>>>(
            fx, fy, fz, Pa, Pb, shell_alpha, shell_coeff, shell_centers,
            shell_atom, shell_L, ao_shell, ao_cart, n_ao, n_prim);
    }
    rhf_forces_scale_kernel<<<cuda::grid_size(natm, threads), threads, 0, s>>>(
        fx, fy, fz, natm);
}

// ── Cartesian l ≤ 3, ragged n_prim (hf-cuda-lgt1) ────────────────────
//
// PySCF `scf.hf.get_jk` calls `scf._vhf.direct` with `intor='int2e_cart'`
// (`pyscf/scf/hf.py` get_veff = J − ½K). One McMurchie–Davidson quartet
// (`native::eri_prim` / libcint `int2e_cart`). Every angular order reuses
// the shared stable Boys sequence: upward recurrence alone is unstable
// at small arguments even for s/p quartets. No second primitive loop and
// no host ERI fill.

static constexpr int kL3E = 7;      // E coeff, ai+bj ≤ 6
static constexpr int kL3I = 5;      // E recurrence rows, ai ≤ 3
static constexpr int kL3R = 7;      // nuclear Hermite index, Ltot ≤ 6
static constexpr int kFDim = 13;    // ERI Hermite index, Ltot ≤ 12
static constexpr int kFOut = 10000; // (ff|ff) = 10^4

__device__ void md_e_l3(int ai, int bj, work_t PA, work_t PB, work_t inv2g,
                        work_t E[kL3E]) {
    for (int t = 0; t < kL3E; ++t)
        E[t] = work_t(0);
    work_t E_buf[kL3I][kL3E];
    for (int i = 0; i < kL3I; ++i)
        for (int t = 0; t < kL3E; ++t)
            E_buf[i][t] = work_t(0);
    E_buf[0][0] = work_t(1);
    for (int i = 0; i < ai; ++i) {
        const int tmax = i + 1;
        for (int t = 0; t <= tmax && t < kL3E; ++t) {
            const work_t etm1 = (t > 0) ? E_buf[i][t - 1] : work_t(0);
            const work_t et = E_buf[i][t];
            const work_t etp1 = (t + 1 < kL3E) ? E_buf[i][t + 1] : work_t(0);
            E_buf[i + 1][t] = inv2g * etm1 + PA * et + work_t(t + 1) * etp1;
        }
    }
    work_t W[2][kL3E];
    int cur = 0;
    for (int t = 0; t < kL3E; ++t)
        W[cur][t] = (ai < kL3I) ? E_buf[ai][t] : work_t(0);
    for (int jj = 0; jj < bj; ++jj) {
        const int nxt = 1 - cur;
        for (int t = 0; t < kL3E; ++t)
            W[nxt][t] = work_t(0);
        const int tmax = ai + jj + 1;
        for (int t = 0; t <= tmax && t < kL3E; ++t) {
            const work_t etm1 = (t > 0) ? W[cur][t - 1] : work_t(0);
            const work_t et = W[cur][t];
            const work_t etp1 = (t + 1 < kL3E) ? W[cur][t + 1] : work_t(0);
            W[nxt][t] = inv2g * etm1 + PB * et + work_t(t + 1) * etp1;
        }
        cur = nxt;
    }
    for (int t = 0; t <= ai + bj && t < kL3E; ++t)
        E[t] = W[cur][t];
}

__device__ work_t nuclear_prim_l3_device(const int a_ang[3], const int b_ang[3],
                                         work_t gamma, const work_t PA[3],
                                         const work_t PB[3], const work_t PC[3],
                                         work_t K_AB) {
    const work_t inv2g = work_t(0.5) / gamma;
    const work_t prefactor = work_t(2) * work_t(ATV::units::pi) / gamma * K_AB;
    const work_t PC2 = PC[0] * PC[0] + PC[1] * PC[1] + PC[2] * PC[2];
    const work_t T = gamma * PC2;
    const int Lx = a_ang[0] + b_ang[0];
    const int Ly = a_ang[1] + b_ang[1];
    const int Lz = a_ang[2] + b_ang[2];
    const int Ltot = Lx + Ly + Lz;
    const int mmax = Ltot < kL3R ? Ltot : kL3R - 1;
    work_t Fm[kL3R];
    if (T < work_t(1e-14)) {
        for (int m = 0; m <= mmax; ++m)
            Fm[m] = work_t(1) / work_t(2 * m + 1);
    } else {
        ATV::boys_sequence(mmax, T, Fm);
    }
    work_t E[3][kL3E];
    for (int d = 0; d < 3; ++d)
        md_e_l3(a_ang[d], b_ang[d], PA[d], PB[d], inv2g, E[d]);
    work_t R_tab[kL3R][kL3R][kL3R][kL3R];
    for (int m = 0; m < kL3R; ++m)
        for (int t = 0; t < kL3R; ++t)
            for (int u = 0; u < kL3R; ++u)
                for (int v = 0; v < kL3R; ++v)
                    R_tab[m][t][u][v] = work_t(0);
    const work_t neg2g = -work_t(2) * gamma;
    work_t neg2g_pow = work_t(1);
    for (int m = 0; m <= mmax; ++m) {
        R_tab[m][0][0][0] = neg2g_pow * Fm[m];
        neg2g_pow *= neg2g;
    }
    for (int m = mmax - 1; m >= 0; --m) {
        const int max_tuv = mmax - m;
        for (int t = 1; t <= Lx && t <= max_tuv; ++t) {
            R_tab[m][t][0][0] = PC[0] * R_tab[m + 1][t - 1][0][0];
            if (t >= 2)
                R_tab[m][t][0][0] += work_t(t - 1) * R_tab[m + 1][t - 2][0][0];
        }
        for (int t = 0; t <= Lx && t <= max_tuv; ++t) {
            for (int u = 1; u <= Ly && t + u <= max_tuv; ++u) {
                R_tab[m][t][u][0] = PC[1] * R_tab[m + 1][t][u - 1][0];
                if (u >= 2)
                    R_tab[m][t][u][0] +=
                        work_t(u - 1) * R_tab[m + 1][t][u - 2][0];
            }
        }
        for (int t = 0; t <= Lx && t <= max_tuv; ++t) {
            for (int u = 0; u <= Ly && t + u <= max_tuv; ++u) {
                for (int v = 1; v <= Lz && t + u + v <= max_tuv; ++v) {
                    R_tab[m][t][u][v] = PC[2] * R_tab[m + 1][t][u][v - 1];
                    if (v >= 2)
                        R_tab[m][t][u][v] +=
                            work_t(v - 1) * R_tab[m + 1][t][u][v - 2];
                }
            }
        }
    }
    work_t Vn = work_t(0);
    for (int t = 0; t <= Lx; ++t)
        for (int u = 0; u <= Ly; ++u)
            for (int v = 0; v <= Lz; ++v)
                Vn += E[0][t] * E[1][u] * E[2][v] * R_tab[0][t][u][v];
    return prefactor * Vn;
}

template <bool FShell>
__global__ void compute_1e_lgt1_kernel(
    prec_t* S, prec_t* T, prec_t* V, const prec_t* prim_alpha,
    const prec_t* prim_coeff, const int* shell_nprim, const int* shell_prim_off,
    const prec_t* shell_centers, const int* shell_L, const int* ao_shell,
    const int* ao_cart, const int* atom_Z, const prec_t* atom_pos, int n_ao,
    int natm) {
    const int n_pairs = n_ao * (n_ao + 1) / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int pair = idx; pair < n_pairs; pair += blockDim.x * gridDim.x) {
        int mu = (int)((-1.0 + sqrt(1.0 + 8.0 * pair)) * 0.5);
        int nu = pair - mu * (mu + 1) / 2;
        if (nu > mu) {
            ++mu;
            nu = pair - mu * (mu + 1) / 2;
        }
        const int si = ao_shell[mu];
        const int sj = ao_shell[nu];
        int a_ang[3], b_ang[3];
        cart_ang_device(shell_L[si], ao_cart[mu], a_ang);
        cart_ang_device(shell_L[sj], ao_cart[nu], b_ang);
        work_t A[3] = {static_cast<work_t>(shell_centers[si * 3 + 0]),
                       static_cast<work_t>(shell_centers[si * 3 + 1]),
                       static_cast<work_t>(shell_centers[si * 3 + 2])};
        work_t B[3] = {static_cast<work_t>(shell_centers[sj * 3 + 0]),
                       static_cast<work_t>(shell_centers[sj * 3 + 1]),
                       static_cast<work_t>(shell_centers[sj * 3 + 2])};
        const int off_i = shell_prim_off[si];
        const int off_j = shell_prim_off[sj];
        const int npi = shell_nprim[si];
        const int npj = shell_nprim[sj];
        work_t acc_S = 0, acc_T = 0, acc_V = 0;
        for (int p = 0; p < npi; ++p) {
            work_t alpha_p = static_cast<work_t>(prim_alpha[off_i + p]);
            work_t c_p = static_cast<work_t>(prim_coeff[off_i + p]);
            for (int q = 0; q < npj; ++q) {
                work_t alpha_q = static_cast<work_t>(prim_alpha[off_j + q]);
                work_t c_q = static_cast<work_t>(prim_coeff[off_j + q]);
                work_t c_pq = c_p * c_q;
                acc_S += c_pq
                         * overlap_prim_3d_device(a_ang, b_ang, alpha_p, A,
                                                  alpha_q, B);
                acc_T += c_pq
                         * kinetic_prim_3d_device(a_ang, b_ang, alpha_p, A,
                                                  alpha_q, B);
                const work_t gamma = alpha_p + alpha_q;
                const work_t mu_ab = alpha_p * alpha_q / gamma;
                work_t R2 = dist2_device(A[0], A[1], A[2], B[0], B[1], B[2]);
                work_t K_AB = exp(-mu_ab * R2);
                work_t inv_g = work_t(1) / gamma;
                work_t PA[3], PB[3], P[3];
                for (int d = 0; d < 3; ++d) {
                    P[d] = (alpha_p * A[d] + alpha_q * B[d]) * inv_g;
                    PA[d] = P[d] - A[d];
                    PB[d] = P[d] - B[d];
                }
                for (int a = 0; a < natm; ++a) {
                    work_t Cpos[3] = {static_cast<work_t>(atom_pos[a * 3 + 0]),
                                      static_cast<work_t>(atom_pos[a * 3 + 1]),
                                      static_cast<work_t>(atom_pos[a * 3 + 2])};
                    work_t PC[3] = {P[0] - Cpos[0], P[1] - Cpos[1],
                                    P[2] - Cpos[2]};
                    if constexpr (FShell) {
                        acc_V += c_pq * work_t(-atom_Z[a])
                                 * nuclear_prim_l3_device(a_ang, b_ang, gamma,
                                                          PA, PB, PC, K_AB);
                    } else {
                        acc_V += c_pq * work_t(-atom_Z[a])
                                 * nuclear_prim_device(a_ang, b_ang, gamma, PA,
                                                       PB, PC, K_AB);
                    }
                }
            }
        }
        const int ij = mu + nu * n_ao;
        const int ji = nu + mu * n_ao;
        S[ij] = acc_S;
        T[ij] = acc_T;
        V[ij] = acc_V;
        if (mu != nu) {
            S[ji] = acc_S;
            T[ji] = acc_T;
            V[ji] = acc_V;
        }
    }
}

static constexpr int kL2Dim = 9;    // Hermite index 0..8 (shell Ltot ≤ 8)
static constexpr int kL2Out = 1296; // (dd|dd) = 6^4

__device__ __forceinline__ int hermite_at(int dim, int m, int t, int u, int v) {
    return ((m * dim + t) * dim + u) * dim + v;
}

// One Hermite R recurrence; all angular orders use the stable Boys batch.
template <int Dim>
__device__ void build_hermite_R(work_t* R, int Ltot, work_t rho, work_t Tval,
                                const work_t PQ[3]) {
    constexpr int n4 = Dim * Dim * Dim * Dim;
    for (int i = 0; i < n4; ++i)
        R[i] = work_t(0);
    const int mmax = Ltot < Dim ? Ltot : Dim - 1;
    work_t Fm[Dim];
    ATV::boys_sequence(mmax, Tval, Fm);
    const work_t neg2 = -work_t(2) * rho;
    work_t powv = work_t(1);
    for (int m = 0; m <= mmax; ++m) {
        R[hermite_at(Dim, m, 0, 0, 0)] = powv * Fm[m];
        powv *= neg2;
    }
    for (int m = mmax - 1; m >= 0; --m) {
        const int max_tuv = mmax - m;
        for (int t = 1; t <= max_tuv && t < Dim; ++t) {
            work_t val = PQ[0] * R[hermite_at(Dim, m + 1, t - 1, 0, 0)];
            if (t >= 2)
                val += work_t(t - 1) * R[hermite_at(Dim, m + 1, t - 2, 0, 0)];
            R[hermite_at(Dim, m, t, 0, 0)] = val;
        }
        for (int t = 0; t <= max_tuv && t < Dim; ++t) {
            for (int u = 1; u + t <= max_tuv && u < Dim; ++u) {
                work_t val = PQ[1] * R[hermite_at(Dim, m + 1, t, u - 1, 0)];
                if (u >= 2)
                    val +=
                        work_t(u - 1) * R[hermite_at(Dim, m + 1, t, u - 2, 0)];
                R[hermite_at(Dim, m, t, u, 0)] = val;
            }
        }
        for (int t = 0; t <= max_tuv && t < Dim; ++t) {
            for (int u = 0; u + t <= max_tuv && u < Dim; ++u) {
                for (int v = 1; v + t + u <= max_tuv && v < Dim; ++v) {
                    work_t val = PQ[2] * R[hermite_at(Dim, m + 1, t, u, v - 1)];
                    if (v >= 2)
                        val += work_t(v - 1)
                               * R[hermite_at(Dim, m + 1, t, u, v - 2)];
                    R[hermite_at(Dim, m, t, u, v)] = val;
                }
            }
        }
    }
}

template <bool L3, int Dim>
__device__ work_t contract_md(const work_t* R, const int a_ang[3],
                              const int b_ang[3], const int c_ang[3],
                              const int d_ang[3], const work_t PA[3],
                              const work_t PB[3], const work_t QC[3],
                              const work_t QD[3], work_t inv2g1, work_t inv2g2,
                              work_t prefactor) {
    constexpr int EDim = L3 ? kL3E : kDeviceMDEDim;
    const int Lx1 = a_ang[0] + b_ang[0];
    const int Ly1 = a_ang[1] + b_ang[1];
    const int Lz1 = a_ang[2] + b_ang[2];
    const int Lx2 = c_ang[0] + d_ang[0];
    const int Ly2 = c_ang[1] + d_ang[1];
    const int Lz2 = c_ang[2] + d_ang[2];
    work_t E1[3][EDim];
    work_t E2[3][EDim];
    for (int d = 0; d < 3; ++d) {
        if constexpr (L3) {
            md_e_l3(a_ang[d], b_ang[d], PA[d], PB[d], inv2g1, E1[d]);
            md_e_l3(c_ang[d], d_ang[d], QC[d], QD[d], inv2g2, E2[d]);
        } else {
            md_e_coeffs_device(a_ang[d], b_ang[d], PA[d], PB[d], inv2g1, E1[d]);
            md_e_coeffs_device(c_ang[d], d_ang[d], QC[d], QD[d], inv2g2, E2[d]);
        }
    }
    work_t eri_val = work_t(0);
    for (int t = 0; t <= Lx1; ++t) {
        for (int u = 0; u <= Ly1; ++u) {
            for (int v = 0; v <= Lz1; ++v) {
                const work_t e1 = E1[0][t] * E1[1][u] * E1[2][v];
                if (e1 == work_t(0))
                    continue;
                for (int tau = 0; tau <= Lx2; ++tau) {
                    for (int nu = 0; nu <= Ly2; ++nu) {
                        for (int phi = 0; phi <= Lz2; ++phi) {
                            const work_t e2 =
                                E2[0][tau] * E2[1][nu] * E2[2][phi];
                            if (e2 == work_t(0))
                                continue;
                            const work_t sign =
                                ((tau + nu + phi) & 1) ? work_t(-1) : work_t(1);
                            eri_val += e1 * e2 * sign
                                       * R[hermite_at(Dim, 0, t + tau, u + nu,
                                                      v + phi)];
                        }
                    }
                }
            }
        }
    }
    return prefactor * eri_val;
}

__device__ int ncart_of_device(int L) { return (L + 1) * (L + 2) / 2; }

// Hoisted shell quartet. FShell=false: l ≤ 2, caller R is kL2Dim^4 and acc
// holds (dd|dd). FShell=true: caller R/acc are the f-shell scratch
// (kFDim^4 + kFOut). Contraction coefficients are already in `prim_coeff`.
template <bool FShell>
__device__ void
eri_quartet_md(prec_t* ERI, const prec_t* prim_alpha, const prec_t* prim_coeff,
               const int* shell_nprim, const int* shell_prim_off,
               const prec_t* shell_centers, int sa, int sb, int sc, int sd,
               int La, int Lb, int Lc, int Ld, int ao_a, int ao_b, int ao_c,
               int ao_d, int n_ao, work_t* R, work_t* acc) {
    const int na = ncart_of_device(La);
    const int nb = ncart_of_device(Lb);
    const int nc = ncart_of_device(Lc);
    const int nd = ncart_of_device(Ld);
    const int n_out = na * nb * nc * nd;
    for (int i = 0; i < n_out; ++i)
        acc[i] = work_t(0);
    work_t A[3] = {static_cast<work_t>(shell_centers[sa * 3 + 0]),
                   static_cast<work_t>(shell_centers[sa * 3 + 1]),
                   static_cast<work_t>(shell_centers[sa * 3 + 2])};
    work_t B[3] = {static_cast<work_t>(shell_centers[sb * 3 + 0]),
                   static_cast<work_t>(shell_centers[sb * 3 + 1]),
                   static_cast<work_t>(shell_centers[sb * 3 + 2])};
    work_t C[3] = {static_cast<work_t>(shell_centers[sc * 3 + 0]),
                   static_cast<work_t>(shell_centers[sc * 3 + 1]),
                   static_cast<work_t>(shell_centers[sc * 3 + 2])};
    work_t D[3] = {static_cast<work_t>(shell_centers[sd * 3 + 0]),
                   static_cast<work_t>(shell_centers[sd * 3 + 1]),
                   static_cast<work_t>(shell_centers[sd * 3 + 2])};
    const int Ltot = La + Lb + Lc + Ld;
    const int oA = shell_prim_off[sa];
    const int oB = shell_prim_off[sb];
    const int oC = shell_prim_off[sc];
    const int oD = shell_prim_off[sd];
    for (int p = 0; p < shell_nprim[sa]; ++p) {
        const work_t ap = static_cast<work_t>(prim_alpha[oA + p]);
        const work_t cp = static_cast<work_t>(prim_coeff[oA + p]);
        for (int q = 0; q < shell_nprim[sb]; ++q) {
            const work_t aq = static_cast<work_t>(prim_alpha[oB + q]);
            const work_t cq = static_cast<work_t>(prim_coeff[oB + q]);
            for (int r = 0; r < shell_nprim[sc]; ++r) {
                const work_t ar = static_cast<work_t>(prim_alpha[oC + r]);
                const work_t cr = static_cast<work_t>(prim_coeff[oC + r]);
                for (int s = 0; s < shell_nprim[sd]; ++s) {
                    const work_t as = static_cast<work_t>(prim_alpha[oD + s]);
                    const work_t cs = static_cast<work_t>(prim_coeff[oD + s]);
                    const work_t gamma1 = ap + aq;
                    const work_t gamma2 = ar + as;
                    const work_t delta = gamma1 + gamma2;
                    const work_t mu1 = ap * aq / gamma1;
                    const work_t mu2 = ar * as / gamma2;
                    const work_t rho = gamma1 * gamma2 / delta;
                    const work_t inv2g1 = work_t(0.5) / gamma1;
                    const work_t inv2g2 = work_t(0.5) / gamma2;
                    work_t P[3], Q[3], PA[3], PB[3], QC[3], QD[3], PQ[3];
                    work_t R2_AB = 0, R2_CD = 0, PQ2 = 0;
                    for (int d = 0; d < 3; ++d) {
                        P[d] = (ap * A[d] + aq * B[d]) / gamma1;
                        Q[d] = (ar * C[d] + as * D[d]) / gamma2;
                        PA[d] = P[d] - A[d];
                        PB[d] = P[d] - B[d];
                        QC[d] = Q[d] - C[d];
                        QD[d] = Q[d] - D[d];
                        PQ[d] = P[d] - Q[d];
                        const work_t dAB = A[d] - B[d];
                        const work_t dCD = C[d] - D[d];
                        R2_AB += dAB * dAB;
                        R2_CD += dCD * dCD;
                        PQ2 += PQ[d] * PQ[d];
                    }
                    const work_t K_AB = exp(-mu1 * R2_AB);
                    const work_t K_CD = exp(-mu2 * R2_CD);
                    const work_t Tval = rho * PQ2;
                    const work_t pi = work_t(ATV::units::pi);
                    const work_t prefactor = work_t(2) * pi * pi * sqrt(pi)
                                             / (gamma1 * gamma2 * sqrt(delta))
                                             * K_AB * K_CD * cp * cq * cr * cs;
                    if constexpr (FShell)
                        build_hermite_R<kFDim>(R, Ltot, rho, Tval, PQ);
                    else
                        build_hermite_R<kL2Dim>(R, Ltot, rho, Tval, PQ);
                    int cout = 0;
                    for (int ia = 0; ia < na; ++ia) {
                        int a_ang[3] = {0, 0, 0};
                        cart_ang_device(La, ia, a_ang);
                        for (int ib = 0; ib < nb; ++ib) {
                            int b_ang[3] = {0, 0, 0};
                            cart_ang_device(Lb, ib, b_ang);
                            for (int ic = 0; ic < nc; ++ic) {
                                int c_ang[3] = {0, 0, 0};
                                cart_ang_device(Lc, ic, c_ang);
                                for (int id = 0; id < nd; ++id) {
                                    int d_ang[3] = {0, 0, 0};
                                    cart_ang_device(Ld, id, d_ang);
                                    if constexpr (FShell) {
                                        acc[cout] += contract_md<true, kFDim>(
                                            R, a_ang, b_ang, c_ang, d_ang, PA,
                                            PB, QC, QD, inv2g1, inv2g2,
                                            prefactor);
                                    } else {
                                        acc[cout] += contract_md<false, kL2Dim>(
                                            R, a_ang, b_ang, c_ang, d_ang, PA,
                                            PB, QC, QD, inv2g1, inv2g2,
                                            prefactor);
                                    }
                                    ++cout;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    const int n2 = n_ao * n_ao;
    const int n3 = n2 * n_ao;
    int cout = 0;
    for (int ia = 0; ia < na; ++ia)
        for (int ib = 0; ib < nb; ++ib)
            for (int ic = 0; ic < nc; ++ic)
                for (int id = 0; id < nd; ++id) {
                    const int mu = ao_a + ia;
                    const int nu = ao_b + ib;
                    const int lam = ao_c + ic;
                    const int sig = ao_d + id;
                    ERI[mu + nu * n_ao + lam * n2 + sig * n3] =
                        static_cast<prec_t>(acc[cout]);
                    ++cout;
                }
}

__device__ void decode_shell_quartet(long long q, long long nsh, int& sa,
                                     int& sb, int& sc, int& sd) {
    sd = static_cast<int>(q / (nsh * nsh * nsh));
    long long rem = q % (nsh * nsh * nsh);
    sc = static_cast<int>(rem / (nsh * nsh));
    rem = rem % (nsh * nsh);
    sb = static_cast<int>(rem / nsh);
    sa = static_cast<int>(rem % nsh);
}

__global__ void
compute_eri_lgt1_kernel(prec_t* ERI, const prec_t* prim_alpha,
                        const prec_t* prim_coeff, const int* shell_nprim,
                        const int* shell_prim_off, const prec_t* shell_centers,
                        const int* shell_L, const int* shell_ao_start,
                        int n_shells, int n_ao) {
    work_t R[kL2Dim * kL2Dim * kL2Dim * kL2Dim];
    work_t acc[kL2Out];
    const long long nsh = n_shells;
    const long long n4 = nsh * nsh * nsh * nsh;
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long q =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         q < n4; q += stride) {
        int sa, sb, sc, sd;
        decode_shell_quartet(q, nsh, sa, sb, sc, sd);
        const int La = shell_L[sa];
        const int Lb = shell_L[sb];
        const int Lc = shell_L[sc];
        const int Ld = shell_L[sd];
        if (La > 2 || Lb > 2 || Lc > 2 || Ld > 2)
            continue;
        eri_quartet_md<false>(ERI, prim_alpha, prim_coeff, shell_nprim,
                              shell_prim_off, shell_centers, sa, sb, sc, sd, La,
                              Lb, Lc, Ld, shell_ao_start[sa],
                              shell_ao_start[sb], shell_ao_start[sc],
                              shell_ao_start[sd], n_ao, R, acc);
    }
}

__global__ void
compute_eri_f_kernel(prec_t* ERI, const prec_t* prim_alpha,
                     const prec_t* prim_coeff, const int* shell_nprim,
                     const int* shell_prim_off, const prec_t* shell_centers,
                     const int* shell_L, const int* shell_ao_start,
                     int n_shells, int n_ao, work_t* scratch, int n_workers) {
    const int worker = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (worker >= n_workers)
        return;
    const int rdim = kFDim * kFDim * kFDim * kFDim;
    work_t* R = scratch + static_cast<long long>(worker) * (rdim + kFOut);
    work_t* acc = R + rdim;
    const long long nsh = n_shells;
    const long long n4 = nsh * nsh * nsh * nsh;
    for (long long q = worker; q < n4; q += n_workers) {
        int sa, sb, sc, sd;
        decode_shell_quartet(q, nsh, sa, sb, sc, sd);
        const int La = shell_L[sa];
        const int Lb = shell_L[sb];
        const int Lc = shell_L[sc];
        const int Ld = shell_L[sd];
        if (La > 3 || Lb > 3 || Lc > 3 || Ld > 3)
            continue;
        if (La < 3 && Lb < 3 && Lc < 3 && Ld < 3)
            continue;
        eri_quartet_md<true>(ERI, prim_alpha, prim_coeff, shell_nprim,
                             shell_prim_off, shell_centers, sa, sb, sc, sd, La,
                             Lb, Lc, Ld, shell_ao_start[sa], shell_ao_start[sb],
                             shell_ao_start[sc], shell_ao_start[sd], n_ao, R,
                             acc);
    }
}

void launch_1e_integrals_lgt1(double* S, double* T, double* V,
                              const double* prim_alpha,
                              const double* prim_coeff, const int* shell_nprim,
                              const int* shell_prim_off,
                              const double* shell_centers, const int* shell_L,
                              const int* ao_shell, const int* ao_cart,
                              const int* atom_Z, const double* atom_pos,
                              int n_ao, int natm, int max_l, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    const int n_pairs = n_ao * (n_ao + 1) / 2;
    const int threads = cuda::kDefaultBlock;
    const int blocks = cuda::grid_size(n_pairs, threads);
    if (max_l >= 3) {
        compute_1e_lgt1_kernel<true><<<blocks, threads, 0, s>>>(
            S, T, V, prim_alpha, prim_coeff, shell_nprim, shell_prim_off,
            shell_centers, shell_L, ao_shell, ao_cart, atom_Z, atom_pos, n_ao,
            natm);
    } else {
        compute_1e_lgt1_kernel<false><<<blocks, threads, 0, s>>>(
            S, T, V, prim_alpha, prim_coeff, shell_nprim, shell_prim_off,
            shell_centers, shell_L, ao_shell, ao_cart, atom_Z, atom_pos, n_ao,
            natm);
    }
}

void launch_eri_lgt1(double* ERI, const double* prim_alpha,
                     const double* prim_coeff, const int* shell_nprim,
                     const int* shell_prim_off, const double* shell_centers,
                     const int* shell_L, const int* shell_ao_start,
                     int n_shells, int n_ao, int max_l, void* stream) {
    if (n_shells > 256) {
        throw std::invalid_argument(
            "launch_eri_lgt1: n_shells=" + std::to_string(n_shells)
            + " exceeds 256 (Cartesian l<=3 device ERI)");
    }
    auto s = static_cast<cudaStream_t>(stream);
    const long long n4 = 1LL * n_shells * n_shells * n_shells * n_shells;
    const int threads = cuda::kDefaultBlock;
    int blocks = static_cast<int>((n4 + threads - 1) / threads);
    if (blocks > 2048)
        blocks = 2048;
    if (blocks < 1)
        blocks = 1;
    compute_eri_lgt1_kernel<<<blocks, threads, 0, s>>>(
        ERI, prim_alpha, prim_coeff, shell_nprim, shell_prim_off, shell_centers,
        shell_L, shell_ao_start, n_shells, n_ao);
    if (max_l >= 3) {
        constexpr int kWorkers = 128;
        const int rdim = kFDim * kFDim * kFDim * kFDim;
        work_t* scratch = nullptr;
        cudaMalloc(&scratch, static_cast<std::size_t>(kWorkers)
                                 * static_cast<std::size_t>(rdim + kFOut)
                                 * sizeof(work_t));
        compute_eri_f_kernel<<<4, 32, 0, s>>>(
            ERI, prim_alpha, prim_coeff, shell_nprim, shell_prim_off,
            shell_centers, shell_L, shell_ao_start, n_shells, n_ao, scratch,
            kWorkers);
        cudaStreamSynchronize(s);
        cudaFree(scratch);
    }
}

} // namespace scf
} // namespace cuda
} // namespace ATV

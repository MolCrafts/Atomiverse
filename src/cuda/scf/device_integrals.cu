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

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/scf/device_integrals.cuh"

#include "atomiverse/cuda/frame/types.cuh"
#include <cmath>
#include <cuda_runtime.h>

namespace ATV {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace scf {

static constexpr work_t kBoysEpsilon = work_t(1e-12);

// ── Device helpers ────────────────────────────────────────────────────

__device__ __forceinline__ work_t boys_f0_device(work_t x) {
    if (x < kBoysEpsilon)
        return work_t(1);
    work_t sx = sqrt(x);
    return work_t(0.5) * sqrt(work_t(ATV::units::pi) / x) * erf(sx);
}

/// Boys F_n via upward recurrence from F_0 (core/gto.h).
__device__ __forceinline__ work_t boys_fn_device(int n, work_t x) {
    if (n == 0)
        return boys_f0_device(x);
    if (x < kBoysEpsilon)
        return work_t(1) / work_t(2 * n + 1);
    work_t ex = exp(-x);
    work_t inv2x = work_t(0.5) / x;
    work_t fn = boys_f0_device(x);
    for (int k = 1; k <= n; ++k)
        fn = (work_t(2 * k - 1) * fn - ex) * inv2x;
    return fn;
}

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
    for (int m = 0; m <= Ltot && m < kDeviceRNuc; ++m)
        Fm[m] = boys_fn_device(m, T);

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

    const int Lx1 = a_ang[0] + b_ang[0];
    const int Ly1 = a_ang[1] + b_ang[1];
    const int Lz1 = a_ang[2] + b_ang[2];
    const int Lx2 = c_ang[0] + d_ang[0];
    const int Ly2 = c_ang[1] + d_ang[1];
    const int Lz2 = c_ang[2] + d_ang[2];
    const int Ltot = Lx1 + Ly1 + Lz1 + Lx2 + Ly2 + Lz2; // ≤5 with force raise

    work_t Fm[kDeviceREri];
    for (int m = 0; m <= Ltot && m < kDeviceREri; ++m)
        Fm[m] = boys_fn_device(m, Tval);

    work_t E1[3][kDeviceMDEDim], E2[3][kDeviceMDEDim];
    for (int d = 0; d < 3; ++d) {
        md_e_coeffs_device(a_ang[d], b_ang[d], PA[d], PB[d], inv2g1, E1[d]);
        md_e_coeffs_device(c_ang[d], d_ang[d], QC[d], QD[d], inv2g2, E2[d]);
    }

    const int Rx_max = Lx1 + Lx2;
    const int Ry_max = Ly1 + Ly2;
    const int Rz_max = Lz1 + Lz2;

    work_t R_tab[kDeviceREri][kDeviceREri][kDeviceREri][kDeviceREri];
    for (int m = 0; m < kDeviceREri; ++m)
        for (int t = 0; t < kDeviceREri; ++t)
            for (int u = 0; u < kDeviceREri; ++u)
                for (int v = 0; v < kDeviceREri; ++v)
                    R_tab[m][t][u][v] = work_t(0);

    work_t neg2rho = -work_t(2) * rho;
    work_t neg2rho_pow = work_t(1);
    for (int m = 0; m <= Ltot; ++m) {
        R_tab[m][0][0][0] = neg2rho_pow * Fm[m];
        neg2rho_pow *= neg2rho;
    }
    for (int m = Ltot - 1; m >= 0; --m) {
        const int max_tuv = Ltot - m;
        for (int t = 1; t <= max_tuv && t <= Rx_max; ++t) {
            R_tab[m][t][0][0] = PQ[0] * R_tab[m + 1][t - 1][0][0];
            if (t >= 2)
                R_tab[m][t][0][0] += work_t(t - 1) * R_tab[m + 1][t - 2][0][0];
        }
        for (int t = 0; t <= Rx_max && t <= max_tuv; ++t) {
            for (int u = 1; u <= Ry_max && t + u <= max_tuv; ++u) {
                R_tab[m][t][u][0] = PQ[1] * R_tab[m + 1][t][u - 1][0];
                if (u >= 2)
                    R_tab[m][t][u][0] +=
                        work_t(u - 1) * R_tab[m + 1][t][u - 2][0];
            }
        }
        for (int t = 0; t <= Rx_max && t <= max_tuv; ++t) {
            for (int u = 0; u <= Ry_max && t + u <= max_tuv; ++u) {
                for (int v = 1; v <= Rz_max && t + u + v <= max_tuv; ++v) {
                    R_tab[m][t][u][v] = PQ[2] * R_tab[m + 1][t][u][v - 1];
                    if (v >= 2)
                        R_tab[m][t][u][v] +=
                            work_t(v - 1) * R_tab[m + 1][t][u][v - 2];
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
                            eri_val += e1 * e2 * sign
                                       * R_tab[0][t + tau][u + nu][v + phi];
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
        out[i] = boys_f0_device(x[i]);
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
                             * boys_f0_device(gamma * PC2);
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
                                  * K_AB * K_CD * boys_f0_device(rho * PQ2);
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
        ERI[q] = result;
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
            work_t F1 = boys_fn_device(1, T);
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

} // namespace scf
} // namespace cuda
} // namespace ATV

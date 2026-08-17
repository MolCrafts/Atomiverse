/// @file cpu/scf/rys_eri.cpp
/// @brief Native Rys-quadrature ERI engine — algorithm identical to
///        libcint's `CINTg0_2e` + `CINTg0_2e_2d` + HRR shift.
///
/// Implements `native::rys_eri_block` using the same 2D-VRR and HRR
/// algebra as libcint (see its `src/g2e.c`). The only dependency on
/// libcint is the closed-form Rys roots generator, which has been
/// ported into native code under `native::rys_roots` (see
/// rys_roots.cpp).
///
/// Design choices:
///  * Output buffer layout and normalization conventions match the
///    existing `native::eri_block` so this engine is a drop-in
///    replacement (callers, tests, and contracted coefficients do not
///    need to change).
///  * Shell-level swaps are avoided: we pick `i_base = li >= lj` and
///    `k_base = lk >= ll` and run the 2D-VRR in the base direction,
///    then apply HRR in the shift direction.
///  * Support range: s, p, d, f shells (l ≤ 3) on all four centers;
///    g shells (l = 4) on the 2-/3-center density-fitting paths, where
///    the root demand stays within kRysMaxRoots (see rys_eri.h).
///  * Scalar fast path first; xsimd / xtensor-blas optimisations are
///    layered on top without changing the public API.

#include "atomiverse/cpu/scf/rys_eri.h"
#include "atomiverse/cpu/scf/cart.h"
#include "atomiverse/cpu/scf/rys_roots.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ATV {
namespace cpu {
namespace scf {
namespace native {

namespace {

constexpr double kTwoPi25 = 2.0 * 17.493418327624863; // 2 * π^(5/2)

// Struct wrapper around the generic int[3] layout from cart.h — gives
// `.nx / .ny / .nz` field access inside the hot inner loop.
struct CartComp {
    int nx, ny, nz;
};

// Root-budget gate shared by the plain and derivative public wrappers.
// g shells (l = 4) are supported for the 2-/3-center density-fitting
// paths, whose heaviest case is Ltot = 10 (6 roots); a 4-center quartet
// that pushes past the 7-root budget (e.g. (gg|gg)) throws instead of
// silently producing garbage roots.
static void check_root_budget(const ShellBlock& A, const ShellBlock& B,
                              const ShellBlock& C, const ShellBlock& D,
                              int extra_l) {
    const int ltot = A.l + B.l + C.l + D.l + extra_l;
    const int nroots = ltot / 2 + 1;
    if (nroots > kRysMaxRoots) {
        throw std::runtime_error(
            "rys_eri_block: shell quartet (l = " + std::to_string(A.l) + ","
            + std::to_string(B.l) + "|" + std::to_string(C.l) + ","
            + std::to_string(D.l) + ") needs " + std::to_string(nroots)
            + " Rys roots (budget " + std::to_string(kRysMaxRoots)
            + "). l = 4 (g) shells are supported only on the 2-/3-center "
              "auxiliary-basis paths; the orbital basis is capped at f.");
    }
}

static void fill_cart_comps_struct(int l, CartComp* out) {
    int tmp[kMaxNCart][3];
    fill_cart_comps(l, tmp);
    const int n = n_cart_of(l);
    for (int c = 0; c < n; ++c) {
        out[c].nx = tmp[c][0];
        out[c].ny = tmp[c][1];
        out[c].nz = tmp[c][2];
    }
}

// ── Rys work buffer layout ───────────────────────────────────────────
//
// The 2D intermediate g_{axis}[i][k] has dimensions
//   dli × dlk  per root, per axis (x / y / z).
// We lay it out flat with root innermost:
//     g[axis * g_size + j*dj + l*dl + k*dk + i*di + root]
// where
//   di = nroots
//   dk = nroots * dli
//   dl = nroots * dli * dlk
//   dj = nroots * dli * dlk * dll
// The bra (i) axis holds li+lj+1 entries before HRR; the ket axis holds
// lk+ll+1 entries. Post-HRR, j / l slots become populated.
//
// All three cartesian axes share the same stride structure, just offset
// by g_size.

struct RysEnv {
    int li, lj, lk, ll;      // angular momenta (i >= j, k >= l assumed).
    int nfi, nfj, nfk, nfl;  // n_cart per shell.
    int nmax, mmax;          // li + lj, lk + ll (bra / ket power sum).
    int nroots;              // Rys order.
    int dli, dlj, dlk, dll;  // g strides per axis.
    int di, dk, dl, dj;      // i/k/l/j multipliers (flat layout).
    int g_size;              // per-axis / per-root buffer length.
    double rirj[3], rkrl[3]; // centre differences used by HRR.
    bool i_base;             // true: 2D-VRR along li direction.
    bool k_base;             // true: 2D-VRR along lk direction.
};

static void init_env(RysEnv& env, const ShellBlock& A, const ShellBlock& B,
                     const ShellBlock& C, const ShellBlock& D) {
    env.li = A.l;
    env.lj = B.l;
    env.lk = C.l;
    env.ll = D.l;
    env.nfi = n_cart_of(env.li);
    env.nfj = n_cart_of(env.lj);
    env.nfk = n_cart_of(env.lk);
    env.nfl = n_cart_of(env.ll);
    env.nmax = env.li + env.lj;
    env.mmax = env.lk + env.ll;
    env.nroots = (env.nmax + env.mmax) / 2 + 1;

    env.i_base = env.li >= env.lj;
    env.k_base = env.lk >= env.ll;

    if (env.i_base) {
        env.dli = env.li + env.lj + 1;
        env.dlj = env.lj + 1;
    } else {
        env.dli = env.li + 1;
        env.dlj = env.li + env.lj + 1;
    }
    if (env.k_base) {
        env.dlk = env.lk + env.ll + 1;
        env.dll = env.ll + 1;
    } else {
        env.dlk = env.lk + 1;
        env.dll = env.lk + env.ll + 1;
    }

    env.di = env.nroots;
    env.dk = env.nroots * env.dli;
    env.dl = env.nroots * env.dli * env.dlk;
    env.dj = env.nroots * env.dli * env.dlk * env.dll;
    env.g_size = env.nroots * env.dli * env.dlk * env.dll * env.dlj;

    for (int d = 0; d < 3; ++d) {
        env.rirj[d] = A.center[d] - B.center[d];
        env.rkrl[d] = C.center[d] - D.center[d];
    }
}

// ── 2D Rys intermediate (VRR) ────────────────────────────────────────
//
// Ports libcint's `CINTg0_2e_2d` almost verbatim, adapted to our layout.
// On entry, gx[root] = gy[root] = 1 and gz[root] = weight[root] * fac.
// Builds g[root + i*di + k*dk] for i ∈ [0, nmax], k ∈ [0, mmax].

struct Rys2eCoeff {
    double c00x[kRysMaxRoots], c00y[kRysMaxRoots], c00z[kRysMaxRoots];
    double c0px[kRysMaxRoots], c0py[kRysMaxRoots], c0pz[kRysMaxRoots];
    double b00[kRysMaxRoots], b10[kRysMaxRoots], b01[kRysMaxRoots];
};

static void g0_2e_2d(double* g, const Rys2eCoeff& bc, const RysEnv& env) {
    const int nroots = env.nroots;
    const int nmax = env.nmax;
    const int mmax = env.mmax;
    const int di = env.di;
    const int dk = env.dk;
    const int g_size = env.g_size;

    double* gx = g;
    double* gy = g + g_size;
    double* gz = g + g_size * 2;

    for (int i = 0; i < nroots; ++i) {
        gx[i] = 1.0;
        gy[i] = 1.0;
        // gz[i] is weight*fac, set by caller.
    }

    for (int irys = 0; irys < nroots; ++irys) {
        const double c00x = bc.c00x[irys];
        const double c00y = bc.c00y[irys];
        const double c00z = bc.c00z[irys];
        const double c0px = bc.c0px[irys];
        const double c0py = bc.c0py[irys];
        const double c0pz = bc.c0pz[irys];
        const double b10 = bc.b10[irys];
        const double b01 = bc.b01[irys];
        const double b00 = bc.b00[irys];

        // ── i-axis recurrence (k=0) ─────────────────────────────────
        if (nmax > 0) {
            double s0x = gx[irys];
            double s0y = gy[irys];
            double s0z = gz[irys];
            double s1x = c00x * s0x;
            double s1y = c00y * s0y;
            double s1z = c00z * s0z;
            gx[irys + di] = s1x;
            gy[irys + di] = s1y;
            gz[irys + di] = s1z;
            for (int n = 1; n < nmax; ++n) {
                double s2x = c00x * s1x + n * b10 * s0x;
                double s2y = c00y * s1y + n * b10 * s0y;
                double s2z = c00z * s1z + n * b10 * s0z;
                gx[irys + (n + 1) * di] = s2x;
                gy[irys + (n + 1) * di] = s2y;
                gz[irys + (n + 1) * di] = s2z;
                s0x = s1x;
                s0y = s1y;
                s0z = s1z;
                s1x = s2x;
                s1y = s2y;
                s1z = s2z;
            }
        }

        // ── k-axis recurrence (i=0) ─────────────────────────────────
        if (mmax > 0) {
            double s0x = gx[irys];
            double s0y = gy[irys];
            double s0z = gz[irys];
            double s1x = c0px * s0x;
            double s1y = c0py * s0y;
            double s1z = c0pz * s0z;
            gx[irys + dk] = s1x;
            gy[irys + dk] = s1y;
            gz[irys + dk] = s1z;
            for (int m = 1; m < mmax; ++m) {
                double s2x = c0px * s1x + m * b01 * s0x;
                double s2y = c0py * s1y + m * b01 * s0y;
                double s2z = c0pz * s1z + m * b01 * s0z;
                gx[irys + (m + 1) * dk] = s2x;
                gy[irys + (m + 1) * dk] = s2y;
                gz[irys + (m + 1) * dk] = s2z;
                s0x = s1x;
                s0y = s1y;
                s0z = s1z;
                s1x = s2x;
                s1y = s2y;
                s1z = s2z;
            }

            // ── (i=1, k) block seeded from (i=1, 0) and (0, k) ─────
            if (nmax > 0) {
                double s0x2 = gx[irys + di];
                double s0y2 = gy[irys + di];
                double s0z2 = gz[irys + di];
                double s1x2 = c0px * s0x2 + b00 * gx[irys];
                double s1y2 = c0py * s0y2 + b00 * gy[irys];
                double s1z2 = c0pz * s0z2 + b00 * gz[irys];
                gx[irys + di + dk] = s1x2;
                gy[irys + di + dk] = s1y2;
                gz[irys + di + dk] = s1z2;
                for (int m = 1; m < mmax; ++m) {
                    double s2x2 =
                        c0px * s1x2 + m * b01 * s0x2 + b00 * gx[irys + m * dk];
                    double s2y2 =
                        c0py * s1y2 + m * b01 * s0y2 + b00 * gy[irys + m * dk];
                    double s2z2 =
                        c0pz * s1z2 + m * b01 * s0z2 + b00 * gz[irys + m * dk];
                    gx[irys + di + (m + 1) * dk] = s2x2;
                    gy[irys + di + (m + 1) * dk] = s2y2;
                    gz[irys + di + (m + 1) * dk] = s2z2;
                    s0x2 = s1x2;
                    s0y2 = s1y2;
                    s0z2 = s1z2;
                    s1x2 = s2x2;
                    s1y2 = s2y2;
                    s1z2 = s2z2;
                }
            }
        }

        // ── (i>=2, k>=1) ────────────────────────────────────────────
        for (int m = 1; m <= mmax; ++m) {
            int off = m * dk;
            int j_idx = off + irys;
            double s0x = gx[j_idx];
            double s0y = gy[j_idx];
            double s0z = gz[j_idx];
            double s1x = gx[j_idx + di];
            double s1y = gy[j_idx + di];
            double s1z = gz[j_idx + di];
            for (int n = 1; n < nmax; ++n) {
                double s2x = c00x * s1x + n * b10 * s0x
                             + m * b00 * gx[j_idx + n * di - dk];
                double s2y = c00y * s1y + n * b10 * s0y
                             + m * b00 * gy[j_idx + n * di - dk];
                double s2z = c00z * s1z + n * b10 * s0z
                             + m * b00 * gz[j_idx + n * di - dk];
                gx[j_idx + (n + 1) * di] = s2x;
                gy[j_idx + (n + 1) * di] = s2y;
                gz[j_idx + (n + 1) * di] = s2z;
                s0x = s1x;
                s0y = s1y;
                s0z = s1z;
                s1x = s2x;
                s1y = s2y;
                s1z = s2z;
            }
        }
    }
}

// ── Horizontal recurrence (HRR / 2D → 4D shift) ──────────────────────
//
// Same algebra as libcint's CINTg0_ik2d_4d: first shift in the l axis
// (fills the ll > 0 slots), then shift in the j axis. Both shifts
// assume the base direction (i, k) already carries li+lj, lk+ll entries
// from the 2D-VRR.
//   g(k,l)     = rkrl · g(k, l-1) + g(k+1, l-1)
//   g(i,j)     = rirj · g(i, j-1) + g(i+1, j-1)

static void g_shift_kl(double* gaxis, const RysEnv& env, double rkrl_d) {
    if (env.ll == 0)
        return;
    const int di = env.di;
    const int dk = env.dk;
    const int dl = env.dl;
    const int nroots = env.nroots;
    const int nmax = env.nmax;
    const int mmax = env.mmax;

    for (int l = 1; l <= env.ll; ++l) {
        for (int k = 0; k <= mmax - l; ++k) {
            for (int i = 0; i <= nmax; ++i) {
                const int ptr = l * dl + k * dk + i * di;
                for (int r = 0; r < nroots; ++r) {
                    gaxis[ptr + r] =
                        rkrl_d * gaxis[ptr + r - dl] + gaxis[ptr + r - dl + dk];
                }
            }
        }
    }
}

static void g_shift_ij(double* gaxis, const RysEnv& env, double rirj_d) {
    if (env.lj == 0)
        return;
    const int di = env.di;
    const int dj = env.dj;
    const int dk = env.dk;
    const int dl = env.dl;
    const int nroots = env.nroots;
    const int nmax = env.nmax;

    for (int j = 1; j <= env.lj; ++j) {
        for (int l = 0; l <= env.ll; ++l) {
            for (int k = 0; k <= env.lk; ++k) {
                const int ptr = j * dj + l * dl + k * dk;
                // i runs from 0 to nmax-j inclusive; each i stride is
                // nroots doubles — flatten the (i, root) loop for the
                // compiler to vectorize.
                const int n_end = nroots * (nmax + 1 - j);
                for (int n = 0; n < n_end; ++n) {
                    gaxis[ptr + n] =
                        rirj_d * gaxis[ptr + n - dj] + gaxis[ptr + n - dj + di];
                }
            }
        }
    }
}

// ── Thread-local scratch buffer ──────────────────────────────────────
// The Rys working buffer is sized per shell-quartet (3 axes × g_size
// doubles). With shell-block direct-SCF calling rys_eri_block millions
// of times, heap allocation on every call is a measurable cost. A
// thread_local vector reuses storage across calls while keeping the
// Fock builder's `#pragma omp parallel` happy — each worker thread has
// its own buffer.
static thread_local std::vector<double> tls_g_buf;

// Precomputed angular index offsets for one canonical quartet. Each worker
// thread reuses one buffer across calls (resize-if-smaller), so the millions
// of shell-block quartets in a direct-SCF Fock build stop heap-allocating a
// fresh vector per quartet. Fully overwritten each call (no zeroing needed);
// per-thread storage keeps the Fock builder's `#pragma omp parallel` race-free.
struct AngIdx {
    int ox, oy, oz, out_idx;
};
static thread_local std::vector<AngIdx> tls_ang_idx;

// Scratch for the non-canonical wrapper's canonical-kernel output. Reused
// per thread (resize-if-smaller); the used prefix is zeroed before each
// canonical accumulation.
static thread_local std::vector<double> tls_swap_scratch;

// ── Canonical-order ERI kernel ───────────────────────────────────────
// Runs the 2D-VRR + HRR assuming `li >= lj` and `lk >= ll`. The public
// `rys_eri_block` wrapper swaps shells and transposes the output buffer
// when those invariants are violated.
//
// omega == 0 is the bare 1/r path (bit-identical to the pre-RSH kernel).
// omega > 0 evaluates the erf(ω r)/r operator by the standard ρ-rescale
// (libcint / PySCF): θ = ω²/(ω²+ρ), ρ ← θ ρ. VRR/HRR are unchanged.
static void rys_eri_block_canonical(const ShellBlock& A, const ShellBlock& B,
                                    const ShellBlock& C, const ShellBlock& D,
                                    double omega, double* out) {
    RysEnv env;
    init_env(env, A, B, C, D);

    const size_t need = static_cast<size_t>(env.g_size) * 3;
    if (tls_g_buf.size() < need)
        tls_g_buf.resize(need);
    double* g = tls_g_buf.data();

    CartComp ci[kMaxNCart], cj[kMaxNCart], ck[kMaxNCart], cl[kMaxNCart];
    fill_cart_comps_struct(env.li, ci);
    fill_cart_comps_struct(env.lj, cj);
    fill_cart_comps_struct(env.lk, ck);
    fill_cart_comps_struct(env.ll, cl);

    const int nb = env.nfj;
    const int nc = env.nfk;
    const int nd = env.nfl;
    const int ncd = nc * nd;
    const int nbcd = nb * ncd;
    const int n_out = env.nfi * nbcd;

    // ── Precompute angular index offsets (invariant across primitives) ──
    // d-shell quartets need up to 6^4 = 1296 entries. The buffer is a
    // per-thread thread_local reused across quartets (resize-if-smaller),
    // fully overwritten below (indices 0..n_out-1), so no zeroing is required
    // and no heap allocation happens on the hot path.
    if (tls_ang_idx.size() < static_cast<size_t>(n_out))
        tls_ang_idx.resize(static_cast<size_t>(n_out));
    AngIdx* ang_idx = tls_ang_idx.data();
    {
        int c = 0;
        for (int ia = 0; ia < env.nfi; ++ia)
            for (int ib = 0; ib < nb; ++ib)
                for (int ic = 0; ic < nc; ++ic)
                    for (int id = 0; id < nd; ++id) {
                        ang_idx[c].ox = ci[ia].nx * env.di + cj[ib].nx * env.dj
                                        + ck[ic].nx * env.dk
                                        + cl[id].nx * env.dl;
                        ang_idx[c].oy = ci[ia].ny * env.di + cj[ib].ny * env.dj
                                        + ck[ic].ny * env.dk
                                        + cl[id].ny * env.dl;
                        ang_idx[c].oz = ci[ia].nz * env.di + cj[ib].nz * env.dj
                                        + ck[ic].nz * env.dk
                                        + cl[id].nz * env.dl;
                        ang_idx[c].out_idx =
                            ia * nbcd + ib * ncd + ic * nd + id;
                        ++c;
                    }
    }

    // ── Precompute ket geometry (invariant across bra primitives) ──────
    struct KetPair {
        double akl, rkl[3], kcd_coeff;
    };
    constexpr int KMAX = 64; // max ket primitive pairs (8×8 for cc-pVTZ)
    KetPair ket_pairs[KMAX];
    const int nket = C.n_prim * D.n_prim;
    {
        int k = 0;
        for (int r = 0; r < C.n_prim; ++r) {
            const double ac = C.alpha[r];
            for (int s = 0; s < D.n_prim; ++s) {
                const double ad = D.alpha[s];
                const double akl = ac + ad;
                ket_pairs[k].akl = akl;
                for (int d = 0; d < 3; ++d)
                    ket_pairs[k].rkl[d] =
                        (ac * C.center[d] + ad * D.center[d]) / akl;
                const double dCD[3] = {C.center[0] - D.center[0],
                                       C.center[1] - D.center[1],
                                       C.center[2] - D.center[2]};
                const double rCD2 =
                    dCD[0] * dCD[0] + dCD[1] * dCD[1] + dCD[2] * dCD[2];
                ket_pairs[k].kcd_coeff =
                    C.coeff[r] * D.coeff[s] * std::exp(-ac * ad / akl * rCD2);
                ++k;
            }
        }
    }

    for (int p = 0; p < A.n_prim; ++p) {
        const double aa = A.alpha[p];
        const double ca = A.coeff[p];

        for (int q = 0; q < B.n_prim; ++q) {
            const double ab = B.alpha[q];
            const double cb = B.coeff[q];
            const double aij = aa + ab;
            const double rij[3] = {
                (aa * A.center[0] + ab * B.center[0]) / aij,
                (aa * A.center[1] + ab * B.center[1]) / aij,
                (aa * A.center[2] + ab * B.center[2]) / aij,
            };
            const double dAB[3] = {
                A.center[0] - B.center[0],
                A.center[1] - B.center[1],
                A.center[2] - B.center[2],
            };
            const double rAB2 =
                dAB[0] * dAB[0] + dAB[1] * dAB[1] + dAB[2] * dAB[2];
            const double K_AB = std::exp(-aa * ab / aij * rAB2);
            const double bra_coeff = ca * cb * K_AB;

            for (int kk = 0; kk < nket; ++kk) {
                const double akl = ket_pairs[kk].akl;
                const double* rkl = ket_pairs[kk].rkl;
                const double prim_coeff = bra_coeff * ket_pairs[kk].kcd_coeff;

                const double a1 = aij * akl;
                const double delta = aij + akl;
                double a0 = a1 / delta;
                if (omega > 0.0) {
                    const double w2 = omega * omega;
                    a0 *= w2 / (w2 + a0);
                }
                const double rijkl[3] = {
                    rij[0] - rkl[0],
                    rij[1] - rkl[1],
                    rij[2] - rkl[2],
                };
                const double rr = rijkl[0] * rijkl[0] + rijkl[1] * rijkl[1]
                                  + rijkl[2] * rijkl[2];
                const double x = a0 * rr;

                // Native convention prefactor — same structure as the
                // existing McMurchie–Davidson `eri_prim`:
                //   prefactor = 2 π^{5/2} / (g1 g2 √(g1+g2)) · K_AB · K_CD · cp·cq·cr·cs
                // Distribute into the Rys quadrature via fac1 that
                // libcint computes internally (`sqrt(a0/(a1³))`):
                //   fac1  = common_factor · √(a0 / a1³)
                //   common_factor = 2 π^{5/2}
                const double fac1 = kTwoPi25 * std::sqrt(a0 / (a1 * a1 * a1));
                const double gz_seed = fac1 * prim_coeff;

                double u[kRysMaxRoots], w[kRysMaxRoots];
                rys_roots(env.nroots, x, u, w);

                Rys2eCoeff bc;
                for (int irys = 0; irys < env.nroots; ++irys) {
                    const double u2 = a0 * u[irys];
                    const double tmp4 = 0.5 / (u2 * delta + a1);
                    const double tmp5 = u2 * tmp4;
                    const double tmp1 = 2.0 * tmp5;
                    const double tmp2 = tmp1 * akl;
                    const double tmp3 = tmp1 * aij;
                    bc.b00[irys] = tmp5;
                    bc.b10[irys] = tmp5 + tmp4 * akl;
                    bc.b01[irys] = tmp5 + tmp4 * aij;
                    // i_base / k_base are true → rx_in_rij = ri, rx_in_rkl = rk.
                    bc.c00x[irys] = (rij[0] - A.center[0]) - tmp2 * rijkl[0];
                    bc.c00y[irys] = (rij[1] - A.center[1]) - tmp2 * rijkl[1];
                    bc.c00z[irys] = (rij[2] - A.center[2]) - tmp2 * rijkl[2];
                    bc.c0px[irys] = (rkl[0] - C.center[0]) + tmp3 * rijkl[0];
                    bc.c0py[irys] = (rkl[1] - C.center[1]) + tmp3 * rijkl[1];
                    bc.c0pz[irys] = (rkl[2] - C.center[2]) + tmp3 * rijkl[2];
                }

                // Seed the z-axis weight. g0_2e_2d resets gx/gy at
                // (i=0, k=0) and the VRR / HRR recurrences overwrite
                // every slot they touch, so the rest of the buffer
                // does not need to be memset — stale values live in
                // slots that the final contraction never reads.
                for (int irys = 0; irys < env.nroots; ++irys) {
                    g[env.g_size * 2 + irys] = w[irys] * gz_seed;
                }

                g0_2e_2d(g, bc, env);

                // HRR: first kl-shift (fills ll > 0 slots), then
                // ij-shift (reads all (k,l) slots). Matches the
                // order used by libcint's CINTg0_ik2d_4d.
                for (int ax = 0; ax < 3; ++ax) {
                    double* ga = g + ax * env.g_size;
                    g_shift_kl(ga, env, env.rkrl[ax]);
                    g_shift_ij(ga, env, env.rirj[ax]);
                }

                // Contract to ERI(a,b,c,d) using precomputed angular indices.
                const int gs = env.g_size;
                for (int c = 0; c < n_out; ++c) {
                    const int ox = ang_idx[c].ox;
                    const int oy = ang_idx[c].oy;
                    const int oz = ang_idx[c].oz;
                    double val = 0.0;
                    for (int irys = 0; irys < env.nroots; ++irys) {
                        val += g[ox + irys] * g[gs + oy + irys]
                               * g[gs * 2 + oz + irys];
                    }
                    out[ang_idx[c].out_idx] += val;
                }
            }
        }
    }
}

// end rys_eri_block_canonical

} // namespace

// ── Derivative kernel: ∂(AB|CD)/∂R_X for X ∈ {A, B, C, D} ────────────
//
// Mirrors libcint's `int2e_ip1` algorithm. The plain ERI kernel above
// builds a 2D Rys grid for each primitive quartet; for the derivative
// we extend that grid by one angular step on the deriv side (bra or
// ket) and then read the derivative out via the nabla recurrence
//    ∂/∂R_{X_d} (ab|cd) = 2·α_X · (shift_+1_d | …) - n_{X,d} · (shift_-1_d | …)
// All three Cartesian axes share the same grid (only the axis-specific
// cartesian stride shifts) — a direct port of CINTnabla1i_2e in libcint.

namespace {

// Variant of g_shift_ij that accepts explicit max_j, max_k, max_l. The
// j-range must be bumped when the deriv is on B (j-side); the k-range
// must be bumped when the deriv is on C (k-side of the ket) because
// nabla reads `g[..., k=lk+1, ...]` at j=lj slots; likewise the l-range
// when the deriv is on D. max_k=env.lk and max_l=env.ll give plain HRR.
static void g_shift_ij_to(double* gaxis, const RysEnv& env, double rirj_d,
                          int max_j, int max_k, int max_l) {
    if (max_j <= 0)
        return;
    const int di = env.di;
    const int dj = env.dj;
    const int dl = env.dl;
    const int dk = env.dk;
    const int nroots = env.nroots;
    const int nmax = env.nmax;

    for (int j = 1; j <= max_j; ++j) {
        for (int l = 0; l <= max_l; ++l) {
            for (int k = 0; k <= max_k; ++k) {
                const int ptr = j * dj + l * dl + k * dk;
                const int n_end = nroots * (nmax + 1 - j);
                for (int n = 0; n < n_end; ++n) {
                    gaxis[ptr + n] =
                        rirj_d * gaxis[ptr + n - dj] + gaxis[ptr + n - dj + di];
                }
            }
        }
    }
}

// Variant of g_shift_kl that lets the caller bump the HRR l-range by one
// (needed when the deriv is on D, the l-side shell).
static void g_shift_kl_to(double* gaxis, const RysEnv& env, double rkrl_d,
                          int max_l) {
    if (max_l <= 0)
        return;
    const int di = env.di;
    const int dk = env.dk;
    const int dl = env.dl;
    const int nroots = env.nroots;
    const int nmax = env.nmax;
    const int mmax = env.mmax;

    for (int l = 1; l <= max_l; ++l) {
        for (int k = 0; k <= mmax - l; ++k) {
            for (int i = 0; i <= nmax; ++i) {
                const int ptr = l * dl + k * dk + i * di;
                for (int r = 0; r < nroots; ++r) {
                    gaxis[ptr + r] =
                        rkrl_d * gaxis[ptr + r - dl] + gaxis[ptr + r - dl + dk];
                }
            }
        }
    }
}

// Populate the enlarged-grid env used by the deriv kernel.
// Input env is the plain init_env output (i_base and k_base enforced true
// by the outer wrapper below). We bump one dimension on the appropriate
// side and recompute strides / nroots.
//
// deriv_center semantics:
//   0 → ∂/∂R_A (bra, i-side): VRR extended by 1 on bra  (dli += 1)
//   1 → ∂/∂R_B (bra, j-side): VRR extended by 1 on bra  (dli += 1)
//                              AND HRR extended by 1 on j (dlj += 1)
//   2 → ∂/∂R_C (ket, k-side): VRR extended by 1 on ket  (dlk += 1)
//   3 → ∂/∂R_D (ket, l-side): VRR extended by 1 on ket  (dlk += 1)
//                              AND HRR extended by 1 on l (dll += 1)
static void enlarge_env_for_deriv(RysEnv& env, int deriv_center) {
    const bool bra_deriv = (deriv_center == 0 || deriv_center == 1);
    const bool j_side = (deriv_center == 1);
    const bool l_side = (deriv_center == 3);

    if (bra_deriv) {
        env.nmax += 1; // extend 2D-VRR on i axis
        env.dli = env.li + env.lj + 2;
        if (j_side)
            env.dlj = env.lj + 2; // extend HRR on j axis
    } else {
        env.mmax += 1; // extend 2D-VRR on k axis
        env.dlk = env.lk + env.ll + 2;
        if (l_side)
            env.dll = env.ll + 2; // extend HRR on l axis
    }

    env.nroots = (env.nmax + env.mmax) / 2 + 1;
    env.di = env.nroots;
    env.dk = env.nroots * env.dli;
    env.dl = env.nroots * env.dli * env.dlk;
    env.dj = env.nroots * env.dli * env.dlk * env.dll;
    env.g_size = env.nroots * env.dli * env.dlk * env.dll * env.dlj;
}

// ── Canonical-order derivative kernel ────────────────────────────────
// Requires A.l >= B.l (i_base) and C.l >= D.l (k_base). The outer
// wrapper `rys_eri_block_deriv` swaps shells to enforce this.
//
// Output layout: out_axes[axis * n_out + ia*nbcd + ib*ncd + ic*nd + id]
// for axis ∈ {0,1,2} = {x,y,z}.
static void rys_eri_block_deriv_canonical(const ShellBlock& A,
                                          const ShellBlock& B,
                                          const ShellBlock& C,
                                          const ShellBlock& D, int deriv_center,
                                          double* out_axes) {
    RysEnv env;
    init_env(env, A, B, C, D);
    enlarge_env_for_deriv(env, deriv_center);

    const size_t need = static_cast<size_t>(env.g_size) * 3;
    if (tls_g_buf.size() < need)
        tls_g_buf.resize(need);
    double* g = tls_g_buf.data();

    CartComp ci[kMaxNCart], cj[kMaxNCart], ck[kMaxNCart], cl[kMaxNCart];
    fill_cart_comps_struct(env.li, ci);
    fill_cart_comps_struct(env.lj, cj);
    fill_cart_comps_struct(env.lk, ck);
    fill_cart_comps_struct(env.ll, cl);

    const int nb = env.nfj;
    const int nc = env.nfk;
    const int nd = env.nfl;
    const int ncd = nc * nd;
    const int nbcd = nb * ncd;
    const int n_out = env.nfi * nbcd;

    // Which shell is the deriv target, and on which Cartesian axis-stride
    // do we shift? For bra_deriv with i-side/j-side we shift by `di`/`dj`
    // on the axis-specific cartesian slot; for ket_deriv by `dk`/`dl`.
    const bool bra_deriv = (deriv_center == 0 || deriv_center == 1);
    const bool j_side = (deriv_center == 1);
    const bool l_side = (deriv_center == 3);
    const int nabla_stride =
        bra_deriv ? (j_side ? env.dj : env.di) : (l_side ? env.dl : env.dk);
    const ShellBlock& deriv_shell = (deriv_center == 0)   ? A
                                    : (deriv_center == 1) ? B
                                    : (deriv_center == 2) ? C
                                                          : D;

    // Precompute angular index offsets.
    // AngIdx.n_xyz_deriv holds the cartesian components of the deriv
    // shell per output-component index — needed for both the raise and
    // lower terms in the nabla recurrence.
    struct AngIdxDeriv {
        int ox, oy, oz, out_idx;
        int n_deriv_x, n_deriv_y, n_deriv_z;
    };
    std::vector<AngIdxDeriv> ang_idx(static_cast<size_t>(n_out));
    {
        int c = 0;
        for (int ia = 0; ia < env.nfi; ++ia)
            for (int ib = 0; ib < nb; ++ib)
                for (int ic = 0; ic < nc; ++ic)
                    for (int id = 0; id < nd; ++id) {
                        AngIdxDeriv& a = ang_idx[c];
                        a.ox = ci[ia].nx * env.di + cj[ib].nx * env.dj
                               + ck[ic].nx * env.dk + cl[id].nx * env.dl;
                        a.oy = ci[ia].ny * env.di + cj[ib].ny * env.dj
                               + ck[ic].ny * env.dk + cl[id].ny * env.dl;
                        a.oz = ci[ia].nz * env.di + cj[ib].nz * env.dj
                               + ck[ic].nz * env.dk + cl[id].nz * env.dl;
                        a.out_idx = ia * nbcd + ib * ncd + ic * nd + id;
                        const CartComp& cd = (deriv_center == 0)   ? ci[ia]
                                             : (deriv_center == 1) ? cj[ib]
                                             : (deriv_center == 2) ? ck[ic]
                                                                   : cl[id];
                        a.n_deriv_x = cd.nx;
                        a.n_deriv_y = cd.ny;
                        a.n_deriv_z = cd.nz;
                        ++c;
                    }
    }

    // Precompute ket geometry (invariant across bra primitives).
    struct KetPair {
        double akl, rkl[3], kcd_coeff;
    };
    constexpr int KMAX = 64;
    KetPair ket_pairs[KMAX];
    const int nket = C.n_prim * D.n_prim;
    {
        int k = 0;
        for (int r = 0; r < C.n_prim; ++r) {
            const double ac = C.alpha[r];
            for (int s = 0; s < D.n_prim; ++s) {
                const double ad = D.alpha[s];
                const double akl = ac + ad;
                ket_pairs[k].akl = akl;
                for (int d = 0; d < 3; ++d)
                    ket_pairs[k].rkl[d] =
                        (ac * C.center[d] + ad * D.center[d]) / akl;
                const double dCD[3] = {C.center[0] - D.center[0],
                                       C.center[1] - D.center[1],
                                       C.center[2] - D.center[2]};
                const double rCD2 =
                    dCD[0] * dCD[0] + dCD[1] * dCD[1] + dCD[2] * dCD[2];
                ket_pairs[k].kcd_coeff =
                    C.coeff[r] * D.coeff[s] * std::exp(-ac * ad / akl * rCD2);
                ++k;
            }
        }
    }

    const int n_deriv_prims = deriv_shell.n_prim;
    (void)n_deriv_prims;

    // HRR ranges — extended on whichever side the deriv lives.
    //   max_j/max_l: the HRR iteration bound in its own shift loop.
    //   max_k/max_l passed into g_shift_ij_to: the set of (k,l) slots at
    //     j=1..max_j that nabla will read. For ket-deriv the nabla
    //     reads beyond env.lk / env.ll so HRR must also cover those
    //     extended slots.
    const bool ket_deriv = !bra_deriv;
    const int hrr_max_j = j_side ? (env.lj + 1) : env.lj;
    const int hrr_max_l = l_side ? (env.ll + 1) : env.ll;
    const int hrr_ij_max_k = ket_deriv ? (env.lk + 1) : env.lk;
    const int hrr_ij_max_l = hrr_max_l;

    for (int p = 0; p < A.n_prim; ++p) {
        const double aa = A.alpha[p];
        const double ca = A.coeff[p];

        for (int q = 0; q < B.n_prim; ++q) {
            const double ab = B.alpha[q];
            const double cb = B.coeff[q];
            const double aij = aa + ab;
            const double rij[3] = {
                (aa * A.center[0] + ab * B.center[0]) / aij,
                (aa * A.center[1] + ab * B.center[1]) / aij,
                (aa * A.center[2] + ab * B.center[2]) / aij,
            };
            const double dAB[3] = {
                A.center[0] - B.center[0],
                A.center[1] - B.center[1],
                A.center[2] - B.center[2],
            };
            const double rAB2 =
                dAB[0] * dAB[0] + dAB[1] * dAB[1] + dAB[2] * dAB[2];
            const double K_AB = std::exp(-aa * ab / aij * rAB2);
            const double bra_coeff = ca * cb * K_AB;

            // Primitive exponent on the deriv shell, picked per primitive.
            // For deriv on A: varies with p. For deriv on B: varies with q.
            // For deriv on C/D: varies inside the ket loop below.
            const double alpha_bra_deriv = (deriv_center == 0)   ? aa
                                           : (deriv_center == 1) ? ab
                                                                 : 0.0;

            for (int kk = 0; kk < nket; ++kk) {
                const double akl = ket_pairs[kk].akl;
                const double* rkl = ket_pairs[kk].rkl;
                const double prim_coeff = bra_coeff * ket_pairs[kk].kcd_coeff;

                // Primitive exponent on the deriv shell for ket targets.
                double alpha_ket_deriv = 0.0;
                if (deriv_center == 2 || deriv_center == 3) {
                    const int r = kk / D.n_prim;
                    const int s = kk % D.n_prim;
                    alpha_ket_deriv =
                        (deriv_center == 2) ? C.alpha[r] : D.alpha[s];
                }
                const double two_alpha = bra_deriv ? (2.0 * alpha_bra_deriv)
                                                   : (2.0 * alpha_ket_deriv);

                const double a1 = aij * akl;
                const double delta = aij + akl;
                const double a0 = a1 / delta;
                const double rijkl[3] = {
                    rij[0] - rkl[0],
                    rij[1] - rkl[1],
                    rij[2] - rkl[2],
                };
                const double rr = rijkl[0] * rijkl[0] + rijkl[1] * rijkl[1]
                                  + rijkl[2] * rijkl[2];
                const double x = a0 * rr;

                const double fac1 = kTwoPi25 * std::sqrt(a0 / (a1 * a1 * a1));
                const double gz_seed = fac1 * prim_coeff;

                double u[kRysMaxRoots], w[kRysMaxRoots];
                rys_roots(env.nroots, x, u, w);

                Rys2eCoeff bc;
                for (int irys = 0; irys < env.nroots; ++irys) {
                    const double u2 = a0 * u[irys];
                    const double tmp4 = 0.5 / (u2 * delta + a1);
                    const double tmp5 = u2 * tmp4;
                    const double tmp1 = 2.0 * tmp5;
                    const double tmp2 = tmp1 * akl;
                    const double tmp3 = tmp1 * aij;
                    bc.b00[irys] = tmp5;
                    bc.b10[irys] = tmp5 + tmp4 * akl;
                    bc.b01[irys] = tmp5 + tmp4 * aij;
                    bc.c00x[irys] = (rij[0] - A.center[0]) - tmp2 * rijkl[0];
                    bc.c00y[irys] = (rij[1] - A.center[1]) - tmp2 * rijkl[1];
                    bc.c00z[irys] = (rij[2] - A.center[2]) - tmp2 * rijkl[2];
                    bc.c0px[irys] = (rkl[0] - C.center[0]) + tmp3 * rijkl[0];
                    bc.c0py[irys] = (rkl[1] - C.center[1]) + tmp3 * rijkl[1];
                    bc.c0pz[irys] = (rkl[2] - C.center[2]) + tmp3 * rijkl[2];
                }

                // Re-seed gx/gy/gz each primitive quartet. g0_2e_2d
                // overwrites the i>=1, k>=0 slots it walks, but the
                // (i=0, k=0) seeds must be fresh; and for the derivative
                // kernel we rely on the full VRR range being built.
                for (int irys = 0; irys < env.nroots; ++irys) {
                    g[0 + irys] = 1.0;
                    g[env.g_size + irys] = 1.0;
                    g[env.g_size * 2 + irys] = w[irys] * gz_seed;
                }

                g0_2e_2d(g, bc, env);

                // HRR (with extended j / k / l ranges when the deriv
                // lives on the corresponding side).
                for (int ax = 0; ax < 3; ++ax) {
                    double* ga = g + ax * env.g_size;
                    g_shift_kl_to(ga, env, env.rkrl[ax], hrr_max_l);
                    g_shift_ij_to(ga, env, env.rirj[ax], hrr_max_j,
                                  hrr_ij_max_k, hrr_ij_max_l);
                }

                // Nabla recurrence → 3 axes of derivative. The "raise"
                // term ((a+1_d b|cd) or analogous) shifts one Cartesian
                // index by +nabla_stride on the axis-specific g-channel;
                // the "lower" term shifts by -nabla_stride (only if the
                // target component's Cartesian exponent is > 0).
                const int gs = env.g_size;
                for (int c = 0; c < n_out; ++c) {
                    const AngIdxDeriv& a = ang_idx[c];
                    const int ox = a.ox, oy = a.oy, oz = a.oz;

                    // x-axis: shift the deriv shell's x-component by ±1
                    double vx_plus = 0.0, vx_minus = 0.0;
                    for (int r = 0; r < env.nroots; ++r)
                        vx_plus += g[ox + nabla_stride + r] * g[gs + oy + r]
                                   * g[gs * 2 + oz + r];
                    if (a.n_deriv_x > 0) {
                        for (int r = 0; r < env.nroots; ++r)
                            vx_minus += g[ox - nabla_stride + r]
                                        * g[gs + oy + r] * g[gs * 2 + oz + r];
                    }
                    const double dx =
                        two_alpha * vx_plus - a.n_deriv_x * vx_minus;

                    // y-axis
                    double vy_plus = 0.0, vy_minus = 0.0;
                    for (int r = 0; r < env.nroots; ++r)
                        vy_plus += g[ox + r] * g[gs + oy + nabla_stride + r]
                                   * g[gs * 2 + oz + r];
                    if (a.n_deriv_y > 0) {
                        for (int r = 0; r < env.nroots; ++r)
                            vy_minus += g[ox + r]
                                        * g[gs + oy - nabla_stride + r]
                                        * g[gs * 2 + oz + r];
                    }
                    const double dy =
                        two_alpha * vy_plus - a.n_deriv_y * vy_minus;

                    // z-axis
                    double vz_plus = 0.0, vz_minus = 0.0;
                    for (int r = 0; r < env.nroots; ++r)
                        vz_plus += g[ox + r] * g[gs + oy + r]
                                   * g[gs * 2 + oz + nabla_stride + r];
                    if (a.n_deriv_z > 0) {
                        for (int r = 0; r < env.nroots; ++r)
                            vz_minus += g[ox + r] * g[gs + oy + r]
                                        * g[gs * 2 + oz - nabla_stride + r];
                    }
                    const double dz =
                        two_alpha * vz_plus - a.n_deriv_z * vz_minus;

                    out_axes[0 * n_out + a.out_idx] += dx;
                    out_axes[1 * n_out + a.out_idx] += dy;
                    out_axes[2 * n_out + a.out_idx] += dz;
                }
            }
        }
    }
}

} // namespace

// ── Public API: rys_eri_block_deriv ──────────────────────────────────

// Helper: map `deriv_center` under a bra-swap (A↔B) or ket-swap (C↔D)
// to the equivalent center in the swapped-shell frame.
//   swap_ab: 0↔1 (A↔B); 2 and 3 unchanged
//   swap_cd: 2↔3 (C↔D); 0 and 1 unchanged
static inline int remap_deriv_center(int dc, bool swap_ab, bool swap_cd) {
    if (swap_ab && (dc == 0 || dc == 1))
        dc ^= 1;
    if (swap_cd && (dc == 2 || dc == 3))
        dc ^= 1;
    return dc;
}

void rys_eri_block_deriv(const ShellBlock& A, const ShellBlock& B,
                         const ShellBlock& C, const ShellBlock& D,
                         int deriv_center, double* out_axes) {
    check_root_budget(A, B, C, D, /*extra_l=*/1);
    const bool swap_ab = A.l < B.l;
    const bool swap_cd = C.l < D.l;

    const ShellBlock& Ae = swap_ab ? B : A;
    const ShellBlock& Be = swap_ab ? A : B;
    const ShellBlock& Ce = swap_cd ? D : C;
    const ShellBlock& De = swap_cd ? C : D;

    const int dc_kernel = remap_deriv_center(deriv_center, swap_ab, swap_cd);

    const int na = A.n_cart;
    const int nb = B.n_cart;
    const int nc = C.n_cart;
    const int nd = D.n_cart;
    const int nae = Ae.n_cart;
    const int nbe = Be.n_cart;
    const int nce = Ce.n_cart;
    const int nde = De.n_cart;

    if (!swap_ab && !swap_cd) {
        rys_eri_block_deriv_canonical(Ae, Be, Ce, De, dc_kernel, out_axes);
        return;
    }

    // Non-canonical: run canonical into scratch, then permute output
    // indices back to the caller's (A, B, C, D) ordering. The axis
    // dimension is independent of bra/ket swaps (shell swaps only
    // affect AO indices, not Cartesian directions).
    const size_t n_out_orig = static_cast<size_t>(na) * nb * nc * nd;
    std::vector<double> scratch(static_cast<size_t>(nae) * nbe * nce * nde * 3,
                                0.0);
    rys_eri_block_deriv_canonical(Ae, Be, Ce, De, dc_kernel, scratch.data());

    const size_t n_out_kernel = static_cast<size_t>(nae) * nbe * nce * nde;
    for (int axis = 0; axis < 3; ++axis) {
        const double* sa = scratch.data() + axis * n_out_kernel;
        double* oa = out_axes + axis * n_out_orig;
        for (int ia = 0; ia < na; ++ia)
            for (int ib = 0; ib < nb; ++ib)
                for (int ic = 0; ic < nc; ++ic)
                    for (int id = 0; id < nd; ++id) {
                        const int iae = swap_ab ? ib : ia;
                        const int ibe = swap_ab ? ia : ib;
                        const int ice = swap_cd ? id : ic;
                        const int ide = swap_cd ? ic : id;
                        oa[((ia * nb + ib) * nc + ic) * nd + id] +=
                            sa[((iae * nbe + ibe) * nce + ice) * nde + ide];
                    }
    }
}

// ── Public API ───────────────────────────────────────────────────────

static void rys_eri_dispatch(const ShellBlock& A, const ShellBlock& B,
                             const ShellBlock& C, const ShellBlock& D,
                             double omega, double* out) {
    check_root_budget(A, B, C, D, /*extra_l=*/0);
    const bool swap_ab = A.l < B.l;
    const bool swap_cd = C.l < D.l;

    if (!swap_ab && !swap_cd) {
        rys_eri_block_canonical(A, B, C, D, omega, out);
        return;
    }

    // Non-canonical shell ordering: run the canonical kernel on the
    // swapped inputs into a scratch buffer, then copy back with the
    // bra / ket indices transposed. Relies on the 8-fold ERI symmetry
    //     (ab|cd) = (ba|cd) = (ab|dc) = (ba|dc)
    // (all bra/ket swaps commute with the integral value).
    const ShellBlock& Ae = swap_ab ? B : A;
    const ShellBlock& Be = swap_ab ? A : B;
    const ShellBlock& Ce = swap_cd ? D : C;
    const ShellBlock& De = swap_cd ? C : D;

    const int nae = Ae.n_cart;
    const int nbe = Be.n_cart;
    const int nce = Ce.n_cart;
    const int nde = De.n_cart;
    // Per-thread reused scratch (resize-if-smaller); the canonical kernel
    // accumulates (+=), so zero the used prefix before the call.
    const size_t sc_need = static_cast<size_t>(nae) * nbe * nce * nde;
    if (tls_swap_scratch.size() < sc_need)
        tls_swap_scratch.resize(sc_need);
    double* scratch = tls_swap_scratch.data();
    std::fill(scratch, scratch + sc_need, 0.0);
    rys_eri_block_canonical(Ae, Be, Ce, De, omega, scratch);

    const int na = A.n_cart;
    const int nb = B.n_cart;
    const int nc = C.n_cart;
    const int nd = D.n_cart;
    for (int ia = 0; ia < na; ++ia) {
        for (int ib = 0; ib < nb; ++ib) {
            for (int ic = 0; ic < nc; ++ic) {
                for (int id = 0; id < nd; ++id) {
                    const int iae = swap_ab ? ib : ia;
                    const int ibe = swap_ab ? ia : ib;
                    const int ice = swap_cd ? id : ic;
                    const int ide = swap_cd ? ic : id;
                    const double v =
                        scratch[((iae * nbe + ibe) * nce + ice) * nde + ide];
                    out[((ia * nb + ib) * nc + ic) * nd + id] += v;
                }
            }
        }
    }
}

void rys_eri_block(const ShellBlock& A, const ShellBlock& B,
                   const ShellBlock& C, const ShellBlock& D, double* out) {
    rys_eri_dispatch(A, B, C, D, /*omega=*/0.0, out);
}

void rys_eri_erf_block(const ShellBlock& A, const ShellBlock& B,
                       const ShellBlock& C, const ShellBlock& D, double omega,
                       double* out) {
    if (omega < 0.0) {
        throw std::invalid_argument(
            "rys_eri_erf_block: omega must be >= 0 [1/Bohr]; "
            "omega = 0 is the bare 1/r operator (identity with rys_eri_block)");
    }
    // omega == 0 is the documented identity: go through the same public
    // 1/r entry so the ω→0 gate is bit-identical, not just numerically close.
    if (omega == 0.0) {
        rys_eri_block(A, B, C, D, out);
        return;
    }
    rys_eri_dispatch(A, B, C, D, omega, out);
}

} // namespace native
} // namespace scf
} // namespace cpu
} // namespace ATV

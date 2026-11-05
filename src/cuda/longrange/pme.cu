/// @file cuda/longrange/pme.cu
/// @brief Smooth Particle-Mesh Ewald reciprocal sum — CUDA (cuFFT).
///        Built only with ATV_BUILD_KSPACE.
///
/// Compile-time spline order and bounded stencil-index wrapping follow the
/// OpenMM PME kernel architecture, platforms/common/src/kernels/pme.cc,
/// SHA 3a2326d0dd091229eaa4c0d7b5ad1001d5fe376e. The Essmann recurrence
/// and FFT normalization remain ATV-native; no source is copied. Splines,
/// mesh and FFT run in force_t; fractional coordinates, influence and energy
/// stay FP64; forces are added to the fixed-point Frame columns.
///
/// Every kernel reads the published device Box (`Box::current()`); the
/// influence/virial coefficient tables remember the Box revision they were
/// computed for, and the captured graph recomputes them in stream order when
/// the published revision differs. Reciprocal virial: Essmann et
/// al., J. Chem. Phys. 103, 8577 (1995), eq. 2.11, evaluated on the discrete
/// mesh energy for both conjugate half-spectrum indices. Lattice vectors are
/// the cell columns (molpy/LAMMPS H): s = cell⁻¹ · r and the wave vector of
/// mode m is Σ_d m_d (row d of cell⁻¹).
///
/// The PPPM setting (`ReciprocalKind::PppmIk`) spreads and gathers on
/// LAMMPS's centred stencil, `src/KSPACE/pppm.cpp` `particle_map` /
/// `make_rho` / `fieldforce_ik` at checkout e8c4c278a8 (Copyright (2003)
/// Sandia Corporation, GPL-2.0-only), through `core/pppm_ik.h`; smooth PME
/// keeps the one-sided Essmann stencil.
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cufft.h>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "atomiverse/core/longrange_math.h"
#include "atomiverse/core/pppm_ik.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/energy_term.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/longrange/pme.cuh"

namespace ATV {

namespace cuda {

static_assert(std::is_same_v<force_t, float>
              || std::is_same_v<force_t, double>);
using PmeComplex = std::conditional_t<std::is_same_v<force_t, double>,
                                      cufftDoubleComplex, cufftComplex>;

// Build-fixed mesh passed by value into kernels.
struct PmeGeom {
    int K[3];
    int order;
};

// Bookkeeping of the cell-dependent coefficient tables (fixed device
// address). The geometry itself is always the published Box.
struct PmeTables {
    unsigned long long revision; ///< Box revision of the tables; ~0 at build.
    int dirty;                   ///< Tables are recomputed on this replay.
};

namespace kernels {

// Canonical Essmann/Amber fill_bspline on device. theta[k]=M_n(frac+n-1-k),
// dtheta[k]=dM/d(frac). order n ≤ 12.
template <int Order>
__device__ __forceinline__ void pme_bspline(force_t frac, int runtime_order,
                                            force_t* theta, force_t* dtheta) {
    const int n = Order > 0 ? Order : runtime_order;
    theta[n - 1] = force_t(0);
    theta[1] = frac;
    theta[0] = force_t(1) - frac;
    for (int k = 3; k < n; ++k) {
        const force_t div = force_t(1) / (k - 1);
        theta[k - 1] = div * frac * theta[k - 2];
        for (int j = 1; j < k - 1; ++j)
            theta[k - 1 - j] = div
                               * ((frac + j) * theta[k - 2 - j]
                                  + (k - j - frac) * theta[k - 1 - j]);
        theta[0] = div * (force_t(1) - frac) * theta[0];
    }
    dtheta[0] = -theta[0];
    for (int j = 1; j < n; ++j)
        dtheta[j] = theta[j - 1] - theta[j];
    const force_t div = force_t(1) / (n - 1);
    theta[n - 1] = div * frac * theta[n - 2];
    for (int j = 1; j < n - 1; ++j)
        theta[n - 1 - j] = div
                           * ((frac + j) * theta[n - 2 - j]
                              + (n - j - frac) * theta[n - 1 - j]);
    theta[0] = div * (force_t(1) - frac) * theta[0];
}

// Each grid dimension is at least order+1 and every stencil index lies in
// [-(order-1), K + order/2] (SPME: [⌊u⌋-(order-1), ⌊u⌋]; PPPM: nx + nlower …
// nx + nupper, nx ≤ K). One adjustment suffices, including the rounded u==K
// edge.
__device__ __forceinline__ int wrap_idx(int v, int K) {
    if (v < 0)
        v += K;
    if (v >= K)
        v -= K;
    return v;
}

/// @brief B-spline weights of one axis and the first grid point they belong to.
///
/// Pppm: LAMMPS centred stencil — θ[a] at nx + nlower + a (core/pppm_ik.h).
/// Otherwise the Essmann one-sided stencil — θ[a] at ⌊u⌋ − (n−1) + a.
template <int Order, bool Pppm>
__device__ __forceinline__ int mesh_stencil(double u, int n, force_t* theta,
                                            force_t* dtheta) {
    if constexpr (Pppm) {
        double frac = 0.0;
        const int nx = longrange::pppm_particle_map(u, n, frac);
        pme_bspline<Order>(force_t(frac), n, theta, dtheta);
        return nx + longrange::pppm_stencil_lower(n);
    } else {
        const int u0 = int(floor(u));
        pme_bspline<Order>(force_t(u - u0), n, theta, dtheta);
        return u0 - (n - 1);
    }
}

/// @brief Spread owned charges; order adjacent threads share one atom.
template <int Order, bool Pppm>
__global__ void
pme_spread(const pos_t* __restrict__ x, const pos_t* __restrict__ y,
           const pos_t* __restrict__ z, const force_t* __restrict__ q,
           int natoms, PmeGeom g, Box box_arg, force_t* __restrict__ grid) {
    const int n = Order > 0 ? Order : g.order;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = gid / n, c = gid % n;
    if (i >= natoms)
        return;
    const force_t qi = force_t(q[i]);
    if (qi == 0.0)
        return;
    const Box box = box_arg.current();
    const double* inv_cell = box.inv_cell;
    const double rx = double(x[i]), ry = double(y[i]), rz = double(z[i]);
    constexpr int extent = Order > 0 ? Order : 12;
    force_t th[3][extent], dth[3][extent];
    int start[3];
    for (int d = 0; d < 3; ++d) {
        const double s = inv_cell[3 * d + 0] * rx + inv_cell[3 * d + 1] * ry
                         + inv_cell[3 * d + 2] * rz;
        double u = s * g.K[d];
        u -= floor(u / g.K[d]) * g.K[d];
        // A non-finite coordinate has no grid point: depositing it would
        // index outside the mesh. Skipping it leaves the gather's NaN force
        // to raise the frame force fault.
        if (!(u >= 0.0 && u <= double(g.K[d])))
            return;
        start[d] = mesh_stencil<Order, Pppm>(u, n, th[d], dth[d]);
    }
    const int K0 = g.K[0], K1 = g.K[1], K2 = g.K[2];
    // As in OpenMM spreadCharge, adjacent threads write consecutive z
    // stencil sites. Each issues n*n atomics instead of n*n*n serial atomics.
    // Keep the existing (q*theta_x)*theta_y*theta_z multiplication order.
    const int g2 = wrap_idx(start[2] + c, K2);
    // Select without dynamically indexing the order-specialized theta
    // array, so the compiler can retain the spline coefficients in registers.
    force_t wz = 0;
#pragma unroll
    for (int k = 0; k < n; ++k)
        if (k == c)
            wz = th[2][k];
    for (int a = 0; a < n; ++a) {
        const int g0 = wrap_idx(start[0] + a, K0);
        const force_t wa = qi * th[0][a];
        for (int b = 0; b < n; ++b) {
            const int g1 = wrap_idx(start[1] + b, K1);
            const force_t wab = wa * th[1][b];
            const int idx = g2 + K2 * (g1 + K1 * g0);
            atomicAdd(&grid[idx], wab * wz);
        }
    }
}

/// @brief Flag the coefficient tables stale when the published cell revision
///        moved; add the neutralizing-background energy (and its virial).
template <bool Virial>
__global__ void pme_refresh_tables(Box box_arg, PmeTables* __restrict__ tables,
                                   double background_coef, EnergyTerm energy,
                                   acc_t* __restrict__ stress) {
    const Box box = box_arg.current();
    PmeTables& t = *tables;
    t.dirty = box.revision != t.revision;
    t.revision = box.revision;
    // E_bg ∝ 1/V: its virial −∂E/∂ε_aa is E_bg on each diagonal component.
    const double background = -background_coef / box.volume();
    if (energy)
        energy.add(background);
    if constexpr (Virial)
        for (int d = 0; d < 3; ++d)
            atomicAdd(stress + d, background);
}

/// @brief Recompute conjugate-averaged influence (and virial coefficients)
///        for every half-spectrum mode after a cell change.
///
/// Each stored mode stands for itself and its full-index conjugate
/// c = (K − m) mod K. Their canonical wave vectors differ only at Nyquist
/// components of even meshes, where triclinic cells give different |m|².
/// Both sides are evaluated and averaged, for the energy kernel H and for
/// the strain derivative coefficients
/// C_ab = ½ Σ_side H (δ_ab − 2(1/m² + π²/α²) m_a m_b).
template <bool Virial>
__global__ void
pme_refresh_influence(const PmeTables* __restrict__ tables, Box box_arg,
                      PmeGeom g, const double* __restrict__ moduli,
                      double inv_pi2_a2, double* __restrict__ influence,
                      double* __restrict__ virial, int nhalf) {
    if (!tables->dirty)
        return;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nhalf)
        return;
    const int K0 = g.K[0], K1 = g.K[1], K2 = g.K[2];
    const int half_z = K2 / 2 + 1;
    const int m[3] = {idx / (half_z * K1), (idx / half_z) % K1, idx % half_z};
    double H = 0.0, coef[6] = {};
    if (idx != 0) {
        const int K[3] = {K0, K1, K2};
        const Box box = box_arg.current();
        const double* inv_cell = box.inv_cell;
        for (int side = 0; side < 2; ++side) {
            int s[3];
            for (int d = 0; d < 3; ++d)
                s[d] = side == 0 || m[d] == 0 ? m[d] : K[d] - m[d];
            double mv[3] = {0.0, 0.0, 0.0};
            for (int d = 0; d < 3; ++d) {
                const int h = s[d] <= K[d] / 2 ? s[d] : s[d] - K[d];
                for (int r = 0; r < 3; ++r)
                    mv[r] += h * inv_cell[3 * d + r];
            }
            const double msq = mv[0] * mv[0] + mv[1] * mv[1] + mv[2] * mv[2];
            const double B =
                moduli[s[0]] * moduli[K0 + s[1]] * moduli[K0 + K1 + s[2]];
            const double h_side = 0.5 * exp(-inv_pi2_a2 * msq) / msq * B;
            H += h_side;
            if constexpr (Virial) {
                const double vt = -2.0 * (1.0 / msq + inv_pi2_a2);
                coef[0] += h_side * (1.0 + vt * mv[0] * mv[0]);
                coef[1] += h_side * (1.0 + vt * mv[1] * mv[1]);
                coef[2] += h_side * (1.0 + vt * mv[2] * mv[2]);
                coef[3] += h_side * vt * mv[0] * mv[1];
                coef[4] += h_side * vt * mv[0] * mv[2];
                coef[5] += h_side * vt * mv[1] * mv[2];
            }
        }
    }
    influence[idx] = H;
    if constexpr (Virial)
        for (int k = 0; k < 6; ++k)
            virial[static_cast<std::size_t>(k) * nhalf + idx] = coef[k];
}

/// @brief Influence function: energy (+ virial) + in-place convolution.
template <bool Virial>
__global__ void pme_influence(PmeComplex* __restrict__ grid, PmeGeom g,
                              Box box_arg, const double* __restrict__ influence,
                              const double* __restrict__ virial, int nhalf,
                              EnergyTerm energy, acc_t* __restrict__ stress) {
    const int K2 = g.K[2];
    const int half_z = K2 / 2 + 1;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const double escale = ATV::units::md_qqr2e
                          / (2.0 * ATV::units::pi * box_arg.current().volume());
    constexpr int components = Virial ? 7 : 1;
    ::cuda::std::array<acc_t, components> values{};
    if (idx < nhalf) {
        if (idx == 0) {
            grid[idx].x = 0;
            grid[idx].y = 0;
        } else {
            const double H = influence[idx];
            const int m2 = idx % half_z;
            // The kz boundary planes already contain both xy-conjugate
            // indices. Only interior kz modes stand for two full modes.
            const double weight =
                (m2 == 0 || (K2 % 2 == 0 && m2 == K2 / 2)) ? 1.0 : 2.0;
            const double qr = grid[idx].x, qi = grid[idx].y;
            const double q2 = weight * escale * (qr * qr + qi * qi);
            values[0] = q2 * H;
            if constexpr (Virial)
                for (int k = 0; k < 6; ++k)
                    values[k + 1] =
                        q2 * virial[static_cast<std::size_t>(k) * nhalf + idx];
            grid[idx].x = H * qr;
            grid[idx].y = H * qi;
        }
    }

    // Tail lanes and the zero mode contribute zero but must participate in
    // the collective. One atomic per block and component. A step without
    // eflag/vflag (null energy, no Virial) skips it; the branch is uniform.
    if (!energy && !Virial)
        return;
    using Reduce =
        cub::BlockReduce<::cuda::std::array<acc_t, components>, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const auto sum =
        Reduce(reduction).Reduce(values, SumComponents<acc_t, components>{});
    if (threadIdx.x == 0) {
        if (energy)
            energy.add(sum[0]);
        if constexpr (Virial)
            for (int k = 0; k < 6; ++k)
                atomicAdd(stress + k, sum[k + 1]);
    }
}

/// @brief Interpolate reciprocal forces onto atoms + per-atom self energy.
template <int Order>
__global__ void
pme_interp(fixed_t* __restrict__ fx, fixed_t* __restrict__ fy,
           fixed_t* __restrict__ fz, ForceFault* fault,
           const pos_t* __restrict__ x, const pos_t* __restrict__ y,
           const pos_t* __restrict__ z, const force_t* __restrict__ q,
           int natoms, PmeGeom g, Box box_arg, const force_t* __restrict__ grid,
           double self_pref, EnergyTerm energy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const double qi = (i < natoms) ? double(q[i]) : 0.0;
    // Reduce before the neutral/tail early exit so every lane reaches the
    // block barriers, including partially populated final blocks. The
    // energy pointer is uniform: a step without eflag skips the reduction.
    if (energy) {
        using Reduce = cub::BlockReduce<double, kDefaultBlock>;
        __shared__ typename Reduce::TempStorage reduction;
        const double block_energy = Reduce(reduction).Sum(self_pref * qi * qi);
        if (threadIdx.x == 0)
            energy.add(block_energy);
    }
    if (i >= natoms || qi == 0.0)
        return;
    const Box box = box_arg.current();
    const double* inv_cell = box.inv_cell;
    const double fscale =
        ATV::units::md_qqr2e / (ATV::units::pi * box.volume());
    const double rx = double(x[i]), ry = double(y[i]), rz = double(z[i]);
    const int n = Order > 0 ? Order : g.order;
    constexpr int extent = Order > 0 ? Order : 12;
    force_t th[3][extent], dth[3][extent];
    int base[3];
    for (int d = 0; d < 3; ++d) {
        const double s = inv_cell[3 * d + 0] * rx + inv_cell[3 * d + 1] * ry
                         + inv_cell[3 * d + 2] * rz;
        double u = s * g.K[d];
        u -= floor(u / g.K[d]) * g.K[d];
        const int u0 = int(floor(u));
        pme_bspline<Order>(force_t(u - u0), n, th[d], dth[d]);
        base[d] = u0;
    }
    const int K0 = g.K[0], K1 = g.K[1], K2 = g.K[2], nm1 = n - 1;
    force_t gd[3] = {0, 0, 0};
    for (int a = 0; a < n; ++a) {
        const int g0 = wrap_idx(base[0] - nm1 + a, K0);
        for (int b = 0; b < n; ++b) {
            const int g1 = wrap_idx(base[1] - nm1 + b, K1);
            for (int c = 0; c < n; ++c) {
                const int g2 = wrap_idx(base[2] - nm1 + c, K2);
                const force_t cval = grid[g2 + K2 * (g1 + K1 * g0)];
                gd[0] += cval * dth[0][a] * th[1][b] * th[2][c];
                gd[1] += cval * th[0][a] * dth[1][b] * th[2][c];
                gd[2] += cval * th[0][a] * th[1][b] * dth[2][c];
            }
        }
    }
    double fxv = 0, fyv = 0, fzv = 0;
    for (int d = 0; d < 3; ++d) {
        const double gk = double(gd[d]) * g.K[d];
        fxv += gk * inv_cell[3 * d + 0];
        fyv += gk * inv_cell[3 * d + 1];
        fzv += gk * inv_cell[3 * d + 2];
    }
    // One thread owns atom i in this kernel.
    add_force_owned(fx, i, force_t(-fscale * qi * fxv), fault);
    add_force_owned(fy, i, force_t(-fscale * qi * fyv), fault);
    add_force_owned(fz, i, force_t(-fscale * qi * fzv), fault);
}

/// @brief Folded half-spectrum PPPM tables (core/pppm_ik.h pppm_half_mode).
///
/// One thread per r2c mode: the energy weight, the three ik planes and the
/// six virial planes, each mode folded with its conjugate partner so the
/// half-spectrum sums equal LAMMPS's full complex mesh (Nyquist planes of an
/// even mesh included). Virial slots hold G·(δ_ab + vt·m_a·m_b), m = k/(2π).
template <bool Virial>
__global__ void
pppm_refresh_influence(const PmeTables* __restrict__ tables, Box box_arg,
                       PmeGeom g, const double* __restrict__ gf_b, double alpha,
                       double* __restrict__ influence,
                       double* __restrict__ virial, double* __restrict__ ik,
                       int nhalf) {
    if (!tables->dirty)
        return;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nhalf)
        return;
    const int K0 = g.K[0], K1 = g.K[1], K2 = g.K[2];
    const int half_z = K2 / 2 + 1;
    const int m0 = idx / (half_z * K1);
    const int m1 = (idx / half_z) % K1;
    const int m2 = idx % half_z;
    const Box box = box_arg.current();
    const int K[3] = {K0, K1, K2};
    const longrange::PppmHalfMode hm = longrange::pppm_half_mode(
        m0, m1, m2, box.cell, K, alpha, g.order, gf_b, box.inv_cell, Virial);
    influence[idx] = hm.green;
    for (int a = 0; a < 3; ++a)
        ik[static_cast<std::size_t>(a) * nhalf + idx] = hm.ik[a];
    if constexpr (Virial)
        for (int c = 0; c < 6; ++c)
            virial[static_cast<std::size_t>(c) * nhalf + idx] = hm.virial[c];
}

/// @brief PPPM energy and ρ̂/V (the ik fields multiply it by the folded ik
///        coefficients). Does not overwrite ρ̂.
template <bool Virial>
__global__ void pppm_energy(const PmeComplex* __restrict__ density,
                            PmeComplex* __restrict__ potential, PmeGeom g,
                            Box box_arg, const double* __restrict__ influence,
                            const double* __restrict__ virial, int nhalf,
                            EnergyTerm energy, acc_t* __restrict__ stress) {
    const int half_z = g.K[2] / 2 + 1;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const Box box = box_arg.current();
    const double volume = box.volume();
    const double escale = ATV::units::md_qqr2e / (2.0 * volume);
    const double inv_vol = 1.0 / volume;
    constexpr int components = Virial ? 7 : 1;
    ::cuda::std::array<acc_t, components> values{};
    if (idx < nhalf) {
        const double G = influence[idx];
        const int m2 = idx % half_z;
        const double weight =
            (m2 == 0 || (g.K[2] % 2 == 0 && m2 == g.K[2] / 2)) ? 1.0 : 2.0;
        const double qr = density[idx].x, qi = density[idx].y;
        const double scaled = weight * escale * (qr * qr + qi * qi);
        values[0] = scaled * G;
        if constexpr (Virial)
            for (int k = 0; k < 6; ++k)
                values[k + 1] =
                    scaled * virial[static_cast<std::size_t>(k) * nhalf + idx];
        potential[idx].x = inv_vol * qr;
        potential[idx].y = inv_vol * qi;
    }
    if (!energy && !Virial)
        return; // uniform: no eflag/vflag this step
    using Reduce =
        cub::BlockReduce<::cuda::std::array<acc_t, components>, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const auto sum =
        Reduce(reduction).Reduce(values, SumComponents<acc_t, components>{});
    if (threadIdx.x == 0) {
        if (energy)
            energy.add(sum[0]);
        if constexpr (Virial)
            for (int k = 0; k < 6; ++k)
                atomicAdd(stress + k, sum[k + 1]);
    }
}

/// @brief Write i·ik_a·ρ̂/V for all three axes into a batched half-spectrum.
///
/// Batch `a` starts at `spectrum + a·nhalf` and is the cuFFT input for
/// ∂V/∂r_a; `ik` holds the folded coefficients (pppm_half_mode), so the c2r
/// transform is LAMMPS's real part of its complex ik transform.
__global__ void pppm_ik_fields(const PmeComplex* __restrict__ potential,
                               const double* __restrict__ ik,
                               PmeComplex* __restrict__ spectrum, int nhalf) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nhalf)
        return;
    const double re = potential[idx].x, im = potential[idx].y;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const double c = ik[static_cast<std::size_t>(axis) * nhalf + idx];
        PmeComplex* out = spectrum + static_cast<long long>(axis) * nhalf;
        out[idx].x = -c * im;
        out[idx].y = c * re;
    }
}

/// @brief Gather all three Cartesian fields with one set of spreading weights.
///
/// `grid` holds three contiguous real meshes of `ntot` points, the batched
/// inverse of `pppm_ik_fields`. F_a += −q · coulomb · Σ θ · grid_a.
/// Self energy is added once.
template <int Order>
__global__ void
pppm_gather(fixed_t* __restrict__ fx, fixed_t* __restrict__ fy,
            fixed_t* __restrict__ fz, ForceFault* fault,
            const pos_t* __restrict__ x, const pos_t* __restrict__ y,
            const pos_t* __restrict__ z, const force_t* __restrict__ q,
            int natoms, PmeGeom g, Box box_arg,
            const force_t* __restrict__ grid, int ntot, double self_pref,
            EnergyTerm energy) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const double qi = (i < natoms) ? double(q[i]) : 0.0;
    if (energy) { // uniform: no eflag skips the self energy
        using Reduce = cub::BlockReduce<double, kDefaultBlock>;
        __shared__ typename Reduce::TempStorage reduction;
        const double block_energy = Reduce(reduction).Sum(self_pref * qi * qi);
        if (threadIdx.x == 0)
            energy.add(block_energy);
    }
    if (i >= natoms || qi == 0.0)
        return;
    const Box box = box_arg.current();
    const double* inv_cell = box.inv_cell;
    const double coulomb = ATV::units::md_qqr2e;
    const double rx = double(x[i]), ry = double(y[i]), rz = double(z[i]);
    const int n = Order > 0 ? Order : g.order;
    constexpr int extent = Order > 0 ? Order : 12;
    force_t th[3][extent], dth[3][extent];
    int start[3];
    for (int d = 0; d < 3; ++d) {
        const double s = inv_cell[3 * d + 0] * rx + inv_cell[3 * d + 1] * ry
                         + inv_cell[3 * d + 2] * rz;
        double u = s * g.K[d];
        u -= floor(u / g.K[d]) * g.K[d];
        if (!(u >= 0.0 && u <= double(g.K[d]))) {
            // Non-finite coordinate: no mesh read; the NaN force raises the
            // frame force fault instead of a wild index.
            add_force_owned(fx, i, force_t(nan("")), fault);
            return;
        }
        start[d] = mesh_stencil<Order, true>(u, n, th[d], dth[d]);
    }
    const int K0 = g.K[0], K1 = g.K[1], K2 = g.K[2];
    const force_t* gx = grid;
    const force_t* gy = grid + static_cast<long long>(ntot);
    const force_t* gz = grid + 2LL * static_cast<long long>(ntot);
    force_t accx = 0, accy = 0, accz = 0;
    for (int a = 0; a < n; ++a) {
        const int g0 = wrap_idx(start[0] + a, K0);
        for (int b = 0; b < n; ++b) {
            const int g1 = wrap_idx(start[1] + b, K1);
            for (int c = 0; c < n; ++c) {
                const int g2 = wrap_idx(start[2] + c, K2);
                const int gid = g2 + K2 * (g1 + K1 * g0);
                const force_t w = th[0][a] * th[1][b] * th[2][c];
                accx += gx[gid] * w;
                accy += gy[gid] * w;
                accz += gz[gid] * w;
            }
        }
    }
    const double qe = -coulomb * qi;
    add_force_owned(fx, i, force_t(qe * double(accx)), fault);
    add_force_owned(fy, i, force_t(qe * double(accy)), fault);
    add_force_owned(fz, i, force_t(qe * double(accz)), fault);
}

} // namespace kernels

namespace {

// The selected complex type, plan kind and execution API must agree in both
// build warmup and capture. Only the configured force_t path is instantiated.
template <typename Real, typename Complex>
cufftResult execute_forward(cufftHandle plan, Real* grid, Complex* spectrum) {
    if constexpr (std::is_same_v<Complex, cufftDoubleComplex>) {
        static_assert(std::is_same_v<Real, double>);
        return cufftExecD2Z(plan, grid, spectrum);
    } else {
        static_assert(std::is_same_v<Complex, cufftComplex>);
        static_assert(std::is_same_v<Real, float>);
        return cufftExecR2C(plan, grid, spectrum);
    }
}

template <typename Real, typename Complex>
cufftResult execute_inverse(cufftHandle plan, Complex* spectrum, Real* grid) {
    if constexpr (std::is_same_v<Complex, cufftDoubleComplex>) {
        static_assert(std::is_same_v<Real, double>);
        return cufftExecZ2D(plan, spectrum, grid);
    } else {
        static_assert(std::is_same_v<Complex, cufftComplex>);
        static_assert(std::is_same_v<Real, float>);
        return cufftExecC2R(plan, spectrum, grid);
    }
}

int good_fft_size(int n) {
    auto ok = [](int m) {
        for (int p : {2, 3, 5, 7})
            while (m % p == 0)
                m /= p;
        return m == 1;
    };
    if (n < 4)
        n = 4;
    while (!ok(n)) {
        if (n == std::numeric_limits<int>::max())
            throw std::overflow_error(
                "KSpacePME: FFT dimension exceeds int indexing");
        ++n;
    }
    return n;
}

std::vector<double> spline_moduli(int K, int n) {
    // M_n at integers via the host mirror of pme_bspline(frac=0).
    std::vector<double> data(n, 0.0), dd(n, 0.0);
    {
        double* theta = data.data();
        double* dtheta = dd.data();
        const double frac = 0.0;
        theta[n - 1] = 0.0;
        theta[1] = frac;
        theta[0] = 1.0 - frac;
        for (int k = 3; k < n; ++k) {
            const double div = 1.0 / (k - 1);
            theta[k - 1] = div * frac * theta[k - 2];
            for (int j = 1; j < k - 1; ++j)
                theta[k - 1 - j] = div
                                   * ((frac + j) * theta[k - 2 - j]
                                      + (k - j - frac) * theta[k - 1 - j]);
            theta[0] = div * (1.0 - frac) * theta[0];
        }
        dtheta[0] = -theta[0];
        for (int j = 1; j < n; ++j)
            dtheta[j] = theta[j - 1] - theta[j];
        const double div = 1.0 / (n - 1);
        theta[n - 1] = div * frac * theta[n - 2];
        for (int j = 1; j < n - 1; ++j)
            theta[n - 1 - j] = div
                               * ((frac + j) * theta[n - 2 - j]
                                  + (n - j - frac) * theta[n - 1 - j]);
        theta[0] = div * (1.0 - frac) * theta[0];
    }
    std::vector<double> mod(K, 0.0);
    for (int i = 0; i < K; ++i) {
        double sr = 0, si = 0;
        for (int k = 0; k < n - 1; ++k) {
            const double arg = 2.0 * ATV::units::pi * i * k / K;
            sr += data[k] * std::cos(arg);
            si += data[k] * std::sin(arg);
        }
        const double den = sr * sr + si * si;
        mod[i] = den > 1e-12 ? 1.0 / den : 0.0;
    }
    for (int i = 0; i < K; ++i)
        if (mod[i] == 0.0)
            mod[i] = 0.5 * (mod[(i - 1 + K) % K] + mod[(i + 1) % K]);
    return mod;
}

} // namespace

KSpacePME::KSpacePME(double alpha, double grid_spacing, int order,
                     ReciprocalKind kind)
    : alpha_(alpha), spacing_(grid_spacing), order_(order), kind_(kind) {
    if (!std::isfinite(alpha) || alpha <= 0.0)
        throw std::invalid_argument("KSpacePME: alpha must be > 0");
    if (!std::isfinite(grid_spacing) || grid_spacing <= 0.0)
        throw std::invalid_argument("KSpacePME: grid spacing must be > 0");
    if (order < 3 || order > 12)
        throw std::invalid_argument("KSpacePME: order must be in [3,12]");
}

KSpacePME::~KSpacePME() {}

KSpacePPPM::KSpacePPPM(double alpha, std::array<int, 3> mesh, int order)
    : KSpacePPPM(alpha, /*grid_spacing=*/1.0, order) {
    for (int d = 0; d < 3; ++d) {
        if (mesh[d] <= 0)
            throw std::invalid_argument(
                "KSpacePPPM: every mesh entry must be > 0");
        fixed_K_[d] = mesh[d];
    }
}

void KSpacePME::build(const BuildContext& ctx) {
    const long long spread_threads =
        static_cast<long long>(ctx.frame.n_local()) * order_;
    if (spread_threads > std::numeric_limits<int>::max() - (kDefaultBlock - 1))
        throw std::overflow_error(
            "KSpacePME: charge-spread thread count exceeds int indexing");
    // The build-time cell only selects the mesh; every replay reads the
    // published device cell.
    const cuda::Box& box = ctx.runtime.host_box;
    const double volume = box.volume();
    if (!std::isfinite(volume) || volume <= 0)
        throw std::invalid_argument(
            "KSpacePME: finite positive periodic cell volume required");

    std::vector<double> moduli;
    const bool pppm = kind_ == ReciprocalKind::PppmIk;
    for (int d = 0; d < 3; ++d) {
        double len = box.length(d);
        if (pppm) {
            const double x = box.cell[d];
            const double y = box.cell[3 + d];
            const double z = box.cell[6 + d];
            len = std::sqrt(x * x + y * y + z * z);
        }
        const double requested =
            fixed_K_[d] > 0 ? double(fixed_K_[d]) : std::ceil(len / spacing_);
        if (!std::isfinite(requested) || requested <= 0
            || requested > std::numeric_limits<int>::max())
            throw std::overflow_error(
                pppm ? "KSpacePPPM: FFT dimension outside int indexing"
                     : "KSpacePME: FFT dimension outside int indexing");
        K_[d] = pppm ? longrange::pppm_fft_size(int(requested))
                     : good_fft_size(int(requested));
        if (K_[d] < order_ + 1)
            K_[d] = pppm ? longrange::pppm_fft_size(order_ + 1)
                         : good_fft_size(order_ + 1);
        if (!pppm) {
            const auto axis = spline_moduli(K_[d], order_);
            moduli.insert(moduli.end(), axis.begin(), axis.end());
        }
    }
    long long real_count = 1;
    const int grid_limit =
        std::numeric_limits<int>::max() - (kDefaultBlock - 1);
    for (int size : K_) {
        if (real_count > grid_limit / size)
            throw std::overflow_error(
                "KSpacePME: FFT grid exceeds int indexing");
        real_count *= size;
    }
    ntot_ = int(real_count);
    const int half_z = K_[2] / 2 + 1;
    nhalf_ = K_[0] * K_[1] * half_z;

    // Net charge of the fixed atom set sets the neutralizing background.
    built_atoms_ = ctx.frame.n_local();
    double net_charge = 0.0;
    if (built_atoms_) {
        std::vector<force_t> charge(built_atoms_);
        ATV_CUDA_CHECK(cudaMemcpy(charge.data(), ctx.frame.atoms.charge,
                                  charge.size() * sizeof(force_t),
                                  cudaMemcpyDeviceToHost));
        for (const force_t q : charge)
            net_charge += double(q);
    }
    background_coef_ = -longrange::background_energy(net_charge, 1.0, alpha_);

    if (pppm) {
        double gf_b[12] = {};
        longrange::pppm_gf_b_fill(order_, gf_b);
        ATV_CUDA_CHECK(
            cudaMalloc(&d_gf_b_, std::size_t(order_) * sizeof(double)));
        ATV_CUDA_CHECK(cudaMemcpy(d_gf_b_, gf_b,
                                  std::size_t(order_) * sizeof(double),
                                  cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMalloc(&d_potential_,
                                  std::size_t(nhalf_) * sizeof(PmeComplex)));
        ATV_CUDA_CHECK(
            cudaMalloc(&d_ik_, 3 * std::size_t(nhalf_) * sizeof(double)));
    } else {
        ATV_CUDA_CHECK(cudaMalloc(&d_moduli_, moduli.size() * sizeof(double)));
        ATV_CUDA_CHECK(cudaMemcpy(d_moduli_, moduli.data(),
                                  moduli.size() * sizeof(double),
                                  cudaMemcpyHostToDevice));
    }
    ATV_CUDA_CHECK(cudaMalloc(&d_tables_, sizeof(PmeTables)));
    // Revision ~0 never matches a published Box: the first replay fills them.
    ATV_CUDA_CHECK(cudaMemset(d_tables_, 0xff, sizeof(PmeTables)));
    ATV_CUDA_CHECK(
        cudaMalloc(&d_influence_, std::size_t(nhalf_) * sizeof(double)));
    // Virial coefficients are refreshed with the influence table on every
    // cell change, so a later vflag step never reads a stale table.
    ATV_CUDA_CHECK(
        cudaMalloc(&d_virial_, 6 * std::size_t(nhalf_) * sizeof(double)));
    // PPPM stores three ik fields back to back. The forward density uses
    // only the first batch; the inverse plan transforms all three at once.
    const int field_batch = pppm ? 3 : 1;
    ATV_CUDA_CHECK(
        cudaMalloc(&d_grid_, std::size_t(field_batch) * std::size_t(ntot_)
                                 * sizeof(force_t)));
    ATV_CUDA_CHECK(
        cudaMalloc(&d_spectrum_, std::size_t(field_batch) * std::size_t(nhalf_)
                                     * sizeof(PmeComplex)));
    cufftHandle forward, inverse;
    constexpr cufftType forward_type =
        std::is_same_v<PmeComplex, cufftDoubleComplex> ? CUFFT_D2Z : CUFFT_R2C;
    constexpr cufftType inverse_type =
        std::is_same_v<PmeComplex, cufftDoubleComplex> ? CUFFT_Z2D : CUFFT_C2R;
    if (cufftPlan3d(&forward, K_[0], K_[1], K_[2], forward_type)
        != CUFFT_SUCCESS)
        throw std::runtime_error("KSpacePME: forward cufftPlan3d failed");
    forward_plan_ = static_cast<int>(forward);
    if (pppm) {
        int n[3] = {K_[0], K_[1], K_[2]};
        int inembed[3] = {K_[0], K_[1], half_z};
        int onembed[3] = {K_[0], K_[1], K_[2]};
        if (cufftPlanMany(&inverse, 3, n, inembed, 1, nhalf_, onembed, 1, ntot_,
                          inverse_type, field_batch)
            != CUFFT_SUCCESS)
            throw std::runtime_error(
                "KSpacePPPM: batched inverse cufftPlanMany failed");
    } else if (cufftPlan3d(&inverse, K_[0], K_[1], K_[2], inverse_type)
               != CUFFT_SUCCESS) {
        throw std::runtime_error("KSpacePME: inverse cufftPlan3d failed");
    }
    inverse_plan_ = static_cast<int>(inverse);
    if (built_atoms_ == 0)
        return;
    const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    if (cufftSetStream(forward, stream) != CUFFT_SUCCESS
        || cufftSetStream(inverse, stream) != CUFFT_SUCCESS)
        throw std::runtime_error("KSpacePME: cufftSetStream failed");
    // cuFFT supports graph capture without callbacks. Allocate/instantiate
    // once here: force evaluation replays the same resident arrays without
    // issuing each FFT stage and spline kernel separately from the host.
    // https://docs.nvidia.com/cuda/cufft/index.html#cuda-graphs-support
    ATV_CUDA_CHECK(cudaMemsetAsync(d_grid_, 0,
                                   std::size_t(field_batch) * std::size_t(ntot_)
                                       * sizeof(force_t),
                                   stream));
    ATV_CUDA_CHECK(cudaMemsetAsync(
        d_spectrum_, 0,
        std::size_t(field_batch) * std::size_t(nhalf_) * sizeof(PmeComplex),
        stream));
    auto* warm_grid = static_cast<force_t*>(d_grid_);
    auto* warm_spectrum = static_cast<PmeComplex*>(d_spectrum_);
    if (execute_forward(forward, warm_grid, warm_spectrum) != CUFFT_SUCCESS)
        throw std::runtime_error(pppm ? "KSpacePPPM: FFT build warmup failed"
                                      : "KSpacePME: FFT build warmup failed");
    // One inverse exec allocates the cuFFT workspace before capture. PPPM's
    // plan already covers all three Cartesian fields.
    if (execute_inverse(inverse, warm_spectrum, warm_grid) != CUFFT_SUCCESS)
        throw std::runtime_error(pppm ? "KSpacePPPM: FFT build warmup failed"
                                      : "KSpacePME: FFT build warmup failed");
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    // One replayable graph per (eflag, vflag) combination; a step picks the
    // one its consumers asked for (LAMMPS ev_set).
    for (int variant = 0; variant < 4; ++variant) {
        cudaGraph_t graph = nullptr;
        ATV_CUDA_CHECK(
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        try {
            enqueue_(LaunchContext{ctx.runtime, ctx.frame, 0, 0},
                     (variant & 1) != 0, (variant & 2) != 0);
        } catch (...) {
            cudaStreamEndCapture(stream, &graph);
            if (graph)
                cudaGraphDestroy(graph);
            throw;
        }
        ATV_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        // Kept: a step segment splices it as a child node.
        graphs_[variant] = graph;
        graph_exec_[variant].instantiate(graph);
    }
}

void KSpacePME::compute(const LaunchContext& ctx) {
    if (ctx.frame.n_local() != built_atoms_)
        throw std::runtime_error(
            "KSpacePME: atom set changed; rebuild the driver");
    const int variant =
        (ctx.energy_requested ? 1 : 0) | (ctx.stress_requested ? 2 : 0);
    const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    if (!built_atoms_)
        return;
    if (graph_capture::active(stream))
        graph_capture::splice(stream,
                              static_cast<cudaGraph_t>(graphs_[variant]));
    else
        ATV_CUDA_CHECK(graph_exec_[variant].launch(stream));
}

void KSpacePME::enqueue_(const LaunchContext& ctx, bool want_energy,
                         bool want_virial) {
    const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const int n = built_atoms_;
    auto* grid = static_cast<force_t*>(d_grid_);
    auto* spectrum = static_cast<PmeComplex*>(d_spectrum_);
    auto* tables = static_cast<PmeTables*>(d_tables_);
    const Box& box = ctx.runtime.host_box;
    const cufftHandle forward = static_cast<cufftHandle>(forward_plan_);
    const cufftHandle inverse = static_cast<cufftHandle>(inverse_plan_);

    PmeGeom geom;
    geom.K[0] = K_[0];
    geom.K[1] = K_[1];
    geom.K[2] = K_[2];
    geom.order = order_;

    const double self_pref = longrange::self_energy_prefactor(alpha_);
    const double inv_pi2_a2 =
        ATV::units::pi * ATV::units::pi / (alpha_ * alpha_);
    const int blk = kDefaultBlock;
    // Self, reciprocal and background energy are pair energy, as on CPU
    // (LAMMPS `epair` includes `elong`).
    const EnergyTerm energy =
        energy_term(frame.scalars, EnergyKind::Pair, want_energy);
    acc_t* stress = frame.scalars->stress;

    const bool pppm = kind_ == ReciprocalKind::PppmIk;
    // Influence and virial coefficients refresh together on a cell change,
    // whatever this variant reports, so every variant reads current tables.
    auto refresh = [&]<bool Virial>() {
        kernels::pme_refresh_tables<Virial><<<1, 1, 0, stream>>>(
            box, tables, background_coef_, energy, stress);
        if (pppm)
            kernels::pppm_refresh_influence<true>
                <<<grid_size(nhalf_), blk, 0, stream>>>(
                    tables, box, geom, d_gf_b_, alpha_, d_influence_, d_virial_,
                    d_ik_, nhalf_);
        else
            kernels::pme_refresh_influence<true>
                <<<grid_size(nhalf_), blk, 0, stream>>>(
                    tables, box, geom, d_moduli_, inv_pi2_a2, d_influence_,
                    d_virial_, nhalf_);
    };
    if (want_virial)
        refresh.template operator()<true>();
    else
        refresh.template operator()<false>();
    ATV_CUDA_CHECK(
        cudaMemsetAsync(grid, 0, std::size_t(ntot_) * sizeof(force_t), stream));
    // PPPM spreads on the LAMMPS centred stencil, SPME on Essmann's.
    auto spread = [&]<int Order, bool Pppm>() {
        kernels::pme_spread<Order, Pppm>
            <<<grid_size(n * order_), blk, 0, stream>>>(
                frame.atoms.x, frame.atoms.y, frame.atoms.z, frame.atoms.charge,
                n, geom, box, grid);
    };
    auto spread_kind = [&]<int Order>() {
        if (pppm)
            spread.template operator()<Order, true>();
        else
            spread.template operator()<Order, false>();
    };
    if (order_ == 5)
        spread_kind.template operator()<5>();
    else if (order_ == 6)
        spread_kind.template operator()<6>();
    else
        spread_kind.template operator()<0>();
    if (execute_forward(forward, grid, spectrum) != CUFFT_SUCCESS)
        throw std::runtime_error(pppm ? "KSpacePPPM: forward FFT failed"
                                      : "KSpacePME: forward FFT failed");
    if (pppm) {
        auto* potential = static_cast<PmeComplex*>(d_potential_);
        if (want_virial)
            kernels::pppm_energy<true><<<grid_size(nhalf_), blk, 0, stream>>>(
                spectrum, potential, geom, box, d_influence_, d_virial_, nhalf_,
                energy, stress);
        else
            kernels::pppm_energy<false><<<grid_size(nhalf_), blk, 0, stream>>>(
                spectrum, potential, geom, box, d_influence_, d_virial_, nhalf_,
                energy, stress);
        kernels::pppm_ik_fields<<<grid_size(nhalf_), blk, 0, stream>>>(
            potential, d_ik_, spectrum, nhalf_);
        if (execute_inverse(inverse, spectrum, grid) != CUFFT_SUCCESS)
            throw std::runtime_error("KSpacePPPM: inverse FFT failed");
        auto gather = [&]<int Order>() {
            kernels::pppm_gather<Order><<<grid_size(n), blk, 0, stream>>>(
                frame.atoms.fx, frame.atoms.fy, frame.atoms.fz,
                frame.force_fault, frame.atoms.x, frame.atoms.y, frame.atoms.z,
                frame.atoms.charge, n, geom, box, grid, ntot_, self_pref,
                energy);
        };
        if (order_ == 5)
            gather.template operator()<5>();
        else if (order_ == 6)
            gather.template operator()<6>();
        else
            gather.template operator()<0>();
    } else {
        if (want_virial)
            kernels::pme_influence<true><<<grid_size(nhalf_), blk, 0, stream>>>(
                spectrum, geom, box, d_influence_, d_virial_, nhalf_, energy,
                stress);
        else
            kernels::pme_influence<false>
                <<<grid_size(nhalf_), blk, 0, stream>>>(spectrum, geom, box,
                                                        d_influence_, d_virial_,
                                                        nhalf_, energy, stress);
        if (execute_inverse(inverse, spectrum, grid) != CUFFT_SUCCESS)
            throw std::runtime_error("KSpacePME: inverse FFT failed");
        auto interpolate = [&]<int Order>() {
            kernels::pme_interp<Order><<<grid_size(n), blk, 0, stream>>>(
                frame.atoms.fx, frame.atoms.fy, frame.atoms.fz,
                frame.force_fault, frame.atoms.x, frame.atoms.y, frame.atoms.z,
                frame.atoms.charge, n, geom, box, grid, self_pref, energy);
        };
        if (order_ == 5)
            interpolate.template operator()<5>();
        else if (order_ == 6)
            interpolate.template operator()<6>();
        else
            interpolate.template operator()<0>();
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}

void KSpacePME::destroy(const DestroyContext& ctx) {
    ATV_CUDA_CHECK(
        cudaStreamSynchronize(static_cast<cudaStream_t>(ctx.runtime.stream)));
    for (auto& ring : graph_exec_)
        ring.reset();
    for (auto& graph : graphs_)
        if (graph) {
            cudaGraphDestroy(static_cast<cudaGraph_t>(graph));
            graph = nullptr;
        }
    built_atoms_ = 0;
    if (forward_plan_ >= 0) {
        cufftDestroy(static_cast<cufftHandle>(forward_plan_));
        forward_plan_ = -1;
    }
    if (inverse_plan_ >= 0) {
        cufftDestroy(static_cast<cufftHandle>(inverse_plan_));
        inverse_plan_ = -1;
    }
    cudaFree(d_tables_);
    d_tables_ = nullptr;
    cudaFree(d_moduli_);
    d_moduli_ = nullptr;
    cudaFree(d_gf_b_);
    d_gf_b_ = nullptr;
    cudaFree(d_potential_);
    d_potential_ = nullptr;
    cudaFree(d_ik_);
    d_ik_ = nullptr;
    cudaFree(d_influence_);
    d_influence_ = nullptr;
    cudaFree(d_virial_);
    d_virial_ = nullptr;
    cudaFree(d_spectrum_);
    d_spectrum_ = nullptr;
    cudaFree(d_grid_);
    d_grid_ = nullptr;
}

} // namespace cuda
} // namespace ATV

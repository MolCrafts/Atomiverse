/// @file cuda/gpw/internal/gpw_forces.cu
/// @brief Device grid force kernels for CUDA GPW (Γ RKS first slice).
///
/// All scientific grid loops run on device — no host loop over grid points
/// in the force hot path. Formulas match CPU `assemble_gpw_forces` grid
/// terms (Quickstep GPW forces).
#include "atomiverse/cuda/gpw/internal/gpw_forces.cuh"

#include <cmath>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

namespace {

constexpr int kBlock = 256;
constexpr double kPi = 3.14159265358979323846;

__device__ __forceinline__ void grid_r(long idx, int n0, int n1, int n2,
                                       double L0, double L1, double L2,
                                       double o0, double o1, double o2,
                                       double r[3]) {
    const int k = static_cast<int>(idx % n2);
    const long t = idx / n2;
    const int j = static_cast<int>(t % n1);
    const int i = static_cast<int>(t / n1);
    r[0] = o0 + L0 * static_cast<double>(i) / static_cast<double>(n0);
    r[1] = o1 + L1 * static_cast<double>(j) / static_cast<double>(n1);
    r[2] = o2 + L2 * static_cast<double>(k) / static_cast<double>(n2);
}

/// One cbgd active point per thread: accumulate V_eff grid Pulay.
__global__ void grid_veff_force_kernel(
    int n_active_d, double w, int nb, const long* __restrict__ cbgd_gidx,
    const std::size_t* __restrict__ cbgd_off, const int* __restrict__ cbgd_fu,
    const double* __restrict__ cbgd_fv, const std::size_t* __restrict__ cbg_off,
    const int* __restrict__ cbg_fu, const double* __restrict__ cbg_fv,
    const int* __restrict__ g2a, const double* __restrict__ veff,
    const double* __restrict__ P, const int* __restrict__ owner,
    double* __restrict__ forces) {
    const int ad = static_cast<int>(blockIdx.x) * blockDim.x
                   + static_cast<int>(threadIdx.x);
    if (ad >= n_active_d)
        return;
    const long g = cbgd_gidx[ad];
    const int a = g2a[g];
    if (a < 0)
        return;
    const std::size_t cs = cbg_off[a];
    const std::size_t ce = cbg_off[a + 1];
    const std::size_t ds = cbgd_off[ad];
    const std::size_t de = cbgd_off[ad + 1];
    const double vg = veff[g];
    const double scale = 2.0 * w * vg;
    for (std::size_t id = ds; id < de; ++id) {
        const int fdu = cbgd_fu[id];
        const int u = fdu / 3;
        const int k = fdu % 3;
        const double chi_dmu = cbgd_fv[id];
        const double* Pu = P + static_cast<std::size_t>(u) * nb;
        double pchi = 0.0;
        for (std::size_t ib = cs; ib < ce; ++ib)
            pchi += Pu[cbg_fu[ib]] * cbg_fv[ib];
        const double f = -scale * chi_dmu * pchi;
        atomicAdd(forces + 3 * owner[u] + k, f);
    }
}

/// One (atom, grid-chunk) via grid-stride: ionic HF.
__global__ void ionic_hf_kernel(int n_atoms, long npts, double w, int n0,
                                int n1, int n2, double L0, double L1, double L2,
                                double o0, double o1, double o2, bool periodic,
                                const DeviceForceAtom* __restrict__ atoms,
                                const double* __restrict__ v_es,
                                double* __restrict__ forces) {
    const long ntot = npts;
    for (int A = 0; A < n_atoms; ++A) {
        const DeviceForceAtom a = atoms[A];
        if (!(a.z_val != 0.0) || !(a.rloc > 0.0))
            continue;
        const double s2 = a.rloc * a.rloc;
        const double norm = a.z_val * pow(2.0 * kPi * s2, -1.5);
        const double inv_s2 = 1.0 / s2;
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (long g = static_cast<long>(blockIdx.x) * blockDim.x
                      + static_cast<long>(threadIdx.x);
             g < ntot; g += static_cast<long>(blockDim.x) * gridDim.x) {
            double r[3];
            grid_r(g, n0, n1, n2, L0, L1, L2, o0, o1, o2, r);
            double d[3] = {r[0] - a.pos[0], r[1] - a.pos[1],
                           r[2] - a.pos[2]};
            if (periodic) {
                d[0] -= L0 * nearbyint(d[0] / L0);
                d[1] -= L1 * nearbyint(d[1] / L1);
                d[2] -= L2 * nearbyint(d[2] / L2);
            }
            const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            const double gauss = norm * exp(-r2 * 0.5 * inv_s2);
            const double vg = v_es[g] * gauss * inv_s2;
            sx += vg * d[0];
            sy += vg * d[1];
            sz += vg * d[2];
        }
        // Block reduce then atomic to force (one block partial per atom).
        __shared__ double s[3][256];
        s[0][threadIdx.x] = sx;
        s[1][threadIdx.x] = sy;
        s[2][threadIdx.x] = sz;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s[0][threadIdx.x] += s[0][threadIdx.x + stride];
                s[1][threadIdx.x] += s[1][threadIdx.x + stride];
                s[2][threadIdx.x] += s[2][threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            atomicAdd(forces + 3 * A + 0, w * s[0][0]);
            atomicAdd(forces + 3 * A + 1, w * s[1][0]);
            atomicAdd(forces + 3 * A + 2, w * s[2][0]);
        }
        __syncthreads();
    }
}

/// Local-PP HF: one atom outer, grid-stride inner (like CPU).
__global__ void local_pp_hf_kernel(int n_atoms, long npts, double w, int n0,
                                   int n1, int n2, double L0, double L1,
                                   double L2, double o0, double o1, double o2,
                                   bool periodic,
                                   const DeviceForceAtom* __restrict__ atoms,
                                   const double* __restrict__ rho,
                                   double* __restrict__ forces) {
    for (int A = 0; A < n_atoms; ++A) {
        const DeviceForceAtom a = atoms[A];
        if (a.n_cloc <= 0 || !(a.rloc > 0.0))
            continue;
        const double inv_s2 = 1.0 / (a.rloc * a.rloc);
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (long g = static_cast<long>(blockIdx.x) * blockDim.x
                      + static_cast<long>(threadIdx.x);
             g < npts; g += static_cast<long>(blockDim.x) * gridDim.x) {
            const double rg = rho[g];
            if (rg == 0.0)
                continue;
            double r[3];
            grid_r(g, n0, n1, n2, L0, L1, L2, o0, o1, o2, r);
            double d[3] = {r[0] - a.pos[0], r[1] - a.pos[1], r[2] - a.pos[2]};
            if (periodic) {
                d[0] -= L0 * nearbyint(d[0] / L0);
                d[1] -= L1 * nearbyint(d[1] / L1);
                d[2] -= L2 * nearbyint(d[2] / L2);
            }
            const double t =
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) * inv_s2;
            double poly = 0.0, dpoly = 0.0;
            double tp = 1.0, tpm1 = 0.0;
            for (int k = 0; k < a.n_cloc; ++k) {
                poly += a.cloc[k] * tp;
                if (k > 0)
                    dpoly += static_cast<double>(k) * a.cloc[k] * tpm1;
                tpm1 = tp;
                tp *= t;
            }
            const double dvdt = exp(-0.5 * t) * (dpoly - 0.5 * poly);
            const double cc = rg * dvdt * 2.0 * inv_s2;
            sx += cc * d[0];
            sy += cc * d[1];
            sz += cc * d[2];
        }
        __shared__ double s[3][256];
        s[0][threadIdx.x] = sx;
        s[1][threadIdx.x] = sy;
        s[2][threadIdx.x] = sz;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s[0][threadIdx.x] += s[0][threadIdx.x + stride];
                s[1][threadIdx.x] += s[1][threadIdx.x + stride];
                s[2][threadIdx.x] += s[2][threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            atomicAdd(forces + 3 * A + 0, w * s[0][0]);
            atomicAdd(forces + 3 * A + 1, w * s[1][0]);
            atomicAdd(forces + 3 * A + 2, w * s[2][0]);
        }
        __syncthreads();
    }
}

} // namespace

void accumulate_grid_veff_forces_device(const CollocateGridDesc& grid,
                                        const DeviceCbgView& cbg,
                                        const DeviceCbgView& cbgd,
                                        const double* d_veff, const double* d_P,
                                        int nb, const int* d_owner,
                                        double* d_forces, cudaStream_t stream) {
    if (cbgd.n_active <= 0 || nb <= 0)
        return;
    const double w = grid.weight();
    const int grid_dim = (cbgd.n_active + kBlock - 1) / kBlock;
    grid_veff_force_kernel<<<grid_dim, kBlock, 0, stream>>>(
        cbgd.n_active, w, nb, cbgd.gidx, cbgd.off, cbgd.fu, cbgd.fv, cbg.off,
        cbg.fu, cbg.fv, cbg.g2a, d_veff, d_P, d_owner, d_forces);
}

void accumulate_ionic_hf_forces_device(const CollocateGridDesc& grid,
                                       const DeviceForceAtom* d_atoms,
                                       int n_atoms, const double* d_v_es,
                                       bool periodic, double* d_forces,
                                       cudaStream_t stream) {
    if (n_atoms <= 0)
        return;
    const long npts = grid.npts();
    if (npts <= 0)
        return;
    const double w = grid.weight();
    // Enough blocks to cover the grid once per atom loop (grid-stride).
    int grid_dim =
        static_cast<int>((npts + kBlock - 1) / static_cast<long>(kBlock));
    if (grid_dim > 1024)
        grid_dim = 1024;
    ionic_hf_kernel<<<grid_dim, kBlock, 0, stream>>>(
        n_atoms, npts, w, grid.n[0], grid.n[1], grid.n[2], grid.L[0],
        grid.L[1], grid.L[2], grid.origin[0], grid.origin[1], grid.origin[2],
        periodic, d_atoms, d_v_es, d_forces);
}

void accumulate_local_pp_hf_forces_device(const CollocateGridDesc& grid,
                                          const DeviceForceAtom* d_atoms,
                                          int n_atoms, const double* d_rho,
                                          bool periodic, double* d_forces,
                                          cudaStream_t stream) {
    if (n_atoms <= 0)
        return;
    const long npts = grid.npts();
    if (npts <= 0)
        return;
    const double w = grid.weight();
    int grid_dim =
        static_cast<int>((npts + kBlock - 1) / static_cast<long>(kBlock));
    if (grid_dim > 1024)
        grid_dim = 1024;
    local_pp_hf_kernel<<<grid_dim, kBlock, 0, stream>>>(
        n_atoms, npts, w, grid.n[0], grid.n[1], grid.n[2], grid.L[0],
        grid.L[1], grid.L[2], grid.origin[0], grid.origin[1], grid.origin[2],
        periodic, d_atoms, d_rho, d_forces);
}

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV

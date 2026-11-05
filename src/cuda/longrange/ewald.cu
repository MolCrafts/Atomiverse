/// @file cuda/longrange/ewald.cu
/// @brief Reciprocal-space Ewald summation implementation (CUDA).
///
/// The integer mode set m is chosen once at build from the build-time cell;
/// every launch reads the published device Box. Wave vectors
/// g = 2π Σ_i m_i b_i (b_i = row i of cell⁻¹: lattice vectors are the cell
/// columns), their weights and virial coefficients live in device tables that
/// are recomputed in stream order whenever `Box::revision` moves, so a
/// changing cell (deform, barostat) needs no rebuild. The cutoff sphere is
/// not re-chosen when the cell changes.
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/longrange_math.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/energy_term.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/longrange/ewald.cuh"

namespace ATV {

namespace cuda {

// Bookkeeping of the cell-dependent k tables (fixed device address).
struct EwaldTables {
    unsigned long long revision; ///< Box revision of the tables; ~0 at build.
    int dirty;                   ///< Tables are recomputed on this launch.
};

namespace kernels {

/// @brief Flag the k tables stale when the published cell revision moved;
///        add the neutralizing-background energy (and its virial).
template <bool Virial>
__global__ void ewald_refresh_tables(Box box_arg,
                                     EwaldTables* __restrict__ tables,
                                     double background_coef, EnergyTerm energy,
                                     acc_t* __restrict__ stress) {
    const Box box = box_arg.current();
    EwaldTables& t = *tables;
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

/// @brief Recompute wave vectors, weights exp(−g²/4α²)/g² and the strain
///        coefficients 1 − 2(1/g² + 1/4α²) g_a g_b for the current cell.
template <bool Virial>
__global__ void
ewald_refresh_k(const EwaldTables* __restrict__ tables, Box box_arg,
                const int* __restrict__ modes, int ng, double inv4a2,
                double* __restrict__ kx, double* __restrict__ ky,
                double* __restrict__ kz, double* __restrict__ kw,
                double* __restrict__ virial) {
    if (!tables->dirty)
        return;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= ng)
        return;
    const Box box = box_arg.current();
    const double tp = 2.0 * ATV::units::pi;
    const int* m = modes + 3 * k;
    double g[3];
    for (int c = 0; c < 3; ++c)
        g[c] = tp
               * (m[0] * box.inv_cell[c] + m[1] * box.inv_cell[3 + c]
                  + m[2] * box.inv_cell[6 + c]);
    const double g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    kx[k] = g[0];
    ky[k] = g[1];
    kz[k] = g[2];
    kw[k] = exp(-g2 * inv4a2) / g2;
    if constexpr (Virial) {
        const double vt = -2.0 * (1.0 / g2 + inv4a2);
        const int pairs[6][2] = {{0, 0}, {1, 1}, {2, 2},
                                 {0, 1}, {0, 2}, {1, 2}};
        for (int c = 0; c < 6; ++c) {
            const int a = pairs[c][0], b = pairs[c][1];
            virial[static_cast<std::size_t>(c) * ng + k] =
                (a == b ? 1.0 : 0.0) + vt * g[a] * g[b];
        }
    }
}

/// @brief Structure factor S(k)=Σ_j q_j exp(i k·r_j). One CUB block per k.
__global__ void ewald_structure_factor(
    const pos_t* __restrict__ x, const pos_t* __restrict__ y,
    const pos_t* __restrict__ z, const force_t* __restrict__ q, int n,
    const double* __restrict__ kx, const double* __restrict__ ky,
    const double* __restrict__ kz, int ng, double* __restrict__ sr,
    double* __restrict__ si) {
    const int k = blockIdx.x;
    const double kxk = kx[k], kyk = ky[k], kzk = kz[k];
    ::cuda::std::array<double, 2> contribution{};
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const double phase =
            kxk * double(x[j]) + kyk * double(y[j]) + kzk * double(z[j]);
        const double qj = double(q[j]);
        contribution[0] += qj * cos(phase);
        contribution[1] += qj * sin(phase);
    }
    using Reduce =
        cub::BlockReduce<::cuda::std::array<double, 2>, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const auto sum =
        Reduce(reduction).Reduce(contribution, SumComponents<double, 2>{});
    if (threadIdx.x == 0) {
        sr[k] = sum[0];
        si[k] = sum[1];
    }
}

/// @brief Reciprocal energy Σ_k K·kw·|S(k)|² (+ virial), K = 2π k_e / V.
template <bool Virial>
__global__ void ewald_energy(const double* __restrict__ kw,
                             const double* __restrict__ virial,
                             const double* __restrict__ sr,
                             const double* __restrict__ si, int ng, Box box_arg,
                             EnergyTerm energy, acc_t* __restrict__ stress) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const double K = ATV::units::md_qqr2e * 2.0 * ATV::units::pi
                     / box_arg.current().volume();
    constexpr int components = Virial ? 7 : 1;
    ::cuda::std::array<acc_t, components> values{};
    if (k < ng) {
        const double e = K * kw[k] * (sr[k] * sr[k] + si[k] * si[k]);
        values[0] = e;
        if constexpr (Virial)
            for (int c = 0; c < 6; ++c)
                values[c + 1] =
                    e * virial[static_cast<std::size_t>(c) * ng + k];
    }
    using Reduce =
        cub::BlockReduce<::cuda::std::array<acc_t, components>, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const auto sum =
        Reduce(reduction).Reduce(values, SumComponents<acc_t, components>{});
    if (threadIdx.x == 0) {
        if (energy)
            energy.add(sum[0]);
        if constexpr (Virial)
            for (int c = 0; c < 6; ++c)
                atomicAdd(stress + c, sum[c + 1]);
    }
}

/// @brief Reciprocal forces on owned atoms + per-atom self energy.
__global__ void
ewald_forces(fixed_t* __restrict__ fx, fixed_t* __restrict__ fy,
             fixed_t* __restrict__ fz, ForceFault* fault,
             const pos_t* __restrict__ x, const pos_t* __restrict__ y,
             const pos_t* __restrict__ z, const force_t* __restrict__ q, int n,
             const double* __restrict__ kx, const double* __restrict__ ky,
             const double* __restrict__ kz, const double* __restrict__ kw,
             const double* __restrict__ sr, const double* __restrict__ si,
             int ng, Box box_arg, double self_pref, EnergyTerm energy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const double qi = i < n ? double(q[i]) : 0;
    if (energy) { // uniform: a step without eflag skips it
        using Reduce = cub::BlockReduce<double, kDefaultBlock>;
        __shared__ typename Reduce::TempStorage reduction;
        const double self_energy = Reduce(reduction).Sum(self_pref * qi * qi);
        if (threadIdx.x == 0)
            energy.add(self_energy);
    }
    if (i >= n)
        return;
    const double K = ATV::units::md_qqr2e * 2.0 * ATV::units::pi
                     / box_arg.current().volume();
    const double xi = double(x[i]), yi = double(y[i]), zi = double(z[i]);
    double fxi = 0, fyi = 0, fzi = 0;
    for (int k = 0; k < ng; ++k) {
        const double phase = kx[k] * xi + ky[k] * yi + kz[k] * zi;
        const double c = cos(phase), s = sin(phase);
        const double fac = 2.0 * K * kw[k] * qi * (s * sr[k] - c * si[k]);
        fxi += fac * kx[k];
        fyi += fac * ky[k];
        fzi += fac * kz[k];
    }
    // One thread owns atom i in this kernel.
    add_force_owned(fx, i, fxi, fault);
    add_force_owned(fy, i, fyi, fault);
    add_force_owned(fz, i, fzi, fault);
}

} // namespace kernels

KSpaceEwald::KSpaceEwald(double alpha, double accuracy)
    : alpha_(alpha), accuracy_(accuracy) {
    if (alpha <= 0.0)
        throw std::invalid_argument("KSpaceEwald: alpha must be > 0");
}

KSpaceEwald::~KSpaceEwald() {}

void KSpaceEwald::build(const BuildContext& ctx) {
    // Mode set from the build-time cell: reciprocal basis b_i = 2π·(row i of
    // inv_cell); keep |Σ m_i b_i| inside the accuracy cutoff.
    const cuda::Box& box = ctx.runtime.host_box;
    const double tp = 2.0 * ATV::units::pi;
    double b[3][3];
    for (int i = 0; i < 3; ++i)
        for (int c = 0; c < 3; ++c)
            b[i][c] = tp * double(box.inv_cell[i * 3 + c]);
    auto blen = [&](int i) {
        return std::sqrt(b[i][0] * b[i][0] + b[i][1] * b[i][1]
                         + b[i][2] * b[i][2]);
    };
    const double gcut = 2.0 * alpha_ * std::sqrt(-std::log(accuracy_));
    const double gcut2 = gcut * gcut;
    int kmax[3];
    for (int i = 0; i < 3; ++i)
        kmax[i] = std::max(1, static_cast<int>(std::ceil(gcut / blen(i))));
    std::vector<int> modes;
    for (int mx = -kmax[0]; mx <= kmax[0]; ++mx)
        for (int my = -kmax[1]; my <= kmax[1]; ++my)
            for (int mz = -kmax[2]; mz <= kmax[2]; ++mz) {
                if (mx == 0 && my == 0 && mz == 0)
                    continue;
                const double gx = mx * b[0][0] + my * b[1][0] + mz * b[2][0];
                const double gy = mx * b[0][1] + my * b[1][1] + mz * b[2][1];
                const double gz = mx * b[0][2] + my * b[1][2] + mz * b[2][2];
                if (gx * gx + gy * gy + gz * gz > gcut2)
                    continue;
                modes.insert(modes.end(), {mx, my, mz});
            }
    ng_ = static_cast<int>(modes.size() / 3);
    const size_t bytes = ng_ * sizeof(double);
    ATV_CUDA_CHECK(cudaMalloc(&d_modes_, modes.size() * sizeof(int)));
    ATV_CUDA_CHECK(cudaMemcpy(d_modes_, modes.data(),
                              modes.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    for (double** table : {&d_kx_, &d_ky_, &d_kz_, &d_kw_, &d_sr_, &d_si_})
        ATV_CUDA_CHECK(cudaMalloc(table, bytes));
    // Strain coefficients refresh with the wave vectors on every cell change,
    // so a later vflag step never reads a stale table.
    ATV_CUDA_CHECK(cudaMalloc(&d_virial_, 6 * bytes));
    ATV_CUDA_CHECK(cudaMalloc(&d_tables_, sizeof(EwaldTables)));
    // Revision ~0 never matches a published Box: the first launch fills them.
    ATV_CUDA_CHECK(cudaMemset(d_tables_, 0xff, sizeof(EwaldTables)));

    // Net charge of the atom set: −k_e π Q² / (2Vα²) background energy.
    const int n = ctx.frame.n_local();
    std::vector<force_t> charge(n);
    if (n)
        ATV_CUDA_CHECK(cudaMemcpy(charge.data(), ctx.frame.atoms.charge,
                                  n * sizeof(force_t), cudaMemcpyDeviceToHost));
    double net_charge = 0.0;
    for (const force_t q : charge)
        net_charge += double(q);
    background_coef_ = -longrange::background_energy(net_charge, 1.0, alpha_);
}

void KSpaceEwald::compute(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const int n = frame.n_local();
    if (ng_ == 0 || n == 0)
        return;

    const cuda::Box& box = ctx.runtime.host_box;
    const double self_pref = longrange::self_energy_prefactor(alpha_);
    const double inv4a2 = 1.0 / (4.0 * alpha_ * alpha_);
    auto* tables = static_cast<EwaldTables*>(d_tables_);
    // Per-step LAMMPS eflag/vflag.
    // Self, reciprocal and background energy are pair energy, as on CPU
    // (LAMMPS `epair` includes `elong`).
    const EnergyTerm energy =
        energy_term(frame.scalars, EnergyKind::Pair, ctx.energy_requested);
    acc_t* stress = frame.scalars->stress;
    const bool virial = ctx.stress_requested;
    const int blk = kDefaultBlock;
    auto run = [&]<bool Virial>() {
        kernels::ewald_refresh_tables<Virial><<<1, 1, 0, stream>>>(
            box, tables, background_coef_, energy, stress);
        kernels::ewald_refresh_k<true><<<grid_size(ng_), blk, 0, stream>>>(
            tables, box, d_modes_, ng_, inv4a2, d_kx_, d_ky_, d_kz_, d_kw_,
            d_virial_);
        kernels::ewald_structure_factor<<<ng_, blk, 0, stream>>>(
            frame.atoms.x, frame.atoms.y, frame.atoms.z, frame.atoms.charge, n,
            d_kx_, d_ky_, d_kz_, ng_, d_sr_, d_si_);
        if (energy || Virial)
            kernels::ewald_energy<Virial><<<grid_size(ng_), blk, 0, stream>>>(
                d_kw_, d_virial_, d_sr_, d_si_, ng_, box, energy, stress);
    };
    if (virial)
        run.template operator()<true>();
    else
        run.template operator()<false>();
    kernels::ewald_forces<<<grid_size(n), blk, 0, stream>>>(
        frame.atoms.fx, frame.atoms.fy, frame.atoms.fz, frame.force_fault,
        frame.atoms.x, frame.atoms.y, frame.atoms.z, frame.atoms.charge, n,
        d_kx_, d_ky_, d_kz_, d_kw_, d_sr_, d_si_, ng_, box, self_pref, energy);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void KSpaceEwald::destroy(const DestroyContext&) {
    for (void* p :
         {static_cast<void*>(d_modes_), static_cast<void*>(d_kx_),
          static_cast<void*>(d_ky_), static_cast<void*>(d_kz_),
          static_cast<void*>(d_kw_), static_cast<void*>(d_sr_),
          static_cast<void*>(d_si_), static_cast<void*>(d_virial_), d_tables_})
        cudaFree(p);
    d_modes_ = nullptr;
    d_kx_ = d_ky_ = d_kz_ = d_kw_ = d_sr_ = d_si_ = d_virial_ = nullptr;
    d_tables_ = nullptr;
    ng_ = 0;
}

} // namespace cuda
} // namespace ATV

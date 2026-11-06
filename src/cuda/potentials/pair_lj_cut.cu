/// @file cuda/potentials/pair_lj_cut.cu
/// @brief CUDA PairLJCut — Lennard-Jones 12-6 shifted (kernel + class).
///        Includes LAMMPS-style `special_bonds` 1-2/1-3/1-4 exclusions.
///
/// Fractional special-bond weights follow LAMMPS `PairLJCut::compute`
/// (`src/pair_lj_cut.cpp`: factor_lj · forcelj and factor_lj · (evdwl −
/// offset) inside cutsq[itype][jtype]); the tail correction follows
/// `PairLJCut::init_one` (etail_ij / ptail_ij, `core/lj_type_table.h`),
/// `src/compute_pe.cpp` (etail/V) and `ComputePressure::virial_compute`
/// (`src/compute_pressure.cpp`, ptail/V on the diagonal). LAMMPS checkout
/// `e8c4c278a8`:
///
///   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
///   https://www.lammps.org/, Sandia National Laboratories
///   LAMMPS development team: developers@lammps.org
///
///   Copyright (2003) Sandia Corporation.  Under the terms of Contract
///   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
///   certain rights in this software.  This software is distributed under
///   the GNU General Public License.

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/lj_type_table.h"
#include "atomiverse/core/pair_kernel.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/energy_term.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/lj_kernel.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "internal/lj_tail.cuh"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ATV {

namespace cuda {
namespace kernels {

// Gather current atom positions into the cluster-sorted slot layout so the
// force kernel reads neighbour positions COALESCED (spx[cj*8+sj] is contiguous
// per cluster) instead of scattering x[orig_index[..]] ~80×/atom. One scattered
// read per slot here amortises the scatter that previously happened per pair —
// the dominant cost of the pre-tile atom-walk force (latency-bound gathers).
__global__ void
gather_cluster_pos(const int* __restrict__ orig_index,
                   const pos_t* __restrict__ x, const pos_t* __restrict__ y,
                   const pos_t* __restrict__ z, const int* __restrict__ type,
                   int n_slots, force_t* __restrict__ spx,
                   force_t* __restrict__ spy, force_t* __restrict__ spz,
                   int* __restrict__ spt) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_slots)
        return;
    int a = orig_index[k];
    if (a >= 0) {
        spx[k] = x[a];
        spy[k] = y[a];
        spz[k] = z[a];
        // Typed path only: the slot's atom type indexes the pair table.
        if (spt != nullptr)
            spt[k] = type[a];
    }
}

// One resolved (type_i, type_j) record of `LJTypeTable` in force_t.
struct LJTypedCoeff {
    force_t lj1, lj2, lj3, lj4, cut_ljsq, offset;
    force_t r4_a, r4_cutsq, r4_rc5inv, r4_e0;
};

/// @brief Atom-centric, **atomic-free** LJ force. One thread owns one atom
///        (cluster slot). It walks its cluster's FULL neighbor-cluster list
///        (col_idx over [row_ptr[ci], row_ptr[ci+1])), computes each pair
///        distance from CURRENT atom positions (passed pre-gathered in the
///        cluster-sorted layout spx/spy/spz, min-image), accumulates the owned
///        atom's force in registers, and writes it ONCE — no atomicAdd, no flat
///        pair list, no per-step export.
///
/// Replaces the old one-thread-per-pair kernel that did 6 atomicAdd/pair
/// (~160-way contention per force slot in a dense melt → the 76%-of-runtime
/// bottleneck). With a FULL neighbor list each interaction is computed twice
/// (once by each endpoint's owning thread), so the force on each atom is
/// complete without writing its neighbors; energy and virial are halved to
/// OpenMM-style tile force: one warp per cluster (CS=32). Neighbor cluster
/// positions are staged in per-warp shared memory; interaction mask selects
/// active j-slots. FULL neighbor + 0.5 energy (both endpoints accumulate).
template <bool Ortho, bool Energy, bool Virial, bool Typed>
__global__ __launch_bounds__(ATV::cuda::kDefaultBlock, 4) void lj_force_tile(
    fixed_t* __restrict__ fx, fixed_t* __restrict__ fy,
    fixed_t* __restrict__ fz, ForceFault* fault, EnergyTerm energy,
    acc_t* __restrict__ stress, const force_t* __restrict__ spx,
    const force_t* __restrict__ spy, const force_t* __restrict__ spz,
    const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const std::uint64_t* __restrict__ cl_mask,
    const int* __restrict__ orig_index, int num_clusters, int n_owned,
    cuda::Box box_arg, bool periodic, force_t epsilon, force_t sigma_sq,
    force_t cutoff_sq, force_t min_pair_dist_sq, force_t v_shift,
    const int* __restrict__ excl_offsets, const int* __restrict__ excl_partners,
    force_t s_excl, const int* __restrict__ spt,
    const LJTypedCoeff* __restrict__ table, int n_types) {
    const auto box = box_arg.current().as<force_t>();
    constexpr int CS = ATV::cuda::kOpenmmClusterSize;
    static_assert(CS == 32, "tile force assumes OpenMM CS=32");

    // Per-warp tile staging: [warp][slot]
    constexpr int kMaxWarps = ATV::cuda::kDefaultBlock / 32;
    __shared__ force_t shx[kMaxWarps][CS];
    __shared__ force_t shy[kMaxWarps][CS];
    __shared__ force_t shz[kMaxWarps][CS];
    __shared__ int sh_aj[kMaxWarps][CS];
    __shared__ int sh_t[Typed ? kMaxWarps : 1][CS];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ci = tid / CS;
    int slot = tid % CS;
    int warp_in_block = threadIdx.x / 32;

    acc_t my_energy = 0;
    acc_t my_vxx = 0, my_vyy = 0, my_vzz = 0;
    acc_t my_vxy = 0, my_vxz = 0, my_vyz = 0;

    const force_t eps_r = force_t(epsilon);
    const force_t sigsq_r = force_t(sigma_sq);
    const force_t cut_r = force_t(cutoff_sq);
    const force_t minr_r = force_t(min_pair_dist_sq);
    const force_t vshift_r = force_t(v_shift);
    const force_t sexcl_r = force_t(s_excl);

    // All lanes of a warp must participate in __syncwarp; pad slots keep
    // ai < 0 and skip force math but still load tiles.
    int ai = (ci < num_clusters) ? orig_index[ci * CS + slot] : -1;
    if (ai >= n_owned)
        ai = -1;
    force_t xi = 0, yi = 0, zi = 0;
    force_t fxi = 0, fyi = 0, fzi = 0;
    const LJTypedCoeff* row = nullptr;
    bool has_excl = false;
    int beg_excl = 0, end_excl = 0;
    if (ai >= 0) {
        xi = spx[ci * CS + slot];
        yi = spy[ci * CS + slot];
        zi = spz[ci * CS + slot];
        if constexpr (Typed)
            row =
                table + static_cast<std::size_t>(spt[ci * CS + slot]) * n_types;
        has_excl = (excl_offsets != nullptr);
        beg_excl = has_excl ? excl_offsets[ai] : 0;
        end_excl = has_excl ? excl_offsets[ai + 1] : 0;
    }

    if (ci < num_clusters) {
        int rstart = row_ptr[ci], rend = row_ptr[ci + 1];
        for (int p = rstart; p < rend; ++p) {
            int cj = col_idx[p];
            shx[warp_in_block][slot] = spx[cj * CS + slot];
            shy[warp_in_block][slot] = spy[cj * CS + slot];
            shz[warp_in_block][slot] = spz[cj * CS + slot];
            sh_aj[warp_in_block][slot] = orig_index[cj * CS + slot];
            if constexpr (Typed)
                sh_t[warp_in_block][slot] =
                    sh_aj[warp_in_block][slot] >= 0 ? spt[cj * CS + slot] : 0;
            __syncwarp();

            if (ai >= 0) {
                unsigned mbits = ATV::cuda::openmm_mask_jbits(cl_mask, p, slot);
                while (mbits) {
                    int sj = __ffs(static_cast<int>(mbits)) - 1;
                    mbits &= mbits - 1;
                    int aj = sh_aj[warp_in_block][sj];
                    if (aj < 0 || aj == ai)
                        continue;
                    force_t dx = shx[warp_in_block][sj] - xi;
                    force_t dy = shy[warp_in_block][sj] - yi;
                    force_t dz = shz[warp_in_block][sj] - zi;
                    if (periodic) {
                        if constexpr (Ortho) {
                            dx -= box.cell[0] * nearbyint(dx * box.inv_l[0]);
                            dy -= box.cell[4] * nearbyint(dy * box.inv_l[1]);
                            dz -= box.cell[8] * nearbyint(dz * box.inv_l[2]);
                        } else {
                            box.min_image(dx, dy, dz);
                        }
                    }
                    force_t r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= cut_r || r2 < minr_r)
                        continue;
                    force_t scale = force_t(1);
                    if (has_excl && end_excl > beg_excl
                        && thrust::binary_search(
                            thrust::seq, excl_partners + beg_excl,
                            excl_partners + end_excl, aj)) {
                        if (sexcl_r == force_t(0))
                            continue;
                        scale = sexcl_r;
                    }
                    force_t inv_r2 = force_t(1) / r2;
                    force_t f_over_r, e_pair = 0;
                    if constexpr (Typed) {
                        // LAMMPS init_one record: own cut_ljsq and offset,
                        // plus the r^-4 term inside its own cutoff.
                        const LJTypedCoeff* c = row + sh_t[warp_in_block][sj];
                        f_over_r = 0;
                        if (r2 < __ldg(&c->cut_ljsq)) {
                            const force_t r6inv = inv_r2 * inv_r2 * inv_r2;
                            f_over_r =
                                inv_r2 * r6inv
                                * (__ldg(&c->lj1) * r6inv - __ldg(&c->lj2));
                            if constexpr (Energy)
                                e_pair = r6inv
                                             * (__ldg(&c->lj3) * r6inv
                                                - __ldg(&c->lj4))
                                         - __ldg(&c->offset);
                        }
                        if (r2 < __ldg(&c->r4_cutsq)) {
                            const R4Term<force_t> t = r4_attraction<force_t>(
                                r2, __ldg(&c->r4_a), __ldg(&c->r4_rc5inv),
                                __ldg(&c->r4_e0));
                            f_over_r += t.fpair;
                            if constexpr (Energy)
                                e_pair += t.energy;
                        }
                        f_over_r *= scale;
                        e_pair *= scale;
                    } else {
                        force_t sr2 = sigsq_r * inv_r2;
                        force_t sr6 = sr2 * sr2 * sr2;
                        force_t sr12 = sr6 * sr6;
                        f_over_r = scale * force_t(24) * eps_r
                                   * (force_t(2) * sr12 - sr6) * inv_r2;
                        if constexpr (Energy)
                            e_pair = scale
                                     * (force_t(4) * eps_r * (sr12 - sr6)
                                        - vshift_r);
                    }
                    fxi += -f_over_r * dx;
                    fyi += -f_over_r * dy;
                    fzi += -f_over_r * dz;
                    if constexpr (Energy)
                        my_energy += acc_t(0.5) * acc_t(e_pair);
                    if constexpr (Virial) {
                        my_vxx += acc_t(0.5) * acc_t(dx * (f_over_r * dx));
                        my_vyy += acc_t(0.5) * acc_t(dy * (f_over_r * dy));
                        my_vzz += acc_t(0.5) * acc_t(dz * (f_over_r * dz));
                        my_vxy += acc_t(0.5) * acc_t(dx * (f_over_r * dy));
                        my_vxz += acc_t(0.5) * acc_t(dx * (f_over_r * dz));
                        my_vyz += acc_t(0.5) * acc_t(dy * (f_over_r * dz));
                    }
                }
            }
            __syncwarp();
        }
        // Full list: each owned row is written by exactly one lane.
        if (ai >= 0) {
            add_force_owned(fx, ai, fxi, fault);
            add_force_owned(fy, ai, fyi, fault);
            add_force_owned(fz, ai, fzi, fault);
        }
    }

    // CUB reduces energy and each virial component in a single collective.
    // A step that asked for neither skips the reduction and every atomic.
    if constexpr (Virial) {
        ::cuda::std::array<acc_t, 7> my_vals = {
            my_energy, my_vxx, my_vyy, my_vzz, my_vxy, my_vxz, my_vyz};

        using Reduce =
            cub::BlockReduce<::cuda::std::array<acc_t, 7>, kDefaultBlock>;
        __shared__ typename Reduce::TempStorage reduction;
        const auto out =
            Reduce(reduction).Reduce(my_vals, SumComponents<acc_t, 7>{});
        if (threadIdx.x == 0) {
            if constexpr (Energy)
                energy.add(out[0]);
            if (stress != nullptr) {
#pragma unroll
                for (int c = 0; c < 6; ++c)
                    atomicAdd(&stress[c], out[c + 1]);
            }
        }
    } else if constexpr (Energy) {
        // Virial gated off: reduce energy alone; the my_v* accumulators stay
        // 0 and are dead-code-eliminated, freeing their registers.
        ::cuda::std::array<acc_t, 1> my_vals = {my_energy};

        using Reduce =
            cub::BlockReduce<::cuda::std::array<acc_t, 1>, kDefaultBlock>;
        __shared__ typename Reduce::TempStorage reduction;
        const auto out =
            Reduce(reduction).Reduce(my_vals, SumComponents<acc_t, 1>{});
        if (threadIdx.x == 0)
            energy.add(out[0]);
    }
}

/// One fractional special-bond pair (i < j) and its table record.
struct LJException {
    int i, j, ab;  ///< Atom rows and table index of (type_i, type_j).
    force_t scale; ///< LAMMPS factor_lj of the pair's nearest shell.
};

/// @brief Sparse pass over fractional special-bond pairs (LAMMPS
///        `factor_lj`): w · (LJ − offset) inside the pair's LJ cutoff and
///        w · r⁻⁴ inside its r⁻⁴ cutoff, minimum image. The tiles excluded
///        these pairs, so each is evaluated exactly once here.
template <bool Energy, bool Virial>
__global__ void
lj_exceptions(const LJException* __restrict__ pairs, int count,
              const pos_t* __restrict__ x, const pos_t* __restrict__ y,
              const pos_t* __restrict__ z,
              const LJTypedCoeff* __restrict__ table, cuda::Box box_arg,
              fixed_t* fx, fixed_t* fy, fixed_t* fz, ForceFault* fault,
              EnergyTerm energy, acc_t* stress) {
    const auto box = box_arg.current().as<force_t>();
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    acc_t e = 0, v[6] = {};
    if (p < count) {
        const LJException t = pairs[p];
        force_t dx = x[t.j] - x[t.i], dy = y[t.j] - y[t.i],
                dz = z[t.j] - z[t.i];
        box.min_image(dx, dy, dz);
        const force_t r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > force_t(0)) {
            const LJTypedCoeff* c = table + t.ab;
            const force_t inv2 = force_t(1) / r2;
            force_t fr = 0, en = 0;
            if (r2 < __ldg(&c->cut_ljsq)) {
                const force_t r6inv = inv2 * inv2 * inv2;
                fr = inv2 * r6inv * (__ldg(&c->lj1) * r6inv - __ldg(&c->lj2));
                en = r6inv * (__ldg(&c->lj3) * r6inv - __ldg(&c->lj4))
                     - __ldg(&c->offset);
            }
            if (r2 < __ldg(&c->r4_cutsq)) {
                const R4Term<force_t> r4 = r4_attraction<force_t>(
                    r2, __ldg(&c->r4_a), __ldg(&c->r4_rc5inv),
                    __ldg(&c->r4_e0));
                fr += r4.fpair;
                en += r4.energy;
            }
            fr *= t.scale;
            en *= t.scale;
            // fr = -(dU/dr)/r with d = x_j - x_i: F_i = -fr d, F_j = +fr d.
            add_force(fx, t.i, -fr * dx, fault);
            add_force(fy, t.i, -fr * dy, fault);
            add_force(fz, t.i, -fr * dz, fault);
            add_force(fx, t.j, fr * dx, fault);
            add_force(fy, t.j, fr * dy, fault);
            add_force(fz, t.j, fr * dz, fault);
            e = acc_t(en);
            v[0] = acc_t(fr * dx * dx);
            v[1] = acc_t(fr * dy * dy);
            v[2] = acc_t(fr * dz * dz);
            v[3] = acc_t(fr * dx * dy);
            v[4] = acc_t(fr * dx * dz);
            v[5] = acc_t(fr * dy * dz);
        }
    }
    if constexpr (Energy || Virial) {
        ::cuda::std::array<acc_t, 7> vals = {e,    v[0], v[1], v[2],
                                             v[3], v[4], v[5]};
        using Reduce =
            cub::BlockReduce<::cuda::std::array<acc_t, 7>, kDefaultBlock>;
        __shared__ typename Reduce::TempStorage reduction;
        const auto out =
            Reduce(reduction).Reduce(vals, SumComponents<acc_t, 7>{});
        if (threadIdx.x == 0) {
            if constexpr (Energy)
                energy.add(out[0]);
            if constexpr (Virial)
                for (int d = 0; d < 6; ++d)
                    atomicAdd(&stress[d], out[d + 1]);
        }
    }
}

/// LAMMPS `pair_modify tail yes`: etail/V and ptail/V from the live cell.
__global__ void lj_tail(cuda::Box box_arg, double etail, double ptail,
                        EnergyTerm energy, acc_t* stress) {
    const cuda::Box box = box_arg.current();
    const double inv_volume = 1.0 / static_cast<double>(box.volume());
    if (energy)
        energy.add(acc_t(etail * inv_volume));
    if (stress != nullptr)
        for (int d = 0; d < 3; ++d)
            atomicAdd(&stress[d], acc_t(ptail * inv_volume));
}

} // namespace kernels

void internal::launch_lj_tail(const Box& box, const LJTail& tail,
                              EnergyTerm energy, acc_t* stress,
                              cudaStream_t stream) {
    if (!energy && stress == nullptr)
        return;
    kernels::lj_tail<<<1, 1, 0, stream>>>(box, tail.energy, tail.virial, energy,
                                          stress);
    ATV_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda

namespace cuda {

PairLJCut::PairLJCut(double epsilon, double sigma, double cutoff)
    : Pair(cutoff), epsilon_(epsilon), sigma_(sigma) {
    v_shift_ = kernels::lj_shift(epsilon, sigma, cutoff);
}

PairLJCut::PairLJCut(std::vector<LJTypeCoeff> coeffs, double cutoff)
    : Pair(cutoff), epsilon_(0.0), sigma_(0.0), v_shift_(0.0) {
    if (coeffs.empty())
        throw std::invalid_argument(
            "cuda::PairLJCut: the per-type constructor needs at least one "
            "coefficient; there is no system-wide epsilon/sigma to fall back "
            "on");
    table_.emplace(std::move(coeffs), cutoff);
    cutoff_ = table_->max_cutoff();
}

PairLJCut& PairLJCut::set_type_coeff(const LJTypeCoeff& row) & {
    if (!table_)
        throw std::logic_error(
            "cuda::PairLJCut::set_type_coeff: this PairLJCut was built with a "
            "single epsilon/sigma and has no type table. Construct it from a "
            "coefficient list instead.");
    table_->set_coeff(row);
    cutoff_ = table_->max_cutoff();
    return *this;
}

PairLJCut PairLJCut::set_type_coeff(const LJTypeCoeff& row) && {
    set_type_coeff(row);
    return std::move(*this);
}

PairLJCut& PairLJCut::set_tail_correction(bool on) & {
    tail_ = on;
    return *this;
}

PairLJCut PairLJCut::set_tail_correction(bool on) && {
    tail_ = on;
    return std::move(*this);
}

PairLJCut& PairLJCut::set_r4_attraction(std::vector<R4TypeCoeff> rows) & {
    if (!table_)
        throw std::logic_error(
            "cuda::PairLJCut::set_r4_attraction: this PairLJCut was built "
            "with a single epsilon/sigma and has no type table; the r^-4 term "
            "is per type pair. Construct it from a coefficient list instead.");
    table_->set_r4(std::move(rows));
    cutoff_ = table_->max_cutoff();
    return *this;
}

PairLJCut PairLJCut::set_r4_attraction(std::vector<R4TypeCoeff> rows) && {
    set_r4_attraction(std::move(rows));
    return std::move(*this);
}

PairLJCut& PairLJCut::set_special_bonds(double s12, double s13, double s14) & {
    for (double w : {s12, s13, s14})
        if (!std::isfinite(w))
            throw std::invalid_argument(
                "cuda::PairLJCut::set_special_bonds: weights must be finite");
    s12_ = s12;
    s13_ = s13;
    s14_ = s14;
    if (s12 == 1.0 && s13 == 1.0 && s14 == 1.0) {
        s_excl_ = 1.0;
        has_exclusions_ = false;
        return *this;
    }
    // The CSR holds every partner whose shell weight is not 1, so the baker
    // drops them all from the tiles (scale 0) even when a nearer shell
    // (s12 = 1) stays in the pair list; fractional weights come back through
    // the exception pass.
    s_excl_ = 0.0;
    has_exclusions_ = true;
    return *this;
}

PairLJCut PairLJCut::set_special_bonds(double s12, double s13, double s14) && {
    // Rvalue overload: delegate to the lvalue body, then move-return so
    // the chained `cuda::PairLJCut{…}.set_special_bonds(…)` expression
    // yields a movable rvalue (not a dangling lvalue reference to a
    // temporary). cf. `cpu::PairLJCut::set_special_bonds(…) &&`.
    this->set_special_bonds(s12, s13, s14);
    return std::move(*this);
}

void PairLJCut::enable_full_exclusions() {
    // cuda::Pair hook: full 1-2/1-3/1-4 exclusion is set_special_bonds(0,0,0).
    set_special_bonds(0.0, 0.0, 0.0);
}

PairLJCut& PairLJCut::set_compute_stress(bool on) & {
    compute_stress_ = on;
    return *this;
}

PairLJCut PairLJCut::set_compute_stress(bool on) && {
    compute_stress_ = on;
    return std::move(*this);
}

void PairLJCut::build(const BuildContext& ctx) {
    const int slots =
        ((ctx.frame.capacity() + kOpenmmClusterSize - 1) / kOpenmmClusterSize)
        * kOpenmmClusterSize;
    if (slots > spos_cap_) {
        cudaFree(spx_);
        cudaFree(spy_);
        cudaFree(spz_);
        cudaFree(spt_);
        spx_ = spy_ = spz_ = nullptr;
        spt_ = nullptr;
        ATV_CUDA_CHECK(cudaMalloc(&spx_, slots * sizeof(force_t)));
        ATV_CUDA_CHECK(cudaMalloc(&spy_, slots * sizeof(force_t)));
        ATV_CUDA_CHECK(cudaMalloc(&spz_, slots * sizeof(force_t)));
        if (table_)
            ATV_CUDA_CHECK(cudaMalloc(&spt_, slots * sizeof(int)));
        spos_cap_ = slots;
    }
    const auto fractional = [](double w) { return w != 0.0 && w != 1.0; };
    const bool any_fractional =
        fractional(s12_) || fractional(s13_) || fractional(s14_);
    if ((any_fractional || tail_) && ctx.frame.n_ghost() != 0)
        throw std::invalid_argument(
            "cuda::PairLJCut: fractional special-bond weights and the tail "
            "correction need a single GPU (no ghost rows)");
    std::vector<int> types;
    if (table_) {
        // LAMMPS init_one over every type the Frame rows carry (owned and,
        // on the multi-GPU path, ghost); a type no row names has no LJ site.
        auto& fr = ctx.frame;
        const int rows = std::max(fr.n_atoms(), fr.n_all());
        if (rows > 0 && !fr.atoms.type)
            throw std::invalid_argument(
                std::string("cuda::PairLJCut: a per-type table needs the "
                            "force-field type column; ")
                + atom_field_how_to("type"));
        types.resize(static_cast<std::size_t>(rows));
        if (rows > 0)
            ATV_CUDA_CHECK(cudaMemcpy(types.data(), fr.atoms.type,
                                      types.size() * sizeof(int),
                                      cudaMemcpyDeviceToHost));
        int max_type = -1;
        for (int t : types) {
            if (t < 0)
                throw std::invalid_argument(
                    "cuda::PairLJCut: atom types must be >= 0 for a per-type "
                    "table");
            max_type = std::max(max_type, t);
        }
        table_->build(max_type + 1, /*shift=*/true);
        n_types_ = table_->n_types();
        std::vector<kernels::LJTypedCoeff> host(table_->pairs().size());
        for (std::size_t k = 0; k < host.size(); ++k) {
            const LJPairCoeff& c = table_->pairs()[k];
            host[k] = kernels::LJTypedCoeff{
                force_t(c.lj1),  force_t(c.lj2),      force_t(c.lj3),
                force_t(c.lj4),  force_t(c.cut_ljsq), force_t(c.offset),
                force_t(c.r4_a), force_t(c.r4_cutsq), force_t(c.r4_rc5inv),
                force_t(c.r4_e0)};
        }
        cudaFree(d_table_);
        d_table_ = nullptr;
        ATV_CUDA_CHECK(
            cudaMalloc(&d_table_, host.size() * sizeof(kernels::LJTypedCoeff)));
        ATV_CUDA_CHECK(cudaMemcpy(d_table_, host.data(),
                                  host.size() * sizeof(kernels::LJTypedCoeff),
                                  cudaMemcpyHostToDevice));
    }
    // LAMMPS init_one tail over the global type counts of this build.
    tail_pref_ = LJTail{};
    if (tail_) {
        const int n = ctx.frame.n_atoms();
        if (table_) {
            std::vector<double> counts(static_cast<std::size_t>(n_types_), 0.0);
            for (int i = 0; i < n; ++i)
                counts[static_cast<std::size_t>(types[i])] += 1.0;
            tail_pref_ = table_->tail(counts);
        } else {
            tail_pref_ =
                lj_tail_pair(epsilon_, sigma_, cutoff_, double(n), double(n));
        }
    }
    cudaFree(d_exceptions_);
    d_exceptions_ = nullptr;
    n_exceptions_ = 0;
    // Re-callable: the multi-GPU bonded path rebuilds the exclusion CSR every
    // reneighbor (the flat topology's local indices change as ghosts move), so
    // free any prior CSR first to avoid leaking on repeated build().
    if (d_excl_offsets_) {
        cudaFree(d_excl_offsets_);
        d_excl_offsets_ = nullptr;
    }
    if (d_excl_partners_) {
        cudaFree(d_excl_partners_);
        d_excl_partners_ = nullptr;
    }
    n_excl_total_ = 0;
    ctx.runtime.md.exclusions_baked = false;
    if (!has_exclusions_) {
        // No special bonds: publish "nothing to bake".
        ctx.runtime.md.excl_offsets = nullptr;
        ctx.runtime.md.excl_partners = nullptr;
        ctx.runtime.md.s_excl = 1.0;
        return;
    }
    auto& f = ctx.frame;
    // n_atoms_ is the owned count on the single-GPU path and owned+ghost
    // after MultiGpu::rebuild_bonded_tuples. The CSR is indexed by the
    // local row the neighbor list stores in orig_index.
    const int n = std::max(f.n_atoms(), f.n_all());
    const int n_owned = f.n_local() > 0 ? f.n_local() : n;
    std::vector<std::set<int>> excl(static_cast<std::size_t>(n));
    std::vector<kernels::LJException> exceptions;

    // Frame::build_special_list is the bond-graph BFS to depth 3. It records
    // a 1-4 with no dihedral term. Partner values are tags; the force kernel
    // binary-searches local indices, so resolve tags through the current
    // tag column (owned rows and halo ghosts). Nearest shell wins, then
    // only scale-0 shells enter the CSR. s12 = 1 leaves the bond pair in
    // the list; s13 = s14 = 0 drops 1-3 and 1-4.
    const int ms = f.atoms.maxspecial;
    const bool have_special =
        f.atoms.nspecial != nullptr && f.atoms.special != nullptr
        && f.atoms.tag != nullptr && ms > 0 && n_owned > 0 && n > 0;
    if (have_special) {
        std::vector<int> ns(static_cast<std::size_t>(n_owned) * 3);
        std::vector<std::int64_t> sp(static_cast<std::size_t>(n_owned)
                                     * static_cast<std::size_t>(ms));
        std::vector<std::int64_t> tags(static_cast<std::size_t>(n));
        ATV_CUDA_CHECK(cudaMemcpy(ns.data(), f.atoms.nspecial,
                                  ns.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
        ATV_CUDA_CHECK(cudaMemcpy(sp.data(), f.atoms.special,
                                  sp.size() * sizeof(std::int64_t),
                                  cudaMemcpyDeviceToHost));
        ATV_CUDA_CHECK(cudaMemcpy(tags.data(), f.atoms.tag,
                                  tags.size() * sizeof(std::int64_t),
                                  cudaMemcpyDeviceToHost));
        std::unordered_map<std::int64_t, int> local_of;
        local_of.reserve(static_cast<std::size_t>(n) * 2);
        for (int row = 0; row < n; ++row)
            local_of.emplace(tags[static_cast<std::size_t>(row)], row);
        const double scale_of[3] = {s12_, s13_, s14_};
        const int nt = table_ ? n_types_ : 1;
        for (int i = 0; i < n_owned && i < n; ++i) {
            auto clamp = [ms](int c) {
                if (c < 0)
                    return 0;
                return c > ms ? ms : c;
            };
            int n12 = clamp(ns[static_cast<std::size_t>(i) * 3 + 0]);
            int n13 = clamp(ns[static_cast<std::size_t>(i) * 3 + 1]);
            int n14 = clamp(ns[static_cast<std::size_t>(i) * 3 + 2]);
            if (n13 < n12)
                n13 = n12;
            if (n14 < n13)
                n14 = n13;
            const int begin[3] = {0, n12, n13};
            const int end[3] = {n12, n13, n14};
            // local partner -> nearest shell (1, 2, or 3).
            std::map<int, int> rel;
            for (int shell = 0; shell < 3; ++shell) {
                for (int s = begin[shell]; s < end[shell]; ++s) {
                    const std::int64_t tj =
                        sp[static_cast<std::size_t>(i)
                               * static_cast<std::size_t>(ms)
                           + static_cast<std::size_t>(s)];
                    const auto found = local_of.find(tj);
                    if (found == local_of.end())
                        continue;
                    const int j = found->second;
                    if (j == i || j < 0 || j >= n)
                        continue;
                    rel.emplace(j, shell + 1);
                }
            }
            for (const auto& entry : rel) {
                const double w = scale_of[entry.second - 1];
                if (w == 1.0)
                    continue; // an ordinary pair: stays in the tiles
                excl[static_cast<std::size_t>(i)].insert(entry.first);
                if (w != 0.0 && i < entry.first)
                    exceptions.push_back(
                        {i, entry.first,
                         table_ ? types[i] * nt + types[entry.first] : 0,
                         force_t(w)});
            }
        }
    } else {
        // No special list on the device (build_special_list was not run).
        // Bonds, angle endpoints, and dihedral endpoints are the only
        // topology this frame uploaded. A 1-4 still requires a dihedral.
        int n_b = f.n_bonds();
        int n_a = f.n_angles();
        int n_d = f.n_dihedrals();
        std::vector<int> bi(n_b), bj(n_b);
        std::vector<int> ai(n_a), ak(n_a);
        std::vector<int> di(n_d), dl(n_d);
        if (n_b > 0) {
            ATV_CUDA_CHECK(cudaMemcpy(bi.data(), f.bonds.atomi,
                                      n_b * sizeof(int),
                                      cudaMemcpyDeviceToHost));
            ATV_CUDA_CHECK(cudaMemcpy(bj.data(), f.bonds.atomj,
                                      n_b * sizeof(int),
                                      cudaMemcpyDeviceToHost));
        }
        if (n_a > 0) {
            ATV_CUDA_CHECK(cudaMemcpy(ai.data(), f.angles.atomi,
                                      n_a * sizeof(int),
                                      cudaMemcpyDeviceToHost));
            ATV_CUDA_CHECK(cudaMemcpy(ak.data(), f.angles.atomk,
                                      n_a * sizeof(int),
                                      cudaMemcpyDeviceToHost));
        }
        if (n_d > 0) {
            ATV_CUDA_CHECK(cudaMemcpy(di.data(), f.dihedrals.atomi,
                                      n_d * sizeof(int),
                                      cudaMemcpyDeviceToHost));
            ATV_CUDA_CHECK(cudaMemcpy(dl.data(), f.dihedrals.atoml,
                                      n_d * sizeof(int),
                                      cudaMemcpyDeviceToHost));
        }
        std::vector<std::map<int, int>> rel(static_cast<std::size_t>(n));
        auto note = [&](int a, int b, int dist) {
            if (a < 0 || b < 0 || a >= n || b >= n || a == b)
                return;
            auto touch = [&](int i, int j) {
                auto it = rel[static_cast<std::size_t>(i)].find(j);
                if (it == rel[static_cast<std::size_t>(i)].end()
                    || dist < it->second)
                    rel[static_cast<std::size_t>(i)][j] = dist;
            };
            touch(a, b);
            touch(b, a);
        };
        for (int k = 0; k < n_b; ++k)
            note(bi[k], bj[k], 1);
        for (int k = 0; k < n_a; ++k)
            note(ai[k], ak[k], 2);
        for (int k = 0; k < n_d; ++k)
            note(di[k], dl[k], 3);
        const double scale_of[4] = {1.0, s12_, s13_, s14_};
        const int nt = table_ ? n_types_ : 1;
        for (int i = 0; i < n; ++i) {
            for (const auto& entry : rel[static_cast<std::size_t>(i)]) {
                const double w = scale_of[entry.second];
                if (w == 1.0)
                    continue;
                excl[static_cast<std::size_t>(i)].insert(entry.first);
                if (w != 0.0 && i < entry.first)
                    exceptions.push_back(
                        {i, entry.first,
                         table_ ? types[i] * nt + types[entry.first] : 0,
                         force_t(w)});
            }
        }
    }

    if (!exceptions.empty()) {
        if (!table_) {
            // Single-type form: one record serves every exception.
            const double s6 = std::pow(sigma_, 6.0);
            const kernels::LJTypedCoeff one{force_t(48.0 * epsilon_ * s6 * s6),
                                            force_t(24.0 * epsilon_ * s6),
                                            force_t(4.0 * epsilon_ * s6 * s6),
                                            force_t(4.0 * epsilon_ * s6),
                                            force_t(cutoff_ * cutoff_),
                                            force_t(v_shift_),
                                            force_t(0),
                                            force_t(0),
                                            force_t(0),
                                            force_t(0)};
            cudaFree(d_table_);
            d_table_ = nullptr;
            ATV_CUDA_CHECK(cudaMalloc(&d_table_, sizeof(one)));
            ATV_CUDA_CHECK(cudaMemcpy(d_table_, &one, sizeof(one),
                                      cudaMemcpyHostToDevice));
        }
        n_exceptions_ = static_cast<int>(exceptions.size());
        ATV_CUDA_CHECK(cudaMalloc(
            &d_exceptions_, exceptions.size() * sizeof(kernels::LJException)));
        ATV_CUDA_CHECK(
            cudaMemcpy(d_exceptions_, exceptions.data(),
                       exceptions.size() * sizeof(kernels::LJException),
                       cudaMemcpyHostToDevice));
    }

    // Pack into CSR.
    std::vector<int> offsets(n + 1, 0);
    for (int i = 0; i < n; ++i)
        offsets[i + 1] = offsets[i] + static_cast<int>(excl[i].size());
    n_excl_total_ = offsets[n];
    std::vector<int> partners(n_excl_total_);
    for (int i = 0; i < n; ++i) {
        int cursor = offsets[i];
        for (int p : excl[i])
            partners[cursor++] = p;
        // std::set already sorted ascending; nothing to do.
    }

    // Upload to device.
    cudaMalloc(&d_excl_offsets_, (n + 1) * sizeof(int));
    cudaMemcpy(d_excl_offsets_, offsets.data(), (n + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    if (n_excl_total_ > 0) {
        cudaMalloc(&d_excl_partners_, n_excl_total_ * sizeof(int));
        cudaMemcpy(d_excl_partners_, partners.data(),
                   n_excl_total_ * sizeof(int), cudaMemcpyHostToDevice);
    }
    // Publish the CSR so NeighborList::launch can bake exclusions into the
    // per-atom Verlet list (skipping the per-pair scan in the force kernel).
    ctx.runtime.md.excl_offsets = d_excl_offsets_;
    ctx.runtime.md.excl_partners = d_excl_partners_;
    ctx.runtime.md.s_excl = s_excl_;
}

void PairLJCut::destroy(const DestroyContext& /*ctx*/) {
    if (d_excl_offsets_) {
        cudaFree(d_excl_offsets_);
        d_excl_offsets_ = nullptr;
    }
    if (d_excl_partners_) {
        cudaFree(d_excl_partners_);
        d_excl_partners_ = nullptr;
    }
    n_excl_total_ = 0;
    if (spx_) {
        cudaFree(spx_);
        cudaFree(spy_);
        cudaFree(spz_);
        cudaFree(spt_);
        spx_ = spy_ = spz_ = nullptr;
        spt_ = nullptr;
        spos_cap_ = 0;
    }
    cudaFree(d_table_);
    d_table_ = nullptr;
    cudaFree(d_exceptions_);
    d_exceptions_ = nullptr;
    n_exceptions_ = 0;
}

void PairLJCut::compute(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const PairList* pairs = ctx.runtime.md.pair_list;
    if (!pairs)
        return;

    const bool baked = ctx.runtime.md.exclusions_baked;
    const int* excl_off =
        (has_exclusions_ && !baked) ? d_excl_offsets_ : nullptr;
    const int* excl_par =
        (has_exclusions_ && !baked) ? d_excl_partners_ : nullptr;
    const force_t cutoff_sq =
        static_cast<force_t>(cutoff_) * static_cast<force_t>(cutoff_);
    const force_t sigma_sq =
        static_cast<force_t>(sigma_) * static_cast<force_t>(sigma_);
    const force_t min_pair_dist_sq =
        static_cast<force_t>(kernels::kMinPairDist)
        * static_cast<force_t>(kernels::kMinPairDist);
    cuda::Box box = ctx.runtime.host_box;
    // Per-step LAMMPS eflag/vflag; set_compute_stress(false) vetoes the virial.
    const bool energy = ctx.energy_requested;
    const bool virial = compute_stress_ && ctx.stress_requested;
    acc_t* stress_ptr = virial ? &frame.scalars->stress[0] : nullptr;

    // OpenMM cluster force — sole CUDA MD nonbonded path (cuda-openmm-md).
    if (pairs->cl_row_ptr == nullptr || pairs->cl_col_idx == nullptr
        || pairs->cl_orig_index == nullptr || pairs->num_clusters <= 0)
        return;

    constexpr int CS = kOpenmmClusterSize;
    const int nc = pairs->num_clusters;
    const int n_slots = nc * CS;
    // Live positions from frame (not rebuild-time cluster_x). Prefer published
    // cl_x only as a gather target cache when present is wrong — always gather
    // current frame.atoms into scratch so force is skin-correct.
    if (n_slots > spos_cap_) {
        throw std::runtime_error(
            "cuda::PairLJCut: cluster slots exceed Frame capacity at build");
    }
    {
        int block = kDefaultBlock;
        int grid = grid_size(n_slots);
        kernels::gather_cluster_pos<<<grid, block, 0, stream>>>(
            pairs->cl_orig_index, frame.atoms.x, frame.atoms.y, frame.atoms.z,
            frame.atoms.type, n_slots, spx_, spy_, spz_,
            table_ ? spt_ : nullptr);
    }

    const uint64_t* mask = pairs->cl_mask;
    int block = kDefaultBlock;
    int grid = grid_size(n_slots);
    constexpr int shmem = 0;

    const auto* d_table = static_cast<const kernels::LJTypedCoeff*>(d_table_);
    auto launch = [&](auto ortho_t, auto energy_t, auto vir_t) {
        constexpr bool O = decltype(ortho_t)::value;
        constexpr bool E = decltype(energy_t)::value;
        constexpr bool V = decltype(vir_t)::value;
        if (table_)
            kernels::lj_force_tile<O, E, V, true>
                <<<grid, block, shmem, stream>>>(
                    frame.atoms.fx, frame.atoms.fy, frame.atoms.fz,
                    frame.force_fault,
                    energy_term(frame.scalars, EnergyKind::Pair, E), stress_ptr,
                    spx_, spy_, spz_, pairs->cl_row_ptr, pairs->cl_col_idx,
                    mask, pairs->cl_orig_index, nc, frame.n_local(), box,
                    /*periodic=*/true, force_t(0), force_t(0), cutoff_sq,
                    min_pair_dist_sq, force_t(0), excl_off, excl_par,
                    static_cast<force_t>(s_excl_), spt_, d_table, n_types_);
        else
            kernels::lj_force_tile<O, E, V, false>
                <<<grid, block, shmem, stream>>>(
                    frame.atoms.fx, frame.atoms.fy, frame.atoms.fz,
                    frame.force_fault,
                    energy_term(frame.scalars, EnergyKind::Pair, E), stress_ptr,
                    spx_, spy_, spz_, pairs->cl_row_ptr, pairs->cl_col_idx,
                    mask, pairs->cl_orig_index, nc, frame.n_local(), box,
                    /*periodic=*/true, static_cast<force_t>(epsilon_), sigma_sq,
                    cutoff_sq, min_pair_dist_sq, static_cast<force_t>(v_shift_),
                    excl_off, excl_par, static_cast<force_t>(s_excl_), nullptr,
                    nullptr, 0);
    };
    auto by_flags = [&](auto ortho_t) {
        using T = std::true_type;
        using F = std::false_type;
        if (energy && virial)
            launch(ortho_t, T{}, T{});
        else if (energy)
            launch(ortho_t, T{}, F{});
        else if (virial)
            launch(ortho_t, F{}, T{});
        else
            launch(ortho_t, F{}, F{});
    };
    if (box.ortho)
        by_flags(std::true_type{});
    else
        by_flags(std::false_type{});

    if (n_exceptions_ > 0) {
        const auto* pairs_d =
            static_cast<const kernels::LJException*>(d_exceptions_);
        auto exceptions = [&](auto energy_t, auto vir_t) {
            constexpr bool E = decltype(energy_t)::value;
            constexpr bool V = decltype(vir_t)::value;
            kernels::lj_exceptions<E, V>
                <<<grid_size(n_exceptions_), kDefaultBlock, 0, stream>>>(
                    pairs_d, n_exceptions_, frame.atoms.x, frame.atoms.y,
                    frame.atoms.z, d_table, box, frame.atoms.fx, frame.atoms.fy,
                    frame.atoms.fz, frame.force_fault,
                    energy_term(frame.scalars, EnergyKind::Pair, E),
                    stress_ptr);
        };
        using T = std::true_type;
        using F = std::false_type;
        if (energy && virial)
            exceptions(T{}, T{});
        else if (energy)
            exceptions(T{}, F{});
        else if (virial)
            exceptions(F{}, T{});
        else
            exceptions(F{}, F{});
    }
    if (tail_)
        internal::launch_lj_tail(
            box, tail_pref_,
            energy_term(frame.scalars, EnergyKind::Pair, energy), stress_ptr,
            stream);
    ATV_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace ATV

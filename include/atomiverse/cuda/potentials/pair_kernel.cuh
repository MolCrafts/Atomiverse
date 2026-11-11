/// @file cuda/potentials/pair_kernel.cuh
/// @brief Pair terms of the CUDA cluster-tile engine: the per-pair math of
///        `core/pair_kernel.h` plus the per-slot parameters a tile carries.
///
/// A term is what one pair style hands the engine
/// (`src/cuda/potentials/internal/cluster_tile.cuh`):
/// - `Param`: the static per-atom parameters of one cluster slot (charge,
///   type, Lorentz–Berthelot halves), gathered into slot order at every
///   neighbor rebuild and exchanged between lanes by `shfl`;
/// - `param(row)`: the parameters of one frame row;
/// - `pair(r2, a, b)`: the `core/pair_kernel.h` contribution — `fpair` with
///   force on i `= -fpair·(r_j - r_i)` and the pair energy — of one pair the
///   engine already holds inside its outer cutoff (`0 < r2 < cut2`); a term
///   masks only its own narrower cutoffs.
///
/// Untyped LJ and real-space Coulomb are `potentials::LJKernel` and
/// `potentials::CoulKernel` themselves; the typed table and the per-atom
/// Lorentz–Berthelot form add only their parameter lookup. `Sum` composes two
/// terms into one tile walk (LAMMPS `lj/cut/coul/long`).
///
/// nvcc only (warp shuffles).
#pragma once

#include <cstddef>

#include "atomiverse/core/lj_type_table.h"
#include "atomiverse/core/pair_kernel.h"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV::cuda::tile {

using Contribution = potentials::PairContribution<force_t>;

inline constexpr unsigned kFullWarp = 0xffffffffu;

/// One resolved (type_i, type_j) record of `LJTypeTable` in force_t
/// (LAMMPS lj1..lj4, cut_ljsq, offset; the ATV r^-4 constants).
struct TypedPairCoeff {
    force_t lj1, lj2, lj3, lj4, cut_ljsq, offset;
    force_t r4_a, r4_cutsq, r4_rc5inv, r4_e0;
};

/// @brief Typed LJ + r⁻⁴ of one record, scaled by `scale`: LAMMPS
///        forcelj·r2inv and evdwl − offset inside cut_ljsq, the r⁻⁴ term
///        inside its own cutoff. Shared with the special-bond passes.
__device__ __forceinline__ Contribution typed_lj(const TypedPairCoeff* c,
                                                 force_t r2, force_t scale) {
    Contribution out{0, 0};
    // rsqrt, not a division: shared with the Coulomb term of a Sum.
    const force_t inv = rsqrt(r2), inv2 = inv * inv;
    if (r2 < __ldg(&c->cut_ljsq)) {
        const force_t r6inv = inv2 * inv2 * inv2;
        out.fpair =
            scale * inv2 * r6inv * (__ldg(&c->lj1) * r6inv - __ldg(&c->lj2));
        out.energy = scale
                     * (r6inv * (__ldg(&c->lj3) * r6inv - __ldg(&c->lj4))
                        - __ldg(&c->offset));
    }
    if (r2 < __ldg(&c->r4_cutsq)) {
        const R4Term<force_t> t = r4_attraction<force_t>(
            r2, __ldg(&c->r4_a), __ldg(&c->r4_rc5inv), __ldg(&c->r4_e0));
        out.fpair += scale * t.fpair;
        out.energy += scale * t.energy;
    }
    return out;
}

/// @brief Single-type shifted LJ (`potentials::LJKernel`); the engine's
///        outer cutoff is the LJ cutoff, so the unmasked form is exact.
struct LJ {
    potentials::LJKernel kernel;
    struct Param {};
    __device__ Param param(int) const { return {}; }
    __device__ static Param shfl(Param p, int) { return p; }
    __device__ Contribution pair(force_t r2, Param, Param) const {
        return kernel.raw(r2, force_t(0), force_t(0));
    }
};

/// @brief Per-type LJ table (+ r⁻⁴), each pair inside its own cut_ljsq.
struct TypedLJ {
    const TypedPairCoeff* table = nullptr;
    int n_types = 0;
    const int* type = nullptr; ///< Frame type column (rebuild gather).
    struct Param {
        int type;
    };
    __device__ Param param(int row) const { return {type[row]}; }
    __device__ static Param shfl(Param p, int lane) {
        return {__shfl_sync(kFullWarp, p.type, lane)};
    }
    __device__ Contribution pair(force_t r2, Param a, Param b) const {
        return typed_lj(table + static_cast<std::size_t>(a.type) * n_types
                            + b.type,
                        r2, force_t(1));
    }
};

/// @brief Lorentz–Berthelot LJ from per-atom halves at one cutoff (the
///        OpenMM-benchmarked parameter fusion): σ_ij = σ_i/2 + σ_j/2,
///        4ε_ij = (2√ε_i)(2√ε_j), optionally shifted at the cutoff.
struct LorentzBerthelotLJ {
    const force_t* sigma_half = nullptr;   ///< Per frame row (rebuild gather).
    const force_t* epsilon_root = nullptr; ///< Per frame row: 2√ε.
    force_t inv_rc2 = 0;                   ///< 1/r_c² of the shift.
    bool shift = false;
    struct Param {
        force_t sigma_half, epsilon_root;
    };
    __device__ Param param(int row) const {
        return {sigma_half[row], epsilon_root[row]};
    }
    __device__ static Param shfl(Param p, int lane) {
        return {__shfl_sync(kFullWarp, p.sigma_half, lane),
                __shfl_sync(kFullWarp, p.epsilon_root, lane)};
    }
    __device__ Contribution pair(force_t r2, Param a, Param b) const {
        const force_t sig = a.sigma_half + b.sigma_half;
        const force_t eps = a.epsilon_root * b.epsilon_root;
        const force_t inv = rsqrt(r2), inv2 = inv * inv;
        const force_t sr2 = sig * sig * inv2, sr6 = sr2 * sr2 * sr2;
        Contribution out{eps * (12 * sr6 - 6) * sr6 * inv2,
                         eps * sr6 * (sr6 - 1)};
        if (shift) { // LAMMPS offset at the common cutoff
            const force_t sc2 = sig * sig * inv_rc2, sc6 = sc2 * sc2 * sc2;
            out.energy -= eps * sc6 * (sc6 - 1);
        }
        return out;
    }
};

/// @brief Real-space Ewald Coulomb (`potentials::CoulKernel`), masked at
///        its own cutoff when that is inside the engine's outer cutoff.
struct Coulomb {
    potentials::CoulKernel kernel;
    const force_t* charge = nullptr; ///< Frame charge column.
    bool masked = false;             ///< Coulomb cutoff < outer cutoff.
    struct Param {
        force_t q;
    };
    __device__ Param param(int row) const { return {charge[row]}; }
    __device__ static Param shfl(Param p, int lane) {
        return {__shfl_sync(kFullWarp, p.q, lane)};
    }
    __device__ Contribution pair(force_t r2, Param a, Param b) const {
        if (masked && !(r2 < force_t(kernel.cutoff_sq)))
            return {0, 0};
        return kernel.raw(r2, a.q, b.q);
    }
};

/// @brief Two terms in one tile walk (`potentials::SumKernel` for tiles).
template <class A, class B>
struct Sum {
    A a;
    B b;
    struct Param {
        typename A::Param a;
        typename B::Param b;
    };
    __device__ Param param(int row) const {
        return {a.param(row), b.param(row)};
    }
    __device__ static Param shfl(Param p, int lane) {
        return {A::shfl(p.a, lane), B::shfl(p.b, lane)};
    }
    __device__ Contribution pair(force_t r2, Param i, Param j) const {
        const Contribution ca = a.pair(r2, i.a, j.a);
        const Contribution cb = b.pair(r2, i.b, j.b);
        return {ca.fpair + cb.fpair, ca.energy + cb.energy};
    }
};

} // namespace ATV::cuda::tile

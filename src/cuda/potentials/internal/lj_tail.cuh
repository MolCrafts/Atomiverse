/// @file cuda/potentials/internal/lj_tail.cuh
/// @brief Private LJ tail-correction launch shared by `cuda::PairLJCut` and
///        `cuda::PairLJCutCoulLong` (LAMMPS `pair_modify tail yes`).
#pragma once

#include <cuda_runtime_api.h>

#include "atomiverse/core/lj_type_table.h"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/energy_term.cuh"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV::cuda::internal {

/// @brief Enqueue E += tail.energy / V and, on each diagonal virial
///        component, += tail.virial / V, with V read on the device from the
///        resident cell (`Box::current()`), so a changing cell (NPT, deform)
///        needs no host read. One thread; graph-capturable, no allocation.
/// @param box By-value box whose `resident` view is the live cell.
/// @param tail Prefactors from `ATV::LJTypeTable::tail` /
///        `ATV::lj_tail_pair` [eV·Å³].
/// @param energy Pair energy sink (total + `energy_pair`) [eV]; empty
///        skips the energy.
/// @param stress Six-component virial accumulator [eV]; null skips it.
/// @param stream Stream to enqueue on.
void launch_lj_tail(const Box& box, const LJTail& tail, EnergyTerm energy,
                    acc_t* stress, cudaStream_t stream);

} // namespace ATV::cuda::internal

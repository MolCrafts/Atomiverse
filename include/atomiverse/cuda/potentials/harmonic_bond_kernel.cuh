/// @file cuda/potentials/harmonic_bond_kernel.cuh
/// @brief Pointer-level harmonic bond kernel -- shared pointer-level kernel for the cuda component.
///
/// V(r) = K (r - r0)^2,  F = -2K(r - r0) r_hat
///
/// Single source of truth for harmonic bond force+energy computation.
#pragma once

#include "atomiverse/cuda/frame/box.cuh"

namespace ATV {
namespace cuda {
namespace kernels {

/// @brief Harmonic bond stretching kernel. One thread per bond.
///
/// Forces accumulated via atomicAdd (Newton's 3rd law).
/// Energy reduced per-block using shared memory, then atomicAdd to global scalar.
/// Uses the simulation cell's minimum image convention.
///
/// @param x,y,z           Position arrays [n_atoms, device].
/// @param fx,fy,fz        Force arrays [n_atoms, device]. Accumulated via atomicAdd.
/// @param energy          Potential energy scalar [1, device]. Accumulated via atomicAdd.
/// @param bi,bj,bt        Bond atom indices and type [n_bonds, device].
/// @param K               Spring constants per bond type [n_types, device].
/// @param r0              Equilibrium distances per bond type [n_types, device].
/// @param box             Simulation cell passed by value.
/// @param n_bonds         Number of bonds.
__global__ void harmonic_bond_k(
    const cuda::real_t* __restrict__ x, const cuda::real_t* __restrict__ y,
    const cuda::real_t* __restrict__ z, cuda::real_t* __restrict__ fx,
    cuda::real_t* __restrict__ fy, cuda::real_t* __restrict__ fz,
    double* __restrict__ energy, double* __restrict__ stress,
    const int* __restrict__ bi, const int* __restrict__ bj,
    const int* __restrict__ bt, const cuda::work_t* __restrict__ K,
    const cuda::work_t* __restrict__ r0, cuda::Box box, int n_bonds);

} // namespace kernels
} // namespace cuda
} // namespace ATV

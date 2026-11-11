/// @file cuda/semiempirical/internal/nddo_hcore.cuh
/// @brief Private native CHNO NDDO one-electron Hamiltonian assembly.
#pragma once

#include "atomiverse/cuda/semiempirical/internal/nddo_fock.cuh"

namespace ATV::cuda::semiempirical::internal {

/// @brief Overwrite dense column-major Hcore and core repulsion [eV].
/// Includes one-center levels, analytic 1s/2s/2p STO resonance, electron-core
/// attraction, and AM1/RM1 Gaussian core corrections. Coordinates are device
/// FP64 Angstrom. All arrays/scalars use separate caller-owned allocations.
/// Owner must validate CHNO identity, s/p AO partition and parameter values at
/// build. Uncovered identities and orbital counts are also reported by the
/// device status. No allocation, pair-cache storage or host scientific transfer.
/// status is overwritten: 0=success, 1=invalid geometry/result, 2=unsupported
/// identity. Owner checks status before publishing results, including n_atoms=1.
void hcore(int n_atoms, int n_orbitals, const NDDOAtom* atoms, const double* x,
           const double* y, const double* z, double* matrix,
           double* core_repulsion, int* status, cudaStream_t stream);

} // namespace ATV::cuda::semiempirical::internal

/// @file cuda/semiempirical/internal/nddo_two_electron.cuh
/// @brief Native CUDA s/p two-center ERI and direct Fock contraction.
#pragma once

#include "atomiverse/cuda/semiempirical/internal/nddo_fock.cuh"

namespace ATV::cuda::semiempirical::internal {

/// @brief Evaluate all distinct atom-pair ERI tiles [eV] on the device.
/// Pair order is (1,0), (2,0), (2,1), ...; each tile occupies 100 doubles,
/// row-major in local packed AO pair order ss, xs, xx, ys, yx, yy, zs, zx,
/// zy, zz. Unused entries for hydrogen are zero. This diagnostic primitive
/// shares the integral kernel with two_center_fock; SCF need not store tiles.
/// Coordinates are device FP64 Angstrom, atom metadata uses the unique
/// AM1/RM1 tables. Owner validates 1/4 orbitals and valid parameter values.
/// status is a separate device int, overwritten with 0 or 1 (invalid pair
/// distance/non-finite integral). Owner must check it before publishing results.
void two_center_integrals(int n_atoms, const NDDOAtom* atoms, const double* x,
                          const double* y, const double* z, double* tiles,
                          int* status, cudaStream_t stream);

/// @brief Accumulate two-center J[P]-K[Pspin] into an existing dense Fock.
/// All matrices are symmetric FP64, column-major. Fock, densities and status
/// must be separate nonoverlapping allocations. Atom AO ranges partition
/// [0,n_orbitals). Pair integrals are contracted immediately from a shared
/// rotation and radial terms; there is no ERI tile storage, runtime allocation
/// or host transfer.
/// Pspin is multiplied by spin_scale; P and 0.5 avoid a closed-shell copy.
/// Adds symmetric entries using FP64 atomics; the accumulation order may vary.
/// status is overwritten as above; no success result may be published on error.
void two_center_fock(int n_atoms, int n_orbitals, const NDDOAtom* atoms,
                     const double* x, const double* y, const double* z,
                     const double* density, const double* spin_density,
                     double* fock, int* status, cudaStream_t stream,
                     double spin_scale = 1.0);

} // namespace ATV::cuda::semiempirical::internal

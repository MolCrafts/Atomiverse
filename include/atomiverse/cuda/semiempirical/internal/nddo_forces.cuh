/// @file cuda/semiempirical/internal/nddo_forces.cuh
/// @brief Fixed-density native CUDA CHNO NDDO analytic pair-force primitive.
#pragma once
#include "atomiverse/cuda/semiempirical/internal/nddo_fock.cuh"
namespace ATV::cuda::semiempirical::internal {
/// Overwrite interleaved forces[3*n_atoms] in eV/Angstrom from nuclear,
/// resonance, electron-core and two-center ERI derivatives. Density is symmetric
/// column-major FP64 total P; restricted exchange uses Pspin=P/2. One-center
/// integrals are coordinate independent. No SCF or density response is computed.
/// Atom descriptors/AO partition/parameters must be validated by the owner.
/// All arrays are separate caller-owned device storage, valid on stream; there
/// is no allocation, host transfer or global pair/derivative cache.
/// status is overwritten: 0=success, 1=nonfinite/invalid geometry or result,
/// 2=unsupported identity/AO bounds. Forces are scratch and must not be
/// published on nonzero status. Repeated calls reset both output and status.
void pair_forces(int n_atoms, int n_orbitals, const NDDOAtom* atoms,
                 const double* x, const double* y, const double* z,
                 const double* density, double* forces, int* status,
                 cudaStream_t stream);
} // namespace ATV::cuda::semiempirical::internal

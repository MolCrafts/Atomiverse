/// @file cuda/semiempirical/internal/nddo_fock.cuh
/// @brief Private s/p NDDO device matrix primitives; no runtime or allocation.
#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace ATV::cuda::semiempirical::internal {

// The owner validates descriptors before upload. Except for the documented
// Hcore/Fock alias, input/output storage must not overlap. Operations enqueue
// on the supplied stream; the owner checks asynchronous completion before
// publishing a successful physical result.

/// @brief Numeric atom descriptor uploaded once when the method builds.
/// Values come from the unique AM1/RM1 parameter tables. No host pointers or
/// parameter lookup occurs in a device kernel. Orbital order is s, px, py, pz.
struct NDDOAtom {
    int first_ao = 0;
    int n_orbitals = 0; // Supported counts: 1 or 4; validated by the owner.
    double core_charge = 0.0;
    double uss = 0.0;
    double upp = 0.0;
    double gss = 0.0;
    double gsp = 0.0;
    double gpp = 0.0;
    double gp2 = 0.0;
    double hsp = 0.0;
    double dd = 0.0;
    double qq = 0.0;
    double am = 0.0;
    double ad = 0.0;
    double aq = 0.0;
    int atomic_number = 0;
    double zs = 0.0;
    double zp = 0.0;
    double betas = 0.0;
    double betap = 0.0;
    double alp = 0.0;
    double guess1[4]{};
    double guess2[4]{};
    double guess3[4]{};
};

/// @brief Overwrite a dense column-major Hcore with its one-center terms [eV].
/// Atom descriptors must partition [0,n_orbitals) without overlap or gaps.
void one_center_hcore(int n_atoms, int n_orbitals, const NDDOAtom* atoms,
                      double* hcore, cudaStream_t stream);

/// @brief Overwrite F=Hcore+J[P]-K[Pspin] with the one-center NDDO terms.
/// All matrices are device FP64, square column-major. P is the total density;
/// Pspin is multiplied by spin_scale (use P and 0.5 for closed shell).
/// F may alias Hcore; neither may alias a density. Caller owns all storage.
void one_center_fock(int n_atoms, int n_orbitals, const NDDOAtom* atoms,
                     const double* hcore, const double* density,
                     const double* spin_density, double* fock,
                     cudaStream_t stream, double spin_scale = 1.0);

/// @brief Overwrite the atom charges with Zcore minus the diagonal population.
void atomic_charges(int n_atoms, int n_orbitals, const NDDOAtom* atoms,
                    const double* density, double* charges,
                    cudaStream_t stream);

/// @brief Required reusable scratch bytes for electronic_energy (no allocation).
std::size_t electronic_energy_workspace_bytes(int n_orbitals);

/// @brief E=0.5 sum(P*(Hcore+F)) [eV] into one device scalar.
/// Uses a CUB transform/reduction: no materialized Hcore+F or host result.
/// Scratch is allocated by the method during build and reused across steps.
/// Output and scratch do not overlap inputs or each other; scratch is not
/// shared by concurrent streams.
void electronic_energy(int n_orbitals, const double* density,
                       const double* hcore, const double* fock, double* energy,
                       void* scratch, std::size_t scratch_bytes,
                       cudaStream_t stream);

} // namespace ATV::cuda::semiempirical::internal

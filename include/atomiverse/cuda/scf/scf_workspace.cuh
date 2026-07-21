/// @file cuda/scf/scf_workspace.cuh
/// @brief cuda::scf::SCFWorkspace -- device-resident SCF electronic-structure
///        scratch.
///
/// Method-specific computed scratch (Fock/ERI/eigenvector buffers, basis
/// data) for the CUDA SCF self-consistency cycle. Per the CLAUDE.md Frame
/// purity rule ("hot-path data structures carry only fundamental physical
/// quantities -- never precomputed indices, caches, or derived maps") this
/// is **not** part of `cuda::Frame`: it is owned by `cuda::scf::SCFSolver`,
/// mirroring how the CPU SCF workspace lives in `cpu::scf::SCFSolver`.
///
/// Allocated by the SCF `internal::*` step helpers in their `build()` and
/// freed in their `destroy()`; `SCFSolver` holds the single instance and
/// passes it by reference into every helper call.
///
/// @see cuda::scf::SCFSolver for the inner solver that owns this workspace
#pragma once

#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {
namespace scf {

/// @brief SCF electronic-structure workspace -- all arrays on device.
///
/// Column-major dense matrices (cuBLAS convention), dimension n x n.
/// Owned by `cuda::scf::SCFSolver`, not by `cuda::Frame`.
struct SCFWorkspace {
    // -- Shared (toy-scf + RHF) --
    prec_t* H0 = nullptr; ///< Base Hamiltonian (toy) [device, n*n].
    prec_t* H = nullptr; ///< Effective Hamiltonian / Fock matrix [device, n*n].
    prec_t* P = nullptr; ///< Density matrix [device, n*n].
    prec_t* P_old = nullptr; ///< Previous density matrix [device, n*n].
    prec_t* C =
        nullptr; ///< Eigenvectors / MO coefficients [device, n*n, col-major].
    prec_t* C_prime =
        nullptr; ///< Previous-iteration orthonormal-basis eigenvectors C'
                 ///< [device, n*n, col-major]. Snapshotted after each
                 ///< diagonalize; the occupied block builds the level-shift
                 ///< projector P'_occ = C'_occ C'_occ^T (hf-robust-03).
    prec_t* eig = nullptr;  ///< Eigenvalues / orbital energies [device, n].
    prec_t* work = nullptr; ///< Scratch buffer [device, n*n].
    int n = 0;              ///< Matrix dimension (host-cached).
    int n_occ = 0;          ///< Occupation count (host-cached).
    bool converged = false; ///< Convergence flag (host-cached).
    int iteration = 0;      ///< Iteration that converged (host-cached).
    double residual = 0;    ///< Last convergence residual (host-cached).

    // -- RHF-specific --
    prec_t* Hcore = nullptr; ///< Core Hamiltonian T+V [device, n*n].
    prec_t* X = nullptr; ///< Orthogonalization matrix S^{-1/2} [device, n*n].
    prec_t* F_prime = nullptr; ///< Transformed Fock X^T*F*X [device, n*n].
    prec_t* ERI = nullptr;     ///< Two-electron integrals [device, n^4].
    double V_nuc = 0;          ///< Nuclear repulsion energy (host-cached).

    // -- Basis data for GPU-native integral kernels --
    prec_t* d_shell_alpha = nullptr; ///< Exponents [device, n_shells * n_prim].
    prec_t* d_shell_coeff =
        nullptr; ///< Normalized coefficients [device, n_shells * n_prim].
    prec_t* d_shell_center =
        nullptr;             ///< Shell centers [device, n_shells * 3] (Bohr).
    int* d_atom_Z = nullptr; ///< Nuclear charges [device, natm].
    prec_t* d_atom_pos = nullptr; ///< Atom positions [device, natm * 3] (Bohr).
    int n_prim = 0;               ///< Primitives per shell (host-cached).
    int n_shells = 0;             ///< Number of shells (host-cached).
    int natm = 0;                 ///< Number of atoms (host-cached).
};

} // namespace scf
} // namespace cuda
} // namespace ATV

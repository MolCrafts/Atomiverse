/// @file torch/frame/frame.cuh
/// @brief torch::Frame -- differentiable system state backed by at::Tensor.
///
/// One backend, one Frame type: this is `torch::Frame`, not a shared layout
/// with CPU or CUDA. Tensors may have requires_grad=true for autograd.
///
/// Data is organized into sub-structs (atoms, bonds, angles, dihedrals)
/// matching the molcrafts naming convention. The SEMANTIC field set
/// mirrors `cpu::Frame` and `cuda::Frame`; the physical representation is
/// independently `at::Tensor` (CPU is blocks+meta+simbox; CUDA is device
/// pointers). Representation work toward a blocks map on torch is deferred.
/// Per the CLAUDE.md Frame purity rule the frame carries only fundamental
/// physical quantities: method-specific scratch (the neighbor pair list) is
/// owned by the component that computes it -- `torch::NeighborList` -- not by
/// the frame.
///
/// @see cpu::Frame for the host-side counterpart
/// @see cuda::Frame for the raw-pointer CUDA counterpart
/// @see core/system_scalars.h for the shared system-scalar field set
/// @see torch::NeighborList for the relocated neighbor pair list
/// @see torch::to_torch() / torch::to_cuda() for transfer
#pragma once

#ifdef ATV_HAS_TORCH

#include <torch/torch.h>

#include "atomiverse/core/system_scalars.h"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace torch {

/// @brief LibTorch type aliases to resolve ATV::torch vs ::torch collision.
/// @{
using Tensor = ::torch::Tensor;
using ScalarType = ::at::ScalarType;
using Device = ::torch::Device;
/// @}

/// @brief Compile-time mapping: cuda::real_t -> Torch ScalarType (per-atom storage).
inline constexpr ScalarType kRealDtype = std::is_same_v<cuda::real_t, float>
                                             ? ScalarType::Float
                                             : ScalarType::Double;

/// @brief Compile-time mapping: cuda::prec_t -> Torch ScalarType (accumulators, energy).
inline constexpr ScalarType kPrecDtype = std::is_same_v<cuda::prec_t, float>
                                             ? ScalarType::Float
                                             : ScalarType::Double;

/// @brief Compile-time mapping: int -> Torch ScalarType.
inline constexpr ScalarType kIntDtype = ScalarType::Int;

/// @brief Compile-time mapping: std::int64_t -> Torch ScalarType (atom tag).
inline constexpr ScalarType kLongDtype = ScalarType::Long;

/// @brief Compile-time mapping: std::int8_t -> Torch ScalarType (PBC image).
inline constexpr ScalarType kImageDtype = ScalarType::Char;

/// @brief Differentiable system state backed by at::Tensor.
///
/// Each field is a CUDA Tensor in SoA layout. Positions and parameters
/// may carry requires_grad=true for autograd; topology and indices do not.
///
/// Usage:
/// @code{.cpp}
/// auto tframe = ATV::torch::to_torch(cuda_frame);
/// auto energy = ATV::torch::lj(tframe, epsilon, sigma, cutoff);
/// energy.backward();
/// @endcode
struct Frame {
    // ---- Per-atom block (molcrafts "atoms" namespace) ----

    /// @brief Per-atom tensors matching molpy atoms namespace.
    struct Atoms {
        Tensor x, y, z;    ///< Positions [Angstrom].
        Tensor vx, vy, vz; ///< Velocities [Angstrom/fs].
        Tensor fx, fy, fz; ///< Forces [eV/Angstrom].
        Tensor mass;       ///< Masses [amu].

        /// @brief Per-atom partial charge [e] (elementary charge).
        ///
        /// Treated as a fundamental physical property (analogue of LAMMPS
        /// atom->q). Same semantic field as CPU `kAtom_charge`. kRealDtype.
        Tensor charge;

        Tensor type; ///< Atom type indices (kIntDtype, no grad).

        /// @brief Global atom identity [n_atoms], persistent across MPI
        /// migration and restart. Same semantic field as CPU `kAtom_tag`.
        /// kLongDtype (std::int64_t), no grad.
        Tensor tag;

        /// @brief Per-atom PBC crossing counts, shape [n_atoms, 3].
        ///
        /// `image[i][d]` increments by +-1 when atom i crosses the box face
        /// along dimension d. Used to reconstruct unwrapped trajectories.
        /// Same semantic field as CPU `kAtom_image`. kImageDtype (std::int8_t),
        /// no grad.
        Tensor image;
    } atoms;

    // ---- Bond topology block (molcrafts "bonds" namespace) ----

    /// @brief Bond topology matching molpy bonds namespace.
    struct Bonds {
        Tensor atomi, atomj; ///< Bond atom indices [n_bonds], kIntDtype.
        Tensor type;         ///< Bond types [n_bonds], kIntDtype.
    } bonds;

    // ---- Angle topology block (molcrafts "angles" namespace) ----

    /// @brief Angle topology matching molpy angles namespace.
    struct Angles {
        Tensor atomi, atomj,
            atomk;   ///< Angle atom indices [n_angles], j is center.
        Tensor type; ///< Angle types [n_angles], kIntDtype.
    } angles;

    // ---- Dihedral topology block (molcrafts "dihedrals" namespace) ----

    /// @brief Dihedral topology matching molpy dihedrals namespace.
    struct Dihedrals {
        Tensor atomi, atomj, atomk,
            atoml;   ///< Dihedral atom indices [n_dihedrals].
        Tensor type; ///< Dihedral types [n_dihedrals], kIntDtype.
    } dihedrals;

    // ---- System scalars ----
    //
    // This Tensor set mirrors `core/system_scalars.h`'s `SystemScalars`
    // field set (energy, kinetic, conserved, stress, stress_kinetic) in the
    // torch device representation: each POD field of `SystemScalars` becomes
    // a CUDA Tensor here, exactly as `cpu::Frame` embeds the value inline
    // and `cuda::Frame` carries a device-resident `SystemScalars*`. See that
    // header for the field-by-field documentation and units.

    Tensor energy;         ///< Potential energy [1], [eV].
    Tensor kinetic;        ///< Kinetic energy [1], [eV].
    Tensor conserved;      ///< Integrator conserved quantity [1], [eV].
    Tensor stress;         ///< Virial stress [6], Voigt [eV].
    Tensor stress_kinetic; ///< Kinetic stress [6], Voigt [eV].
    Tensor box;            ///< Simulation cell [9], row-major [Angstrom].

    // ---- Metadata (host-cached) ----

    /// @brief Number of atoms in the system.
    /// @return Atom count.
    int n_atoms() const { return n_atoms_; }

    /// @brief Number of bonds.
    /// @return Bond count.
    int n_bonds() const { return n_bonds_; }

    /// @brief Number of angles.
    /// @return Angle count.
    int n_angles() const { return n_angles_; }

    /// @brief Number of dihedrals.
    /// @return Dihedral count.
    int n_dihedrals() const { return n_dihedrals_; }

    int n_atoms_ = 0;     ///< Host-cached atom count.
    int n_bonds_ = 0;     ///< Host-cached bond count.
    int n_angles_ = 0;    ///< Host-cached angle count.
    int n_dihedrals_ = 0; ///< Host-cached dihedral count.
};

} // namespace torch
} // namespace ATV

#endif // ATV_HAS_TORCH

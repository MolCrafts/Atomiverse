/// @file cuda/frame/frame.cuh
/// @brief cuda::Frame -- GPU-side system state (SoA device pointers).
///
/// ALL array data lives on the GPU. Host-cached metadata (n_atoms, n_bonds,
/// etc.) avoids D2H transfers during the simulation loop.
/// CUDA-free header (contains only pointer declarations).
///
/// One backend, one Frame type: this is `cuda::Frame`, not a shared layout
/// with CPU. The SEMANTIC field set mirrors `cpu::Frame`
/// (same physical quantities), but the topology REPRESENTATION is
/// deliberately dual on this backend: the single-GPU path uses the flat
/// `BondList/AngleList/DihedralList` device arrays (transfer-friendly;
/// the CUDA kernels have not been ported to per-atom topology, so
/// `cpu::to_cuda` re-flattens from the CPU per-atom layout on every
/// H2D transfer), while the multi-GPU domain-decomposition path carries
/// the per-atom SoA (global-tag partners) needed for migration. CPU is
/// blocks+meta+box with per-atom topology columns only. Per the
/// CLAUDE.md Frame purity rule the frame carries only fundamental physical
/// quantities: method-specific scratch (the SCF workspace, the neighbor
/// pair list) is owned by the components that compute it, not by the frame.
///
/// @see cpu::Frame for the host-side counterpart
/// @see core/system_scalars.h for the shared system-scalar bundle
/// @see cuda/scf/scf_workspace.cuh for the relocated SCF scratch
/// @see cuda/neighbor_list.cuh for the relocated neighbor pair list
/// @see cpu::to_cuda / cuda::to_cpu for transfer
#pragma once

#include <cstdint>

#include "atomiverse/core/system_scalars.h"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {

/// @brief Bond topology list -- all arrays on device, molcrafts naming.
struct BondList {
    int* atomi = nullptr;   ///< First atom index [device].
    int* atomj = nullptr;   ///< Second atom index [device].
    int* type = nullptr;    ///< Bond type index [device].
    int* n_bonds = nullptr; ///< Bond count [device, 1 element].
};

/// @brief Angle topology list -- all arrays on device, molcrafts naming.
struct AngleList {
    int* atomi = nullptr;    ///< First atom index [device].
    int* atomj = nullptr;    ///< Center atom index [device].
    int* atomk = nullptr;    ///< Third atom index [device].
    int* type = nullptr;     ///< Angle type index [device].
    int* n_angles = nullptr; ///< Angle count [device, 1 element].
};

/// @brief Dihedral topology list -- all arrays on device, molcrafts naming.
struct DihedralList {
    int* atomi = nullptr;       ///< First atom index [device].
    int* atomj = nullptr;       ///< Second atom index [device].
    int* atomk = nullptr;       ///< Third atom index [device].
    int* atoml = nullptr;       ///< Fourth atom index [device].
    int* type = nullptr;        ///< Dihedral type index [device].
    int* n_dihedrals = nullptr; ///< Dihedral count [device, 1 element].
};

/// @brief GPU-side system state -- device pointers + host-cached metadata.
///
/// Device pointer fields are GPU-resident. Metadata fields (n_atoms_,
/// n_bonds_, etc.) are host-cached counts set at materialize time to avoid
/// D2H transfers.
///
/// Created by cpu::to_cuda() / cuda::from_molrs(), freed by cuda::destroy().
struct Frame {
    // -- Per-atom arrays (device pointers, molcrafts "atoms" block) --

    /// @brief Per-atom data block matching molpy atoms namespace.
    struct Atoms {
        real_t* x = nullptr; ///< x positions [Angstrom, device].
        real_t* y = nullptr; ///< y positions [Angstrom, device].
        real_t* z = nullptr; ///< z positions [Angstrom, device].
        real_t* x_correction =
            nullptr; ///< Mixed x residual [Angstrom, device].
        real_t* y_correction =
            nullptr; ///< Mixed y residual [Angstrom, device].
        real_t* z_correction =
            nullptr;              ///< Mixed z residual [Angstrom, device].
        motion_t* vx = nullptr;   ///< x velocities [Angstrom/fs, device].
        motion_t* vy = nullptr;   ///< y velocities [Angstrom/fs, device].
        motion_t* vz = nullptr;   ///< z velocities [Angstrom/fs, device].
        real_t* fx = nullptr;     ///< x forces [eV/Angstrom, device].
        real_t* fy = nullptr;     ///< y forces [eV/Angstrom, device].
        real_t* fz = nullptr;     ///< z forces [eV/Angstrom, device].
        motion_t* mass = nullptr; ///< Masses [amu, device].
        real_t* charge = nullptr; ///< Partial charges [e, device].
        int* type = nullptr;      ///< Atom types [device].

        /// @brief Global atom identity [device], persistent across MPI
        /// migration and restart. Same semantic field as CPU `kAtom_tag`.
        std::int64_t* tag = nullptr;

        /// @brief Per-atom PBC crossing counts, flat row-major [device,
        /// 3 * n_atoms], `image[i * 3 + d]` for atom i along dimension d.
        /// Same semantic field as CPU `kAtom_image`.
        std::int8_t* image = nullptr;

        int* d_size = nullptr; ///< Atom count [device, 1 element].

        // -- Per-atom bonded topology (multi-GPU bonded path ONLY) --
        //
        // The single-GPU / standard `materialize` path leaves all of these null
        // and uses the flat BondList/AngleList/DihedralList below. The
        // multi-GPU domain decomposition (`MultiGpu`) allocates and fills them so
        // each subdomain carries, per OWNED atom, the OWNER-filtered tuples it is
        // responsible for, with partners stored as GLOBAL TAGS (not local rows).
        // This lets `migrate` move topology with the atom and lets each
        // reneighbor RE-materialize the flat tuple lists by resolving partner
        // tags → local (owned or ghost) rows via the device tag→local map.
        // Strides mirror the SOURCE cpu::Atom's runtime topology strides
        // (`bond_per_atom` / `angle_per_atom` / `dihedral_per_atom` /
        // `maxspecial`), captured below at materialize time; arrays are sized
        // to `capacity` rows (ghost headroom).
        int* num_bond = nullptr; ///< [capacity] owner-filtered bond count.
        std::int64_t* bond_atom =
            nullptr; ///< [capacity*bond_per_atom] partner GLOBAL tags.
        int* bond_type = nullptr; ///< [capacity*bond_per_atom] bond type ids.
        int* num_angle =
            nullptr; ///< [capacity] owner(center)-filtered angle count.
        std::int64_t* angle_atom =
            nullptr; ///< [capacity*2*angle_per_atom] non-center tags.
        int* angle_type =
            nullptr; ///< [capacity*angle_per_atom] angle type ids.
        int* num_dihedral =
            nullptr; ///< [capacity] owner(2nd-atom)-filtered count.
        std::int64_t* dihedral_atom =
            nullptr; ///< [capacity*3*dihedral_per_atom] non-owner tags.
        int* dihedral_type =
            nullptr; ///< [capacity*dihedral_per_atom] dihedral type ids.
        int* nspecial =
            nullptr; ///< [capacity*3] special-bond tier prefix counts.
        std::int64_t* special =
            nullptr; ///< [capacity*maxspecial] special-bond partner tags.

        // Host-cached runtime strides of the wide arrays above (copied from
        // the source cpu::Atom at materialize time; 0 until then). Kernel
        // launches and D2D/peer copies must use these, never a compile-time
        // constant.
        int bond_per_atom = 0;  ///< Columns of bond_atom / bond_type.
        int angle_per_atom = 0; ///< angle_atom has 2x this; angle_type this.
        int dihedral_per_atom =
            0;              ///< dihedral_atom has 3x this; dihedral_type this.
        int maxspecial = 0; ///< Columns of special.
    } atoms;

    // -- System scalars (device-resident SystemScalars) --

    /// @brief Per-frame system scalars (energy, kinetic, conserved, stress,
    /// kinetic stress) -- one device-resident `SystemScalars` struct.
    ///
    /// Embeds `core/system_scalars.h`'s `SystemScalars` exactly as
    /// `cpu::Frame` does; the only difference is the device representation
    /// (a raw device pointer here, an inline value on the CPU frame). See
    /// that header for the field-by-field documentation.
    SystemScalars* scalars = nullptr;

    prec_t* box =
        nullptr; ///< Simulation cell 3x3 row-major [Angstrom, device, 9].

    // -- Topology sub-structures --

    BondList bonds;   ///< Bond topology (transferred by cpu::to_cuda).
    AngleList angles; ///< Angle topology (transferred by cpu::to_cuda).
    DihedralList
        dihedrals; ///< Dihedral topology (transferred by cpu::to_cuda).

    // -- Host-cached metadata (set at materialize, avoids D2H in loop) --

    /// @brief Number of atoms cached on the host side.
    /// @return Atom count.
    int n_atoms() const { return n_atoms_; }
    /// @brief Number of bonds cached on the host side.
    /// @return Bond count.
    int n_bonds() const { return n_bonds_; }
    /// @brief Number of angles cached on the host side.
    /// @return Angle count.
    int n_angles() const { return n_angles_; }
    /// @brief Number of dihedrals cached on the host side.
    /// @return Dihedral count.
    int n_dihedrals() const { return n_dihedrals_; }

    int n_atoms_ = 0;     ///< Host-cached atom count.
    int n_bonds_ = 0;     ///< Host-cached bond count.
    int n_angles_ = 0;    ///< Host-cached angle count.
    int n_dihedrals_ = 0; ///< Host-cached dihedral count.

    // -- Domain-decomposition counts (single-process multi-GPU) --
    // Owned atoms occupy [0, n_local); halo ghosts received from peer GPUs
    // occupy [n_local, n_local + n_ghost); the per-atom arrays are allocated to
    // `capacity` (>= n_local + n_ghost) so ghosts can be appended each step
    // without reallocation. SINGLE-GPU invariant: n_local == n_atoms,
    // n_ghost == 0, capacity == n_atoms (set at materialize) — so existing
    // kernels that read n_atoms() are unchanged.

    /// Owned atoms this subdomain integrates (force kernels write [0,n_local)).
    int n_local() const { return n_local_; }
    /// Ghost atoms received from peer GPUs (read-only halo copies).
    int n_ghost() const { return n_ghost_; }
    /// Owned + ghost — the range force kernels must READ positions over.
    int n_all() const { return n_local_ + n_ghost_; }
    /// Allocated per-atom array length (headroom for ghosts).
    int capacity() const { return capacity_; }

    int n_local_ = 0;
    int n_ghost_ = 0;
    int capacity_ = 0;
};

} // namespace cuda
} // namespace ATV

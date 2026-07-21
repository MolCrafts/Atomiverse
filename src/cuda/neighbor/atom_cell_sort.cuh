#pragma once

/// @file cuda/neighbor/atom_cell_sort.cuh
/// @brief Device-side cell-sort of one device Frame's OWNED atoms.
///
/// Spec multi-gpu-atom-cell-sort-01-permute. Reorders the owned atoms
/// ([0, n_local)) of a `cuda::Frame` by their neighbor-grid cell index — the
/// SAME cell convention `nlist_direct.cu::cell_of` uses to bin atoms for the
/// per-atom Verlet build — permuting EVERY per-atom array consistently:
///   - LJ SoA:  x y z vx vy vz fx fy fz mass charge type tag image
///   - bonded SoA (multi-GPU per-atom topology, partners as GLOBAL tags):
///       num_bond bond_atom bond_type
///       num_angle angle_atom angle_type
///       num_dihedral dihedral_atom dihedral_type
///       nspecial special
///
/// Row strides mirror `transfer.cu::materialize_per_atom_bonded` / the
/// multi-GPU bonded path EXACTLY (row-major [capacity, cols]):
///   bond_atom  [cap * MB]     bond_type     [cap * MB]
///   angle_atom [cap * 2*MA]   angle_type    [cap * MA]
///   dihedral_atom [cap * 3*MD] dihedral_type[cap * MD]
///   nspecial   [cap * 3]      special       [cap * MS]
/// where MB/MA/MD/MS are the frame's RUNTIME topology strides
/// (`Frame::atoms.bond_per_atom` / `angle_per_atom` / `dihedral_per_atom` /
/// `maxspecial`). image is [3 * cap] row-major image[i*3+d].
///
/// The bonded SoA pointers may be null (single-GPU `to_cuda` leaves them
/// null); they are skipped when null. Ghosts ([n_local, n_all)) are never
/// touched — caller cell-sorts BEFORE the halo is built.
///
/// This is a standalone primitive: it has NO call site in `multi_gpu.cu` yet.

#include "nlist_memory.cuh"

#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

#include <cstdint>

namespace ATV::cuda::md::nlist {

/// @brief Scratch buffers owned across cell-sort calls (resized on demand).
///
/// Owns the cub radix-sort temp plus the key/index ping-pong arrays and a
/// single real_t/int64 staging buffer reused to gather each per-atom column.
struct AtomCellSortBuffers {
    DeviceVector<int> cell_key;     ///< [n] cell index per owned atom.
    DeviceVector<int> cell_key_out; ///< [n] sorted keys (cub output).
    DeviceVector<int> perm;         ///< [n] identity 0..n-1 (cub values in).
    DeviceVector<int> perm_out;     ///< [n] sort order: perm_out[k] = old row.
    DeviceVector<int> inverse;      ///< [n] inverse permutation (optional).
    DeviceVector<char> sort_temp;   ///< cub temp storage.

    DeviceVector<real_t> stage_r;         ///< staging for one real_t column.
    DeviceVector<std::int64_t> stage_i64; ///< staging for one int64 column.
    DeviceVector<int> stage_i;            ///< staging for one int column.
    DeviceVector<std::int8_t> stage_i8;   ///< staging for the image column.

    void resize(int n) {
        cell_key.resize(n);
        cell_key_out.resize(n);
        perm.resize(n);
        perm_out.resize(n);
        inverse.resize(n);
        // Widest per-atom column is the int64 bonded arrays at stride 3*MD;
        // size the i64 stage to the largest and reuse for the narrower ones.
        // Caller sizes the bonded stages explicitly via resize_bonded().
        stage_r.resize(n);
        stage_i.resize(n);
        stage_i8.resize(static_cast<std::size_t>(n) * 3);
    }

    /// Size the int64 staging to hold the widest bonded column (n * max_cols).
    void resize_bonded(int n, int max_cols) {
        stage_i64.resize(static_cast<std::size_t>(n) * max_cols);
        if (stage_i.size() < static_cast<std::size_t>(n) * max_cols)
            stage_i.resize(static_cast<std::size_t>(n) * max_cols);
    }
};

/// @brief Cell-sort the owned atoms of `frame` in place.
///
/// Computes a cell key per owned atom from (x,y,z) using the orthorhombic
/// grid (Lx,Ly,Lz, dims dx*dy*dz) — identical to nlist_direct.cu's `cell_of`
/// — cub-radix-sorts row indices by key into `buf.perm_out`, then gathers
/// every (non-null) per-atom array through `perm_out` and writes it back so
/// row k holds what was row `perm_out[k]`. `buf.inverse` is filled so a
/// subsequent `apply_inverse_permutation` restores the original order.
///
/// @param frame    device Frame (mutated in place, rows [0, n_local) only).
/// @param Lx,Ly,Lz orthorhombic box edge lengths [Angstrom].
/// @param dx,dy,dz neighbor-grid cell counts per axis.
/// @param periodic wrap positions into [0,L) before binning.
/// @param buf      scratch holder (resized as needed).
/// @param stream   CUDA stream.
void cell_sort_owned(Frame& frame, real_t Lx, real_t Ly, real_t Lz, int dx,
                     int dy, int dz, bool periodic, AtomCellSortBuffers& buf,
                     cudaStream_t stream = 0);

/// @brief Apply an arbitrary permutation `order` (order[k] = source row) to
/// every per-atom array of `frame`'s owned rows, in place via `buf` staging.
/// Used by `cell_sort_owned`; exposed so the inverse can be applied to undo a
/// previous sort (`order == buf.inverse`).
void apply_permutation_owned(Frame& frame, const int* order, int n,
                             AtomCellSortBuffers& buf, cudaStream_t stream = 0);

} // namespace ATV::cuda::md::nlist

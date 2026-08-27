#pragma once

/// @file cuda/comm/domain_decomposition.cuh
/// @brief SpatialGrid — single-process 3D spatial split of an
///        orthorhombic box across N GPUs. Pure host-side topology (no device
///        code): factorizes N into a balanced 3D grid (the single-process
///        analog of MPI_Dims_create), assigns each GPU slot its Cartesian
///        coords, owned slab [sublo, subhi), and 6 face-neighbor slots (with
///        periodic wrap), and maps a wrapped position to its owner slot.
///
/// Mirrors cpu::md::mpi::DomainDecomposition's topology, minus the MPI handles.
/// The multi-GPU driver uses it to partition atoms into per-device subdomains
/// and to drive P2P halo/migration between face neighbors.

#include <array>

namespace ATV {
namespace cuda {

class SpatialGrid {
  public:
    /// Face order matches cpu: 0:-x 1:+x 2:-y 3:+y 4:-z 5:+z.
    enum Face { MX = 0, PX, MY, PY, MZ, PZ };

    /// @param n_gpus  Number of subdomains (>= 1).
    /// @param Lx,Ly,Lz  Orthorhombic box edge lengths [Angstrom].
    SpatialGrid(int n_gpus, double Lx, double Ly, double Lz);

    int count() const { return n_; }
    std::array<int, 3> dims() const { return {dims_[0], dims_[1], dims_[2]}; }

    /// Cartesian coords of a slot in the dims_ grid.
    std::array<int, 3> coords(int slot) const;
    /// Owned slab lower / upper bounds [Angstrom] for a slot.
    std::array<double, 3> sublo(int slot) const;
    std::array<double, 3> subhi(int slot) const;
    /// Neighbor slot across `face` (periodic wrap). May equal `slot` when
    /// dims along that axis == 1 (the slot is its own periodic neighbor).
    int face_neighbor(int slot, int face) const;

    /// Owner slot of a position (wrapped into [0,L) first).
    int owner_of(double x, double y, double z) const;

    /// Linearize / delinearize Cartesian coords <-> slot (x-fastest).
    int slot_of(int cx, int cy, int cz) const;

  private:
    int n_ = 1;
    int dims_[3] = {1, 1, 1};
    double L_[3] = {0, 0, 0};
};

} // namespace cuda
} // namespace ATV

/// @file cpu/longrange/grid_remap.h
/// @brief Plan-based redistribution of a 3-D grid between two per-rank
///        box tilings of the same global mesh.
///
/// Ported from LAMMPS `src/KSPACE/remap.cpp` (`remap_3d`,
/// `remap_3d_create_plan`, `remap_3d_collide`) and `src/pack.h` (`pack_3d`,
/// `unpack_3d`, `unpack_3d_permute1_n`, `unpack_3d_permute2_n`), checkout
/// e8c4c278a8: the point-to-point plan with non-blocking sends.
/// LAMMPS: Copyright (2003) Sandia Corporation. GPL-2.0-only.
///
/// **Layout.** A box is six inclusive global indices `{ilo, ihi, jlo, jhi,
/// klo, khi}`; a rank owning nothing passes `hi < lo`. Input data is
/// contiguous with `i` fastest, `j` middle, `k` slowest, `nqty` values per
/// element. The union of all input boxes and the union of all output boxes
/// each tile the mesh without overlap.
///
/// **Permutation** of the *output* storage order (LAMMPS `permute`):
/// 0 keeps (i, j, k); 1 stores (j, k, i), i.e. `j` fastest; 2 stores
/// (k, i, j), i.e. `k` fastest. The pencil FFT uses it to hand the next
/// transform axis over as the contiguous one.
///
/// Not ported: the collective (`MPI_Alltoallv`) plan and the caller-provided
/// scratch (`memory = 0`); the plan owns its scratch. LAMMPS duplicates the
/// communicator per plan; this plan takes a tag instead, which is enough
/// because every exchange completes before the next one starts and messages
/// between one pair of ranks do not overtake.
#pragma once

#ifdef ATV_HAS_MPI

#include <array>
#include <cstddef>
#include <vector>

#include <mpi.h>

namespace ATV::longrange {

/// @brief One redistribution plan (LAMMPS `remap_plan_3d`).
class GridRemap {
  public:
    /// @brief Build the plan. Collective over @p comm (two `MPI_Allgather`).
    /// @param comm    Communicator whose ranks own the two tilings.
    /// @param in      This rank's input box (inclusive global indices).
    /// @param out     This rank's output box (inclusive global indices).
    /// @param nqty    Values per element (count), ≥ 1.
    /// @param permute Output storage permutation, 0, 1 or 2.
    /// @param tag     MPI tag of this plan's messages.
    /// @throws std::invalid_argument on a bad @p nqty or @p permute.
    GridRemap(MPI_Comm comm, const std::array<int, 6>& in,
              const std::array<int, 6>& out, int nqty, int permute, int tag);

    /// @brief Move the data: post receives, pack and send, copy the part
    ///        that stays on this rank, unpack.
    /// @param in  Input values, laid out as the input box.
    /// @param out Output values, laid out as the (permuted) output box. May
    ///            alias @p in: everything is packed before anything is
    ///            unpacked.
    void perform(const double* in, double* out);

    /// @brief Heap bytes held by the plan.
    /// @return Bytes of the send buffer, the scratch and the message tables.
    std::size_t memory_bytes() const;

  private:
    /// LAMMPS `pack_plan_3d`.
    struct PackPlan {
        int nfast = 0;         ///< elements (permuted) or values along i
        int nmid = 0;          ///< elements along j
        int nslow = 0;         ///< elements along k
        int nstride_line = 0;  ///< value stride between successive lines
        int nstride_plane = 0; ///< value stride between successive planes
    };
    struct Message {
        int proc = -1;  ///< partner rank
        int offset = 0; ///< first value in the data array
        int size = 0;   ///< values in the message
        int bufloc = 0; ///< first value in the send buffer / scratch
        PackPlan plan;
    };

    void pack_(const double* data, double* buf, const PackPlan& p) const;
    void unpack_(const double* buf, double* data, const PackPlan& p) const;

    MPI_Comm comm_;
    int nqty_;
    int permute_;
    int tag_;
    std::vector<Message> sends_; ///< other ranks only
    std::vector<Message> recvs_; ///< other ranks only
    bool self_ = false;
    Message self_send_;
    Message self_recv_;
    std::vector<double> sendbuf_;
    std::vector<double> scratch_;
    std::vector<MPI_Request> requests_;
};

} // namespace ATV::longrange

#endif // ATV_HAS_MPI

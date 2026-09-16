/// @file cpu/component/comm.h
/// @brief `ForwardPacker` — LAMMPS Fix-style extra-field forward protocol.
///
/// Algorithm transplanted from LAMMPS `src/fix.h` `pack_forward_comm` /
/// `unpack_forward_comm` (checkout `78a7ddf`, 2026-08-25). Public API is
/// ATV-native: a packer fills a `double` buffer from its own arrays; `Comm`
/// replays the Halo sendlists and posts MPI. Plugins include this header,
/// never `atom_pack.h`.
///
/// Copyright (2003) Sandia Corporation.  Under the terms of Contract
/// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
/// certain rights in this software.  This software is distributed under
/// the GNU General Public License (GPL-2.0-only).
///
/// References:
///   * S. Plimpton, J. Comput. Phys. **117**, 1 (1995),
///     DOI 10.1006/jcph.1995.1039
///   * A. P. Thompson et al., Comput. Phys. Commun. **271**, 108171 (2022),
///     DOI 10.1016/j.cpc.2021.108171
#pragma once

#include "atomiverse/cpu/communication/comm.h"

namespace ATV {
namespace cpu {

/// @brief Extra-field packer invoked by `Comm::forward_comm(Packer&)`.
///
/// Default is a no-op (`comm_forward() == 0`, pack returns 0, unpack is
/// empty). Concrete plugins override the trio and call
/// `ctx.md.comm->forward_comm(*this)` after they dirty owned rows.
/// Coordinates may apply a box image shift \(p_d L_d\) [Å]; velocities
/// [Å/fs] and charges [e] must not.
class ForwardPacker {
  public:
    virtual ~ForwardPacker() = default;

    /// @brief Doubles packed per atom, used to size the receive buffer.
    /// @return Count per atom [dimensionless]. Default 0 (no MPI).
    virtual int comm_forward() const { return 0; }

    /// @brief Pack `n` sendlist rows into `buf`.
    /// @param n Number of local rows in `list` [dimensionless].
    /// @param list Local row indices to pack.
    /// @param buf Output buffer; layout is packer-defined.
    /// @param pbc_flag 1 when this swap crosses a periodic image.
    /// @param pbc Per-axis image flags in `{-1,0,+1}`; shift is
    ///        \(p_d L_d\) [Å] when the packer applies PBC.
    /// @return Number of doubles written.
    virtual int pack_forward_comm(int n, const int* list, double* buf,
                                  int pbc_flag, const int* pbc) {
        (void)n;
        (void)list;
        (void)buf;
        (void)pbc_flag;
        (void)pbc;
        return 0;
    }

    /// @brief Unpack `n` received rows starting at ghost slot `first`.
    /// @param n Number of ghost rows [dimensionless].
    /// @param first First destination row index.
    /// @param buf Packed doubles from the matching `pack_forward_comm`.
    virtual void unpack_forward_comm(int n, int first, const double* buf) {
        (void)n;
        (void)first;
        (void)buf;
    }
};

} // namespace cpu
} // namespace ATV

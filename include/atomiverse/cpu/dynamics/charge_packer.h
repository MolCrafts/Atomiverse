/// @file cpu/dynamics/charge_packer.h
/// @brief `ChargePacker` — forward `kAtom_charge` [e] with no PBC shift.
///
/// Shared by the six owned-charge solvers. Charge is a scalar; a periodic
/// image flag must not translate it. LAMMPS twin: `fix_electrode_conp.cpp`
/// `pack_forward_comm` (checkout `78a7ddf`).
///
/// Copyright (2003) Sandia Corporation.  Under the terms of Contract
/// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
/// certain rights in this software.  This software is distributed under
/// the GNU General Public License (GPL-2.0-only).
///
/// References:
///   * L. J. V. Ahrens-Iwers et al., J. Chem. Phys. **157**, 084801 (2022),
///     DOI 10.1063/5.0100191
#pragma once

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {

/// @brief Packer for per-atom charge [e]. Ignores `pbc_flag` / `pbc`.
class ChargePacker : public ForwardPacker {
  public:
    /// @param frame Frame whose `kAtom_charge` column [e] is packed.
    explicit ChargePacker(cpu::Frame& frame) : frame_(frame) {}

    /// @brief One double per atom (charge [e]).
    /// @return 1.
    int comm_forward() const override { return 1; }

    /// @brief Pack owner charges [e]; PBC arguments are ignored.
    /// @param n Number of local rows in `list` [dimensionless].
    /// @param list Local row indices to pack.
    /// @param buf Output buffer of charges [e].
    /// @param pbc_flag Unused (charge is translation-invariant).
    /// @param pbc Unused.
    /// @return Number of doubles written.
    int pack_forward_comm(int n, const int* list, double* buf, int pbc_flag,
                          const int* pbc) override {
        (void)pbc_flag;
        (void)pbc;
        const auto q = frame_.span(kAtom_charge);
        int m = 0;
        for (int i = 0; i < n; ++i)
            buf[m++] = static_cast<double>(q[list[i]]);
        return m;
    }

    /// @brief Unpack received charges [e] into ghost rows `[first, first+n)`.
    /// @param n Number of ghost rows [dimensionless].
    /// @param first First destination row index.
    /// @param buf Packed charges [e].
    void unpack_forward_comm(int n, int first, const double* buf) override {
        auto q = frame_.span(kAtom_charge);
        int m = 0;
        for (int i = 0; i < n; ++i)
            q[first + i] = static_cast<StorageReal>(buf[m++]);
    }

    /// @brief Post `Comm::forward_comm(ChargePacker)` when a communicator
    ///        is published. Single-rank / null `ctx.md.comm` is a no-op.
    /// @param ctx Launch context (frame + published communicator).
    static void forward(const LaunchContext& ctx) {
        if (cpu::Comm* comm = ctx.runtime.md.comm) {
            ChargePacker packer{ctx.frame};
            comm->forward_comm(packer);
        }
    }

  private:
    cpu::Frame& frame_;
};

} // namespace cpu
} // namespace ATV

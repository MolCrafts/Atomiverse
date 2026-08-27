/// @file cpu/communication/comm.h
/// @brief `cpu::Comm` — abstract brick/tiled communicator (LAMMPS `Comm`).
///
/// Algorithm and communication structure transplanted from LAMMPS
/// `src/comm.h` (checkout `78a7ddf`, 2026-08-25). Public API is ATV-native:
/// methods take `LaunchContext` (Frame + DD) rather than a LAMMPS `Pointers`
/// bag. `forward_comm(Packer&)` replays the Halo sendlists through a
/// `ForwardPacker` (`component/comm.h`).
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

namespace ATV {
namespace cpu {

struct LaunchContext;

/// @brief Extra-field packer. Complete type lives in `component/comm.h`.
class ForwardPacker;
using Packer = ForwardPacker;

/// @brief Abstract spatial communicator: borders / forward / reverse /
///        exchange.
///
/// Owned by `communication::DomainRuntime` and published non-owning as
/// `ctx.md.comm`. Users never construct a concrete communicator.
class Comm {
  public:
    virtual ~Comm() = default;

    /// @brief Derive the swap table from the current decomposition and box.
    /// @param ctx Launch context (frame + published DD).
    virtual void setup(const LaunchContext& ctx) = 0;

    /// @brief Refresh ghost rows from the existing swap table (Å positions).
    /// @param ctx Launch context (frame + published DD).
    virtual void forward_comm(const LaunchContext& ctx) = 0;

    /// @brief Accumulate ghost-row forces onto owners [eV/Å].
    /// @param ctx Launch context (frame + published DD).
    virtual void reverse_comm(const LaunchContext& ctx) = 0;

    /// @brief Move owned atoms that left this rank's subdomain.
    /// @param ctx Launch context (frame + published DD).
    virtual void exchange(const LaunchContext& ctx) = 0;

    /// @brief Rebuild send lists and ghost slots from current positions [Å].
    /// @param ctx Launch context (frame + published DD).
    virtual void borders(const LaunchContext& ctx) = 0;

    /// @brief Extra-field forward: replay Halo sendlists through `packer`.
    ///        Single-rank and empty tables are no-ops; posts no MPI then.
    /// @param packer Plugin packer (`pack_forward_comm` / `unpack_forward_comm`).
    virtual void forward_comm(Packer& packer) = 0;

    /// @brief Ghost-shell width used by `setup` / `borders`.
    /// @param cutoff Pair-interaction cutoff [Angstrom].
    virtual void set_ghost_cutoff(double cutoff) = 0;
};

} // namespace cpu
} // namespace ATV

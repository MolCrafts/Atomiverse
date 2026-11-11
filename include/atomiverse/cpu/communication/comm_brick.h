/// @file cpu/communication/comm_brick.h
/// @brief `cpu::comm::CommBrick` — LAMMPS brick (6-way stencil) communicator.
///
/// Algorithm and communication structure transplanted from LAMMPS
/// `src/comm_brick.cpp` (checkout `78a7ddf`, 2026-08-25). Structural
/// reimplementation of `CommBrick::forward_comm` / `reverse_comm` /
/// `exchange` / `borders` over a private `comm::Halo` swap table — not a
/// verbatim dump. Public API is ATV-native (`cpu::Comm`).
///
/// Extra-field reverse protocol follows LAMMPS `src/comm_brick.h`, checkout
/// `a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b`.
///
/// Copyright (2003) Sandia Corporation.  Under the terms of Contract
/// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
/// certain rights in this software.  This software is distributed under
/// the GNU General Public License (GPL-2.0-only).
///
/// The constructor is private: `communication::DomainRuntime` is the sole
/// owner and publishes the instance as `ctx.md.comm`. There is no
/// `make_*` / `create_*` factory.
///
/// References:
///   * S. Plimpton, J. Comput. Phys. **117**, 1 (1995),
///     DOI 10.1006/jcph.1995.1039
///   * A. P. Thompson et al., Comput. Phys. Commun. **271**, 108171 (2022),
///     DOI 10.1016/j.cpc.2021.108171
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "atomiverse/core/ghost_contract.h"
#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/communication/halo.h"

namespace ATV {
namespace cpu {

namespace communication {
class DomainRuntime;
}

namespace comm {

namespace detail {
struct CommBuffers;
} // namespace detail

/// @brief Brick-decomposition communicator. Private `Halo`; MPI in the four
///        methods lives in `comm_brick.cpp` once.
class CommBrick : public Comm {
    friend class communication::DomainRuntime;

  public:
    ~CommBrick() override;
    CommBrick(CommBrick&&) noexcept;
    CommBrick& operator=(CommBrick&&) noexcept;
    CommBrick(const CommBrick&) = delete;
    CommBrick& operator=(const CommBrick&) = delete;

    void setup(const LaunchContext& ctx) override;
    void forward_comm(const LaunchContext& ctx) override;
    void reverse_comm(const LaunchContext& ctx) override;
    void exchange(const LaunchContext& ctx) override;
    void borders(const LaunchContext& ctx) override;
    void forward_comm(Packer& packer) override;
    void reverse_comm(ReversePacker& packer) override;
    void set_ghost_cutoff(double cutoff) override;

    /// @brief Resident heap bytes of the shared send/recv buffers.
    std::size_t memory_bytes() const;

  private:
    explicit CommBrick(md::mpi::DomainDecomposition& domain);

    /// The run's ghost contract, cached at `setup()` from the Driver's
    /// fold. Each pass compiles its own `PackPlan` off this; there is no
    /// per-wire copy.
    GhostContract contract_;

    /// @brief Whether `setup()` has cached the wire schemas for this run.
    ///        `bufs_` cannot stand in for it: `exchange()` and
    ///        `reverse_comm()` allocate the buffers on demand too.
    bool setup_done_ = false;

    Halo swaps_;
    std::unique_ptr<detail::CommBuffers> bufs_;
    double ghost_cutoff_ = 0.0;
    md::mpi::DomainDecomposition* dd_ = nullptr;
    std::uint64_t last_checked_halo_generation_{};
    /// Wire-schema fingerprints anchored collectively in `setup()`; the
    /// per-step / per-rebuild paths compare against them locally.
    std::uint64_t verified_border_fingerprint_{};
    std::uint64_t verified_reverse_fingerprint_{};
    /// Exchange wire schema, anchored on the first `exchange()` (which can
    /// run without a ghost cutoff, so `setup()` is not a reliable hook) and
    /// compared locally on every migration after it.
    std::uint64_t verified_exchange_fingerprint_{};
    bool exchange_schema_anchored_ = false;
    /// @brief Widest extra-field forward/reverse packer seen this run; used
    ///        to size shared packer buffers at the end of `borders()`.
    int packer_nsize_ = 0;
};

} // namespace comm
} // namespace cpu
} // namespace ATV

/// @file cpu/component/owned_range.h
/// @brief Authoritative owned / force atom counts — serial vs multi-rank split.
///
/// **Decision (split paths, one entry):** without multi-rank MPI there is
/// **no spatial domain decomposition**. Owned atoms are simply
/// `frame.nlocal()`. `DomainDecomposition` exists only for multi-rank
/// ownership / ghosts / halo. Call sites must use these helpers — do not
/// open-code `mpi ? dd->n_local : f.nlocal()` (that re-entangles serial
/// code with a DD type it does not need).
///
/// Multi-rank (`dd` non-null, `mpi_size > 1`, and a live Cartesian communicator):
///   - owned rows: `[0, dd->n_local)`
///   - ghost rows: `[n_local, n_local + n_ghost)` — force partners only;
///     integrators / zeroing must not touch them (CommBrick returns forces).
///
/// Serial / single-rank MPI / serial-on-replica: return `frame.nlocal()`.
#pragma once

#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {

/// @brief True multi-rank spatial decomposition (not "MPI linked, 1 rank").
/// @param rt Launch/Build `Context` (field name is historically `runtime`).
inline bool multi_rank_dd(const cpu::Context& rt) noexcept {
#ifdef ATV_HAS_MPI
    // serial_on_replica keeps a DD infrastructure object, but deliberately
    // has no Cartesian topology and does not partition the Frame.
    return rt.md.dd != nullptr && rt.mpi_size > 1
           && rt.md.dd->cart_comm != MPI_COMM_NULL;
#else
    (void)rt;
    return false;
#endif
}

/// @brief Owned-atom count at `launch()` time.
/// @return `dd->n_local` if multi-rank, else `frame.nlocal()`.
inline int owned_count(const cpu::LaunchContext& ctx) {
    if (multi_rank_dd(ctx.runtime))
        return ctx.runtime.md.dd->n_local;
    return ctx.frame.nlocal();
}

/// @brief Owned-atom count at `build()` time.
inline int owned_count(const cpu::BuildContext& ctx) {
    if (multi_rank_dd(ctx.runtime))
        return ctx.runtime.md.dd->n_local;
    return ctx.frame.nlocal();
}

/// @brief Force-loop extent: owned + ghosts under multi-rank, else nlocal.
/// Pair / real-space Coulomb iterate this range; integrators use
/// `owned_count` only.
inline int force_count(const cpu::LaunchContext& ctx) {
    if (multi_rank_dd(ctx.runtime)) {
        const auto* dd = ctx.runtime.md.dd;
        return dd->n_local + dd->n_ghost;
    }
    return ctx.frame.nlocal();
}

} // namespace cpu
} // namespace ATV

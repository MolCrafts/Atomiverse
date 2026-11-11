/// @file cpu/component/owned_range.h
/// @brief Authoritative owned / force atom counts from Frame.
///
/// Owned rows are integrated; ghost rows are force partners only. Serial
/// triclinic halos can have self-image ghosts without multi-rank ownership.
/// Consumers must not infer the absence of ghosts from the MPI rank count.
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
    return rt.md.dd != nullptr && rt.mpi_size > 1
           && rt.md.dd->cart_comm != MPI_COMM_NULL;
#else
    (void)rt;
    return false;
#endif
}

/// @brief Owned-atom count at `launch()` time.
/// @return `frame.nlocal()` for every execution topology.
inline int owned_count(const cpu::LaunchContext& ctx) {
    return ctx.frame.nlocal();
}

/// @brief Owned-atom count at `build()` time.
inline int owned_count(const cpu::BuildContext& ctx) {
    return ctx.frame.nlocal();
}

/// @brief Force-loop extent: owned + ghosts, including serial periodic images.
/// Pair / real-space Coulomb iterate this range; integrators use
/// `owned_count` only.
inline int force_count(const cpu::LaunchContext& ctx) {
    return ctx.frame.n_total();
}

} // namespace cpu
} // namespace ATV

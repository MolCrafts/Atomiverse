/// @file cpu/component/owned_range.h
/// @brief `owned_count()` helpers — the authoritative "how many locally
///        owned atoms are on this rank?" answer used by integrators,
///        thermostats, optimizers, and observers across every domain.
///
/// Under multi-rank (`dd` non-null and `mpi_size > 1`) the owned atoms
/// live in rows `[0, dd->n_local)`; rows `[n_local, n_local + n_ghost)`
/// hold halo ghosts that must not be touched by integrators or by the
/// local zeroing pass (HaloRecv ships ghost force contributions
/// back to their owners). Single-rank / DD-less contexts simply return
/// the full atom count.
///
/// Shared backend-root infrastructure (consumed by md/, minimizer/,
/// observer/). The decomposition state itself stays in
/// `md/mpi/domain.h` because `cpu::Context` already publishes it as the
/// canonical non-owning pointer.
#pragma once

#include "atomiverse/cpu/comm/domain.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {

/// @brief Owned-atom count at `launch()` time.
/// @param ctx Current launch context.
/// @return `dd->n_local` under multi-rank, `ctx.frame.nlocal()` otherwise.
inline int owned_count(const cpu::LaunchContext& ctx) {
    const auto* dd = ctx.runtime.md.dd;
    if (dd && ctx.runtime.mpi_size > 1)
        return dd->n_local;
    return ctx.frame.nlocal();
}

/// @brief Owned-atom count at `build()` time. Identical semantics to the
/// `LaunchContext` overload; separate because `BuildContext` and
/// `LaunchContext` do not share a common base.
/// @param ctx Current build context.
/// @return `dd->n_local` under multi-rank, `ctx.frame.nlocal()` otherwise.
inline int owned_count(const cpu::BuildContext& ctx) {
    const auto* dd = ctx.runtime.md.dd;
    if (dd && ctx.runtime.mpi_size > 1)
        return dd->n_local;
    return ctx.frame.nlocal();
}

} // namespace cpu
} // namespace ATV

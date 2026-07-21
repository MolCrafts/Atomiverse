/// @file cpu/minimizer/global_reduce.h
/// @brief Scalar cross-rank reductions for optimizer components.
///
/// The optimizer algorithms (LBFGS, CG) and the OptLog recorder compute
/// inner products and force/displacement metrics over the owned rows
/// `[0, dd->n_local)`; under multi-rank these per-rank partials must be
/// combined before any decision (step size, Armijo accept/reject,
/// convergence verdict) so every rank acts identically. These helpers
/// wrap the `MPI_Allreduce` with the standard single-rank /
/// MPI-less-build no-op shortcuts.
///
/// All ranks must call them unconditionally (they are collectives) —
/// never behind a rank-dependent branch.
#pragma once

#include "atomiverse/cpu/comm/domain.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace minimizer {

namespace detail {
#ifdef ATV_HAS_MPI
inline bool multi_rank(const cpu::LaunchContext& ctx) {
    return ctx.runtime.mpi_size > 1 && ctx.runtime.md.dd
           && ctx.runtime.md.dd->cart_comm != MPI_COMM_NULL;
}
#endif
} // namespace detail

/// @brief In-place `MPI_Allreduce(SUM)` of `n` doubles. Single-rank /
/// MPI-less builds: no-op (the local partial already is the global value).
inline void global_sum(const cpu::LaunchContext& ctx, double* vals, int n) {
#ifdef ATV_HAS_MPI
    if (detail::multi_rank(ctx))
        MPI_Allreduce(MPI_IN_PLACE, vals, n, MPI_DOUBLE, MPI_SUM,
                      ctx.runtime.md.dd->cart_comm);
#else
    (void)ctx;
    (void)vals;
    (void)n;
#endif
}

/// @brief Convenience scalar overload of `global_sum`.
inline double global_sum(const cpu::LaunchContext& ctx, double v) {
    global_sum(ctx, &v, 1);
    return v;
}

/// @brief In-place `MPI_Allreduce(MAX)` of `n` doubles. Single-rank /
/// MPI-less builds: no-op.
inline void global_max(const cpu::LaunchContext& ctx, double* vals, int n) {
#ifdef ATV_HAS_MPI
    if (detail::multi_rank(ctx))
        MPI_Allreduce(MPI_IN_PLACE, vals, n, MPI_DOUBLE, MPI_MAX,
                      ctx.runtime.md.dd->cart_comm);
#else
    (void)ctx;
    (void)vals;
    (void)n;
#endif
}

/// @brief Convenience scalar overload of `global_max`.
inline double global_max(const cpu::LaunchContext& ctx, double v) {
    global_max(ctx, &v, 1);
    return v;
}

} // namespace minimizer
} // namespace cpu
} // namespace ATV

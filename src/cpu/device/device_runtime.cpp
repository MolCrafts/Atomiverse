/// @file cpu/device/device_runtime.cpp
/// @brief cpu::device::DeviceRuntime implementation (moved out of the former
///        CPU execution engine by backend-mirror-05-dissolve).

#include "atomiverse/cpu/device/device_runtime.h"

#include "atomiverse/cpu/component/context.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

// Optional: OpenBLAS exposes its own thread count. Weak so a build against any
// other BLAS still links. See the threading note in configure_threads().
extern "C" int openblas_get_num_threads(void) __attribute__((weak));

namespace ATV {
namespace cpu {
namespace device {

void DeviceRuntime::set_threads(Context& ctx, int n) {
    ctx.n_threads = (n < 1) ? 0 : n;
}

void DeviceRuntime::configure_threads(Context& ctx) {
#ifdef _OPENMP
    if (ctx.n_threads > 0)
        omp_set_num_threads(ctx.n_threads);
    else
        ctx.n_threads = omp_get_max_threads();
#else
    ctx.n_threads = 1;
#endif

    // Thread affinity is a **trade-off**, not a free win, and the direction
    // depends on which hot path a run is in:
    //
    //   * ATV's own OpenMP loops (pair styles, neighbour build, the SK and γ
    //     walks) want pinned threads — migration costs NUMA and cache.
    //   * BLAS/LAPACK-bound paths (SCF, GPW, plane-wave, and the DFTB SCC,
    //     whose cost is one dense generalized eigenproblem per iteration) can
    //     lose an order of magnitude: an OpenBLAS built `USE_OPENMP` collapses
    //     to a single thread as soon as **any** OpenMP affinity is requested.
    //     Measured on OpenBLAS 0.3.32 (ZEN, USE_OPENMP): with
    //     OMP_NUM_THREADS=16, `openblas_get_num_threads()` returns 16 with no
    //     affinity set and 1 under `OMP_PROC_BIND=close`, `=spread`, `=true`,
    //     or any `OMP_PLACES` — dgemm 807 → 106 GFLOP/s, and a 2432² dsygvd
    //     0.52 s → 9.7 s.
    //
    // So this does not recommend a setting. It reports the one state that is
    // unambiguously wrong: several OpenMP threads while BLAS runs on one.
    static bool s_threading_warned = false;
    if (!s_threading_warned && ctx.n_threads > 1) {
        int blas_threads = 0;
        if (openblas_get_num_threads)
            blas_threads = openblas_get_num_threads();
        if (blas_threads == 1) {
            int my_rank = 0;
#ifdef ATV_HAS_MPI
            int mpi_init_check = 0;
            MPI_Initialized(&mpi_init_check);
            if (mpi_init_check)
                MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif
            if (my_rank == 0) {
                const char* bind = std::getenv("OMP_PROC_BIND");
                const char* places = std::getenv("OMP_PLACES");
                std::fprintf(
                    stderr,
                    "[atv] warning: n_threads=%d but BLAS reports 1 thread"
                    " (OMP_PROC_BIND=%s, OMP_PLACES=%s). Any OpenMP affinity"
                    " setting collapses an OpenMP-flavour OpenBLAS to one"
                    " thread; unset both if this run is BLAS-bound (SCF, GPW,"
                    " plane-wave, DFTB SCC).\n",
                    ctx.n_threads, bind ? bind : "(unset)",
                    places ? places : "(unset)");
                std::fflush(stderr);
            }
            s_threading_warned = true;
        }
    }
}

void DeviceRuntime::init_timing(Context& ctx) {
    // Per-component timing is ON BY DEFAULT (near-zero overhead) so every driver
    // can emit the built-in end-of-run time+memory summary. Opt OUT with
    // ATV_TIMER=0/off. Re-apply the remembered preference so a pre-run
    // enable_timing(false) sticks; ATV_TIMER overrides.
    timing_.set_enabled(timing_pref_);
    if (const char* tv = std::getenv("ATV_TIMER")) {
        const std::string v = tv;
        if (v == "0" || v == "false" || v == "FALSE" || v == "off")
            timing_.set_enabled(false);
        else if (v == "1" || v == "true" || v == "TRUE" || v == "on")
            timing_.set_enabled(true);
    }
    ctx.timer = &timing_;
}

} // namespace device
} // namespace cpu
} // namespace ATV

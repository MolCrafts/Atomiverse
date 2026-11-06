/// @file cpu/device/fftw_threads.h
/// @brief One-time FFTW OpenMP threading for every CPU FFT plan.
///
/// `fftw_plan_with_nthreads` is a process-global setting that applies to
/// plans created after it, so one call before the first plan threads every
/// later plan (GPW grids, PPPM/PME meshes). Built with `ATV_FFTW_THREADS`
/// when libfftw3_omp is linked; otherwise plans stay single-threaded.
/// Include only from translation units that already include <fftw3.h>.
#pragma once

#include <mutex>

#include <fftw3.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {
namespace device {

/// @brief Initialise FFTW threads once and size them to the OpenMP team.
///        Call before creating a plan; a no-op after the first call and in
///        builds without libfftw3_omp.
inline void ensure_fftw_threads() {
#ifdef ATV_FFTW_THREADS
    static std::once_flag once;
    std::call_once(once, [] {
        if (fftw_init_threads() == 0)
            return; // plans stay single-threaded
        int n = 1;
#ifdef _OPENMP
        n = omp_get_max_threads();
#endif
        fftw_plan_with_nthreads(n < 1 ? 1 : n);
    });
#endif
}

} // namespace device
} // namespace cpu
} // namespace ATV

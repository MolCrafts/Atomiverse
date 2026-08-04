/// @file src/cpu/gpw/internal/fft_cache.cpp
/// @brief Thread-local FFTW plan/buffer cache implementation.

#include "atomiverse/cpu/gpw/internal/fft_cache.h"

#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {
namespace gpw {
namespace {

// FFTW planner is process-global and not thread-safe. Guard create/destroy only;
// execute of ESTIMATE plans is fine concurrently (and under OpenMP FFTW threads).
std::mutex& fftw_plan_mutex() {
    static std::mutex m;
    return m;
}

/// One-time: multi-threaded FFTW for large 3D GGA gradient / Poisson transforms
/// (linked with libfftw3_omp when `ATV_GPW_FFTW_THREADS`). Plans created after
/// this use all OMP threads during `fftw_execute`. KS builds are serial; do not
/// call FFT from nested OpenMP workers concurrently.
void ensure_fftw_threads() {
#ifdef ATV_GPW_FFTW_THREADS
    static std::once_flag once;
    std::call_once(once, [] {
        if (fftw_init_threads() == 0)
            return; // fall back to single-thread plans
        int n = 1;
#ifdef _OPENMP
        n = omp_get_max_threads();
        if (n < 1)
            n = 1;
#endif
        fftw_plan_with_nthreads(n);
    });
#else
    (void)0;
#endif
}

using ShapeKey = std::tuple<int, int, int>;

} // namespace

// ── R2cFft ──────────────────────────────────────────────────────────────────

R2cFft::R2cFft(int n0, int n1, int n2)
    : n0_(n0), n1_(n1), n2_(n2), n2c_(n2 / 2 + 1),
      ntot_(static_cast<long>(n0) * n1 * n2),
      ncplx_(static_cast<long>(n0) * n1 * n2c_) {
    if (n0 <= 0 || n1 <= 0 || n2 <= 0)
        throw std::invalid_argument("R2cFft: non-positive grid shape");
    r_.resize(static_cast<std::size_t>(ntot_));
    c_.resize(static_cast<std::size_t>(ncplx_));
    std::lock_guard<std::mutex> lock(fftw_plan_mutex());
    ensure_fftw_threads();
    fwd_ = fftw_plan_dft_r2c_3d(n0_, n1_, n2_, r(), c(), FFTW_ESTIMATE);
    bwd_ = fftw_plan_dft_c2r_3d(n0_, n1_, n2_, c(), r(), FFTW_ESTIMATE);
    if (!fwd_ || !bwd_) {
        if (fwd_)
            fftw_destroy_plan(fwd_);
        if (bwd_)
            fftw_destroy_plan(bwd_);
        fwd_ = bwd_ = nullptr;
        throw std::runtime_error("R2cFft: fftw_plan_dft_* failed");
    }
}

R2cFft::~R2cFft() {
    std::lock_guard<std::mutex> lock(fftw_plan_mutex());
    if (fwd_)
        fftw_destroy_plan(fwd_);
    if (bwd_)
        fftw_destroy_plan(bwd_);
    fwd_ = bwd_ = nullptr;
}

double* R2cFft::r2() {
    if (r2_.size() != static_cast<std::size_t>(ntot_))
        r2_.resize(static_cast<std::size_t>(ntot_));
    return r2_.data();
}

fftw_complex* R2cFft::c2() {
    if (c2_.size() != static_cast<std::size_t>(ncplx_))
        c2_.resize(static_cast<std::size_t>(ncplx_));
    return c2_.data();
}

void R2cFft::execute_c2r_alt() {
    fftw_execute_dft_c2r(bwd_, c2(), r2());
}

R2cFft& R2cFft::for_shape(int n0, int n1, int n2) {
    // Thread-local map: each OpenMP thread owns its plans + scratch (no
    // buffer races). First touch of a shape pays plan cost once.
    thread_local std::map<ShapeKey, R2cFft*> cache;
    const ShapeKey key{n0, n1, n2};
    auto it = cache.find(key);
    if (it != cache.end())
        return *it->second;
    // Leaked deliberately until thread exit: plans must outlive all GPW calls
    // on this thread; process teardown destroys FFTW state anyway.
    auto* p = new R2cFft(n0, n1, n2);
    cache.emplace(key, p);
    return *p;
}

// ── C2cFft ──────────────────────────────────────────────────────────────────

C2cFft::C2cFft(int n0, int n1, int n2)
    : n0_(n0), n1_(n1), n2_(n2),
      ntot_(static_cast<long>(n0) * n1 * n2) {
    if (n0 <= 0 || n1 <= 0 || n2 <= 0)
        throw std::invalid_argument("C2cFft: non-positive grid shape");
    in_.resize(static_cast<std::size_t>(ntot_));
    out_.resize(static_cast<std::size_t>(ntot_));
    std::lock_guard<std::mutex> lock(fftw_plan_mutex());
    ensure_fftw_threads();
    fwd_ = fftw_plan_dft_3d(n0_, n1_, n2_, in(), out(), FFTW_FORWARD,
                            FFTW_ESTIMATE);
    bwd_ = fftw_plan_dft_3d(n0_, n1_, n2_, in(), out(), FFTW_BACKWARD,
                            FFTW_ESTIMATE);
    if (!fwd_ || !bwd_) {
        if (fwd_)
            fftw_destroy_plan(fwd_);
        if (bwd_)
            fftw_destroy_plan(bwd_);
        fwd_ = bwd_ = nullptr;
        throw std::runtime_error("C2cFft: fftw_plan_dft_3d failed");
    }
}

C2cFft::~C2cFft() {
    std::lock_guard<std::mutex> lock(fftw_plan_mutex());
    if (fwd_)
        fftw_destroy_plan(fwd_);
    if (bwd_)
        fftw_destroy_plan(bwd_);
    fwd_ = bwd_ = nullptr;
}

C2cFft& C2cFft::for_shape(int n0, int n1, int n2) {
    thread_local std::map<ShapeKey, C2cFft*> cache;
    const ShapeKey key{n0, n1, n2};
    auto it = cache.find(key);
    if (it != cache.end())
        return *it->second;
    auto* p = new C2cFft(n0, n1, n2);
    cache.emplace(key, p);
    return *p;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV

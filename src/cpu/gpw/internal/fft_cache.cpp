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
// execute of MEASURE plans is fine concurrently (and under OpenMP FFTW threads).
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

// Prefer MEASURE (timed once per shape) over ESTIMATE for production SCF grids.
// MEASURE overwrites the planning buffers — only call after buffers exist and
// before any physics content is written.
constexpr unsigned kPlanFlags = FFTW_MEASURE;
// howmany=3 multi-out plans are larger and only used on GGA paths; ESTIMATE
// avoids a second cold MEASURE tax when LDA/Poisson already paid for r2c/c2r.
// Per-execute cost is essentially identical for these fused 3-transforms on
// production grids (verified: no wall regression on Padé-only twins that never
// touch many_*).
constexpr unsigned kManyPlanFlags = FFTW_ESTIMATE;

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
    // Single-transform plans only at first shape touch. howmany=3 multi-out
    // plans (GGA gradient / divergence) are built lazily on first many_* use
    // so LDA / MT Poisson twins (UKS O₂ Padé) do not pay 2× MEASURE cold cost.
    std::lock_guard<std::mutex> lock(fftw_plan_mutex());
    ensure_fftw_threads();
    fwd_ = fftw_plan_dft_r2c_3d(n0_, n1_, n2_, r(), c(), kPlanFlags);
    bwd_ = fftw_plan_dft_c2r_3d(n0_, n1_, n2_, c(), r(), kPlanFlags);
    if (!fwd_ || !bwd_) {
        if (fwd_)
            fftw_destroy_plan(fwd_);
        if (bwd_)
            fftw_destroy_plan(bwd_);
        fwd_ = bwd_ = nullptr;
        throw std::runtime_error("R2cFft: fftw_plan_dft_* failed");
    }
}

// Lazy howmany=3 plans + buffers for GGA paths only (see R2cFft ctor).
void R2cFft::ensure_many_plans() {
    if (many_fwd_ && many_bwd_)
        return;
    r_many_.resize(static_cast<std::size_t>(3 * ntot_));
    c_many_.resize(static_cast<std::size_t>(3 * ncplx_));
    std::lock_guard<std::mutex> lock(fftw_plan_mutex());
    if (many_fwd_ && many_bwd_)
        return; // another caller won the race under the lock
    ensure_fftw_threads();
    const int n[3] = {n0_, n1_, n2_};
    // Use raw buffer pointers — never r_many()/c_many() here: those call
    // ensure_many_plans() and would re-enter fftw_plan_mutex (non-recursive).
    double* rbuf = r_many_.data();
    fftw_complex* cbuf = c_many_.data();
    if (!many_fwd_)
        many_fwd_ = fftw_plan_many_dft_r2c(
            /*rank=*/3, n, /*howmany=*/3, rbuf, /*inembed=*/nullptr,
            /*istride=*/1, /*idist=*/static_cast<int>(ntot_), cbuf,
            /*onembed=*/nullptr, /*ostride=*/1,
            /*odist=*/static_cast<int>(ncplx_), kManyPlanFlags);
    if (!many_bwd_)
        many_bwd_ = fftw_plan_many_dft_c2r(
            /*rank=*/3, n, /*howmany=*/3, cbuf, /*inembed=*/nullptr,
            /*istride=*/1, /*idist=*/static_cast<int>(ncplx_), rbuf,
            /*onembed=*/nullptr, /*ostride=*/1,
            /*odist=*/static_cast<int>(ntot_), kManyPlanFlags);
    if (!many_fwd_ || !many_bwd_) {
        if (many_fwd_)
            fftw_destroy_plan(many_fwd_);
        if (many_bwd_)
            fftw_destroy_plan(many_bwd_);
        many_fwd_ = many_bwd_ = nullptr;
        throw std::runtime_error("R2cFft: fftw_plan_many_dft_* failed");
    }
}

R2cFft::~R2cFft() {
    std::lock_guard<std::mutex> lock(fftw_plan_mutex());
    if (fwd_)
        fftw_destroy_plan(fwd_);
    if (bwd_)
        fftw_destroy_plan(bwd_);
    if (many_fwd_)
        fftw_destroy_plan(many_fwd_);
    if (many_bwd_)
        fftw_destroy_plan(many_bwd_);
    fwd_ = bwd_ = many_fwd_ = many_bwd_ = nullptr;
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

double* R2cFft::r_many() {
    ensure_many_plans();
    return r_many_.data();
}

fftw_complex* R2cFft::c_many() {
    ensure_many_plans();
    return c_many_.data();
}

void R2cFft::execute_c2r_alt() {
    fftw_execute_dft_c2r(bwd_, c2(), r2());
}

void R2cFft::execute_many_r2c() {
    ensure_many_plans();
    fftw_execute_dft_r2c(many_fwd_, r_many_.data(), c_many_.data());
}

void R2cFft::execute_many_c2r() {
    ensure_many_plans();
    fftw_execute_dft_c2r(many_bwd_, c_many_.data(), r_many_.data());
}

R2cFft& R2cFft::for_shape(int n0, int n1, int n2) {
    // Thread-local map: each OpenMP thread owns its plans + scratch (no
    // buffer races). First touch of a shape pays MEASURE plan cost once.
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
                            kPlanFlags);
    bwd_ = fftw_plan_dft_3d(n0_, n1_, n2_, in(), out(), FFTW_BACKWARD,
                            kPlanFlags);
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

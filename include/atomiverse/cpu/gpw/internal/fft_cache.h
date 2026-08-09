/// @file cpu/gpw/internal/fft_cache.h
/// @brief Thread-local FFTW plan + r/c buffer cache keyed by grid shape.
///
/// GPW Poisson / GGA / multigrid previously called `fftw_plan_*` + `fftw_malloc`
/// on every transform. Geometry is fixed for a `run()`, and SCF reuses the same
/// few grid shapes (primary + multigrid levels). This cache mirrors the
/// plane-wave `FFTPlan3D` / `FFTWBuffer` pattern without a public API: one
/// r2c/c2r (or c2c) plan pair per thread per `(n0,n1,n2)`, with reusable
/// aligned scratch. Plan creation is mutex-guarded (FFTW planner is not
/// thread-safe); execute uses new-array FFTW APIs so multi-alpha callers can
/// point at their own buffers.
///
/// Plans use FFTW_MEASURE (timed once at first shape touch). Single r2c/c2r
/// plans are created eagerly; GGA howmany=3 multi-out plans are **lazy** on
/// first many_* use so LDA/Poisson-only twins skip that cold MEASURE cost.
#pragma once

#include <cstddef>
#include <new>
#include <fftw3.h>

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief RAII `fftw_malloc` buffer (SIMD-aligned). Idempotent `resize`.
/// @tparam T Element type (`double` or `fftw_complex`).
template <typename T>
class FftwBuffer {
    T* p_ = nullptr;
    std::size_t n_ = 0;

  public:
    FftwBuffer() = default;
    explicit FftwBuffer(std::size_t n) { resize(n); }
    ~FftwBuffer() {
        if (p_)
            fftw_free(p_);
    }
    FftwBuffer(const FftwBuffer&) = delete;
    FftwBuffer& operator=(const FftwBuffer&) = delete;
    FftwBuffer(FftwBuffer&& o) noexcept : p_(o.p_), n_(o.n_) {
        o.p_ = nullptr;
        o.n_ = 0;
    }
    FftwBuffer& operator=(FftwBuffer&& o) noexcept {
        if (this != &o) {
            if (p_)
                fftw_free(p_);
            p_ = o.p_;
            n_ = o.n_;
            o.p_ = nullptr;
            o.n_ = 0;
        }
        return *this;
    }

    /// @brief Ensure capacity for `n` elements (no-op when unchanged).
    void resize(std::size_t n) {
        if (n == n_)
            return;
        if (p_)
            fftw_free(p_);
        p_ = static_cast<T*>(fftw_malloc(sizeof(T) * n));
        if (!p_ && n > 0)
            throw std::bad_alloc();
        n_ = n;
    }
    T* data() { return p_; }
    const T* data() const { return p_; }
    std::size_t size() const { return n_; }
};

/// @brief Cached r2c + c2r plans and primary scratch for shape `(n0,n1,n2)`.
///
/// Layout matches existing GPW callers: `fftw_plan_dft_r2c_3d(n0,n1,n2,…)` with
/// real length `n0*n1*n2` and Hermitian length `n0*n1*(n2/2+1)`. Extra `r2`/`c2`
/// slots support the GGA gradient path (separate work arrays). Multi-out
/// (`howmany=3`) buffers/plans fuse Cartesian gradient/divergence FFTs.
class R2cFft {
  public:
    /// @brief Thread-local instance for this shape (created on first use).
    static R2cFft& for_shape(int n0, int n1, int n2);

    int n0() const { return n0_; }
    int n1() const { return n1_; }
    int n2() const { return n2_; }
    int n2c() const { return n2c_; }
    long ntot() const { return ntot_; }
    long ncplx() const { return ncplx_; }

    double* r() { return r_.data(); }
    fftw_complex* c() { return c_.data(); }
    /// @brief Second real buffer (lazy); used as c2r destination in gradient.
    double* r2();
    /// @brief Second complex buffer (lazy); used as spectral work in gradient.
    fftw_complex* c2();

    /// @brief Contiguous howmany=3 real batch: layout `[d][i]` with
    ///        `idist = ntot`. Allocated on first multi-out use.
    double* r_many();
    /// @brief Contiguous howmany=3 complex batch: layout `[d][k]` with
    ///        `odist = ncplx`.
    fftw_complex* c_many();

    /// @brief Forward r2c on the primary buffers: `r` → `c`.
    void execute_r2c() { fftw_execute_dft_r2c(fwd_, r(), c()); }
    /// @brief Backward c2r on the primary buffers: `c` → `r`.
    void execute_c2r() { fftw_execute_dft_c2r(bwd_, c(), r()); }
    /// @brief Backward c2r: `c2` → `r2` (gradient / multi-buffer paths).
    void execute_c2r_alt();
    /// @brief Forward r2c on caller-owned arrays of matching size/alignment.
    void execute_r2c(double* in, fftw_complex* out) {
        fftw_execute_dft_r2c(fwd_, in, out);
    }
    /// @brief Backward c2r on caller-owned arrays of matching size/alignment.
    void execute_c2r(fftw_complex* in, double* out) {
        fftw_execute_dft_c2r(bwd_, in, out);
    }
    /// @brief howmany=3 forward: `r_many` → `c_many` (divergence inputs).
    void execute_many_r2c();
    /// @brief howmany=3 backward: `c_many` → `r_many` (gradient outputs).
    void execute_many_c2r();

    /// @brief Build the howmany=3 plans and their 3×-sized buffers (idempotent).
    /// Call during SCF setup for GGA so the first KS does not pay the cold
    /// buffer allocation inside the OT/DIIS wall.
    void ensure_many_plans();

  private:
    R2cFft(int n0, int n1, int n2);
    ~R2cFft();
    R2cFft(const R2cFft&) = delete;
    R2cFft& operator=(const R2cFft&) = delete;

    int n0_ = 0, n1_ = 0, n2_ = 0, n2c_ = 0;
    long ntot_ = 0, ncplx_ = 0;
    fftw_plan fwd_ = nullptr;
    fftw_plan bwd_ = nullptr;
    fftw_plan many_fwd_ = nullptr;
    fftw_plan many_bwd_ = nullptr;
    FftwBuffer<double> r_;
    FftwBuffer<fftw_complex> c_;
    FftwBuffer<double> r2_;
    FftwBuffer<fftw_complex> c2_;
    FftwBuffer<double> r_many_;
    FftwBuffer<fftw_complex> c_many_;
};

/// @brief Cached out-of-place c2c forward/backward plans for multigrid DFT.
class C2cFft {
  public:
    /// @brief Thread-local instance for this shape (created on first use).
    static C2cFft& for_shape(int n0, int n1, int n2);

    int n0() const { return n0_; }
    int n1() const { return n1_; }
    int n2() const { return n2_; }
    long ntot() const { return ntot_; }

    fftw_complex* in() { return in_.data(); }
    fftw_complex* out() { return out_.data(); }

    /// @brief Forward c2c: `in` → `out` (FFTW_FORWARD, unnormalized).
    void execute_fwd() { fftw_execute_dft(fwd_, in(), out()); }
    /// @brief Backward c2c: `in` → `out` (FFTW_BACKWARD, unnormalized).
    void execute_bwd() { fftw_execute_dft(bwd_, in(), out()); }

  private:
    C2cFft(int n0, int n1, int n2);
    ~C2cFft();
    C2cFft(const C2cFft&) = delete;
    C2cFft& operator=(const C2cFft&) = delete;

    int n0_ = 0, n1_ = 0, n2_ = 0;
    long ntot_ = 0;
    fftw_plan fwd_ = nullptr;
    fftw_plan bwd_ = nullptr;
    FftwBuffer<fftw_complex> in_;
    FftwBuffer<fftw_complex> out_;
};

} // namespace gpw
} // namespace cpu
} // namespace ATV

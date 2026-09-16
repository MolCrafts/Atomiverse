/// @file cpu/planewave/internal/fft.h
/// @brief RAII wrapper around FFTW3 3D transforms.
#pragma once

#include "atomiverse/core/precision.h"
#include <cstddef>
#include <fftw3.h>
#include <new>
#include <stdexcept>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

class FFTPlan3D {
    fftw_plan fwd_ = nullptr;    // r2c: real → complex (Hermitian)
    fftw_plan bwd_ = nullptr;    // c2r: complex (Hermitian) → real
    fftw_plan fwd_cc_ = nullptr; // c2c forward: complex → complex
    fftw_plan bwd_cc_ = nullptr; // c2c backward: complex → complex
    int nx_, ny_, nz_;
    int nr_, nc_; // real size, complex (half) size
    AccReal* rbuf_ = nullptr;
    fftw_complex* cbuf_ = nullptr;
    fftw_complex* cbuf2_ = nullptr; // for c2c
  public:
    FFTPlan3D(int nx, int ny, int nz);
    ~FFTPlan3D();
    FFTPlan3D(const FFTPlan3D&) = delete;
    FFTPlan3D& operator=(const FFTPlan3D&) = delete;

    void r2c(const AccReal* in, fftw_complex* out);
    void c2r(const fftw_complex* in, AccReal* out);
    void c2c_fwd(fftw_complex* inout); // in-place c2c forward
    void c2c_bwd(fftw_complex* inout); // in-place c2c backward
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    int nr() const { return nr_; }
    int nc() const { return nc_; }
};

/// @brief RAII `fftw_malloc` buffer (SIMD-aligned), for pre-allocated
///        step-helper scratch. `resize()` is idempotent for equal sizes,
///        so `prepare()` can be re-invoked across geometry changes without
///        reallocating when the grid is unchanged.
/// @tparam T Element type (`AccReal` or `fftw_complex`).
template <typename T>
class FFTWBuffer {
    T* p_ = nullptr;
    std::size_t n_ = 0;

  public:
    FFTWBuffer() = default;
    explicit FFTWBuffer(std::size_t n) { resize(n); }
    ~FFTWBuffer() {
        if (p_)
            fftw_free(p_);
    }
    FFTWBuffer(const FFTWBuffer&) = delete;
    FFTWBuffer& operator=(const FFTWBuffer&) = delete;
    FFTWBuffer(FFTWBuffer&& o) noexcept : p_(o.p_), n_(o.n_) {
        o.p_ = nullptr;
        o.n_ = 0;
    }
    FFTWBuffer& operator=(FFTWBuffer&& o) noexcept {
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
        if (!p_)
            throw std::bad_alloc();
        n_ = n;
    }
    T* data() { return p_; }
    const T* data() const { return p_; }
    std::size_t size() const { return n_; }
};

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV

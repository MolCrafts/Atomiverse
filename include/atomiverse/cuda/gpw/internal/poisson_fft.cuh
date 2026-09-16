/// @file cuda/gpw/internal/poisson_fft.cuh
/// @brief Γ-point periodic FFT Poisson on the GPW uniform grid — device path.
///
/// Thin CUDA foothold mirroring `cpu::gpw::solve_poisson`: r2c → V(G)=4πρ(G)/|G|²
/// (G=0 zeroed) → c2r / N, plus Hartree energy 0.5 w Σ ρ V. Layout matches the
/// CPU FFTW plan `fftw_plan_dft_r2c_3d(n0,n1,n2)` (last axis contiguous /
/// Hermitian-halved). All scientific work stays on device; host only owns the
/// plan object and optional final D2H of results.
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

/// @brief Orthorhombic real-space grid descriptor (Bohr lengths).
struct GpwGridDesc {
    int n[3] = {0, 0, 0};    ///< points per axis (n0, n1, n2).
    double L[3] = {0, 0, 0}; ///< box lengths [Bohr].

    /// @brief Total real-space points.
    long npts() const {
        return static_cast<long>(n[0]) * n[1] * n[2];
    }
    /// @brief Hermitian half-complex length along the fast axis.
    int n2c() const { return n[2] / 2 + 1; }
    /// @brief Half-complex buffer length = n0 * n1 * n2c.
    long ncplx() const {
        return static_cast<long>(n[0]) * n[1] * n2c();
    }
    /// @brief Integration weight w = V_cell / npts [Bohr³].
    double weight() const {
        return (L[0] * L[1] * L[2]) / static_cast<double>(npts());
    }
};

/// @brief Device-resident Γ-point periodic Poisson solver for one grid shape.
///
/// Plans and scratch are owned here; density / potential buffers are caller-
/// owned device pointers. Matches `cpu::gpw::solve_poisson` conventions:
/// unnormalized r2c/c2r, physical G_d = 2π m_d / L_d, V_hat(0)=0, divide by N
/// after c2r, e_H = 0.5 w Σ_g ρ_g V_g.
class PoissonFftDevice {
    int n0_ = 0, n1_ = 0, n2_ = 0, n2c_ = 0;
    long ntot_ = 0, ncplx_ = 0;
    double L_[3] = {0, 0, 0};
    double weight_ = 0.0;
    double inv_n_ = 0.0;
    double gx_unit_ = 0.0, gy_unit_ = 0.0, gz_unit_ = 0.0;

    cufftHandle r2c_ = 0;
    cufftHandle c2r_ = 0;
    cudaStream_t stream_ = nullptr;
    cufftDoubleComplex* d_cbuf_ = nullptr; ///< half-complex work (r2c out / c2r in)
    cufftDoubleComplex* d_c_scratch_ = nullptr; ///< preserves c2r input
    double* d_e_scratch_ = nullptr;        ///< single-slot Hartree energy

  public:
    /// @brief Build cuFFT plans + scratch for grid shape / cell.
    /// @param grid Orthorhombic grid (n, L in Bohr).
    /// @param stream Stream for plans and kernels (default stream if null).
    explicit PoissonFftDevice(const GpwGridDesc& grid,
                              cudaStream_t stream = nullptr);
    ~PoissonFftDevice();

    PoissonFftDevice(const PoissonFftDevice&) = delete;
    PoissonFftDevice& operator=(const PoissonFftDevice&) = delete;

    /// @brief Solve ∇²V = −4πρ on device.
    /// @param d_rho Device density, length npts [a.u.]. Must not alias `d_v`
    ///        (Hartree energy needs ρ after the c2r write into `d_v`).
    /// @param d_v   Device potential out, length npts [Hartree].
    /// @param d_e_hartree Optional device scalar (1 double) for e_H [Hartree];
    ///        if null, energy is written only to the internal scratch and may
    ///        be read via `last_e_hartree_device()`.
    void solve(const double* d_rho, double* d_v,
               double* d_e_hartree = nullptr);

    /// @brief Device pointer to last computed e_hartree (owned by this object).
    const double* last_e_hartree_device() const { return d_e_scratch_; }

    int n0() const { return n0_; }
    int n1() const { return n1_; }
    int n2() const { return n2_; }
    int n2c() const { return n2c_; }
    long ntot() const { return ntot_; }
    long ncplx() const { return ncplx_; }
    double weight() const { return weight_; }
    cudaStream_t stream() const { return stream_; }
};

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV

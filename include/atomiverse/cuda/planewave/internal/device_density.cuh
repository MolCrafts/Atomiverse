/// @file cuda/planewave/internal/device_density.cuh
/// @brief Device-resident density-side state of the plane-wave SCF —
///        ρ(G), V_H, V_xc, V_eff, the Pulay history and the per-iteration
///        energy/residual reductions.
///
/// Owned by `cuda::pw::PW`, never registered in any Driver Plan (same
/// internal-helper contract as `cuda::scf::internal::*`). One method per CPU
/// stage helper — the SCF order is composed by `PW::run`, not hidden here.
///
/// Every buffer is sized ONCE from the density sphere `prob.gv0.count()` and
/// the FFT grid `ws.nr` / `ws.nc`. It never reads `ws.n_g`, which the SCF loop
/// rebinds to the wavefunction sphere around every band solve.
///
/// After construction the host `PWWorkspace` arrays (rho_G, V_H_G, V_xc_G,
/// V_eff_r, …) are NEVER updated again: the host copies hold the initial
/// atomic density only. The host side of the loop may read `ws` solely for
/// per-k table binding and for `ws.E_xc` (written back from `reduce()` so the
/// shared `log_iteration_energy` formatter keeps working).
///
/// Stream: construction creates one non-default `cudaStream_t` and runs every
/// density-side kernel, cuFFT transform and cuBLAS call on it. The band
/// solver borrows the same stream via `stream()` so density and band work
/// stay strictly ordered without events. Host decisions (Pulay coefficients,
/// SCF residual) synchronize that stream only — never the default stream.
///
/// NUMERICAL PARITY CONTRACT vs `cpu::pw::PW`
/// ------------------------------------------
/// The device path is algorithmically identical to the CPU stage helpers
/// (`PoissonSolver`, `XCBuilder` with `XCType::LSDA` or `XCType::PBE`,
/// `VeffBuilder`, `DensityBuilder::finish`, `pulay_mix`, `ConvergenceCheck`,
/// `iteration_energy`), including every cutoff, floor and G = 0 special case.
/// It is deliberately NOT bit-exact against the CPU:
///   * cuFFT and FFTW use different butterfly orders, so ρ(r) / V_eff(r) and
///     everything downstream differ at the ~1e-15 relative level.
///   * Reductions here are fixed-geometry trees; the CPU sums serially. Tree
///     summation carries √N·ε error against the CPU's N·ε, i.e. the device is
///     slightly MORE accurate, not less.
///   * libdevice and libm transcendentals differ by ≲ 1 ulp (see xc_lda.cu).
/// Each device reduction IS bitwise reproducible run-to-run for a fixed
/// `nr` / `n_g`, because the launch geometry and therefore the summation
/// partition are fixed constants, not functions of the problem size.
/// The one catastrophic-cancellation site of the energy expression,
/// `e_one = e_band − 2·e_h − e_xc_int`, stays on the host in the CPU's exact
/// arithmetic order and is never reassociated on device.
#pragma once

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include "atomiverse/cuda/planewave/internal/fft.cuh"
#include <complex>
#include <cublas_v2.h>
#include <cufft.h>

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

/// @brief Which device G-sphere array a diagnostic readback should fetch.
enum class GField {
    rho,       ///< ρ(G) [Bohr⁻³]
    v_hartree, ///< V_H(G) [Hartree]
    v_xc,      ///< V_xc(G) [Hartree]
    v_loc,     ///< V_loc(G) incl. structure factor [Hartree]
    v_eff,     ///< V_eff(G) [Hartree]
};

/// @brief Device-resident density-side buffers and stages of one PW SCF run.
class DeviceDensity {
  public:
    /// @brief Per-iteration scalars produced by the device reductions.
    struct IterationScalars {
        double e_xc = 0;     ///< E_xc = Σ_r ρ ε_xc dV [Hartree].
        double e_h = 0;      ///< Ω Σ_G ½ Re[ρ*(G) V_H(G)] [Hartree].
        double e_xc_int = 0; ///< Ω Σ_G Re[ρ*(G) V_xc(G)] [Hartree].
        double e_loc = 0;    ///< Ω Σ_G Re[ρ*(G) V_loc(G)] [Hartree].
        double resid = 0;    ///< ‖ρ − ρ_old‖₂ after mixing [Bohr⁻³].
    };

    /// @brief Allocate and seed every density-side device buffer.
    ///
    /// Uploads (once per run, never per iteration): the density-sphere Miller
    /// indices h/k/l, |G|², the complex V_loc(G), the atomic ρ(G) seed from
    /// `prob.ws.rho_G`, and — for a GGA functional — the Cartesian G
    /// components feeding the spectral ∇ρ / divergence passes.
    ///
    /// @param prob        Assembled problem from `setup_pw_problem` (must
    ///                    outlive this object; `ws.gvec` aliases `gv0`).
    /// @param mix_beta    Density mixing parameter (dimensionless).
    /// @param pulay_depth Pulay history window ≥ 1 (dimensionless).
    /// @param xc          Functional `exchange_correlation()` evaluates. Only
    ///                    `XCType::LSDA` and `XCType::PBE` have device
    ///                    kernels; anything else throws (no fallback).
    /// @throws std::invalid_argument if `pulay_depth < 1`, the problem has
    ///         an empty G-sphere / FFT grid, or `xc` has no device kernel.
    /// @throws std::runtime_error on any CUDA / cuFFT / cuBLAS failure.
    DeviceDensity(const cpu::pw::internal::PWProblem& prob, double mix_beta,
                  int pulay_depth, cpu::scf::XCType xc);
    ~DeviceDensity();
    DeviceDensity(const DeviceDensity&) = delete;
    DeviceDensity& operator=(const DeviceDensity&) = delete;

    /// @brief V_H(G) = 4π ρ(G)/|G|², G = 0 zeroed [Hartree].
    ///        Mirrors `cpu::pw::internal::PoissonSolver::step`, same
    ///        |G|² < 1e-16 floor.
    void hartree();

    /// @brief ρ(G)→ρ(r), V_xc(r)/ε_xc(r), E_xc, then V_xc(G) [Hartree].
    ///        Mirrors `cpu::pw::internal::XCBuilder::step(ws, xc)` for the
    ///        constructor's functional, including the ρ < 1e-12 cutoff and
    ///        zeroing Im[V_xc(G = 0)]. For `XCType::PBE` this adds the CPU's
    ///        GGA passes in the same order: spectral ∇ρ from i G ρ(G), the
    ///        pointwise PBE evaluation with flux assembly, and
    ///        v_xc −= ∇·flux via three r2c + one c2r transform.
    void exchange_correlation();

    /// @brief V_eff(G) = V_loc + V_H + V_xc, then the UNNORMALIZED c2r to
    ///        V_eff(r) [Hartree]. Mirrors `VeffBuilder::step`.
    void assemble_veff();

    /// @brief Zero the real-space density accumulator the band solver fills.
    ///        Mirrors `DensityBuilder::begin`.
    void zero_accumulator();

    /// @brief ρ_old ← ρ (device-to-device), then ρ_acc(r) --r2c--> ρ(G)
    ///        scaled by 1/nr. Mirrors the first half of
    ///        `DensityBuilder::finish`.
    void collect();

    /// @brief Pulay/DIIS density mixing on the G-sphere.
    ///
    /// Pushes F = ρ − ρ_old and ρ_old onto a circular history of
    /// `pulay_depth` slots (same sliding-window ordering as the CPU's
    /// front-trimmed vector), builds B_ij = Re⟨F_i|F_j⟩ with one
    /// `cublasZgemm`, solves the augmented Lagrangian system with the SHARED
    /// host helper `cpu::pw::internal::solve_pulay_coefficients`, and
    /// recombines ρ = Σ_i c_i(ρ_in^i + β F^i). Falls back to
    /// ρ = ρ_old + β F when the history holds fewer than 2 entries, exactly
    /// like the CPU.
    ///
    /// @param iter Zero-based SCF iteration index (selects the ring slot).
    void mix(int iter);

    /// @brief Device reductions producing every per-iteration scalar.
    ///
    /// Reads the MIXED ρ(G) (call after `mix()`), reproducing the CPU order
    /// `pulay_mix` → `ConvergenceCheck::step` → `iteration_energy`. `e_xc`
    /// is the value `exchange_correlation()` computed from this iteration's
    /// INPUT density, matching `ws.E_xc` on the CPU.
    ///
    /// @param vol_bohr Cell volume Ω [Bohr³].
    /// @return Energy terms [Hartree] and the density-change L2 norm.
    IterationScalars reduce(double vol_bohr);

    /// @brief Real-space density accumulator ρ_acc(r), nr doubles (device).
    ///        The band solver writes here; it never frees the buffer.
    double* density_accumulator() { return d_rho_r_acc_; }

    /// @brief Real-space effective potential V_eff(r), nr doubles (device).
    ///        The band solver reads here; it never frees the buffer.
    const double* effective_potential() const { return d_V_eff_r_; }

    /// @brief Diagnostic readback of one G-sphere array (device-to-host).
    ///        Never called inside the SCF loop — tests and diagnostics only.
    /// @param which Field selector.
    /// @param out   Host buffer of `prob.gv0.count()` complex doubles.
    void fetch_g(GField which, std::complex<double>* out) const;

    /// @brief Diagnostic readback of V_eff(r) (device-to-host).
    ///        Never called inside the SCF loop — tests and diagnostics only.
    /// @param out Host buffer of `ws.nr` doubles [Hartree].
    void fetch_veff_r(double* out) const;

    /// @brief Non-default CUDA stream every density stage executes on.
    ///
    /// Never null after a successful constructor. Pass to
    /// `DeviceBandSolver` so the band side shares the same serial chain.
    /// @return Owned stream handle (destroyed in the destructor).
    cudaStream_t stream() const { return stream_; }

  private:
    void free_all();

    int ng0_ = 0, nr_ = 0, nc_ = 0;
    int depth_ = 0;
    double mix_beta_ = 0;
    double omega_ = 0;
    cpu::scf::XCType xc_ = cpu::scf::XCType::LSDA;
    bool gga_ = false;
    cudaStream_t stream_ = nullptr; ///< Non-default; owned.
    FFTPlan3D fft_;
    cublasHandle_t blas_ = nullptr;

    // Density-sphere arrays (ng0_)
    int* d_h_ = nullptr;
    int* d_k_ = nullptr;
    int* d_l_ = nullptr;
    double* d_g2_ = nullptr;
    cufftDoubleComplex* d_rho_G_ = nullptr;
    cufftDoubleComplex* d_rho_G_old_ = nullptr;
    cufftDoubleComplex* d_V_H_G_ = nullptr;
    cufftDoubleComplex* d_V_xc_G_ = nullptr;
    cufftDoubleComplex* d_V_loc_G_ = nullptr;
    cufftDoubleComplex* d_V_eff_G_ = nullptr;

    // Pulay history ring (depth_ × ng0_ each) + Gram / coefficients
    cufftDoubleComplex* d_res_hist_ = nullptr;
    cufftDoubleComplex* d_rho_hist_ = nullptr;
    cufftDoubleComplex* d_gram_ = nullptr; ///< depth_ × depth_, column-major.
    double* d_coeff_ = nullptr;            ///< depth_ mixing coefficients.

    // Grids
    cufftDoubleComplex* d_half_ = nullptr; ///< nc_ half-complex staging.
    double* d_rho_r_in_ = nullptr;         ///< ρ(r) input to XC.
    double* d_rho_r_acc_ = nullptr;        ///< band-solver accumulator.
    double* d_V_xc_r_ = nullptr;
    double* d_exc_r_ = nullptr;
    double* d_V_eff_r_ = nullptr;

    // GGA-only scratch (allocated only when is_gga(xc_); else null).
    double* d_gx_ = nullptr; ///< Cartesian G_x per sphere entry [1/Bohr].
    double* d_gy_ = nullptr; ///< Cartesian G_y per sphere entry [1/Bohr].
    double* d_gz_ = nullptr; ///< Cartesian G_z per sphere entry [1/Bohr].
    cufftDoubleComplex* d_sph_ = nullptr; ///< ng0_ G-sphere staging
                                          ///< (i·G_d·ρ and flux_d(G)).
    double* d_grad_x_r_ = nullptr; ///< ∂ρ/∂x, overwritten with flux_x.
    double* d_grad_y_r_ = nullptr; ///< ∂ρ/∂y, overwritten with flux_y.
    double* d_grad_z_r_ = nullptr; ///< ∂ρ/∂z, overwritten with flux_z.
    double* d_div_r_ = nullptr;    ///< ∇·flux on the real grid.

    // Reduction scratch and outputs
    double* d_partials_ = nullptr;    ///< 4 × kReduceBlocks stage-1 partials.
    double* d_xc_partials_ = nullptr; ///< kXcEnergyPartials for xc_energy.
    double* d_scalars_ = nullptr;     ///< 5 doubles: E_xc, e_h, e_xc_int,
                                      ///< e_loc, resid.
};

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV

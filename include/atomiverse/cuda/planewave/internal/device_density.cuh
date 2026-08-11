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
    /// @param spin_polarized Collinear UKS (nspin=2). Supports `XCType::LSDA`
    ///                    and `XCType::PBE` (true spin-GGA); seeds ρ_↑/ρ_↓
    ///                    from the atomic total density with fractions
    ///                    n_alpha/n_elec and n_beta/n_elec.
    /// @param n_alpha     Occupied ↑ bands (spin path only; may be 0).
    /// @param n_beta      Occupied ↓ bands (spin path only; may be 0).
    /// @param n_elec      Valence electron count used for the spin seed.
    /// @throws std::invalid_argument if `pulay_depth < 1`, the problem has
    ///         an empty G-sphere / FFT grid, or `xc` has no device kernel.
    /// @throws std::runtime_error on any CUDA / cuFFT / cuBLAS failure.
    DeviceDensity(const cpu::pw::internal::PWProblem& prob, double mix_beta,
                  int pulay_depth, cpu::scf::XCType xc,
                  bool spin_polarized = false, int n_alpha = 0,
                  int n_beta = 0, int n_elec = 0);
    ~DeviceDensity();
    DeviceDensity(const DeviceDensity&) = delete;
    DeviceDensity& operator=(const DeviceDensity&) = delete;

    /// @brief V_H(G) = 4π ρ(G)/|G|², G = 0 zeroed [Hartree].
    ///        Mirrors `cpu::pw::internal::PoissonSolver::step`, same
    ///        |G|² < 1e-16 floor. Spin path: call `sync_total_rho()` first.
    void hartree();

    /// @brief ρ(G)→ρ(r), V_xc(r)/ε_xc(r), E_xc, then V_xc(G) [Hartree].
    ///        Mirrors `cpu::pw::internal::XCBuilder::step(ws, xc)` for the
    ///        constructor's functional, including the ρ < 1e-12 cutoff and
    ///        zeroing Im[V_xc(G = 0)]. For `XCType::PBE` this adds the CPU's
    ///        GGA passes in the same order: spectral ∇ρ from i G ρ(G), the
    ///        pointwise PBE evaluation with flux assembly, and
    ///        v_xc −= ∇·flux via three r2c + one c2r transform.
    ///        Restricted path only — collinear uses `exchange_correlation_spin`.
    void exchange_correlation();

    /// @brief Collinear spin XC: ρ_σ(G)→ρ_σ(r), V_xc^σ / ε_xc / E_xc /
    ///        e_xc_int_spin, then V_xc_σ(G). Mirrors `XCBuilder::step_spin`
    ///        for `XCType::LSDA` and true spin-GGA `XCType::PBE` (spectral
    ///        ∇ρ_σ + flux divergence).
    void exchange_correlation_spin();

    /// @brief V_eff(G) = V_loc + V_H + V_xc, then the UNNORMALIZED c2r to
    ///        V_eff(r) [Hartree]. Mirrors `VeffBuilder::step`.
    void assemble_veff();

    /// @brief Collinear: V_eff_σ from V_loc + V_H + V_xc_σ, then V_eff(r).
    /// @param spin 0 = α, 1 = β.
    void assemble_veff_spin(int spin);

    /// @brief Zero the real-space density accumulator the band solver fills.
    ///        Mirrors `DensityBuilder::begin`.
    void zero_accumulator();

    /// @brief ρ_old ← ρ (device-to-device), then ρ_acc(r) --r2c--> ρ(G)
    ///        scaled by 1/nr. Mirrors the first half of
    ///        `DensityBuilder::finish`. Restricted path only.
    void collect();

    /// @brief Collinear: r2c ρ_acc → ρ_out_σ(G) into a scratch slot for spin
    ///        `spin` (does not overwrite mixed ρ_σ yet). Call `mix_spin_linear`
    ///        after both channels are collected.
    void collect_spin_out(int spin);

    /// @brief Collinear linear mix of both spin channels (CPU twin: no Pulay
    ///        on the fixed-occupation collinear molecule path), then
    ///        ρ_total = ρ_↑ + ρ_↓ and residual ‖Δρ_total‖₂.
    void mix_spin_linear();

    /// @brief Collinear: ρ_total(G) = ρ_↑(G) + ρ_↓(G) into d_rho_G_ for
    ///        Poisson / energy. Call at the top of every SCF iteration.
    void sync_total_rho();

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

    /// @brief Collinear energy + residual reductions.
    ///
    /// Hartree / local from total ρ; e_xc_int is the real-space spin integral
    /// (`e_xc_int_spin`); residual is ‖Δρ_total‖₂ after `mix_spin_linear`.
    /// @param vol_bohr Cell volume Ω [Bohr³].
    IterationScalars reduce_spin(double vol_bohr);

    /// @brief True when this object was constructed for collinear UKS.
    bool spin_polarized() const { return spin_; }

    /// @brief Real-space density accumulator ρ_acc(r), nr doubles (device).
    ///        The band solver writes here; it never frees the buffer.
    double* density_accumulator() { return d_rho_r_acc_; }

    /// @brief Real-space effective potential V_eff(r), nr doubles (device).
    ///        The band solver reads here; it never frees the buffer.
    const double* effective_potential() const { return d_V_eff_r_; }

    /// @brief Mixed density ρ(G) on the density sphere (device), ng0 entries.
    ///        Used by analytic ionic forces after SCF; never freed by the
    ///        caller. Valid after the last `mix()` of a completed run.
    const cufftDoubleComplex* device_rho_G() const { return d_rho_G_; }

    /// @brief Density-sphere G-count (`prob.gv0.count()`).
    int n_g0() const { return ng0_; }

    /// @brief Real-space FFT grid point count (`ws.nr`).
    int nr() const { return nr_; }

    /// @brief Cell volume Ω [Bohr³] used by the density side.
    double omega() const { return omega_; }

    /// @brief Diagnostic readback of one G-sphere array (device-to-host).
    ///        Never called inside the SCF loop — tests and diagnostics only.
    /// @param which Field selector.
    /// @param out   Host buffer of `prob.gv0.count()` complex doubles.
    void fetch_g(GField which, std::complex<double>* out) const;

    /// @brief Diagnostic readback of V_eff(r) (device-to-host).
    ///        Never called inside the SCF loop — tests and diagnostics only.
    /// @param out Host buffer of `ws.nr` doubles [Hartree].
    void fetch_veff_r(double* out) const;

    /// @brief Real-space density ρ(r) [Bohr⁻³] via unnormalized c2r of ρ(G).
    ///
    /// Used by Nielsen–Martin cell stress after the reference SCF. Mutates
    /// staging buffers (`d_half_`, `d_rho_r_in_`); not on the SCF hot path.
    ///
    /// @param out Host buffer of `nr()` doubles.
    void fetch_rho_r(double* out);

    /// @brief Overwrite ρ(G) from a real-space density, scaled by `scale`.
    ///
    /// Twin of CPU `load_rho_real_scaled`: ρ'(r) = ρ₀(r)·scale then r2c /
    /// gather with 1/nr. When `nr0 != nr()` falls back to a uniform mean
    /// density (rare under small strain at fixed ecut).
    ///
    /// @param rho_r0 Host real-space density [Bohr⁻³], length `nr0`.
    /// @param nr0    Length of `rho_r0`.
    /// @param scale  Volume remap Ω₀/Ω (dimensionless).
    void load_rho_r_scaled(const double* rho_r0, int nr0, double scale);

    /// @brief Real-space collinear spin density ρ_σ(r) via c2r of ρ_σ(G).
    ///
    /// Used by Nielsen–Martin spin stress after the reference SCF. Mutates
    /// staging buffers; not on the SCF hot path.
    ///
    /// @param spin 0 = α, 1 = β.
    /// @param out  Host buffer of `nr()` doubles [Bohr⁻³].
    /// @throws std::invalid_argument if not a spin density object.
    void fetch_rho_spin_r(int spin, double* out);

    /// @brief Overwrite both spin ρ_σ(G) and total ρ from real-space densities.
    ///
    /// Twin of CPU `load_rho_spin_real_scaled`: ρ'_σ(r) = ρ_σ0(r)·scale then
    /// r2c / gather, then ρ_total = ρ_↑ + ρ_↓.
    ///
    /// @param rho_a0 Host ρ_↑ [Bohr⁻³], length `nr0`.
    /// @param rho_b0 Host ρ_↓ [Bohr⁻³], length `nr0`.
    /// @param nr0    Length of each host buffer.
    /// @param scale  Volume remap Ω₀/Ω (dimensionless).
    /// @throws std::invalid_argument if not a spin density object.
    void load_rho_spin_r_scaled(const double* rho_a0, const double* rho_b0,
                                int nr0, double scale);

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
    bool spin_ = false;
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

    // Collinear spin channels (allocated only when spin_ == true).
    cufftDoubleComplex* d_rho_spin_G_[2] = {nullptr, nullptr};
    cufftDoubleComplex* d_rho_spin_G_old_[2] = {nullptr, nullptr};
    cufftDoubleComplex* d_rho_spin_out_G_[2] = {nullptr, nullptr};
    cufftDoubleComplex* d_V_xc_spin_G_[2] = {nullptr, nullptr};
    double* d_rho_spin_r_[2] = {nullptr, nullptr};
    double* d_V_xc_spin_r_[2] = {nullptr, nullptr};

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
    // Restricted PBE uses the α (unsigned) set only; collinear spin-GGA
    // also allocates the β (dn) gradients / div.
    double* d_gx_ = nullptr; ///< Cartesian G_x per sphere entry [1/Bohr].
    double* d_gy_ = nullptr; ///< Cartesian G_y per sphere entry [1/Bohr].
    double* d_gz_ = nullptr; ///< Cartesian G_z per sphere entry [1/Bohr].
    cufftDoubleComplex* d_sph_ = nullptr; ///< ng0_ G-sphere staging
                                          ///< (i·G_d·ρ and flux_d(G)).
    double* d_grad_x_r_ = nullptr; ///< ∂ρ/∂x or ∂ρ_α/∂x → flux.
    double* d_grad_y_r_ = nullptr;
    double* d_grad_z_r_ = nullptr;
    double* d_div_r_ = nullptr; ///< ∇·flux (restricted or α).
    double* d_grad_dn_x_r_ = nullptr; ///< ∂ρ_β/∂x → flux_β (spin-GGA).
    double* d_grad_dn_y_r_ = nullptr;
    double* d_grad_dn_z_r_ = nullptr;
    double* d_div_dn_r_ = nullptr; ///< ∇·flux_β (spin-GGA).

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

/// @file cpu/planewave/internal/density_builder.h
/// @brief internal::DensityBuilder — ρ from wavefunctions + Pulay mixing.
///
/// Facade-private step helper owned by `cpu::pw::PW`. NOT a
/// `cpu::Component` subclass and never registered into a Plan.
///
/// Restricted densities use Pulay mixing; collinear channels use joint
/// Pulay mixing with a joint residual that retains both spin errors.
#pragma once

#include "atomiverse/cpu/planewave/internal/pw_workspace.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Accumulates ρ(r) over k-points, FFTs to ρ(G), Pulay-mixes.
///
/// Per SCF iteration: `begin()` → one `accumulate_kpoint()` per k-point
/// (with `ws.gvec`/`ws.n_g`/`ws.wfc` set to that k-point) → `finish()`.
/// The real-space accumulation buffer is exposed via `accumulation()`
/// so the k-point-parallel MPI path can Allreduce partial densities
/// before `finish()`.
class DensityBuilder {
  public:
    /// @brief Default construct. Mixing parameters are NOT snapshotted here:
    ///        `finish()` reads the live `ws.mix_beta` / `ws.pulay_depth`, so a
    ///        caller changing them between run()s takes effect.
    DensityBuilder() = default;

    /// @brief Size FFT and joint-spin density scratch to the workspace.
    void prepare(const PWWorkspace& ws);

    /// @brief Clear Pulay history (once per SCF run).
    void reset(PWWorkspace& ws);

    /// @brief Zero the real-space density accumulation buffer.
    void begin(const PWWorkspace& ws);

    /// @brief Add this k-point's band densities: ρ(r) += w·spin_degen·|ψ_b|²/Ω.
    /// @param ws         Workspace with `gvec`/`n_g`/`wfc` at the current k.
    /// @param weight     k-point weight (Monkhorst-Pack).
    /// @param n_bands    Occupied band count.
    /// @param spin_degen Occupation weight per band (2 restricted, 1 collinear).
    void accumulate_kpoint(const PWWorkspace& ws, AccReal weight, int n_bands,
                           AccReal spin_degen = AccReal(2));
    /// @brief Fractional occupations: ρ += w·f_b·|ψ_b|²/Ω with f_b in [0,2].
    void accumulate_kpoint(const PWWorkspace& ws, AccReal weight, int n_bands,
                           const AccReal* occ);

    /// @brief Real-space accumulation buffer (length `ws.nr`) — exposed
    ///        for the MPI k-point Allreduce between accumulation and
    ///        `finish()`.
    AccReal* accumulation() { return rho_r_accum_.data(); }

    /// @brief Save ρ_old, extract the new ρ(G) on the density G-sphere,
    ///        and Pulay-mix toward self-consistency.
    /// @return The unmixed ||rho_out-rho_in||_2, independent of damping
    ///         and Pulay history. Only this residual measures self-consistency.
    AccReal finish(PWWorkspace& ws);

    /// @brief Mix both spin densities and return their unmixed joint residual.
    /// @param ws Collinear workspace; reads live Pulay controls.
    /// @param alpha Output alpha density on the density G-sphere.
    /// @param beta Output beta density on the density G-sphere.
    /// @return sqrt(sum_sigma,G |rho_out-rho_in|^2), before damping.
    AccReal mix_spin(PWWorkspace& ws,
                     const xt::xarray<std::complex<AccReal>>& alpha,
                     const xt::xarray<std::complex<AccReal>>& beta);

    /// @brief FFT the real-space accumulation onto `rho_out` (density G-sphere).
    ///        No mixing — used by the collinear spin channel before a total mix.
    void extract(const PWWorkspace& ws,
                 xt::xarray<std::complex<AccReal>>& rho_out);

  private:
    xt::xarray<std::complex<AccReal>> spin_input_;
    xt::xarray<std::complex<AccReal>> spin_output_;
    FFTWBuffer<AccReal> rho_r_accum_;
    FFTWBuffer<fftw_complex> psi_cc_;
    FFTWBuffer<fftw_complex> rho_g_full_;
};

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV

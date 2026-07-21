/// @file cpu/pw/internal/density_builder.h
/// @brief internal::DensityBuilder — ρ from wavefunctions + Pulay mixing.
///
/// Facade-private step helper owned by `cpu::pw::PWSolver`. NOT a
/// `cpu::Component` subclass and never registered into a Plan.
///
/// Single canonical implementation of the density build (the previous
/// tree carried two divergent copies: a dead `PW_Density` component and
/// an inline copy in the original monolithic solver — consolidated here with the k-point
/// weighting of the production copy and the correct sliding-window Pulay
/// history of the component copy).
#pragma once

#include "atomiverse/cpu/pw/internal/pw_workspace.h"

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
    /// @param mix_beta    Linear/Pulay mixing parameter (QE mixing_beta).
    /// @param pulay_depth Pulay history window depth.
    explicit DensityBuilder(AccReal mix_beta = AccReal(0.7),
                            int pulay_depth = 5)
        : mix_beta_(mix_beta), pulay_depth_(pulay_depth) {}

    /// @brief Size scratch to the workspace's FFT grid (idempotent).
    void prepare(const PWWorkspace& ws);

    /// @brief Clear the Pulay history on the workspace (once per SCF run).
    void reset(PWWorkspace& ws);

    /// @brief Zero the real-space density accumulation buffer.
    void begin(const PWWorkspace& ws);

    /// @brief Add this k-point's band densities: ρ(r) += w·2·|ψ_b(r)|²/Ω.
    /// @param ws      Workspace with `gvec`/`n_g`/`wfc` at the current k.
    /// @param weight  k-point weight (Monkhorst-Pack).
    /// @param n_bands Occupied band count.
    void accumulate_kpoint(const PWWorkspace& ws, AccReal weight, int n_bands);

    /// @brief Real-space accumulation buffer (length `ws.nr`) — exposed
    ///        for the MPI k-point Allreduce between accumulation and
    ///        `finish()`.
    AccReal* accumulation() { return rho_r_accum_.data(); }

    /// @brief Save ρ_old, extract the new ρ(G) on the density G-sphere,
    ///        and Pulay-mix toward self-consistency.
    void finish(PWWorkspace& ws);

  private:
    AccReal mix_beta_ = AccReal(0.7);
    int pulay_depth_ = 5;
    FFTWBuffer<AccReal> rho_r_accum_;
    FFTWBuffer<fftw_complex> psi_cc_;
    FFTWBuffer<fftw_complex> rho_g_full_;
};

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV

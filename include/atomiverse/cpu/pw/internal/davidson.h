/// @file cpu/pw/internal/davidson.h
/// @brief internal::DavidsonSolver — matrix-free iterative diagonalization.
///
/// Facade-private step helper owned by `cpu::pw::PWSolver`. NOT a
/// `cpu::Component` subclass and never registered into a Plan.
#pragma once

#include "atomiverse/cpu/pw/internal/pw_workspace.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Test hook: one H·ψ application on the bound k-sphere of `ws`
/// (same code path DavidsonSolver uses). For validation oracles only.
void pw_apply_h_for_tests(const PWWorkspace& ws,
                          const std::complex<AccReal>* psi_g,
                          std::complex<AccReal>* hpsi_g);

/// @brief Block Davidson with FFT-based H·ψ and KB non-local potential.
///
/// Diagonalizes the current-k-point Hamiltonian (T_G diagonal +
/// real-space V_eff + KB projectors) for the lowest `n_bands` states,
/// writing `ws.wfc` and `ws.eigenvalues`.
class DavidsonSolver {
  public:
    /// @param n_bands  Number of lowest eigenpairs to compute.
    /// @param tol      Residual tolerance [Hartree].
    /// @param max_iter Maximum Davidson expansion sweeps.
    DavidsonSolver(int n_bands, AccReal tol = AccReal(1e-8), int max_iter = 50)
        : n_bands_(n_bands), tol_(tol), max_iter_(max_iter) {}

    /// @brief Size the FFT work buffer to the workspace grid (idempotent).
    void prepare(const PWWorkspace& ws);

    /// @brief Diagonalize at the workspace's current k-point G-sphere.
    /// @param ws Plane-wave workspace (`gvec`/`n_g`/`T_G` set per k-point).
    void step(PWWorkspace& ws);

  private:
    int n_bands_ = 4;
    AccReal tol_ = AccReal(1e-8);
    int max_iter_ = 50;
    FFTWBuffer<fftw_complex> work_buf_;
};

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV

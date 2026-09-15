/// @file cpu/planewave/internal/davidson.h
/// @brief internal::DavidsonSolver — matrix-free iterative diagonalization.
///
/// Facade-private step helper owned by `cpu::pw::PW`. NOT a
/// `cpu::Component` subclass and never registered into a Plan.
#pragma once

#include "atomiverse/cpu/planewave/internal/pw_workspace.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief One H·ψ application on the bound k-sphere of `ws` — the exact
/// code path DavidsonSolver uses (T_G diagonal + real-space V_eff + KB).
/// Shared by the DFPT Sternheimer solver and validation tests.
/// @param work_buf Caller-owned FFT scratch of size `ws.nr`.
void pw_apply_h(const PWWorkspace& ws, const std::complex<AccReal>* psi_g,
                std::complex<AccReal>* hpsi_g, fftw_complex* work_buf);

/// @brief Convenience overload allocating its own FFT scratch per call.
void pw_apply_h(const PWWorkspace& ws, const std::complex<AccReal>* psi_g,
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

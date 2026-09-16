/// @file cpu/scf/internal/cc_diis.h
/// @brief internal::CcAmplitudeDiis — Pulay DIIS accelerator for CC amplitudes.
///
/// **Internal post-HF helper.** Does NOT inherit `cpu::Component` and is not
/// registered into any Driver Plan. Owned by the CCSD amplitude loop (spatial
/// closed-shell `ccsd_energy_spatial` and the spin-orbital `so_ccsd_kernel`)
/// and by tests, via direct include of this header.
///
/// Where `internal::DIISMixer` extrapolates spin-resolved Fock **matrices** for
/// the SCF loop, this class extrapolates the FLATTENED coupled-cluster
/// amplitude vector `T = [t1; t2]` from a short history, with the DIIS error
/// vector the Jacobi increment `e = Δt = R/D`. Naming, construction, and the
/// `step`/`reset`/`size` surface deliberately mirror `DIISMixer` so the two
/// accelerators read alike; the deque history and the augmented-normal-equation
/// solve are the only shared shape (this class does NOT generalize or touch
/// `diis_mixer.*`).
///
/// References:
///   - Scuseria, Lee, Schaefer, Chem. Phys. Lett. 130, 236 (1986)
///   - Pulay, Chem. Phys. Lett. 73, 393 (1980)
#pragma once

#include <deque>
#include <vector>

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Pulay DIIS accelerator over flattened CC amplitude vectors.
///
/// Stores a history of `(amp, err)` pairs — `amp = [t1; t2]` flattened, `err`
/// the Jacobi increment `Δt` — and solves the augmented normal equations
/// \f$\begin{bmatrix} B & -1 \\ -1^{T} & 0 \end{bmatrix}
///    \begin{bmatrix} c \\ \lambda \end{bmatrix}
///  = \begin{bmatrix} 0 \\ -1 \end{bmatrix}\f$, with
/// \f$B_{ij} = e_i\!\cdot\!e_j\f$, then returns the extrapolant
/// \f$T = \sum_i c_i\,T_i\f$. Beyond `max_vectors` the oldest pair is dropped;
/// a near-singular \f$B\f$ (near-parallel error vectors) is handled by dropping
/// the oldest history vector until the solve returns a finite extrapolant.
class CcAmplitudeDiis {
  public:
    /// @brief Construct with a bounded history depth.
    /// @param max_vectors Maximum number of stored `(amp, err)` pairs
    ///        (dimensionless; oldest dropped beyond it).
    explicit CcAmplitudeDiis(int max_vectors = 8) : max_vectors_(max_vectors) {}

    /// @brief Store a new `(amp, err)` pair and return the extrapolated amp.
    ///
    /// With a single stored pair the input `amp` is returned verbatim
    /// (\f$c_0 = 1\f$). Otherwise the Pulay augmented system is solved and the
    /// coefficient-weighted amplitude combination is returned.
    /// @param amp Flattened current amplitude vector \f$[t_1; t_2]\f$
    ///        (dimensionless).
    /// @param err Flattened Jacobi increment \f$\Delta t = R/D\f$ used as the
    ///        DIIS error vector (dimensionless).
    /// @return The extrapolated flattened amplitude vector (same length as
    ///         `amp`), always finite.
    std::vector<double> step(const std::vector<double>& amp,
                             const std::vector<double>& err);

    /// @brief Number of stored `(amp, err)` pairs.
    /// @return History length (dimensionless).
    int size() const { return static_cast<int>(amp_hist_.size()); }

    /// @brief Clear all stored history.
    void reset() {
        amp_hist_.clear();
        err_hist_.clear();
    }

  private:
    int max_vectors_;
    std::deque<std::vector<double>> amp_hist_;
    std::deque<std::vector<double>> err_hist_;
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

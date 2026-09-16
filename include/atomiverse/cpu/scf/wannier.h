/// @file cpu/scf/wannier.h
/// @brief Foster–Boys maximally localized Wannier functions (molecular Γ).
///
/// After a closed-shell mean-field `run()`, occupied spatial MOs are localized
/// by successive Jacobi rotations that maximize
/// \(\sum_i|\langle\mathbf r\rangle_i|^2\) (Foster & Boys, Rev. Mod. Phys.
/// 32, 300 (1960)). Multipole AO matrices come from `MultipoleIntegrals`
/// about the Cartesian origin. Parameters are members of `WannierCenters` —
/// no public `*Config` bag, no factories.
///
/// Foothold scope: restricted RHF/RKS with integer occupations. Periodic
/// Marzari–Vanderbilt / multi-k MLWF is deferred (roadmap §11).
///
/// @code
///   cpu::scf::RHF rhf;
///   rhf.basis = BasisSet::STO3G;
///   rhf.run(frame);
///   cpu::scf::WannierCenters wb;
///   auto r = wb.compute(rhf.result_handle(), frame);
///   // r.centers[i] [Å], r.spreads[i] [Å²]
/// @endcode
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/mean_field_method.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Result of a Foster–Boys localization of occupied MOs.
struct WannierResult {
    /// Orbital centroids \(\langle\mathbf r\rangle_i\) [Å].
    std::vector<std::array<double, 3>> centers;
    /// Orbital spreads \(\Omega_i = \langle r^2\rangle_i - |\langle\mathbf r\rangle_i|^2\)
    /// [Å²].
    std::vector<double> spreads;
    /// Total spread \(\sum_i\Omega_i\) after localization [Å²].
    double total_spread = 0.0;
    /// Total spread of the input occupied MOs before localization [Å²].
    double total_spread_initial = 0.0;
    /// Number of full pair-sweeps executed (dimensionless).
    int sweeps = 0;
    /// Whether the Jacobi loop stopped with \(\max|\gamma| < \texttt{tol}\).
    bool converged = false;
};

/// @brief Foster–Boys Jacobi localization of occupied closed-shell MOs.
///
/// Consumes a restricted `MeanFieldOutputs` view (determinant + Cartesian
/// shells + optional c2s). Origin of the multipole integrals is the
/// simulation Cartesian origin (0,0,0) in Bohr.
class WannierCenters {
  public:
    /// @brief Maximum number of full occupied-pair Jacobi sweeps
    ///        (dimensionless, default 200).
    int max_sweeps = 200;
    /// @brief Convergence: stop when the largest absolute pair angle
    ///        \(\max|\gamma|\) in a sweep is below this value [rad]
    ///        (default 1e-10).
    double tol = 1e-10;

    /// @brief Localize occupied MOs of a restricted mean-field result.
    /// @param outs  Shared view after `method.run(frame)` (must be ready,
    ///              restricted, non-generalized).
    /// @param frame Geometry [Å]; used only for validation context (positions
    ///              enter via shells already stored on `outs`).
    /// @return Owned `WannierResult` with centers [Å] and spreads [Å²].
    /// @throws std::invalid_argument when `outs` is not ready, open-shell,
    ///         generalized, or has no occupied orbitals; std::invalid_argument
    ///         when `max_sweeps < 1` or `tol <= 0`.
    WannierResult compute(const MeanFieldOutputs& outs,
                          const cpu::Frame& frame) const;
};

/// @brief Optimal Foster–Boys pair angle \(\gamma = \frac14\operatorname{atan2}(B,A)\)
///        [rad] (pure math; unit-testable without SCF).
///
/// With \(A^a = R_{ii}^a - R_{jj}^a\), \(B^a = 2 R_{ij}^a\),
/// \(A = \sum_a((A^a)^2 - (B^a)^2)\), \(B = \sum_a 2 A^a B^a\).
///
/// @param A Boys pair scalar \(A\) (Bohr²-squared scale; dimensionless ratio).
/// @param B Boys pair scalar \(B\) (same units as \(A\)).
/// @return Rotation angle \(\gamma\) [rad] in \((-\pi/4,\pi/4]\).
double boys_pair_angle(double A, double B);

} // namespace scf
} // namespace cpu
} // namespace ATV

/// @file cpu/gpw/periodic_wannier.h
/// @brief Periodic multi-k maximally localized Wannier functions (CPU GPW).
///
/// Marzari–Vanderbilt construction on the occupied manifold of a GPW SCF
/// (Marzari & Vanderbilt, Phys. Rev. B 56, 12847 (1997); Marzari et al.,
/// Rev. Mod. Phys. 84, 1419 (2012)). After multi-k SCF, neighbor AO overlaps
/// \(M^{(\mathbf k,\mathbf b)}\) feed a projection initialization
/// (bond-centered Gaussians for elemental semiconductors) and optional MV
/// steepest-descent polish. Centers come from the Berry-phase formula
/// \(\mathbf r_n = -N_k^{-1}\sum_{\mathbf k,\mathbf b} w_b\,\mathbf b\,
/// \operatorname{Im}\ln M_{nn}\).
///
/// **Collinear spin:** when `GpwDft::is_uks()` (`n_alpha`/`n_beta`), α and β
/// occupied manifolds are localized independently from `C_occ_k` /
/// `C_occ_k_beta` (fixed occupations; insulator only). Centers are
/// concatenated α then β; see `PeriodicWannierResult::spin`.
///
/// Product: closed-shell insulator or collinear UKS insulator, orthorhombic
/// cell, no metallic smearing / disentanglement. Parameters are members of
/// `PeriodicWannier` — no public `*Config` bag, no factories.
///
/// @code
///   cpu::gpw::GpwDft dft;
///   dft.cell[0] = dft.cell[1] = a / std::sqrt(2.0); // BCT Si [Bohr]
///   dft.cell[2] = a;
///   dft.cutoff_ry = 100.0;
///   dft.n_alpha = 8;   // collinear UKS (M = 0): 8↑ + 8↓
///   dft.n_beta  = 8;
///   dft.xc = cpu::gpw::XcKind::PADE;
///   dft.build(atoms);
///
///   cpu::gpw::PeriodicWannier mlwf;   // members: k_mesh, max_iter, …
///   auto r = mlwf.compute(dft, atoms);
///   // r.centers[i] [Å], r.spreads[i] [Å²], r.n_wann_up / r.n_wann_dn
/// @endcode
#pragma once

#include <array>
#include <complex>
#include <vector>

#include "atomiverse/cpu/gpw/gpw_solver.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Result of a periodic multi-k MLWF calculation on GPW.
struct PeriodicWannierResult {
    /// Wannier centroids \(\langle\mathbf r\rangle_n\) [Å].
    /// Collinear spin: α manifold first, then β.
    std::vector<std::array<double, 3>> centers;
    /// Orbital spreads \(\Omega_n=\langle r^2\rangle_n-|\langle\mathbf r\rangle_n|^2\)
    /// [Å²] (SMV discretization).
    std::vector<double> spreads;
    /// Spin index per center: 0 = ↑, 1 = ↓. Empty when restricted (RKS).
    std::vector<int> spin;
    /// \(\sum_n\Omega_n\) after localization [Å²] (all channels).
    double total_spread = 0.0;
    /// \(\sum_n\Omega_n\) of the unrotated (identity-gauge) manifold [Å²].
    double total_spread_initial = 0.0;
    /// MV steepest-descent iterations executed (max over spin channels).
    int iterations = 0;
    /// Whether every spin channel's SD loop stopped with
    /// \(\|\mathrm{G}\|_\infty < \texttt{tol}\) (or `max_iter == 0`).
    bool converged = false;
    /// Whether the preceding SCF converged.
    bool scf_converged = false;
    /// SCF total energy [Hartree].
    double energy_ha = 0.0;
    /// Number of Wannier functions (RKS: occupied bands; UKS:
    /// \(n_\alpha+n_\beta\)).
    int n_wann = 0;
    /// Occupied α count when collinear UKS; 0 when restricted (RKS).
    int n_wann_up = 0;
    /// Occupied β count when collinear UKS; 0 when restricted (RKS).
    int n_wann_dn = 0;
};

/// @brief Multi-k Marzari–Vanderbilt MLWF on the GPW method (AO basis).
///
/// Occupied-only manifold; projection initialization onto `n_wann` real-space
/// Gaussians at caller-supplied or auto BCT diamond bond midpoints when the
/// system is 4 identical Si-like atoms with 8 occupied orbitals **per channel**.
/// Collinear UKS localizes each spin channel separately (see
/// `PeriodicWannierResult`).
class PeriodicWannier {
  public:
    /// @brief Monkhorst–Pack mesh per dimension for the MLWF k-grid
    ///        (dimensionless; default 2). Applied to `GpwDft::k_mesh` for the
    ///        SCF that feeds localization (`k_time_reversal` forced off).
    int k_mesh = 2;
    /// @brief Maximum MV steepest-descent iterations (dimensionless).
    int max_iter = 200;
    /// @brief Stop when \(\max_{k,mn}|G_{mn}(k)| < \texttt{tol}\) (dimensionless).
    double tol = 1e-8;
    /// @brief SD step length \(\alpha\) in \(U\leftarrow U\exp(-\alpha G)\)
    ///        (dimensionless).
    double step = 0.5;
    /// @brief Gaussian width for projection trial orbitals [Å]
    ///        (default 1.0 — order of a Si–Si bond half-length).
    double proj_alpha_ang = 1.0;
    /// @brief If true (default), initialize \(U(\mathbf k)\) by projection onto
    ///        bond-centered Gaussians. If false, start from identity.
    bool use_projection = true;
    /// @brief Optional explicit trial centers [Å] (length must equal n_wann
    ///        **per spin channel** when non-empty — same trials for both spins
    ///        under UKS). Empty → auto BCT diamond midpoints for 4-atom
    ///        Si-like cells with 8 WFs/channel, else crystal origin replicates.
    std::vector<std::array<double, 3>> trial_centers;

    /// @brief Multi-k SCF + MLWF for the geometry in `atoms`.
    ///
    /// Configures `method.k_mesh = {k_mesh,k_mesh,k_mesh}`, disables time
    /// reversal, runs `method.run(atoms)`, then localizes the occupied Bloch
    /// manifold from `last_result()` (RKS: one channel; UKS: α then β).
    ///
    /// @param method GPW DFT method (members + `build()` already done; k-mesh
    ///               and TR flag are overwritten for the MLWF SCF).
    /// @param atoms  Atom set at the current geometry [Bohr] (same as `build`).
    /// @return Owned `PeriodicWannierResult` (centers [Å], spreads [Å²]).
    /// @throws std::invalid_argument on metallic smearing, non-positive
    ///         mesh / step / tol, empty occupied manifold, or `!built`.
    /// @throws std::logic_error when `method` was not `build()`'d.
    PeriodicWannierResult compute(GpwDft& method,
                                  const std::vector<GpwAtom>& atoms) const;
};

/// @brief Eight Si–Si bond midpoints of the diamond BCT 4-atom orthorhombic
///        cell with edges \((a/\sqrt2,\,a/\sqrt2,\,a)\) [Å].
///
/// Atoms at fractional \((0,0,0)\), \((½,½,½)\), \((½,0,¼)\), \((0,½,¾)\).
/// Each of the four sites has four tetrahedral neighbors at distance
/// \(a\sqrt3/4\); the eight unique bonds yield eight midpoints (one MLWF each).
///
/// @param a_ang Cubic lattice constant [Å] (must be > 0).
/// @return Eight Cartesian centers [Å] inside/near the BCT cell.
/// @throws std::invalid_argument when `a_ang <= 0`.
std::array<std::array<double, 3>, 8> diamond_bct_bond_midpoints(double a_ang);

/// @brief SMV weight \(w_b = 1/(2 b^2)\) for a cubic six-neighbor shell [Å²].
/// @param b_len Neighbor distance \(|\mathbf b|\) [1/Å] (must be > 0).
/// @return Weight [Å²].
/// @throws std::invalid_argument when `b_len <= 0`.
double mv_pair_weight_cubic(double b_len);

/// @brief Recover a Wannier center from diagonal phases on a single-k
///        six-neighbor cubic shell (pure math; unit-testable).
///
/// \(\mathbf r = -\sum_b w_b\,\mathbf b\,\operatorname{Im}\ln M_b\) (Nk=1).
///
/// @param b_cart Neighbor \(\mathbf b\) vectors [1/Å], length 6.
/// @param w_b    Weights [Å²], length 6.
/// @param M_diag Diagonal \(M_{nn}^{(\mathbf b)}\) phases as complex values.
/// @return Cartesian center [Å].
/// @throws std::invalid_argument on size mismatch or empty input.
std::array<double, 3>
center_from_diagonal_phases(const std::vector<std::array<double, 3>>& b_cart,
                            const std::vector<double>& w_b,
                            const std::vector<std::complex<double>>& M_diag);

} // namespace gpw
} // namespace cpu
} // namespace ATV

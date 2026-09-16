/// @file cpu/planewave/periodic_wannier.h
/// @brief Periodic multi-k maximally localized Wannier functions (CPU PW).
///
/// Marzari–Vanderbilt construction on the occupied manifold of a plane-wave
/// SCF (Marzari & Vanderbilt, Phys. Rev. B 56, 12847 (1997); Marzari et al.,
/// Rev. Mod. Phys. 84, 1419 (2012)). After SCF, Bloch states are recomputed
/// non-self-consistently on the frozen \(V_\mathrm{eff}\) at every
/// Monkhorst–Pack k-point; neighbor overlap matrices
/// \(M^{(\mathbf k,\mathbf b)}\) feed a projection initialization
/// (bond-centered Gaussians for elemental semiconductors) and optional MV
/// steepest-descent polish. Centers come from the Berry-phase formula
/// \(\mathbf r_n = -N_k^{-1}\sum_{\mathbf k,\mathbf b} w_b\,\mathbf b\,
/// \operatorname{Im}\ln M_{nn}\).
///
/// **Collinear spin:** when `PW::spin_polarized` is true, α and β occupied
/// manifolds are localized independently on frozen
/// \(V_{\mathrm{eff},\sigma}\) (fixed occupations; insulator only). Centers
/// are concatenated α then β; see `PeriodicWannierResult::spin`.
///
/// Parameters are members of `PeriodicWannier` — no public `*Config` bag,
/// no factories. No metallic smearing / disentanglement in this product path.
///
/// @code
///   cpu::pw::PW pw;
///   pw.upf_path = "Si.upf";
///   pw.ecut_ry = 15.0;
///   pw.k_mesh = 2;
///   pw.xc = cpu::scf::XCType::LSDA;
///   pw.spin_polarized = true;
///   pw.multiplicity = 1;           // M = 0 closed-shell collinear
///
///   cpu::pw::PeriodicWannier mlwf;   // members: k_mesh, max_iter, …
///   auto r = mlwf.compute(pw, frame);
///   // r.centers[i] [Å], r.spreads[i] [Å²], r.n_wann_up / r.n_wann_dn
/// @endcode
#pragma once

#include <array>
#include <complex>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

namespace ATV {
namespace cpu {
namespace pw {

/// @brief Result of a periodic multi-k MLWF calculation.
struct PeriodicWannierResult {
    /// Wannier centroids \(\langle\mathbf r\rangle_n\) [Å].
    /// Collinear spin: α manifold first, then β.
    std::vector<std::array<double, 3>> centers;
    /// Orbital spreads \(\Omega_n=\langle r^2\rangle_n-|\langle\mathbf r\rangle_n|^2\)
    /// [Å²] (SMV discretization).
    std::vector<double> spreads;
    /// Spin index per center: 0 = ↑, 1 = ↓. Empty when restricted.
    std::vector<int> spin;
    /// \(\sum_n\Omega_n\) after localization [Å²] (all channels).
    double total_spread = 0.0;
    /// \(\sum_n\Omega_n\) of the unrotated (identity-gauge) manifold [Å²].
    double total_spread_initial = 0.0;
    /// MV steepest-descent iterations executed (max over spin channels).
    int iterations = 0;
    /// Whether every spin channel's SD loop stopped with
    /// \(\|\mathrm{G}\|_\infty < \texttt{tol}\) (or max_iter==0).
    bool converged = false;
    /// Whether the preceding SCF converged.
    bool scf_converged = false;
    /// SCF total energy [Ry] (0 if not evaluated on the spin foothold path).
    double energy_ry = 0.0;
    /// Number of Wannier functions (restricted: occupied bands;
    /// spin: \(n_\uparrow+n_\downarrow\)).
    int n_wann = 0;
    /// Occupied α count when collinear; 0 when restricted.
    int n_wann_up = 0;
    /// Occupied β count when collinear; 0 when restricted.
    int n_wann_dn = 0;
};

/// @brief Multi-k Marzari–Vanderbilt MLWF on the PW method.
///
/// Occupied-only manifold; projection initialization onto `n_wann` real-space
/// Gaussians at caller-supplied or auto diamond bond midpoints when the frame
/// has two identical atoms on the diamond basis. Collinear spin localizes each
/// channel separately (see `PeriodicWannierResult`).
class PeriodicWannier {
  public:
    /// @brief Monkhorst–Pack mesh per dimension for the MLWF k-grid
    ///        (dimensionless; default 2). Independent of `PW::k_mesh` used for
    ///        the density SCF (which still comes from the method members).
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
    ///        bond-centered Gaussians (diamond auto midpoints, or
    ///        `trial_centers` if non-empty). If false, start from identity.
    bool use_projection = true;
    /// @brief Optional explicit trial centers [Å] (length must equal n_wann
    ///        per spin channel when non-empty — same trials for both spins).
    ///        Empty → auto diamond bond midpoints for 2-atom Si-like
    ///        cells with 4 WFs/spin, else crystal origin replicates.
    std::vector<std::array<double, 3>> trial_centers;

    /// @brief SCF + multi-k MLWF for the geometry in `frame`.
    /// @param method Plane-wave method (members only; not mutated).
    /// @param frame  Atom positions [Å], types (Z), simulation cell.
    /// @return Owned `PeriodicWannierResult` (centers [Å], spreads [Å²]).
    /// @throws std::invalid_argument on metallic method, non-positive
    ///         mesh / step / tol, empty occupied manifold, or incompatible
    ///         spin multiplicity.
    PeriodicWannierResult compute(const PW& method,
                                  const cpu::Frame& frame) const;
};

/// @brief Four tetrahedral Si–Si bond midpoints of the diamond 2-atom cell
///        with atoms at \(\mathbf 0\) and \((a/4)(1,1,1)\) [Å].
///
/// Midpoints: \((a/8)(\pm1,\pm1,\pm1)\) with an even number of minus signs.
///
/// @param a_ang Cubic lattice constant [Å] (must be > 0).
/// @return Four Cartesian centers [Å].
/// @throws std::invalid_argument when `a_ang <= 0`.
std::array<std::array<double, 3>, 4> diamond_bond_midpoints(double a_ang);

/// @brief SMV weight \(w_b = 3/(N_b b^2)\) for a cubic six-neighbor shell
///        (\(N_b=6\Rightarrow w_b=1/(2b^2)\)) [Å²].
/// @param b_len Neighbor distance \(|\mathbf b|\) [1/Å] (must be > 0).
/// @return Weight [Å²].
/// @throws std::invalid_argument when `b_len <= 0`.
double mv_pair_weight_cubic(double b_len);

/// @brief Recover a Wannier center from diagonal phases on a single-k
///        six-neighbor cubic shell (pure math; unit-testable).
///
/// \(\mathbf r = -\sum_b w_b\,\mathbf b\,\operatorname{Im}\ln M_b\) with
/// \(\sum\) over the six ±b neighbors (Nk=1).
///
/// @param b_cart Neighbor \(\mathbf b\) vectors [1/Å], length 6.
/// @param w_b    Weights [Å²], length 6.
/// @param M_diag Diagonal \(M_{nn}^{(\mathbf b)}\) phases as complex values.
/// @return Cartesian center [Å].
/// @throws std::invalid_argument on size mismatch or empty input.
std::array<double, 3> center_from_diagonal_phases(
    const std::vector<std::array<double, 3>>& b_cart,
    const std::vector<double>& w_b,
    const std::vector<std::complex<double>>& M_diag);

} // namespace pw
} // namespace cpu
} // namespace ATV

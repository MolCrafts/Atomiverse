/// @file cpu/semiempirical/internal/dftb_gamma_lattice.h
/// @brief **PRIVATE WORKSPACE** — the geometry-fixed SCC charge kernel
///        \f$\gamma_{AB}\f$, Ewald-summed under periodic boundaries.
///
/// \f$\gamma\f$ is a function of the geometry alone: within one SCC solve the
/// charges move and \f$\gamma\f$ does not. This type builds it **once** and
/// then answers \f$V_A=\sum_B\gamma_{AB}\Delta q_B\f$ and
/// \f$E^{(2)}=\tfrac12\sum_{AB}\gamma_{AB}\Delta q_A\Delta q_B\f$ by dense
/// matrix algebra.
///
/// ## Why Ewald
///
/// \f[
///   \gamma_{AB}(R) = \frac{1}{R} - S(\tau_A,\tau_B,R), \qquad
///   \gamma_{AA} = U_A
/// \f]
///
/// The short-range piece \f$S\f$ (Elstner `expGamma`, `dftb_gamma.h`) decays
/// exponentially and is safe to truncate. The \f$1/R\f$ piece is **not**: its
/// lattice sum is only conditionally convergent, so cutting it at a sphere
/// leaves a \f$\gamma\f$ that is not positive definite. Doing that gave
/// \f$E^{(2)}<0\f$ and a 37 eV error on a 24-atom water cell against DFTB+.
/// The \f$1/R\f$ lattice sum is therefore taken as a standard Ewald sum with a
/// neutralising background, reusing `ATV::longrange::EwaldSum` for the
/// reciprocal tables.
///
/// Derived from the published DFTB2 functional (Elstner et al., *Phys. Rev. B*
/// **58**, 7260 (1998), DOI 10.1103/PhysRevB.58.7260) and Ewald's lattice sum
/// (P. P. Ewald, *Ann. Phys.* **369**, 253 (1921), DOI
/// 10.1002/andp.19213690304) — not transcribed from DFTB+, which is LGPL-3.0
/// and so cannot be absorbed by this GPL-2.0-only tree. DFTB+ is the numerical
/// oracle for these terms, never the source.
///
/// ## Matrix assembly
///
/// With \f$\sum_T'\f$ excluding \f$(A{=}B,\,T{=}0)\f$ and \f$k\f$ running over
/// the positive reciprocal hemisphere (`EwaldSum` convention, prefactor
/// \f$4\pi/V\f$):
///
/// \f[
///   \gamma_{AB} = \underbrace{\delta_{AB}U_A}_{\text{on-site}}
///   \; \underbrace{-\sum_T{}' S(\tau_A,\tau_B,r)}_{\text{short range}}
///   \; +\; C\Big[ \sum_T{}' \frac{\mathrm{erfc}(\alpha r)}{r}
///        + \frac{8\pi}{V}\sum_k w_k\cos(\mathbf k\!\cdot\!\mathbf R_{AB})
///        - \frac{\pi}{\alpha^2 V}
///        - \delta_{AB}\frac{2\alpha}{\sqrt\pi} \Big]
/// \f]
///
/// with \f$C=\f$ `units::coulomb` [eV·Å], \f$w_k=e^{-k^2/4\alpha^2}/k^2\f$.
/// The factor \f$8\pi/V\f$ (not \f$4\pi/V\f$) is the half-sphere sum written
/// as a symmetric quadratic form.
///
/// ## Cluster mode
///
/// `periodic = false` reproduces an isolated molecule: no images, no Ewald,
/// \f$\gamma_{AB}=\gamma(R_{AB})\f$ directly. That is DFTB+'s `C` geometry,
/// and it is a different Hamiltonian from a supercell — not an approximation
/// of one.
///
/// ## Cost
///
/// Storage is dense \f$N^2\f$ doubles (9.5 MB at N = 1088). DFTB itself is
/// \f$O(N^3)\f$ dense diagonalisation, so this is not the binding limit; if it
/// ever becomes one the answer is a matrix-free short-range neighbour list
/// plus cached \f$\cos/\sin\f$ tables, not a smaller cutoff.
///
/// Internal — never on a public include path. Units: positions [Å], Hubbard U
/// [eV], \f$\gamma\f$ [eV], \f$\alpha\f$ [1/Å].
#pragma once

#include <cstddef>
#include <vector>

#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/longrange/ewald_sum.h"

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace internal {

/// @brief Dense geometry-fixed SCC charge kernel \f$\gamma_{AB}\f$ [eV].
class GammaLattice {
  public:
    /// @brief Build \f$\gamma\f$ for one geometry.
    /// @param n_at         Atom count (> 0).
    /// @param u_eV         Per-atom Hubbard U [eV], length @p n_at.
    /// @param x            Cartesian x [Å], length @p n_at.
    /// @param y            Cartesian y [Å], length @p n_at.
    /// @param z            Cartesian z [Å], length @p n_at.
    /// @param box          Cell; read only when @p periodic.
    /// @param periodic     True for a 3-D lattice (Ewald), false for an
    ///                     isolated cluster (no images).
    /// @param real_cutoff  Ewald real-space cutoff [Å]; must be > 0. Also sets
    ///                     \f$\alpha\f$ through the LAMMPS heuristic.
    /// @param accuracy     Ewald relative accuracy in (0, 1), sets \f$|k|\f$.
    /// @throws std::invalid_argument on a non-positive count or cutoff, or an
    ///         accuracy outside (0, 1).
    /// @throws std::runtime_error when @p periodic and the cell has
    ///         non-positive volume.
    GammaLattice(int n_at, const double* u_eV, const double* x, const double* y,
                 const double* z, const cpu::Box& box, bool periodic,
                 double real_cutoff, double accuracy);

    /// @brief Atom count.
    int size() const noexcept { return n_; }

    /// @brief \f$\gamma_{AB}\f$ [eV].
    double pair(int a, int b) const {
        return g_[static_cast<std::size_t>(a) * static_cast<std::size_t>(n_)
                  + static_cast<std::size_t>(b)];
    }

    /// @brief Row pointer \f$\gamma_{A\cdot}\f$ [eV] — for a caller's own
    ///        contraction.
    const double* row(int a) const {
        return g_.data()
               + static_cast<std::size_t>(a) * static_cast<std::size_t>(n_);
    }

    /// @brief SCC potential \f$V_A=\sum_B\gamma_{AB}\Delta q_B\f$ [eV].
    /// @param dq  Mulliken fluctuations [e], length `size()`.
    /// @param out Destination [eV], length `size()`; overwritten.
    void potential(const double* dq, double* out) const;

    /// @brief \f$E^{(2)}=\tfrac12\sum_{AB}\gamma_{AB}\Delta q_A\Delta q_B\f$
    ///        [eV].
    double energy(const double* dq) const;

    /// @brief Accumulate the \f$\gamma\f$ share of the nuclear gradient and
    ///        the cell derivative at **fixed** \f$\Delta q\f$.
    ///
    /// The value and the derivative live on one type deliberately: the SCC
    /// kernel is assembled from a real-space walk, an Ewald reciprocal sum, a
    /// self term and a background term, and \f$F=-\nabla E\f$ only holds if
    /// the gradient walks exactly those same four pieces. Splitting them
    /// across two files is how a truncated \f$1/R\f$ ends up in one and not
    /// the other.
    ///
    /// @param dq     Mulliken fluctuations [e], length `size()`.
    /// @param fx     Force x accumulator [eV/Å], length `size()`; nullable.
    /// @param fy     Force y accumulator [eV/Å]; nullable.
    /// @param fz     Force z accumulator [eV/Å]; nullable.
    /// @param dE_de  \f$\partial E^{(2)}/\partial\varepsilon\f$ Voigt
    ///               `(xx,yy,zz,xy,xz,yz)` [eV], accumulated; nullable. The
    ///               off-diagonal entries carry both \f$ij\f$ and \f$ji\f$,
    ///               matching the caller's pair-virial convention
    ///               \f$\sigma=-\mathrm{d}E/\mathrm{d}\varepsilon/\Omega\f$.
    void add_gradient(const double* dq, double* fx, double* fy, double* fz,
                      double* dE_de) const;

    /// @brief Ewald splitting parameter \f$\alpha\f$ [1/Å]; 0 in cluster mode.
    double alpha() const noexcept { return alpha_; }

    /// @brief Whether the lattice sum (Ewald) path was taken.
    bool periodic() const noexcept { return periodic_; }

    /// @brief Range at which the exponential short-range piece was dropped
    ///        [Å]; 0 in cluster mode, where nothing is dropped.
    ///
    /// Derived from \(S\)'s own decay, not from the SK grid range — the two
    /// are unrelated quantities that used to share one number, which silently
    /// truncated \(S\) at 5.81 Å and cost 0.4 meV on a 24-atom water cell.
    double short_cutoff() const noexcept { return short_cutoff_; }

  private:
    int n_ = 0;
    bool periodic_ = false;
    double alpha_ = 0.0;
    double short_cutoff_ = 0.0;
    double real_cutoff_ = 0.0;
    double volume_ = 0.0;
    Mat3 cell_{};
    int nmax_[3] = {0, 0, 0};
    std::vector<double> x_, y_, z_, u_;
    std::vector<double> g_;
    longrange::EwaldSum ewald_{1.0, 1.0e-8};
};

} // namespace internal
} // namespace semiempirical
} // namespace cpu
} // namespace ATV

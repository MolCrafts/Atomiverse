/// @file cpu/semiempirical/internal/dftb3_scc.h
/// @brief **PRIVATE WORKSPACE** for `DFTB3` — second-/third-order SCC charge
///        kernel. Users never construct this; Frame is the only I/O currency.
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace internal {

/// @brief Second- and third-order SCC energy and atomic potential for DFTB3.
///
/// Freezes the per-atom Hubbard data and the interatomic distances of one
/// geometry, then evaluates — for a Mulliken fluctuation vector Δq [e] —
///
/// - \f$E^{(2)} = \tfrac{1}{2}\sum_{AB}\gamma_{AB}\,\Delta q_A\,\Delta q_B\f$
///   (Elstner et al., *Phys. Rev. B* **58**, 7260 (1998))
/// - \f$E^{(3)} = \tfrac{1}{3}\sum_{AB}\Delta q_A^2\,\Delta q_B\,\Gamma_{AB}\f$
///   (Gaus, Cui, Elstner, *J. Chem. Theory Comput.* **7**, 931 (2011))
///
/// with \f$\Gamma_{AB} = \partial\gamma_{AB}/\partial q_A\f$ and the on-site
/// values \f$\gamma_{AA} = U_A\f$, \f$\Gamma_{AA} = U^{\mathrm{d}}_A\f$.
///
/// **\f$\Gamma_{AB}\neq\Gamma_{BA}\f$**, so \f$\partial E^{(3)}/\partial\Delta
/// q_C\f$ has two distinct contractions and dropping either one silently
/// breaks the Euler identity \f$\sum_A\Delta q_A V^{(3)}_A = 3E^{(3)}\f$:
///
/// \f[ V^{(3)}_C = \tfrac{2}{3}\,\Delta q_C\sum_B \Gamma_{CB}\,\Delta q_B
///                + \tfrac{1}{3}\sum_B \Delta q_B^2\,\Gamma_{BC} \f]
///
/// Because \f$E^{(2)}\f$ / \f$E^{(3)}\f$ are homogeneous of degree 2 / 3 in
/// Δq, the full potential obeys \f$\sum_A \Delta q_A V_A = 2E^{(2)} +
/// 3E^{(3)}\f$ — i.e. a band-structure energy built from this potential
/// double-counts by \f$E^{(2)} + 2E^{(3)}\f$, not \f$E^{(2)} + E^{(3)}\f$.
///
/// Units: Hubbard U and \f$U^{\mathrm{d}}\f$ [eV], distances [Å], energies
/// [eV], potentials [eV/e].
class Dftb3Scc {
  public:
    /// @brief Freeze the Hubbard data and geometry of one SCC solve.
    /// @param hubbard_u Per-atom Hubbard U [eV], size n.
    /// @param hubbard_derivative Per-atom \f$U^{\mathrm{d}}=dU/dq\f$ [eV],
    ///        size n.
    /// @param distance Row-major n×n interatomic distances [Å] (minimum
    ///        image already applied; diagonal ignored).
    /// @throws std::invalid_argument on an empty or size-mismatched input,
    ///         or when any Hubbard derivative is zero or non-finite.
    Dftb3Scc(std::vector<double> hubbard_u,
             std::vector<double> hubbard_derivative,
             std::vector<double> distance);

    /// @brief Number of atoms this kernel was built for (dimensionless).
    /// @return Atom count n.
    int size() const { return n_; }

    /// @brief Charge-charge kernel \f$\gamma_{AB}\f$ [eV]; on-site \f$U_A\f$.
    /// @param a Atom index A (0-based).
    /// @param b Atom index B (0-based).
    /// @return \f$\gamma_{AB}\f$ [eV].
    double gamma(int a, int b) const;

    /// @brief \f$\Gamma_{AB}=\partial\gamma_{AB}/\partial q_A\f$ [eV/e];
    ///        on-site \f$U^{\mathrm{d}}_A\f$. Not symmetric in (A,B).
    /// @param a Atom index A (0-based) — the differentiated charge.
    /// @param b Atom index B (0-based).
    /// @return \f$\Gamma_{AB}\f$ [eV/e].
    double gamma_charge_derivative(int a, int b) const;

    /// @brief \f$\partial\gamma_{AB}/\partial R_{AB}\f$ [eV/Å].
    ///
    /// The on-site value \f$\gamma_{AA}=U_A\f$ is geometry-independent, so
    /// A == B returns 0.
    /// @param a Atom index A (0-based).
    /// @param b Atom index B (0-based).
    /// @return \f$\partial\gamma_{AB}/\partial R\f$ [eV/Å].
    double gamma_distance_derivative(int a, int b) const;

    /// @brief \f$\partial^2\gamma_{AB}/\partial q_A\,\partial R_{AB} =
    ///        \partial\Gamma_{AB}/\partial R_{AB}\f$ [eV/(e·Å)].
    ///
    /// The on-site value \f$\Gamma_{AA}=U^{\mathrm{d}}_A\f$ is
    /// geometry-independent, so A == B returns 0. Not symmetric in (A,B).
    /// @param a Atom index A (0-based) — the differentiated charge.
    /// @param b Atom index B (0-based).
    /// @return \f$\partial\Gamma_{AB}/\partial R\f$ [eV/(e·Å)].
    double gamma_charge_distance_derivative(int a, int b) const;

    /// @brief Radial derivative of the A–B pair contribution to
    ///        \f$E^{(2)}+E^{(3)}\f$ at **fixed** Δq [eV/Å].
    ///
    /// Both ordered terms of the symmetric double sums in
    /// @ref second_order_energy / @ref third_order_energy collapse onto the
    /// unordered pair (A,B), A ≠ B:
    ///
    /// \f[ \frac{\partial}{\partial R_{AB}}\Big[
    ///        \gamma_{AB}\Delta q_A\Delta q_B
    ///      + \tfrac13\big(\Delta q_A^2\Delta q_B\Gamma_{AB}
    ///                   + \Delta q_B^2\Delta q_A\Gamma_{BA}\big)\Big] \f]
    ///
    /// Dropping either Γ ordering breaks \f$F=-\nabla E\f$ by exactly the same
    /// asymmetry that makes \f$V^{(3)}\f$ need two contractions. A == B
    /// returns 0 (on-site terms carry no geometry).
    /// @param a Atom index A (0-based).
    /// @param b Atom index B (0-based).
    /// @param dq Mulliken charge fluctuations Δq [e], size n.
    /// @return \f$\partial(E^{(2)}+E^{(3)})_{AB}/\partial R\f$ [eV/Å].
    double pair_energy_distance_derivative(int a, int b,
                                           const std::vector<double>& dq) const;

    /// @brief Second-order SCC energy \f$E^{(2)}\f$ [eV].
    /// @param dq Mulliken charge fluctuations Δq [e], size n.
    /// @return \f$E^{(2)}\f$ [eV].
    double second_order_energy(const std::vector<double>& dq) const;

    /// @brief Third-order SCC energy \f$E^{(3)}\f$ [eV].
    /// @param dq Mulliken charge fluctuations Δq [e], size n.
    /// @return \f$E^{(3)}\f$ [eV].
    double third_order_energy(const std::vector<double>& dq) const;

    /// @brief Second-order atomic potential
    ///        \f$V^{(2)}_A=\partial E^{(2)}/\partial\Delta q_A\f$ [eV/e].
    /// @param dq Mulliken charge fluctuations Δq [e], size n.
    /// @return Per-atom \f$V^{(2)}\f$ [eV/e], size n.
    std::vector<double>
    second_order_potential(const std::vector<double>& dq) const;

    /// @brief Third-order atomic potential
    ///        \f$V^{(3)}_A=\partial E^{(3)}/\partial\Delta q_A\f$ [eV/e].
    /// @param dq Mulliken charge fluctuations Δq [e], size n.
    /// @return Per-atom \f$V^{(3)}\f$ [eV/e], size n.
    std::vector<double>
    third_order_potential(const std::vector<double>& dq) const;

    /// @brief Full SCC atomic potential
    ///        \f$\partial(E^{(2)}+E^{(3)})/\partial\Delta q_A\f$ [eV/e].
    /// @param dq Mulliken charge fluctuations Δq [e], size n.
    /// @return Per-atom \f$V^{(2)}+V^{(3)}\f$ [eV/e], size n.
    std::vector<double> potential(const std::vector<double>& dq) const;

  private:
    void require_size(const std::vector<double>& dq) const;

    std::vector<double> u_;
    std::vector<double> ud_;
    std::vector<double> r_;
    int n_ = 0;
};

} // namespace internal
} // namespace semiempirical
} // namespace cpu
} // namespace ATV

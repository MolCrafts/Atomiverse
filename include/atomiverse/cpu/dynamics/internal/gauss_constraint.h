/// @file cpu/dynamics/internal/gauss_constraint.h
/// @brief Gauss's least-constraint kinetic thermostats (isokinetic, flow,
///        transverse): the constrained velocity components and the
///        multiplier. Used by `cpu::md::GearPC`.
///
/// A thermostat holds \(K_w = \tfrac12\sum_i m_i |w_i|^2\) constant, \(w_i\)
/// the PECULIAR velocity of the constrained components: on each axis \(d\),
///  - `Barycentric`: \(w_{id} = v_{id} - V_d\), \(V\) the centre-of-mass
///    velocity;
///  - `Species`: \(w_{id} = v_{id} - u_{\nu d}\), \(u_\nu\) the
///    mass-weighted mean velocity of atom \(i\)'s species \(\nu\)
///    (`kAtom_type`), Petravic & Delhommelle eq. (3);
///  - `Free`: the axis is not constrained, \(w_{id} = 0\).
///
/// Gauss's principle of least constraint gives
/// \(\dot v_i = a_i - \alpha w_i\) with
/// \[
///   \alpha = \frac{\sum_i m_i\, w_i\cdot a_i}{\sum_i m_i |w_i|^2},
/// \]
/// \(a_i\) the acceleration from the TOTAL force (an external field
/// included). Because \(\sum_{i\in g} m_i w_i = 0\) for every group \(g\),
/// \(\mathrm{d}K_w/\mathrm{d}t = 0\) exactly, and the group mean velocities
/// (centre of mass, species flows) evolve as without the thermostat. With
/// zero centre-of-mass momentum, \(\alpha\) is J. Petravic and J.
/// Delhommelle, J. Chem. Phys. 118, 7477 (2003), eq. (9) (`Kinetic`), eq.
/// (10) (`Flow`; the peculiar force of eq. 10 gives the same sum because the
/// species peculiar momenta sum to zero) and eq. (11) (`Transverse`).
///
/// Degrees of freedom: each constrained axis contributes \(N - n_g\),
/// \(n_g\) its number of groups, so `Kinetic` has \(3N-3\), `Flow`
/// \(3N - 2 - n_\nu\) and `Transverse` \(2N - 2\).
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/dynamics/gauss_thermostat.h"

namespace ATV::cpu::md::internal {

/// @brief How one Cartesian axis enters the constrained kinetic energy.
enum class GaussAxis : unsigned char { Barycentric, Species, Free };

/// @brief Three read-only per-atom rows (x, y, z components).
using ConstRows3 = std::array<const double*, 3>;
/// @brief Three writable per-atom rows (x, y, z components).
using Rows3 = std::array<double*, 3>;

/// @brief The constraint geometry of one `GaussThermostat`.
class GaussConstraint {
  public:
    GaussConstraint() = default;
    /// @param kind Which kinetic temperature is constrained.
    /// @param axis Field axis 0..2 for `Flow` / `Transverse`.
    /// @throws std::invalid_argument for an axis outside 0..2.
    GaussConstraint(GaussKind kind, int axis);

    /// @brief Number the species present in rows [0, n) of `type`.
    /// @throws std::invalid_argument for a negative type.
    void bind(int n, const int* type);

    /// @brief Constrained degrees of freedom (class comment).
    long long dof() const;

    /// @brief How axis `d` is constrained.
    GaussAxis mode(int d) const { return mode_[d]; }

    /// @brief Peculiar velocities of rows [0, n): `w = v − group mean` on
    ///        constrained axes, 0 on a free axis.
    /// @throws std::invalid_argument for a type that was not bound.
    void peculiar(int n, const double* mass, const int* type, ConstRows3 v,
                  Rows3 w) const;

    /// @brief \(\alpha = \sum m\,w\cdot a / \sum m |w|^2\) [1/time].
    static double multiplier(int n, const double* mass, ConstRows3 w,
                             ConstRows3 a);

    /// @brief \(\sum_i m_i |w_i|^2\) (twice the constrained kinetic energy,
    ///        mass·velocity² units).
    static double twice_kinetic(int n, const double* mass, ConstRows3 w);

  private:
    std::array<GaussAxis, 3> mode_{
        GaussAxis::Barycentric, GaussAxis::Barycentric, GaussAxis::Barycentric};
    int n_atoms_ = 0;
    std::vector<int> species_of_type_; ///< type → dense species, −1 absent
    int n_species_ = 0;
};

} // namespace ATV::cpu::md::internal

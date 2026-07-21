/// @file cpu/potentials/eam_potential.h
/// @brief EAMPotential — analytic embedded-atom potential, constructed by
///        (`molrs::Element`, name) lookup into the in-tree static tables. No parsing.
#pragma once

#include <cmath>
#include <string_view>

#include "atomiverse/molrs.hpp"

namespace ATV {
namespace cpu {

/// @brief Analytic embedded-atom-method potential for one element.
///
/// The EAM ansatz (Daw & Baskes, *Phys. Rev. B* **29**, 6443 (1984)):
///
///     E = sum_i F(rho_i) + 1/2 sum_i sum_{j!=i} phi(r_ij)
///     rho_i = sum_{j!=i} f(r_ij)
///
/// The embedding energy `F` is a **nonlinear** function of the local density
/// `rho_i`, which is what makes EAM many-body: unlike a pair potential, the
/// energy does not decompose into a sum over independent pairs, and the force on
/// atom i depends on the environment of its neighbours, not just on the
/// separations from them. That is the whole point of this class as an
/// architecture probe — it needs a per-atom intermediate computed in a first pass
/// before any force can be evaluated in a second.
///
/// The parameterization is Sutton-Chen (*Philos. Mag. Lett.* **61**, 139 (1990)),
/// an analytic long-range Finnis-Sinclair form:
///
///     phi(r) = eps * (a/r)^n         f(r) = (a/r)^m         F(rho) = -eps*c*sqrt(rho)
///
/// Constructed by lookup, never by a factory function; an unsupported element or
/// parameterization name throws (no fallback).
class EAMPotential {
  public:
    /// @brief Look up an analytic-EAM parameterization.
    /// @param element Chemical element (molrs periodic-table enum).
    /// @param name Parameterization name, e.g. `"sutton-chen"`.
    /// @throws std::out_of_range if the (element, name) pair is not tabulated.
    EAMPotential(molrs::Element element, std::string_view name);

    /// @brief Pair term phi(r) [eV].
    /// @param r Separation [Angstrom].
    double phi(double r) const { return eps_ * std::pow(a_ / r, n_); }

    /// @brief d(phi)/dr [eV/Angstrom].
    /// @param r Separation [Angstrom].
    double dphi(double r) const {
        return -n_ * eps_ * std::pow(a_ / r, n_) / r;
    }

    /// @brief Density contribution f(r) of one neighbour [dimensionless].
    /// @param r Separation [Angstrom].
    double f(double r) const { return std::pow(a_ / r, m_); }

    /// @brief d(f)/dr [1/Angstrom].
    /// @param r Separation [Angstrom].
    double df(double r) const { return -m_ * std::pow(a_ / r, m_) / r; }

    /// @brief Embedding energy F(rho) [eV].
    /// @param rho Local density [dimensionless].
    double F(double rho) const { return -eps_ * c_ * std::sqrt(rho); }

    /// @brief dF/d(rho) [eV].
    ///
    /// Diverges as rho -> 0 (the sqrt has infinite slope at the origin). An atom
    /// with no neighbours inside the cutoff has rho = 0 and, correctly, no
    /// embedding force — the caller must special-case rho == 0 rather than
    /// evaluate this. Guarded here so a stray isolated atom cannot inject a NaN
    /// into the whole force array.
    /// @param rho Local density [dimensionless].
    double dF(double rho) const {
        if (rho <= 0.0)
            return 0.0;
        return -0.5 * eps_ * c_ / std::sqrt(rho);
    }

    /// @brief Energy scale eps [eV].
    double eps() const { return eps_; }
    /// @brief Length scale a — the fcc lattice constant [Angstrom].
    double a() const { return a_; }
    /// @brief Embedding coefficient c [dimensionless].
    double c() const { return c_; }
    /// @brief Repulsive-pair exponent n.
    int n() const { return n_; }
    /// @brief Density exponent m.
    int m() const { return m_; }

  private:
    int n_ = 0;
    int m_ = 0;
    double eps_ = 0.0;
    double c_ = 0.0;
    double a_ = 0.0;
};

} // namespace cpu
} // namespace ATV

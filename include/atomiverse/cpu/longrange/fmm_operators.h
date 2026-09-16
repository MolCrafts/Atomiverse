/// @file cpu/longrange/fmm_operators.h
/// @brief FMM translation operators M2M, M2L, L2L (White & Head-Gordon).
///
/// Backend-agnostic: no Frame / Component / MPI. Coefficient tables for the
/// real/complex solid-harmonic addition theorems are built in the constructor;
/// #apply is allocation-free aside from reading the precomputed solid
/// harmonics of the shift vector.
///
/// Reference: White & Head-Gordon, *J. Chem. Phys.* **101**, 6593 (1994),
/// DOI: 10.1063/1.468354. M2M and L2L are exact within truncation order p;
/// only M2L introduces a truncation error that decays geometrically with p
/// under the well-separated criterion (Greengard & Rokhlin 1987).
///
/// Kernel: dimensionless Coulomb Σ q_i q_j / r_ij (eV conversion in link 21).
#pragma once

#include <stdexcept>

#include "atomiverse/cpu/longrange/multipole_expansion.h"

namespace ATV {
namespace longrange {

/// @brief Multipole-to-multipole translation (child → parent centre).
///
/// Exact within order p. Shift vector is taken as
/// t = src.center() − dst.center() (child minus parent).
class M2M {
  public:
    /// @param order Truncation order p ≥ 0 (must match expansions).
    /// @throws std::invalid_argument if order < 0.
    explicit M2M(int order);

    int order() const { return order_; }

    /// @brief Add the translated source multipole into dst (does not clear dst).
    ///
    /// @param src Child multipole expansion.
    /// @param dst Parent multipole expansion (same order as src).
    /// @throws std::invalid_argument on order mismatch.
    void apply(const MultipoleExpansion& src, MultipoleExpansion& dst) const;

  private:
    int order_;
};

/// @brief Multipole-to-local translation (well-separated far field).
///
/// Truncation error decays geometrically with p for separation ratio c > 1
/// (Greengard–Rokhlin bound ∼ (1/(c−1)) (1/c)^p).
class M2L {
  public:
    /// @param order Truncation order p ≥ 0.
    /// @throws std::invalid_argument if order < 0.
    explicit M2L(int order);

    int order() const { return order_; }

    /// @brief Add the local expansion of src's multipole into dst.
    ///
    /// Shift t = src.center() − dst.center() (multipole centre relative to
    /// local centre). Requires t ≠ 0.
    ///
    /// @param src Source multipole.
    /// @param dst Destination local expansion (same order).
    void apply(const MultipoleExpansion& src, LocalExpansion& dst) const;

  private:
    int order_;
};

/// @brief Local-to-local translation (parent → child centre).
///
/// Exact within order p. Shift t = dst.center() − src.center() (child minus
/// parent).
class L2L {
  public:
    /// @param order Truncation order p ≥ 0.
    /// @throws std::invalid_argument if order < 0.
    explicit L2L(int order);

    int order() const { return order_; }

    /// @brief Add the translated parent local expansion into dst.
    ///
    /// @param src Parent local expansion.
    /// @param dst Child local expansion (same order).
    void apply(const LocalExpansion& src, LocalExpansion& dst) const;

  private:
    int order_;
};

} // namespace longrange
} // namespace ATV

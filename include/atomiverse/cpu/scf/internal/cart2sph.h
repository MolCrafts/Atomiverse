/// @file cpu/scf/internal/cart2sph.h
/// @brief CartToSph — per-shell Cartesian→real-spherical AO transform matrices.
///
/// Two closely related matrices per angular momentum l ≤ kMaxL:
///
///   - `matrix(l)` — the PySCF/libcint convention coefficients, identical to
///     `pyscf.gto.mole.cart2sph(l, normalized='sp')`: s/p are identities,
///     d/f carry the real-Y_lm normalization. These are the community-standard
///     constants (real solid harmonics in the m = −l..+l ordering of libcint;
///     Schlegel & Frisch, Int. J. Quantum Chem. 54, 83 (1995), DOI
///     10.1002/qua.560540202; Helgaker–Jørgensen–Olsen ch. 6) and are pinned
///     to 1e-12 by tests/cpu/scf/test_cart2sph.cpp.
///
///   - `block(l)` — the same transform expressed in THIS engine's Cartesian
///     convention (each Cartesian component unit-normalized by `ShellList`):
///     block(l) = D(l) · matrix(l), where the diagonal
///     D_i = √(4π (2lx−1)!!(2ly−1)!!(2lz−1)!! / (2l+1)!!) absorbs the
///     double-factorial normalization ratio between components (√3 between
///     d_xx and d_xy — the classic pitfall). Sandwiching the unit-normalized
///     Cartesian overlap with block(l) yields exactly the spherical overlap.
///
/// The matrices are a static data table (l ≤ 4; the g rows are needed by
/// the def2-JKFIT auxiliary basis of the density-fitting path),
/// not generated at run time: the community m-ordering/phase convention is
/// exactly what downstream comparisons against PySCF require, and
/// `core/real_harmonics.h`'s generator deliberately leaves m-ordering
/// unspecified. The unit test cross-checks the spanned space against the
/// closed-form overlap identity instead.
///
/// Why a table and not the `real_solid_harmonics(l)` generator: that
/// generator deliberately leaves the m-ordering unspecified, which makes a
/// convention-pinned transform from it ill-defined; the constants are
/// therefore tabulated in the community convention and verified against
/// PySCF values at 1e-12.
#pragma once

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Cartesian→spherical transform matrices for shells with l ≤ 4
///        (g rows serve the density-fitting auxiliary basis only).
class CartToSph {
  public:
    /// @brief Build the transform tables (constructor computes block(l) from
    ///        matrix(l) by absorbing the double-factorial diagonal).
    CartToSph();

    /// @brief PySCF-convention coefficients, (l+1)(l+2)/2 × (2l+1).
    /// @param l Angular momentum (0..4); throws std::out_of_range above.
    /// @return Coefficient matrix identical to
    ///         `pyscf.gto.mole.cart2sph(l, normalized='sp')` [dimensionless].
    const xt::xtensor<double, 2>& matrix(int l) const;

    /// @brief Engine-convention transform (unit-normalized Cartesian AOs),
    ///        (l+1)(l+2)/2 × (2l+1).
    /// @param l Angular momentum (0..4); throws std::out_of_range above.
    /// @return D(l)·matrix(l) — sandwiching the engine's Cartesian matrices
    ///         with this block yields the spherical-basis matrices
    ///         [dimensionless].
    const xt::xtensor<double, 2>& block(int l) const;

  private:
    xt::xtensor<double, 2> matrix_[5];
    xt::xtensor<double, 2> block_[5];
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

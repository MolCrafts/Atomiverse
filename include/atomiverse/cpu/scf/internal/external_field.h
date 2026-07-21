/// @file cpu/scf/internal/external_field.h
/// @brief Private uniform external electric-field contribution to the core
///        Hamiltonian and classical nuclear energy.
///
/// Origin R₀ is the center of nuclear charge — the same convention as the
/// multipole observer (link 03). The one-electron operator is
/// h += F · ⟨(r − R₀)⟩ (electron charge −1 a.u.); the nuclear term is
/// E += −Σ_A Z_A F · (R_A − R₀).
///
/// **Finite-field collapse hazard:** the energy is unbounded below under a
/// static field; keep |F| ≲ 1e-4 a.u. (especially with diffuse bases) or the
/// SCF may "ionize". Documented in ch20.
///
/// Obara & Saika, J. Chem. Phys. 84, 3963 (1986), DOI 10.1063/1.450106.
#pragma once

#include <array>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Uniform external electric field in the core Hamiltonian.
class ExternalField {
  public:
    /// @brief Construct from field [a.u.] and QM nuclei; computes R₀.
    /// @param field      Electric field (Ex, Ey, Ez) [a.u. = Ha/(e·Bohr)].
    /// @param atoms      QM nuclei (Z, positions [Bohr]).
    /// @throws std::invalid_argument on non-finite field or empty atoms
    ///         with non-zero field, or zero total nuclear charge.
    ExternalField(const std::array<double, 3>& field,
                  const std::vector<native::Atom>& atoms);

    /// @brief One-electron field matrix Σ_a F_a ⟨μ|(r−R₀)_a|ν⟩.
    /// @param shells Cartesian AO shell descriptors (engine convention).
    /// @return (n_cart × n_cart) matrix [Hartree]; caller sandwiches with
    ///         c2s when the pack is spherical.
    [[nodiscard]] xt::xtensor<double, 2>
    core_matrix(const std::vector<native::Shell>& shells) const;

    /// @brief Classical nuclear field energy −Σ_A Z_A F · (R_A − R₀).
    /// @param atoms QM nuclei (Z, positions [Bohr]); typically the same
    ///              list used at construction.
    /// @return Energy contribution [Hartree].
    [[nodiscard]] double
    nuclear_energy(const std::vector<native::Atom>& atoms) const;

    /// @brief Expansion origin R₀ (center of nuclear charge) [Bohr].
    [[nodiscard]] const std::array<double, 3>& origin_bohr() const {
        return origin_;
    }

    /// @brief Field components [a.u.].
    [[nodiscard]] const std::array<double, 3>& field() const { return field_; }

    /// @brief True when |F| is numerically zero (no-op path).
    [[nodiscard]] bool is_zero() const;

  private:
    std::array<double, 3> field_{};  ///< Field [a.u.].
    std::array<double, 3> origin_{}; ///< Nuclear-charge center [Bohr].
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

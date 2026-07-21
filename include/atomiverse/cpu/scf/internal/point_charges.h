/// @file cpu/scf/internal/point_charges.h
/// @brief Private point-charge embedding contributions to the core
///        Hamiltonian and classical energy (QM/MM electrostatic embedding,
///        QM side).
///
/// Positions enter in Å and are converted to Bohr at construction. The
/// one-electron matrix reuses the nuclear-attraction primitive
/// (`native::nuclear_block`) with charge scaling. No free-function
/// factories — construct `PointChargeEmbedding{sites}` only.
///
/// Warshel & Levitt, J. Mol. Biol. 103, 227 (1976),
/// DOI 10.1016/0022-2836(76)90311-9; Obara & Saika, J. Chem. Phys. 84,
/// 3963 (1986), DOI 10.1063/1.450106.
#pragma once

#include <array>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief One external MM-like point charge (input units).
struct PointChargeSite {
    double x = 0; ///< Position x [Å].
    double y = 0; ///< Position y [Å].
    double z = 0; ///< Position z [Å].
    double q = 0; ///< Charge [e].
};

/// @brief Point-charge electrostatic embedding: Hcore and classical energy.
///
/// Owns Bohr-converted sites after construction. Throw on non-finite input.
class PointChargeEmbedding {
  public:
    /// @brief Construct from external charges; converts Å → Bohr.
    /// @param sites Point charges (positions [Å], charges [e]).
    /// @throws std::invalid_argument on empty-ok; non-finite values throw.
    explicit PointChargeEmbedding(const std::vector<PointChargeSite>& sites);

    /// @brief One-electron embedding matrix V_μν = Σ_C −q_C ⟨μ|1/|r−R_C||ν⟩.
    /// @param shells Cartesian AO shell descriptors (engine convention).
    /// @return (n_cart × n_cart) matrix [Hartree]; caller sandwiches with
    ///         c2s when the pack is spherical.
    [[nodiscard]] xt::xtensor<double, 2>
    core_matrix(const std::vector<native::Shell>& shells) const;

    /// @brief Classical QM–MM nuclear attraction Σ_A Σ_C Z_A q_C / |R_A−R_C|.
    /// @param atoms QM nuclei (Z, positions [Bohr]).
    /// @return Energy contribution [Hartree].
    [[nodiscard]] double
    nuclear_energy(const std::vector<native::Atom>& atoms) const;

    /// @brief Classical MM–MM self energy Σ_{C<C'} q_C q_C' / |R_C−R_C'|.
    /// @return Energy contribution [Hartree] (0 for fewer than two charges).
    [[nodiscard]] double self_energy() const;

    /// @brief Number of embedded charges (dimensionless).
    [[nodiscard]] int n_charges() const {
        return static_cast<int>(q_.size());
    }

  private:
    std::vector<std::array<double, 3>> pos_bohr_; ///< Charge positions [Bohr].
    std::vector<double> q_;                       ///< Charges [e].
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

/// @file cpu/scf/internal/point_charges.h
/// @brief Private point-charge embedding: Hcore, classical energy, and
///        analytic QM/MM embedding gradients (electron + nucleus + site).
///
/// Positions enter in Å and are converted to Bohr at construction. The
/// one-electron matrix reuses the nuclear-attraction primitive
/// (`native::nuclear_block`) with charge scaling. Gradients contract
/// primitive nuclear-attraction derivatives on the fly (no dV tensor).
///
/// Warshel & Levitt, J. Mol. Biol. 103, 227 (1976); Obara & Saika, J. Chem.
/// Phys. 84, 3963 (1986).
#pragma once

#include <array>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/component/qm_engine.h"
#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Point-charge electrostatic embedding: Hcore, classical energy,
///        and analytic gradients on QM nuclei and MM sites.
///
/// Owns Bohr-converted sites after construction. Throw on non-finite input.
class PointChargeEmbedding {
  public:
    /// @brief Construct from external charges; converts Å → Bohr.
    /// @param sites Point charges (positions [Å], charges [e]).
    /// @throws std::invalid_argument on non-finite values.
    explicit PointChargeEmbedding(const std::vector<cpu::EmbeddingSite>& sites);

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

    /// @brief Analytic ∂E_embed/∂R_A on QM nuclei from electron–MM + nucleus–MM.
    ///
    /// Contracts Tr[P · dV/dR_A] with the nuclear-attraction raising formula
    /// plus the classical Coulomb term. All [Hartree/Bohr].
    /// @param[out] grad [n_atoms, 3] accumulator (added into).
    /// @param shells Cartesian AO shells (same order as P_cart).
    /// @param P_cart Total density in the Cartesian AO basis (dimensionless).
    /// @param shell_to_atom Map shell index → atom index (size = n_shells).
    /// @param atoms QM nuclei for the classical nucleus–MM leg.
    void accumulate_qm_center_gradient(
        xt::xtensor<double, 2>& grad,
        const std::vector<native::Shell>& shells,
        const xt::xtensor<double, 2>& P_cart,
        const std::vector<int>& shell_to_atom,
        const std::vector<native::Atom>& atoms) const;

    /// @brief Analytic ∂E_embed/∂R_C on embedding sites (electron + nucleus).
    ///
    /// Electron leg uses translational invariance of the nuclear-attraction
    /// integral; nucleus leg is the sign-flipped classical Coulomb. Does
    /// **not** include MM–MM self-energy (see `accumulate_self_gradient`).
    /// @param[out] site_grad [n_charges, 3] [Hartree/Bohr] (added into).
    /// @param shells Cartesian AO shells.
    /// @param P_cart Total density in the Cartesian AO basis.
    /// @param atoms QM nuclei.
    void accumulate_site_gradient(xt::xtensor<double, 2>& site_grad,
                                  const std::vector<native::Shell>& shells,
                                  const xt::xtensor<double, 2>& P_cart,
                                  const std::vector<native::Atom>& atoms) const;

    /// @brief Analytic ∂E_self/∂R_C of the MM–MM Coulomb self-energy.
    /// @param[out] site_grad [n_charges, 3] [Hartree/Bohr] (added into).
    void accumulate_self_gradient(xt::xtensor<double, 2>& site_grad) const;

    /// @brief Number of embedded charges (dimensionless).
    [[nodiscard]] int n_charges() const { return static_cast<int>(q_.size()); }

  private:
    std::vector<std::array<double, 3>> pos_bohr_; ///< Charge positions [Bohr].
    std::vector<double> q_;                       ///< Charges [e].
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

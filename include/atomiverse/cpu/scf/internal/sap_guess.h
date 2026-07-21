/// @file cpu/scf/internal/sap_guess.h
/// @brief internal::SapGuess — superposition-of-atomic-potentials (SAP)
///        initial-guess Fock matrix, F = T + V_SAP, assembled by quadrature
///        on the molecular DFT grid.
///
/// **Internal SCF helper.** Does NOT inherit `cpu::Component` and is not
/// registered into any Plan. Owned transiently by the concrete method's
/// initial-guess dispatch (`MeanFieldMethod::drive_scf_core_`) and by tests.
///
/// SAP guess (S. Lehtola, J. Chem. Theory Comput. 15, 1593 (2019),
/// DOI 10.1021/acs.jctc.8b01089): the effective one-electron potential is a
/// superposition of tabulated exchange-only LDA effective atomic radial
/// potentials,
///
///   V_SAP(r) = Σ_A V_A(|r − R_A|),
///   (V_μν)   = Σ_g w_g φ_μ(r_g) V_SAP(r_g) φ_ν(r_g),   F = T + V_SAP,
///
/// with V_A in the Gaussian-fitted error-function expansion form of
/// sap_data.h (S. Lehtola, L. Visscher, E. Engel, J. Chem. Phys. 152, 144105
/// (2020), DOI 10.1063/5.0004046, transcribed from PySCF's
/// sap_grasp_large.dat). One diagonalization of `F` (done by the caller)
/// yields the initial density — no density superposition step is needed.
///
/// **Spin-averaging limitation:** V_SAP is spin-free, so open-shell runs
/// derive both spin channels from the same orbitals (the existing split +
/// HOMO-LUMO-mixing rule at the dispatch site); the guess cannot break spin
/// symmetry by itself.
///
/// Unsupported elements throw (`SapAtomicPotential`) — no fallback values.
#pragma once

#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief SAP guess Fock matrix in the working AO basis.
///
/// Constructor-only class (no factory free function). The quadrature runs in
/// the Cartesian AO basis of `shells`; for spherical-convention bases the
/// caller passes the pack's Cartesian→spherical transform so `V` and `F`
/// come out in the same working dimension as `T`.
class SapGuess {
  public:
    /// @brief Assemble F = T + V_SAP by quadrature on `grid`.
    /// @param shells Cartesian AO shell descriptors (one entry per Cartesian
    ///               AO; centers in Bohr) — `IntegralPack::shells`.
    /// @param T      Kinetic-energy matrix [Hartree] in the WORKING AO basis
    ///               (spherical when the basis convention is spherical) —
    ///               `IntegralPack::T`.
    /// @param atoms  Atom descriptors (Z, position in Bohr) whose tabulated
    ///               SAP potentials are superposed — `IntegralPack::atoms`.
    /// @param grid   Molecular quadrature grid (points + weights, Bohr).
    /// @param c2s    Block-diagonal Cartesian→spherical transform
    ///               (n_cart × n_work) — `IntegralPack::c2s`; pass an EMPTY
    ///               tensor for Cartesian-convention bases.
    /// @throws std::invalid_argument on an element without a tabulated SAP
    ///         potential, or on a T/c2s dimension mismatch. No fallback.
    SapGuess(const std::vector<native::Shell>& shells, const RealMatrix& T,
             const std::vector<native::Atom>& atoms, const MolecularGrid& grid,
             const xt::xtensor<double, 2>& c2s);

    /// @brief SAP potential matrix V_SAP [Hartree] in the working AO
    ///        dimension of `T` (symmetric).
    RealMatrix V;

    /// @brief The assembled guess Fock matrix F = T + V_SAP [Hartree], in
    ///        the working AO dimension of `T`.
    RealMatrix F;
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

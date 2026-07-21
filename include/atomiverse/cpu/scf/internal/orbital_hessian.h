/// @file cpu/scf/internal/orbital_hessian.h
/// @brief **PRIVATE** MO-basis orbital Hessian A/B kernel for the closed-shell
///        (RHF) reference: matrix-free σ-vectors σ± = (A ± B)·v plus explicit
///        dense builds for small systems — the shared foundation of stability
///        analysis (this link), SOSCF, CPHF, and TDHF.
///
/// Closed-shell spatial-orbital blocks (real orbitals, chemist notation
/// (pq|rs), all values Hartree; i,j occupied, a,b virtual):
///
///   ¹A_ai,bj = δ_ab δ_ij (ε_a − ε_i) + 2(ai|jb) − (ij|ab)
///   ¹B_ai,bj = 2(ai|bj) − (aj|bi)
///   ³A_ai,bj = δ_ab δ_ij (ε_a − ε_i) − (ij|ab)
///   ³B_ai,bj = −(aj|bi)
///
/// For real orbitals (ai|jb) = (ai|bj), so the two channel sums reduce to
///   ¹(A+B) = δδΔε + 4(ai|bj) − (ij|ab) − (aj|bi)
///   ³(A+B) = δδΔε           − (ij|ab) − (aj|bi)
///   (A−B)  = δδΔε           − (ij|ab) + (aj|bi)   (channel-independent)
///
/// The constructor performs the O(N⁵) occ-virt quarter transform of the AO
/// ERI tensor once; every σ product afterwards is an O(n_ov²) contraction of
/// the stored MO blocks — no n_ov × n_ov matrix is ever formed on the σ path.
/// Composite index convention: p = a·n_occ + i (virtual-major).
///
/// References: Seeger & Pople, J. Chem. Phys. 66, 3045 (1977),
/// DOI 10.1063/1.434318; Szabo & Ostlund §3.8.7; Helgaker, Jørgensen, Olsen,
/// *Molecular Electronic-Structure Theory*, ch. 10.
#pragma once

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Spin-adapted excitation channel of the closed-shell orbital
///        Hessian: `singlet` (RHF→RHF, internal stability) or `triplet`
///        (RHF→UHF, external / spin-symmetry-breaking stability).
enum class Channel { singlet, triplet };

/// @brief The closed-shell MO orbital Hessian A/B kernel: constructor does
///        the occ-virt MO-ERI transform once; members provide matrix-free
///        σ± products and explicit dense (A ± B) builds per channel.
class OrbitalHessian {
  public:
    /// @brief Transform the AO ERI tensor to the occ-virt MO blocks
    ///        (ai|bj) and (ij|ab) needed by both channels.
    /// @param det    Converged restricted determinant supplying C, ε and the
    ///               integer occupations (occupied ⇔ f_i = 1). ε in Hartree.
    /// @param eri_ao Dense AO two-electron tensor (μν|λσ) [Hartree] in the
    ///               determinant's working AO dimension (spherical count for
    ///               spherical-convention bases — same dimension as C rows).
    /// @throws std::invalid_argument on an unrestricted determinant,
    ///         fractional occupations, an AO-dimension mismatch between C
    ///         and `eri_ao`, or n_virt = 0 (no orbital rotations exist —
    ///         e.g. He/STO-3G; no fallback).
    OrbitalHessian(const SlaterDeterminant& det,
                   const xt::xtensor<double, 4>& eri_ao);

    /// @brief Number of occupied spatial orbitals (f_i = 1 columns).
    /// @return n_occ (dimensionless count).
    int n_occ() const { return no_; }
    /// @brief Number of virtual spatial orbitals (f_a = 0 columns).
    /// @return n_virt (dimensionless count).
    int n_virt() const { return nv_; }
    /// @brief Dimension of the occ-virt rotation space.
    /// @return n_occ · n_virt (dimensionless count).
    int n_ov() const { return no_ * nv_; }

    /// @brief Matrix-free product σ = (A + B)_ch · v.
    /// @param v  Trial vector, length n_ov(), composite index a·n_occ + i
    ///           (dimensionless amplitudes).
    /// @param ch Excitation channel (singlet or triplet).
    /// @return (A + B)·v [Hartree].
    /// @throws std::invalid_argument when v.size() != n_ov().
    RealVector sigma_plus(const RealVector& v, Channel ch) const;

    /// @brief Matrix-free product σ = (A − B)_ch · v. For real closed-shell
    ///        orbitals (A − B) is channel-independent; `ch` is kept so the
    ///        RPA-facing interface stays uniform for CPHF/TDHF reuse.
    /// @param v  Trial vector, length n_ov() (dimensionless amplitudes).
    /// @param ch Excitation channel (singlet or triplet).
    /// @return (A − B)·v [Hartree].
    /// @throws std::invalid_argument when v.size() != n_ov().
    RealVector sigma_minus(const RealVector& v, Channel ch) const;

    /// @brief Explicit dense (A + B)_ch — small systems and validation.
    /// @param ch Excitation channel.
    /// @return Symmetric n_ov × n_ov matrix [Hartree].
    RealMatrix dense_plus(Channel ch) const;

    /// @brief Explicit dense (A − B)_ch — small systems and validation.
    /// @param ch Excitation channel.
    /// @return Symmetric n_ov × n_ov matrix [Hartree].
    RealMatrix dense_minus(Channel ch) const;

    /// @brief Diagonal of (A + B)_ch — the standard Davidson preconditioner
    ///        for the matrix-free lowest-root solve (also reusable by the
    ///        CPHF/TDHF iterative solvers).
    /// @param ch Excitation channel.
    /// @return Length-n_ov diagonal [Hartree].
    RealVector diagonal_plus(Channel ch) const;

  private:
    /// @brief Full four-index MO transform of a dense chemist-order tensor:
    ///        out[p,q,r,s] = Σ_{μνλσ} C0(μ,p) C1(ν,q) C2(λ,r) C3(σ,s) T[μνλσ].
    static xt::xtensor<double, 4>
    transform_block_(const xt::xtensor<double, 4>& eri_ao,
                     const xt::xtensor<double, 2>& C0,
                     const xt::xtensor<double, 2>& C1,
                     const xt::xtensor<double, 2>& C2,
                     const xt::xtensor<double, 2>& C3);

    int no_ = 0; ///< Occupied spatial orbitals.
    int nv_ = 0; ///< Virtual spatial orbitals.
    RealVector eps_occ_;  ///< ε_i of the occupied columns [Hartree].
    RealVector eps_virt_; ///< ε_a of the virtual columns [Hartree].
    xt::xtensor<double, 4> ovov_; ///< (ai|bj), layout [a,i,b,j] [Hartree].
    xt::xtensor<double, 4> oovv_; ///< (ij|ab), layout [i,j,a,b] [Hartree].
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

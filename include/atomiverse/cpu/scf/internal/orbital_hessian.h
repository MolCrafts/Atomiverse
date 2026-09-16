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

/// @brief Result of the shared direct random-phase-approximation (dRPA) /
///        Casida (linear-response) Hermitian reduction
///        \((A-B)^{1/2}(A+B)(A-B)^{1/2}\) — the single solve behind both
///        `RHF::direct_rpa` (reads `omega` + `e_c`) and `RHF::g0w0` (reads
///        `omega` + the `X`/`Y` amplitude matrices its screened Coulomb
///        consumes). Lightweight value object; \(T,T^{-1}\) stay method-private.
struct DirectRpaResult {
    /// @brief Positive dRPA excitation energies \(\Omega_s\) [Hartree],
    ///        ascending, length n_ov (= √eigenvalues of the reduced matrix).
    RealVector omega;
    /// @brief dRPA correlation energy \(E_c = \tfrac12(\sum_s\Omega_s -
    ///        \mathrm{Tr}\,A)\) [Hartree].
    double e_c = 0.0;
    /// @brief \(\mathrm{Tr}\,A\) of the supplied \(A\) matrix [Hartree]
    ///        (diagnostic; enters the plasmon formula).
    double tr_A = 0.0;
    /// @brief RPA X amplitudes, row s = state s [n_ov × n_ov] (dimensionless).
    RealMatrix X;
    /// @brief RPA Y amplitudes, row s = state s [n_ov × n_ov] (dimensionless).
    RealMatrix Y;
};

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

    /// @brief Explicit dense Casida **A** matrix with global-hybrid exact-
    ///        exchange fraction `alpha_hf` (dimensionless in [0, 1]).
    ///
    /// Closed-shell spatial blocks (Hartree; real orbitals):
    ///   ¹A_ai,bj = δ_ab δ_ij (ε_a − ε_i) + 2(ai|jb) − α_hf (ij|ab)
    ///   ³A_ai,bj = δ_ab δ_ij (ε_a − ε_i)           − α_hf (ij|ab)
    ///
    /// `alpha_hf = 1` recovers the pure-HF A used by CIS / RHF-TDA;
    /// hybrid TDDFT-TDA uses the functional's `hybrid_exchange_fraction`
    /// (0.20 B3LYP, 0.25 PBE0) and adds the semi-local \(f^{\mathrm{xc}}\)
    /// kernel separately (Dreuw & Head-Gordon, Chem. Rev. 105, 4009 (2005)).
    /// @param ch        Excitation channel.
    /// @param alpha_hf  Exact-exchange admixture in [0, 1].
    /// @return Symmetric n_ov × n_ov matrix [Hartree].
    RealMatrix dense_A(Channel ch, double alpha_hf) const;

    /// @brief Explicit dense Casida **B** matrix with global-hybrid exact-
    ///        exchange fraction `alpha_hf` (dimensionless in [0, 1]).
    ///
    /// Closed-shell spatial blocks (Hartree; real orbitals):
    ///   ¹B_ai,bj = 2(ai|bj) − α_hf (aj|bi)
    ///   ³B_ai,bj =           − α_hf (aj|bi)
    ///
    /// `alpha_hf = 1` recovers pure-HF B (TDHF); pure KS (`α = 0`) keeps only
    /// the Coulomb coupling on the singlet channel. Semi-local \(f^{\mathrm{xc}}\)
    /// is added by the full LR-TDDFT path (same adiabatic kernel on A and B).
    /// @param ch        Excitation channel.
    /// @param alpha_hf  Exact-exchange admixture in [0, 1].
    /// @return Symmetric n_ov × n_ov matrix [Hartree] (no orbital-energy
    ///         diagonal — that lives only on A).
    RealMatrix dense_B(Channel ch, double alpha_hf) const;

    /// @brief Diagonal of (A + B)_ch — the standard Davidson preconditioner
    ///        for the matrix-free lowest-root solve (also reusable by the
    ///        CPHF/TDHF iterative solvers).
    /// @param ch Excitation channel.
    /// @return Length-n_ov diagonal [Hartree].
    RealVector diagonal_plus(Channel ch) const;

    /// @brief Dense direct-RPA / Casida Hermitian reduction on this reference:
    ///        \(A=\)`dense_A(ch,alpha_hf)`, \(B=\)`dense_B(ch,alpha_hf)`,
    ///        \(M=(A-B)^{1/2}(A+B)(A-B)^{1/2}\), then a symmetric
    ///        eigendecomposition \(M=Z\,\omega^2\,Z^\top\). Returns the positive
    ///        excitation energies \(\Omega_s=\sqrt{\omega_s^2}\), the plasmon
    ///        correlation energy \(E_c=\tfrac12(\sum_s\Omega_s-\mathrm{Tr}\,A)\),
    ///        and the RPA \(X,Y\) amplitudes. The one solve shared by
    ///        `RHF::direct_rpa` (dRPA correlation) and `RHF::g0w0` (screened W).
    ///
    /// The \((A-B)^{\pm1/2}\) factors use `linalg::sym_power`; a positive-
    /// definiteness guard on \(\lambda_{\min}(A-B)\le\)`kAmBFloor` and the
    /// \(\omega^2\) guard at `kOmega2Tol` are preserved from the pre-refactor
    /// per-method solves (no fallback — both throw).
    /// @param ch       Excitation channel (dRPA uses `singlet`).
    /// @param alpha_hf Exact-exchange admixture in [0, 1] (dRPA uses 0).
    /// @return Excitation energies, \(E_c\), \(\mathrm{Tr}\,A\), and \(X,Y\).
    /// @throws std::runtime_error when \(A-B\) is not positive definite or an
    ///         \(\omega^2\) root is negative / numerically zero (no fallback).
    DirectRpaResult direct_rpa(Channel ch, double alpha_hf) const;

  private:
    /// @brief Full four-index MO transform of a dense chemist-order tensor:
    ///        out[p,q,r,s] = Σ_{μνλσ} C0(μ,p) C1(ν,q) C2(λ,r) C3(σ,s) T[μνλσ].
    static xt::xtensor<double, 4> transform_block_(
        const xt::xtensor<double, 4>& eri_ao, const xt::xtensor<double, 2>& C0,
        const xt::xtensor<double, 2>& C1, const xt::xtensor<double, 2>& C2,
        const xt::xtensor<double, 2>& C3);

    int no_ = 0;                  ///< Occupied spatial orbitals.
    int nv_ = 0;                  ///< Virtual spatial orbitals.
    RealVector eps_occ_;          ///< ε_i of the occupied columns [Hartree].
    RealVector eps_virt_;         ///< ε_a of the virtual columns [Hartree].
    xt::xtensor<double, 4> ovov_; ///< (ai|bj), layout [a,i,b,j] [Hartree].
    xt::xtensor<double, 4> oovv_; ///< (ij|ab), layout [i,j,a,b] [Hartree].
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

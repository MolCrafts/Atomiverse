/// @file cpu/scf/internal/shell_pair.h
/// @brief ShellPairList: canonical shell pairs with Cauchy-Schwarz bounds for direct SCF.
///
/// Pre-computes Schwarz bounds Q_ij = sqrt((ij|ij)) for all canonical pairs (i>=j).
/// Used by direct-SCF Fock builders to skip negligible shell-quartet contributions.
///
/// @see ShellPairList::ShellPairList(const internal::IntegralView&, double)
/// @see ShellPairList::g_rhf, ShellPairList::jk_uhf, ShellPairList::j
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/integral_view.h"
#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
namespace scf {

struct ShellList; // forward declare

/// @brief Single canonical shell pair with Cauchy-Schwarz bound.
struct ShellPair {
    int i;          ///< Basis function index (i >= j).
    int j;          ///< Basis function index.
    double schwarz; ///< Cauchy-Schwarz bound sqrt((ij|ij)).
};

/// @brief Precomputed canonical shell pair list for direct SCF.
///
/// Owns copies of shells, atoms, and shell_data so it can outlive the view.
/// Only pairs with schwarz > threshold are retained.
struct ShellPairList {
    std::vector<ShellPair> pairs;      ///< Screened canonical pairs.
    std::vector<native::Shell> shells; ///< Owned shell copies.
    std::vector<native::Atom> atoms;   ///< Owned atom copies.
    std::vector<std::vector<double>>
        shell_data;  ///< Backing [α..., c...] per shell (length 2*n_prim).
    int n_basis = 0; ///< Total basis functions.
    double threshold = 1e-10; ///< Schwarz screening threshold.

    /// @brief Empty pair list (fields filled by the caller or left default).
    ShellPairList() = default;

    /// @brief Build the canonical shell pair list from method integral fields.
    ///
    /// Computes Q_ij = sqrt((ij|ij)) [Hartree^(1/2)] for all i >= j, retaining
    /// pairs where Q_ij > thr. The list owns all shell data (safe after the
    /// view's storage is rebuilt).
    ///
    /// @param ints Non-owning view (shells, shell_data, atoms).
    /// @param thr  Cauchy-Schwarz screening threshold [Hartree^(1/2),
    ///             default 1e-10].
    explicit ShellPairList(const internal::IntegralView& ints,
                           double thr = 1e-10);

    /// @brief Direct RHF G = J - 0.5*K via on-the-fly ERI with 8-fold symmetry
    ///        and Cauchy-Schwarz screening.
    ///
    /// Evaluates canonical shell quartets (pair_ij >= pair_kl, i>=j, k>=l) once
    /// and scatters contributions to G using full 8-fold permutation symmetry.
    /// Thread-local reduction ensures race-free OpenMP parallelism.
    ///
    /// @param P_total Total RHF density matrix P = 2 * P_occ [n x n, unitless].
    /// @param[out] G  Fused G = J - 0.5*K [n x n, Hartree].
    void g_rhf(const RealMatrix& P_total, RealMatrix& G) const;

    /// @brief Direct UHF J, K_alpha, K_beta via one on-the-fly ERI pass.
    ///
    /// Computes three matrices in a single quartet loop:
    ///   J       from P_total = P_alpha + P_beta  (Coulomb)
    ///   K_alpha from P_alpha                     (alpha exchange)
    ///   K_beta  from P_beta                      (beta exchange)
    ///
    /// @param P_total  Total density P_alpha + P_beta [n x n, unitless].
    /// @param P_alpha  Alpha spin density [n x n, unitless].
    /// @param P_beta   Beta spin density [n x n, unitless].
    /// @param[out] J       Coulomb matrix [n x n, Hartree].
    /// @param[out] K_alpha Alpha exchange matrix [n x n, Hartree].
    /// @param[out] K_beta  Beta exchange matrix [n x n, Hartree].
    void jk_uhf(const RealMatrix& P_total, const RealMatrix& P_alpha,
                const RealMatrix& P_beta, RealMatrix& J, RealMatrix& K_alpha,
                RealMatrix& K_beta) const;

    /// @brief Direct Coulomb-only J matrix for UKS (exchange via XC functional).
    ///
    /// @param P_total Total density P_alpha + P_beta [n x n, unitless].
    /// @param[out] J  Coulomb matrix [n x n, Hartree].
    void j(const RealMatrix& P_total, RealMatrix& J) const;
};

// ── Shell-block pair list (libcint/PySCF approach) ────────────────────

/// @brief Canonical shell-block pair with block-level Cauchy-Schwarz bound.
///
/// The Schwarz bound is Q_{AB} = sqrt(max_{ia,ib} |(ia ib|ia ib)|), covering
/// all AO combinations within the block pair.  Used to screen entire block
/// quartets before computing eri_block().
struct ShellBlockPair {
    int si;         ///< Block index i (si >= sj, canonical ordering).
    int sj;         ///< Block index j.
    double schwarz; ///< Block-level Schwarz bound sqrt(max_diag (AB|AB)).
};

/// @brief Precomputed canonical shell-block pair list for direct SCF.
///
/// Groups AO-level shells into shell blocks (s-shells: n_cart=1;
/// p-shells: n_cart=3 grouping px/py/pz together).  For STO-3G aspirin
/// (73 AOs), this gives 47 shell blocks vs 73 AO-level shells.
///
/// The list owns all shell-block data so it can outlive the integral view.
struct ShellBlockPairList {
    std::vector<ShellBlockPair> pairs;      ///< Screened canonical block pairs.
    std::vector<native::ShellBlock> blocks; ///< Owned shell-block descriptors.
    std::vector<std::vector<double>>
        block_data;  ///< Backing [α..., c...] per block (length 2*n_prim).
    int n_basis = 0; ///< Total AO basis functions.
    double threshold = 1e-10; ///< Schwarz screening threshold.

    /// @brief Empty block pair list (fields filled by the caller or left
    ///        default).
    ShellBlockPairList() = default;

    /// @brief Build the canonical shell-block pair list from method fields.
    ///
    /// Groups consecutive AO-level shells into shell blocks, then computes
    /// block-level Schwarz bounds Q_{AB} = sqrt(max_{ia,ib} (ia ib|ia ib))
    /// [Hartree^(1/2)] using eri_block().  Pairs with Q_{AB} ≤ thr are dropped.
    ///
    /// @param ints Non-owning integral view.
    /// @param thr  Cauchy-Schwarz screening threshold [Hartree^(1/2),
    ///             default 1e-10].
    explicit ShellBlockPairList(const internal::IntegralView& ints,
                                double thr = 1e-10);

    /// @brief Shell-block direct RHF G = J - 0.5*K (libcint/PySCF approach).
    ///
    /// Iterates canonical shell-block pair quartets (block_left >= block_right)
    /// and calls native::eri_block() to evaluate all AO ERIs within a
    /// shell-block at once, amortizing Boys function and R-table computation.
    ///
    /// Key performance gain over the AO-level ShellPairList::g_rhf:
    ///   For a (pp|pp) shell-block quartet with 3 primitives (STO-3G), this
    ///   evaluates 81 primitive quartets × (1 Boys+R-table per quartet + 81
    ///   cheap inner angular loops) vs 81×81 = 6561 full eri_prim_md() calls.
    ///
    /// OpenMP parallelism via thread-local G reduction (same as the AO-level
    /// variant). Scatter uses the same 8-fold ERI symmetry with the exchange
    /// mirror-permutation fix from the AO-level ShellPairList::g_rhf.
    ///
    /// @param P_total Total RHF density matrix P = 2*P_occ [n x n, unitless].
    /// @param[out] G  Fused G = J - 0.5*K [n x n, Hartree].
    void g_rhf(const RealMatrix& P_total, RealMatrix& G) const;

    /// @brief Shell-block direct exchange-only matrix K (KS-hybrid path).
    ///
    /// Builds the full exchange matrix `K_ij = sum_kl P_kl (ik|jl)` (NO 0.5
    /// factor, no Coulomb) using the same shell-block ERI amortization as
    /// `g_rhf`. `K` is normalized so that the pure RHF Fock is
    /// `F = Hcore + J - 0.5*K`; a KS global hybrid uses
    /// `F = Hcore + J - 0.5*alpha_hf*K + Vxc`.
    ///
    /// Used by `HamiltonianBuilder` for the B3LYP / PBE0 restricted hybrid
    /// Fock build. Far faster than the AO-level ShellPairList::jk_uhf exchange
    /// path on benzene-scale systems (shell-block Boys/R-table amortization).
    ///
    /// @param P Total RHF density matrix P = 2*P_occ [n x n, unitless].
    /// @param[out] K Exchange matrix [n x n, Hartree], symmetric, no 0.5 factor.
    void k(const RealMatrix& P, RealMatrix& K) const;

    /// @brief Exchange matrix from the erf-attenuated operator erf(ω r)/r.
    ///
    /// Same normalization as `k()` (bare K, no 0.5, no α/β). `omega == 0`
    /// is the bare-Coulomb identity (`k()`). Used by the CAM-B3LYP Fock
    /// mix \(K = \alpha K[1/r] + \beta K[\mathrm{erf}]\).
    ///
    /// @param P     Total RHF density [n x n, unitless].
    /// @param omega Range-separation parameter [1/Bohr]; must be ≥ 0.
    /// @param[out] K Exchange matrix [n x n, Hartree].
    void k_erf(const RealMatrix& P, double omega, RealMatrix& K) const;

    /// @brief Fused one-sweep restricted J and K from a single shell-block
    ///        ERI quartet scan.
    ///
    /// Produces the bare Coulomb matrix `J_ij = sum_kl P_kl (ij|kl)` and the
    /// bare exchange matrix `K_ij = sum_kl P_kl (ik|jl)` (NO 0.5 factor, no
    /// `alpha_hf`) in ONE `eri_block` pass — the Coulomb accumulation of
    /// `g_rhf` and the 8-term exchange accumulation of `k` share the same
    /// evaluated integrals, so this is algebraically exact vs the two-pass
    /// `g_rhf` (J = G + 0.5*K) + `k`, differing only in floating-point add
    /// order. The restricted global hybrid then assembles
    /// `F = Hcore + J - 0.5*alpha_hf*K + Vxc`.
    ///
    /// K is emitted with the SAME (unscaled) normalization as
    /// `ShellBlockPairList::k`, kept separable from J so a later
    /// range-separated (erf-attenuated) exchange operator can add a second K
    /// without J+K being fused into an opaque single matrix.
    ///
    /// @param P_total Total RHF density matrix P = 2*P_occ [n x n, unitless].
    /// @param[out] J  Bare Coulomb matrix [n x n, Hartree].
    /// @param[out] K  Bare exchange matrix [n x n, Hartree], symmetric, no
    ///                0.5 factor (RSH-separable).
    void jk_rhf(const RealMatrix& P_total, RealMatrix& J, RealMatrix& K) const;
};

/// @brief Group a flat vector of AO-level `native::Shell` entries into
///        libcint-style l-blocks without materializing Schwarz bounds.
///
/// Consecutive shells sharing an angular momentum (same `l`) collapse
/// into one `native::ShellBlock` with `n_cart = n_cart_of(l)`. Callers
/// feed this with `ShellList::shells` (one-electron integral path) or
/// `internal::IntegralWorkspace::shells` (Fock-build Schwarz path) — both use the same
/// flat layout.
///
/// Returned blocks alias the input's `alpha` / `coeff` pointers, so the
/// source storage must outlive the returned vector.
[[nodiscard]] std::vector<native::ShellBlock>
blocks_from_shells(const std::vector<native::Shell>& shells);

} // namespace scf
} // namespace cpu
} // namespace ATV

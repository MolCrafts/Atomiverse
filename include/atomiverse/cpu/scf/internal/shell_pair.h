/// @file cpu/scf/internal/shell_pair.h
/// @brief ShellBlockPairList: canonical shell-block pairs with block-level
///        Cauchy-Schwarz bounds, consumed by ExactJK and the ERI-derivative
///        builders.
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/integral_view.h"
#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
namespace scf {

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

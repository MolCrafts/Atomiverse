/// @file cpu/scf/xc_grid_engine.h
/// @brief Blocked molecular XC grid integration (PySCF numint block scheme).
///
/// The grid is processed in blocks of at most `XCGridBasis::kBlockSize`
/// points. Per block, only AOs that are numerically significant on that
/// block enter (value or gradient above `kAoCutoff`), and every contraction
/// is a level-3 BLAS call on the compacted block:
///   1. T = φ P_sub (one GEMM per spin channel),
///      ρ = rowsum(T ∘ φ), ∇ρ = 2 rowsum(T ∘ ∇φ);
///   2. one batched functional evaluation;
///   3. V_sub = φᵀ X + Xᵀ φ with X = ½ w v_ρ φ + w g·∇φ (one GEMM per spin,
///      LDA and GGA alike).
/// A restricted density runs a single channel (α = β). Blocks are
/// distributed over OpenMP threads; BLAS then runs on the calling thread.
///
/// The compacted AO values (and gradients for GGA) are geometry-fixed and
/// cached by `XCGridBasis`; memory scales with the screened, not the full,
/// G × N product.
///
/// @see PySCF numint.py for the reference algorithm.
#pragma once

#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/basis_eval.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/xc/xc_functional.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Result of batched XC grid integration.
struct XCGridResult {
    xt::xtensor<double, 2>
        Vxc_alpha; ///< XC potential matrix (alpha) [n_basis × n_basis].
    xt::xtensor<double, 2>
        Vxc_beta; ///< XC potential matrix (beta) [n_basis × n_basis].
    double E_xc;  ///< Total XC energy [Hartree].
    double n_alpha_integrated; ///< Grid-integrated alpha electron count.
    double n_beta_integrated;  ///< Grid-integrated beta electron count.
};

/// @brief Geometry-fixed, block-compacted AO-on-grid basis (internal
///        cache owned by the Fock builder and the response kernel; not a
///        user-facing configuration).
struct XCGridBasis {
    /// Grid points per block (PySCF BLKSIZE order of magnitude).
    static constexpr int kBlockSize = 128;
    /// AO significance cutoff on |φ| and |∇φ| over a block.
    static constexpr double kAoCutoff = 1e-14;

    /// One block of consecutive grid points and its significant AOs.
    struct Block {
        int g0 = 0;            ///< First grid point.
        int ng = 0;            ///< Number of points.
        std::vector<int> ao;   ///< Significant Cartesian AO indices, ascending.
        RealMatrix phi;        ///< AO values [ng × m] (column-major).
        RealMatrix dx, dy, dz; ///< AO gradients [ng × m] (GGA only).
    };

    std::vector<Block> blocks; ///< Blocks covering the grid in order.
    int n_basis = 0;           ///< AO dimension of the produced matrices.
    bool gga = false;          ///< true → gradient blocks are populated.
    bool built = false; ///< true → build() has run for the current geometry.

    /// @brief Evaluate and compact the AO values (and gradients for GGA) of
    ///        every block, reusing the per-point basis evaluation of the cube
    ///        writers (`eval_basis` / `eval_basis_val_grad`).
    /// @param grid    Molecular quadrature grid.
    /// @param shells  Basis-function shell descriptors [n_basis].
    /// @param n_basis Number of basis functions.
    /// @param is_gga  true → also build gradient blocks.
    void build(const MolecularGrid& grid,
               const std::vector<native::Shell>& shells, int n_basis,
               bool is_gga);
};

/// @brief Blocked XC integration on a precomputed AO-on-grid basis.
///
/// @param aog     Block-compacted AO basis built for this grid.
/// @param grid    Molecular quadrature grid (weights; the grid `aog` used).
/// @param P_alpha Alpha density matrix [n_basis × n_basis].
/// @param P_beta  Beta density matrix [n_basis × n_basis]; ignored for
///                `XcSpin::Restricted`, which requires P_beta = P_alpha.
/// @param functional Batched XC mathematics.
/// @param spin Restricted (one channel) or collinear spin convention.
/// @return XCGridResult with Vxc matrices, E_xc, and integration diagnostics.
/// @throws std::invalid_argument for functionals needing τ or ∇²ρ, or a
///         density / basis dimension mismatch.
XCGridResult evaluate_xc_batched(
    const XCGridBasis& aog, const MolecularGrid& grid,
    const xt::xtensor<double, 2>& P_alpha, const xt::xtensor<double, 2>& P_beta,
    const ATV::electronic::XcFunctional& functional,
    ATV::electronic::XcSpin spin = ATV::electronic::XcSpin::Collinear);

/// @brief Blocked XC integration that builds the AO basis for this call
///        (collinear spin convention).
/// @param grid    Molecular quadrature grid.
/// @param shells  Basis function shell descriptors [n_basis].
/// @param n_basis Number of basis functions.
/// @param P_alpha Alpha density matrix [n_basis × n_basis].
/// @param P_beta  Beta density matrix [n_basis × n_basis].
/// @param functional Batched XC mathematical provider.
/// @return XCGridResult with Vxc matrices, E_xc, and integration diagnostics.
XCGridResult
evaluate_xc_batched(const MolecularGrid& grid,
                    const std::vector<native::Shell>& shells, int n_basis,
                    const xt::xtensor<double, 2>& P_alpha,
                    const xt::xtensor<double, 2>& P_beta,
                    const ATV::electronic::XcFunctional& functional);

} // namespace scf
} // namespace cpu
} // namespace ATV

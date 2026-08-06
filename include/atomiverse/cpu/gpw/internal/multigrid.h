/// @file cpu/gpw/internal/multigrid.h
/// @brief CP2K-style CUTOFF/REL_CUTOFF multigrid for GPW collocation.
///
/// A single uniform grid forces every Gaussian onto the finest spacing: diffuse
/// Gaussians then span huge bounding boxes (many points) — slow — and a single
/// grid leaves an O(mHa) discretization residual vs CP2K's multigrid. CP2K maps
/// each Gaussian (by its exponent) to the COARSEST grid level that still
/// resolves it (REL_CUTOFF), collocates there (compact support, far fewer points
/// for diffuse functions), interpolates the per-level densities to a common fine
/// grid for the (nonlinear) Hartree + XC, then restricts the fine potential back
/// to each level to integrate the KS matrix.
///
/// Transfer between levels is real-space trilinear interpolation (prolong) and
/// its exact transpose (restrict). The two being adjoints (R = c·Pᵀ) keeps the
/// integrated KS matrix the exact gradient of the collocated energy, so the SCF
/// stays variational/convergent; the discretization is faithful (energy → the
/// fine-grid limit as the grids refine), not an uncontrolled approximation.
///
/// Gated: `ngrids == 1` bypasses all of this and uses the single-grid path
/// unchanged (bit-identical).
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/gpw/internal/collocate.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief A multigrid: `levels[0]` is the finest (at CUTOFF); coarser levels
/// follow at geometrically smaller cutoffs. The finest grid is the reference on
/// which Poisson + XC run.
struct MultiGrid {
    std::vector<UniformGrid> levels;     ///< finest first.
    std::vector<double> level_cutoff_ry; ///< per-level density cutoff [Ry].
};

/// @brief Build `ngrids` geometrically-spaced grid levels for an orthorhombic
///        cell. Finest cutoff = `cutoff_ry`; each coarser level divides the
///        cutoff by a fixed ratio so the whole REL_CUTOFF..CUTOFF range is
///        covered. `ngrids == 1` returns the single CUTOFF grid.
/// @param cell_abc Box lengths a,b,c [Bohr].
/// @param cutoff_ry Finest (plane-wave) density cutoff [Ry].
/// @param ngrids Number of grid levels (>= 1).
/// @param rel_cutoff Reference cutoff [Ry] for the exponent->level mapping; sets
///        the coarsest level's cutoff ~ rel_cutoff and the geometric ratio.
MultiGrid make_multigrid(const double cell_abc[3], double cutoff_ry, int ngrids,
                         double rel_cutoff);

/// @brief Assign each basis function to a grid level by its most-diffuse
///        exponent. A Gaussian of exponent α needs an effective density cutoff
///        ~ rel_cutoff·(α/α_ref); it goes on the coarsest level whose cutoff
///        still meets that need (tight α -> fine level 0; diffuse α -> coarse).
/// @return level index per basis function, length basis.size().
std::vector<int> assign_levels(const MultiGrid& mg,
                               const std::vector<GridBasisFunction>& basis,
                               double rel_cutoff);

/// @brief Prolong (interpolate) a coarse-grid field onto the fine grid by
///        periodic trilinear interpolation. @p periodic wraps indices.
std::vector<double> prolong_trilinear(const UniformGrid& coarse,
                                      const UniformGrid& fine,
                                      const std::vector<double>& f_coarse,
                                      bool periodic);

/// @brief Restrict a fine-grid field onto the coarse grid as the EXACT transpose
///        of `prolong_trilinear` (the same trilinear weights, scattered) times
///        the cell-volume-per-point ratio so that ⟨V_fine, P f_coarse⟩_fine =
///        ⟨R V_fine, f_coarse⟩_coarse — i.e. integration is the adjoint of
///        collocation across the level transfer.
std::vector<double> restrict_transpose(const UniformGrid& fine,
                                       const UniformGrid& coarse,
                                       const std::vector<double>& f_fine,
                                       bool periodic);

/// @brief Prolong (interpolate) a coarse-grid field onto the fine grid by
///        SPECTRAL (band-limited / Fourier) interpolation — the shape-exact
///        replacement for `prolong_trilinear`.
///
/// Forward-FFT @p f_coarse on the coarse grid, copy every coarse Fourier mode to
/// the SAME physical frequency on the fine grid (zero-padding the higher fine
/// frequencies), then inverse-FFT on the fine grid. Because the coarse and fine
/// grids share the box `L`, integer mode `m` maps to the same physical frequency
/// `G = 2π m / L` on both, so the embedding is a copy by mode index. The
/// normalization (×N_fine/N_coarse on each mode) preserves VALUES (a field
/// band-limited below the coarse Nyquist is reproduced exactly at every fine
/// point) and conserves mass (∫prolong(f) = ∫f). Coarse Nyquist modes (index
/// ±n_c/2 on any axis) are split half/half between +n_c/2 and −n_c/2 on the fine
/// grid so the interpolant stays real and Hermitian.
///
/// Orthorhombic + periodic only (the production multigrid case). For
/// `periodic == false` it falls back to `prolong_trilinear`.
/// @param coarse Source grid.
/// @param fine Target grid (sizes even, n_fine >= n_coarse per axis, same L).
/// @param f_coarse Coarse-grid field, length coarse.npts().
/// @param periodic When false, defers to `prolong_trilinear`.
/// @return Fine-grid field, length fine.npts().
std::vector<double> prolong_fft(const UniformGrid& coarse,
                                const UniformGrid& fine,
                                const std::vector<double>& f_coarse,
                                bool periodic);

/// @brief Restrict a fine-grid field onto the coarse grid as the EXACT ADJOINT
///        of `prolong_fft` w.r.t. the grid integration inner products:
///        ⟨V_fine, prolong_fft(g)⟩_fine = ⟨restrict_fft(V_fine), g⟩_coarse.
///
/// In G-space this is the transpose of the zero-pad embedding: forward-FFT
/// @p f_fine on the fine grid, TRUNCATE to the coarse modes (the transpose of
/// zero-padding; Nyquist modes summed back from their split halves with the same
/// half/half weights as `prolong_fft`), inverse-FFT on the coarse grid with the
/// transpose normalization (×N_coarse/N_fine on each mode). Adjointness keeps the
/// integrated KS matrix the exact gradient of the collocated energy, so the
/// multigrid SCF stays variational.
///
/// Orthorhombic + periodic only. For `periodic == false` it falls back to
/// `restrict_transpose`.
/// @param fine Source grid.
/// @param coarse Target grid (sizes even, n_coarse <= n_fine per axis, same L).
/// @param f_fine Fine-grid field, length fine.npts().
/// @param periodic When false, defers to `restrict_transpose`.
/// @return Coarse-grid field, length coarse.npts().
std::vector<double> restrict_fft(const UniformGrid& fine,
                                 const UniformGrid& coarse,
                                 const std::vector<double>& f_fine,
                                 bool periodic);

/// @brief One primitive-pair PRODUCT Gaussian, assigned to a grid level.
///
/// The density ρ = Σ_μν P_μν χ_μ χ_ν is a sum of primitive-pair products. For a
/// primitive a∈μ (exponent α_a, Cartesian powers la, center r_a) and b∈ν
/// (α_b, lb, r_b), the product is a single Gaussian of exponent p = α_a+α_b
/// centered at P_ab = (α_a r_a + α_b r_b)/p, with the Gaussian-product prefactor
/// K = exp(−α_a α_b/p |r_a−r_b|²) folded into `gweight`, times the polynomial
/// (r−r_a)^la (r−r_b)^lb. Periodic images of the two factors are expanded as
/// separate products (each carries its own shifted centers `ra`/`rb`), exactly
/// reproducing the cross-image Γ-point product Σ_{T,T'} φ_μ(r−T) φ_ν(r−T').
///
/// `gweight` is geometry-fixed (it excludes P_μν); the per-SCF density weight is
/// P[mu*nb+nu]·gweight, applied at collocation time. The KS scatter target is the
/// (mu,nu) element.
struct GridProduct {
    double p = 0.0;           ///< combined exponent α_a + α_b.
    double pc[3] = {0, 0, 0}; ///< product center P_ab [Bohr].
    double ra[3] = {0, 0, 0}; ///< bra primitive center (shifted image) [Bohr].
    double rb[3] = {0, 0, 0}; ///< ket primitive center (shifted image) [Bohr].
    int la[3] = {0, 0, 0};    ///< bra Cartesian powers.
    int lb[3] = {0, 0, 0};    ///< ket Cartesian powers.
    double k = 0.0; ///< Gaussian product prefactor exp(−α_a α_b/p |r_a−r_b|²).
    int ua = 0,
        ub = 0; ///< indices into ProductSet::prims (bra, ket cartesian prim).
    int level = 0; ///< assigned grid level (index into MultiGrid::levels).
    int t[3] = {0, 0, 0}; ///< ket image shift (lattice cells; bra stays home).
    int tcell = 0; ///< index of `t` in ProductSet::cells (Bloch phase lookup).
};

/// @brief One spherical basis function μ that a cartesian primitive feeds, with
///        its real-solid-harmonic expansion coefficient.
struct PrimContrib {
    int mu = 0; ///< basis-function index (KS row/col target).
    double coeff =
        0.0; ///< harmonic·contraction coefficient of this cartesian term in μ.
};

/// @brief A UNIQUE cartesian primitive (α, center, Cartesian powers l) shared by
///        every spherical basis function whose expansion contains it. Collocating
///        the product between two of these once — then scattering to all (μ,ν)
///        contributor pairs — removes the spherical→cartesian redundancy (the same
///        monomial recurs across d_z²/d_x²-y², etc.) and the per-monomial window
///        recompute. Mirrors CP2K's primitive-cartesian collocation.
struct PrimRef {
    double alpha = 0.0;           ///< Gaussian exponent.
    double center[3] = {0, 0, 0}; ///< home-cell center [Bohr].
    int l[3] = {0, 0, 0};         ///< Cartesian powers.
    int cbeg = 0,
        cn = 0; ///< contributor slice [cbeg, cbeg+cn) in ProductSet::contribs.
    double maxc =
        0.0; ///< max |coeff| over contributors (for the build screen).
};

/// @brief Geometry-fixed collint shell task — contiguous monoms in
///        `ProductSet::by_level[l]` that share one collint pol table
///        (p, ra, rb, pc). Built once in `build_products` (CP2K-style task
///        list beyond a flat cartesian-prim monom array). Per-SCF collocate
///        / integrate iterates tasks; monoms stay the expansion of one
///        geometry batch (shell-prim pair × angular products × image).
struct CollintTask {
    int beg = 0; ///< inclusive start index into `by_level[l]`.
    int end = 0; ///< exclusive end index into `by_level[l]`.
    int lpx = 0; ///< max monom (la_x + lb_x) in the task.
    int lpy = 0; ///< max monom (la_y + lb_y).
    int lpz = 0; ///< max monom (la_z + lb_z).
    int Ltot = 0; ///< max |la|+|lb| over monoms (window / radius).
    /// max_i maxc(ua)·maxc(ub)·|k| — geometry magnitude for integrate screen.
    double geom_mag = 0.0;
};

/// @brief Products partitioned by grid level, plus the unique cartesian-primitive
///        table and the flat contributor pool the products index into.
struct ProductSet {
    std::vector<std::vector<GridProduct>>
        by_level; ///< products per level (index ua/ub into prims).
    /// Per-level shell tasks (geometry batches). Parallel to `by_level`; each
    /// task's [beg, end) indexes a contiguous monom run of one geometry.
    std::vector<std::vector<CollintTask>> tasks_by_level;
    std::vector<PrimRef> prims; ///< unique cartesian primitives.
    std::vector<PrimContrib>
        contribs; ///< flat contributor pool (sliced by PrimRef).
    /// Distinct ket image shifts over all products, lexicographically sorted;
    /// `GridProduct::tcell` indexes this list. Size 1 (the home cell) for an
    /// isolated system; the Γ path ignores it (every cell weighs by the same P).
    std::vector<std::array<int, 3>> cells;
};

/// @brief Build the geometry-fixed primitive-pair product list (the product-based
///        density expansion), assigning each product to the coarsest grid level
///        whose cutoff resolves its combined exponent p (mirrors `assign_levels`
///        with the exponent = p). Periodic image pairs are expanded so the
///        collocated density reproduces the Γ-point cross-image product.
/// @param mg The multigrid (levels finest-first).
/// @param basis Global contracted-Gaussian basis.
/// @param L Box lengths [Bohr].
/// @param rel_cutoff CP2K REL_CUTOFF [Ry] — exponent→level calibration.
/// @param eps Gaussian-tail screening tolerance (drops negligible product
///        prefactors and bounds the per-product collocation window).
/// @param periodic When true, expand lattice images of both factors; false
///        (isolated) uses the single in-box pair.
ProductSet build_products(const MultiGrid& mg,
                          const std::vector<GridBasisFunction>& basis,
                          const double L[3], double rel_cutoff, double eps,
                          bool periodic);

/// @brief Collocate every product assigned to one level into ρ_level, weighted by
///        the density matrix: ρ_level(r) += Σ_prod P[mu*nb+nu]·gweight·poly·
///        exp(−p|r−P_ab|²). Uses the same unwrapped-window + wrap-index + sum
///        convention as `build_compact_basis_grid` so the periodic product
///        density matches the single-grid path.
/// @param grid The level's uniform grid.
/// @param prods Products assigned to this level (from `ProductSet::by_level[l]`).
/// @param tasks Shell tasks for this level (`ProductSet::tasks_by_level[l]`).
/// @param nb Basis size (row stride of `P`).
/// @param P Density matrix, row-major nb×nb.
/// @param eps Collocation-window screening tolerance.
std::vector<double> collocate_products(const UniformGrid& grid,
                                       const std::vector<GridProduct>& prods,
                                       const std::vector<CollintTask>& tasks,
                                       const std::vector<PrimRef>& prims,
                                       const std::vector<PrimContrib>& contribs,
                                       int nb, const std::vector<double>& P,
                                       double eps);

/// @brief Integrate a level potential against every product on that level and
///        scatter into the KS matrix — the EXACT transpose of
///        `collocate_products`: V[mu*nb+nu] += Σ_g w·gweight·poly(r_g)·
///        exp(−p|r_g−P_ab|²)·V_level(r_g). Adjointness keeps the KS matrix the
///        exact gradient of the collocated energy.
/// @return Row-major nb×nb KS contribution from this level.
std::vector<double> integrate_products(
    const UniformGrid& grid, const std::vector<GridProduct>& prods,
    const std::vector<CollintTask>& tasks, const std::vector<PrimRef>& prims,
    const std::vector<PrimContrib>& contribs, int nb,
    const std::vector<double>& V_level, double eps);

/// @brief `collocate_products` with a per-image-cell density-matrix stack —
///        the Bloch-sampled product collocation. Each product weighs by
///        `D[tcell][μ·nb+ν]` instead of one shared `P`: for k-point sampling
///        D(t) = Σ_k w_k Re[P^k e^{−2πi κ·t}] (t = the product's ket image
///        shift), which reduces to the Γ path when every D[c] is the same P.
/// @param D One row-major nb×nb real matrix per `ProductSet::cells` entry.
/// @return Level density ρ_level, length grid.npts().
std::vector<double> collocate_products_cells(
    const UniformGrid& grid, const std::vector<GridProduct>& prods,
    const std::vector<CollintTask>& tasks, const std::vector<PrimRef>& prims,
    const std::vector<PrimContrib>& contribs, int nb,
    const std::vector<std::vector<double>>& D, double eps);

/// @brief `integrate_products` resolved per image cell — the EXACT transpose of
///        `collocate_products_cells`: each product's ∫V·shape scatters into the
///        real matrix of ITS ket image shift, V_t[tcell][μ·nb+ν]. The per-k
///        grid KS matrix is the caller's phase sum Σ_t e^{+2πi κ·t}·V_t
///        (Hermitian: the lattice-shifted window makes V_{−t} = V_tᵀ exactly).
/// @param ncells Number of image cells (`ProductSet::cells.size()`).
/// @return One row-major nb×nb real matrix per image cell.
std::vector<std::vector<double>> integrate_products_cells(
    const UniformGrid& grid, const std::vector<GridProduct>& prods,
    const std::vector<CollintTask>& tasks, const std::vector<PrimRef>& prims,
    const std::vector<PrimContrib>& contribs, int nb,
    const std::vector<double>& V_level, std::size_t ncells, double eps);

} // namespace gpw
} // namespace cpu
} // namespace ATV

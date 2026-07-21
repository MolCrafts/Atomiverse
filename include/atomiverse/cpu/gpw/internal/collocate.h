/// @file cpu/gpw/internal/collocate.h
/// @brief Gaussian density collocation onto a uniform real-space grid, and the
///        adjoint integration of a grid potential back into KS matrix elements.
///
/// This is the core GPW bridge (spec D2): the density matrix in the contracted
/// Gaussian basis is mapped to ρ(r) on a uniform grid (`collocate_density`), and
/// a real-space potential V(r) is integrated back to V_μν = ∫ φ_μ V φ_ν dr
/// (`integrate_potential`). The FFT Poisson step that turns ρ into V_Hartree is
/// added on top in a later task; this module is FFT-free and validated by the
/// round trips ∫ρ = Tr(P S) and integrate(V≡1) = S.
///
/// Phase 1 scope: orthorhombic cell, isolated/non-periodic evaluation (Gaussians
/// assumed well inside the box — periodic image sums are a documented TODO for
/// the periodic Ar/H2 milestone). Dense evaluation, suited to the small systems
/// of the early milestones.
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief One primitive Cartesian Gaussian:
/// f(r) = coeff * (x-cx)^lx (y-cy)^ly (z-cz)^lz * exp(-alpha |r-c|^2).
struct GridPrimitive {
    double alpha = 0.0;           ///< Gaussian exponent.
    double center[3] = {0, 0, 0}; ///< center c [Bohr].
    int l[3] = {0, 0, 0};         ///< Cartesian powers (lx, ly, lz).
    double coeff = 1.0; ///< contraction coefficient (incl. any normalization).
};

/// @brief A basis function is a contracted sum of primitives. φ(r) = Σ_p prim_p(r).
using GridBasisFunction = std::vector<GridPrimitive>;

/// @brief Uniform orthorhombic real-space grid covering [0,L) per axis.
struct UniformGrid {
    int n[3] = {0, 0, 0};         ///< points per axis.
    double L[3] = {0, 0, 0};      ///< box lengths [Bohr].
    double origin[3] = {0, 0, 0}; ///< lower corner [Bohr].

    /// @brief Total number of grid points.
    long npts() const { return static_cast<long>(n[0]) * n[1] * n[2]; }
    /// @brief Uniform integration weight per point = cell_volume / npts.
    double weight() const {
        return (L[0] * L[1] * L[2]) / static_cast<double>(npts());
    }
    /// @brief Cartesian coordinates of linear grid index `idx` (k fastest).
    void point(long idx, double r[3]) const;
};

/// @brief Build a uniform grid for an orthorhombic cell from a plane-wave-style
///        density cutoff. n_i = ceil((L_i/pi) sqrt(cutoff_ry)) rounded up to even.
/// @param cell_abc Box lengths a,b,c [Bohr].
/// @param cutoff_ry Density grid cutoff [Ry].
UniformGrid make_uniform_grid(const double cell_abc[3], double cutoff_ry);

/// @brief Compact (per-grid-point) basis representation for collocation.
///
/// Each Gaussian basis function is non-negligible only within a finite radius of
/// its center, so at any grid point only a handful of functions are nonzero. This
/// stores, per ACTIVE grid point (any function above `eps`), the list of
/// (function index, value) pairs — a CSR-by-point layout. It replaces the dense
/// `npts x nbasis` matrix (which is O(ngrid*nbasis) in both memory — prohibitive
/// at scale — and per-iter work O(na*nbasis^2)). Collocation/integration over it
/// cost O(sum_g k_g^2) with k_g = nonzero functions at point g, not nbasis^2.
/// Built once per geometry; the min-image periodic wrap (and the non-periodic
/// raw-displacement path) match `eval_basis_on_grid`.
struct CompactBasisGrid {
    std::vector<long> gidx; ///< active grid-point linear indices (length na).
    std::vector<std::size_t> off; ///< CSR offsets into fu/fv (length na+1).
    std::vector<int> fu;          ///< function index per nonzero (length nnz).
    std::vector<double> fv;       ///< φ_fu(r_g) value per nonzero (length nnz).
    long npts = 0;                ///< total grid points (for output sizing).
    double weight = 0.0;          ///< grid integration weight per point.
};

/// @brief Radius [Bohr] beyond which a contracted Gaussian basis function is
///        below `eps` in magnitude. Solves |coeff| r^l exp(-alpha r^2) = eps per
///        primitive (the r^l growth folded in by a fixed-point correction) and
///        returns the max over primitives. Drives both the collocation window and
///        the analytic lattice-sum image count (`image_counts`) from one cutoff.
double basis_function_radius(const GridBasisFunction& bf, double eps);

/// @brief Build the compact per-point basis representation from per-Gaussian
///        bounding boxes (no dense npts x nbasis matrix is ever formed).
/// @param eps drop a function at a point when |φ| <= eps; chosen tight enough
///        (default 1e-12) that the dropped Gaussian tails are below the SCF
///        energy tolerance, so collocation/integration are bit-equivalent to the
///        dense path. @param periodic see `eval_basis_on_grid`.
CompactBasisGrid
build_compact_basis_grid(const UniformGrid& grid,
                         const std::vector<GridBasisFunction>& basis,
                         bool periodic, double eps = 1e-12);

/// @brief Collocate ρ(r) = Σ_uv P_uv φ_u(r) φ_v(r) over the compact representation.
std::vector<double> collocate_density(const CompactBasisGrid& cbg, int nbasis,
                                      const std::vector<double>& P);

/// @brief Integrate V_uv = Σ_g w φ_u(g) V(g) φ_v(g) over the compact representation.
std::vector<double> integrate_potential(const CompactBasisGrid& cbg, int nbasis,
                                        const std::vector<double>& V_r);

/// @brief Evaluate every basis function at every grid point.
/// @param periodic When true, the Gaussian-to-grid displacement uses the
///        minimum image per axis, so atoms outside [0,L) (unwrapped periodic
///        slab atoms) collocate through their nearest image. False = isolated
///        (Martyna-Tuckerman); raw displacement, Gaussians assumed inside the box.
/// @return Row-major (npts x nbasis) matrix Phi, Phi[g*nbasis + u] = φ_u(r_g).
std::vector<double>
eval_basis_on_grid(const UniformGrid& grid,
                   const std::vector<GridBasisFunction>& basis,
                   bool periodic = false);

/// @brief Collocate the electron density ρ(r) = Σ_uv P_uv φ_u(r) φ_v(r).
/// @param P Symmetric density matrix, row-major nbasis x nbasis.
/// @return ρ(r) sampled on the grid, length npts.
std::vector<double>
collocate_density(const UniformGrid& grid,
                  const std::vector<GridBasisFunction>& basis,
                  const std::vector<double>& P);

/// @brief Collocate density from a precomputed basis-on-grid matrix `phi`
///        (row-major npts x nbasis, from `eval_basis_on_grid`). The grid and
///        basis are geometry-fixed, so caching `phi` across SCF iterations avoids
///        recomputing every Gaussian at every point.
std::vector<double> collocate_density(const UniformGrid& grid,
                                      const std::vector<double>& phi,
                                      int nbasis, const std::vector<double>& P);

/// @brief Grid-point indices where any basis function exceeds `eps` in magnitude.
///        The molecule occupies a small part of a large box, so collocation /
///        integration over just these "active" points skips the empty vacuum.
std::vector<long> active_grid_points(const std::vector<double>& phi, long npts,
                                     int nbasis, double eps = 1e-12);

/// @brief Cached + screened collocation: precompute `phi` and `active` once
///        (both geometry-fixed) and reuse across SCF iterations.
std::vector<double> collocate_density(const UniformGrid& grid,
                                      const std::vector<double>& phi,
                                      int nbasis, const std::vector<double>& P,
                                      const std::vector<long>& active);

/// @brief Integrate a grid potential into KS matrix elements
///        V_uv = Σ_g w φ_u(r_g) V(r_g) φ_v(r_g).
/// @param V_r Potential sampled on the grid, length npts.
/// @return Row-major nbasis x nbasis matrix.
std::vector<double>
integrate_potential(const UniformGrid& grid,
                    const std::vector<GridBasisFunction>& basis,
                    const std::vector<double>& V_r);

/// @brief Integrate a grid potential into KS matrix elements using a precomputed
///        basis-on-grid matrix `phi` (row-major npts x nbasis). See the cached
///        `collocate_density` overload.
std::vector<double> integrate_potential(const UniformGrid& grid,
                                        const std::vector<double>& phi,
                                        int nbasis,
                                        const std::vector<double>& V_r);

/// @brief Cached + screened integration over `active` grid points.
std::vector<double> integrate_potential(const UniformGrid& grid,
                                        const std::vector<double>& phi,
                                        int nbasis,
                                        const std::vector<double>& V_r,
                                        const std::vector<long>& active);

} // namespace gpw
} // namespace cpu
} // namespace ATV

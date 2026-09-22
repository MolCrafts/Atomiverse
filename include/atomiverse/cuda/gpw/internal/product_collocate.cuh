/// @file cuda/gpw/internal/product_collocate.cuh
/// @brief Thin CUDA foothold: collocate product Gaussians onto a uniform grid.
///
/// Mirrors the continuous product density that CPU `collint_collocate_product`
/// (CP2K collint) samples on `UniformGrid`:
///   shape(r) = weight · (r−ra)^{la} (r−rb)^{lb} · exp(−p |r−pc|²)
/// Periodic orthorhombic cells use a **sphere-window** collocate matching CPU
/// `collocate_one` / collint wrap: evaluate at unwrapped r near the product
/// center, wrap the storage index (cuda-gpw-product-collocate-pbc). Device
/// kernels; non-periodic foothold still walks the full primary grid. Not a
/// full GPW SCF product: no multigrid, no P-scatter, no geometry batching.
#pragma once

#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

/// @brief Orthorhombic real-space grid for product collocation (Bohr).
///
/// Layout matches `cpu::gpw::UniformGrid`: point (i,j,k) has coordinates
/// `origin[d] + L[d] * index[d] / n[d]`, with k (= axis 2) contiguous.
struct CollocateGridDesc {
    int n[3] = {0, 0, 0};         ///< points per axis (n0, n1, n2).
    double L[3] = {0, 0, 0};      ///< box lengths [Bohr].
    double origin[3] = {0, 0, 0}; ///< lower corner [Bohr].
    /// @brief When true, collocate/integrate use sphere-window PBC wrap.
    /// Default true — CUDA GPW product rejects Martyna–Tuckerman.
    bool periodic = true;
    /// @brief Sphere window eps for product_radius.
    /// Default mass-margin class (eps_default·1e-4) matches product-collocate-pbc
    /// Si charge recovery; group-max `DeviceProduct.mag` carries collint window
    /// mag (cuda-gpw-multigrid-device-ks). Do not use collint `eps·0.02` here —
    /// that under-sizes monom continuous spheres vs collint cxyz refine.
    double eps_win = 1.0e-14;

    /// @brief Total real-space points.
    long npts() const { return static_cast<long>(n[0]) * n[1] * n[2]; }
    /// @brief Integration weight w = V_cell / npts [Bohr³].
    double weight() const {
        return (L[0] * L[1] * L[2]) / static_cast<double>(npts());
    }
};

/// @brief One primitive-pair product Gaussian for device collocation.
///
/// Same fields as `cpu::gpw::GridProduct` that enter the collint shape, plus
/// a caller-resolved density weight (P·coeffs·k folded in). Angular powers
/// supported for the foothold are modest (l ≤ 2 per axis in tests); the
/// kernel is general for non-negative la/lb.
struct DeviceProduct {
    double p = 0.0;           ///< combined exponent α_a + α_b.
    double pc[3] = {0, 0, 0}; ///< product center P_ab [Bohr].
    double ra[3] = {0, 0, 0}; ///< bra primitive center [Bohr].
    double rb[3] = {0, 0, 0}; ///< ket primitive center [Bohr].
    int la[3] = {0, 0, 0};    ///< bra Cartesian powers.
    int lb[3] = {0, 0, 0};    ///< ket Cartesian powers.
    double weight = 0.0;      ///< density weight (includes k · P · coeffs).
    /// @brief Window magnitude for sphere radius (collint group max\|w\|).
    /// Zero ⇒ collocate falls back to \|weight\| (foothold / unit tests).
    double mag = 0.0;
    /// @brief Angular L for sphere radius (collint group max Ltot).
    /// Pack sets monom L; `apply_group_max_mag` lifts to group max (collint
    /// twin). Negative ⇒ collocate/integrate recompute monom L from la+lb.
    int L_win = -1;
};

/// @brief Collocate product Gaussians onto ρ (device path).
///
/// For each grid point g and product q:
///   ρ_g += weight_q · ∏_d (r_d−ra_d)^{la_d} (r_d−rb_d)^{lb_d}
///          · exp(−p_q |r−pc|²)
/// Accumulates into `d_rho` (caller zeros). All scientific work is on device;
/// no host loop over grid points.
///
/// @param grid    Orthorhombic grid descriptor (host).
/// @param d_prods Device array of products, length nprod.
/// @param nprod   Number of products.
/// @param d_rho   Device density buffer, length grid.npts() [e/Bohr³].
/// @param stream  CUDA stream (default stream if null).
void collocate_products(const CollocateGridDesc& grid,
                        const DeviceProduct* d_prods, int nprod, double* d_rho,
                        cudaStream_t stream = nullptr);

/// @brief Collocate a single product (convenience wrapper).
///
/// @param grid   Orthorhombic grid descriptor (host).
/// @param d_prod Device pointer to one product.
/// @param d_rho  Device density buffer, length grid.npts().
/// @param stream CUDA stream (default stream if null).
void collocate_product(const CollocateGridDesc& grid,
                       const DeviceProduct* d_prod, double* d_rho,
                       cudaStream_t stream = nullptr);

/// @brief Integrate ρ over the grid: Σ_g ρ_g · w → device scalar.
///
/// @param d_rho  Device density, length nr.
/// @param nr     Number of real-space points.
/// @param w      Grid weight V_cell/npts [Bohr³].
/// @param d_sum  Output scalar (device), overwritten.
/// @param stream CUDA stream (default stream if null).
void collocate_integrate_rho(const double* d_rho, long nr, double w,
                             double* d_sum, cudaStream_t stream = nullptr);

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV

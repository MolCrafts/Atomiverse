/// @file cuda/gpw/internal/product_integrate.cuh
/// @brief Thin CUDA foothold: integrate V against product Gaussian shapes.
///
/// Transpose of product collocation. Mirrors the continuous product that CPU
/// `collint_integrate_product` (CP2K collint) contracts against a real-space
/// potential on `UniformGrid`:
///   shape(r) = (r−ra)^{la} (r−rb)^{lb} · exp(−p |r−pc|²)
///   acc_q    = Σ_g V_g · shape_q(r_g)
/// (no `DeviceProduct.weight`, no grid weight — matches collint; caller folds
/// k · w_cell later). Device kernels evaluate the formula at every grid point
/// (z-fast layout matching `cpu::gpw::UniformGrid::point`). Not a full GPW
/// SCF product: no multigrid, no P-scatter, no geometry batching.
#pragma once

#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"

#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

/// @brief Integrate V against product shapes → per-product accumulators.
///
/// For each product q:
///   acc_q = Σ_g V_g · ∏_d (r_d−ra_d)^{la_d} (r_d−rb_d)^{lb_d}
///           · exp(−p_q |r−pc|²)
/// `DeviceProduct.weight` is ignored (collint integrate is shape-only).
/// Overwrites `d_acc[0..nprod)` (zeros empty / screened-as-zero cases).
/// All scientific work is on device; no host loop over grid points.
///
/// @param grid    Orthorhombic grid descriptor (host).
/// @param d_prods Device array of products, length nprod.
/// @param nprod   Number of products.
/// @param d_V     Device potential, length grid.npts() [Hartree].
/// @param d_acc   Device per-product accumulators, length nprod.
/// @param stream  CUDA stream (default stream if null).
void integrate_products(const CollocateGridDesc& grid,
                        const DeviceProduct* d_prods, int nprod,
                        const double* d_V, double* d_acc,
                        cudaStream_t stream = nullptr);

/// @brief Integrate a single product (convenience wrapper).
///
/// @param grid   Orthorhombic grid descriptor (host).
/// @param d_prod Device pointer to one product.
/// @param d_V    Device potential, length grid.npts().
/// @param d_acc  Device scalar accumulator (overwritten).
/// @param stream CUDA stream (default stream if null).
void integrate_product(const CollocateGridDesc& grid,
                       const DeviceProduct* d_prod, const double* d_V,
                       double* d_acc, cudaStream_t stream = nullptr);

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV

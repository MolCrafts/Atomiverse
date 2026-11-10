#pragma once

#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"

namespace ATV::cuda::gpw::internal {

inline constexpr int kProductWindowBlock = 256;
inline constexpr int kProductAxisCacheSize = 1024;

__device__ __forceinline__ double ipow_nonneg(double x, int n) {
    double value = 1.0;
    for (int i = 0; i < n; ++i)
        value *= x;
    return value;
}

/// Radius where the angular-weighted Gaussian falls below eps.
__device__ inline double product_radius(double mag, double p, int angular,
                                        double eps) {
    if (!(mag > 0.0) || !(p > 0.0) || !(eps > 0.0))
        return 0.0;
    const double log_mag = log(fmax(mag, 1e-300));
    const double log_eps = log(fmax(eps, 1e-300));
    double r2 = fmax(0.0, (log_mag - log_eps) / p);
    for (int it = 0; it < 4; ++it) {
        const double r = sqrt(fmax(r2, 1e-30));
        r2 = fmax(0.0, (log_mag + angular * log(r) - log_eps) / p);
    }
    return sqrt(fmax(r2, 0.0));
}

struct ProductAxisPoint {
    long offset;
    double distance2;
    double factor;
};

struct ProductWindow {
    long start[3];
    long length[3];
    double radius2;
};

__device__ inline ProductWindow product_window(const CollocateGridDesc& grid,
                                               const DeviceProduct& product,
                                               double magnitude) {
    const int monomial = product.la[0] + product.la[1] + product.la[2]
                         + product.lb[0] + product.lb[1] + product.lb[2];
    const int angular = product.L_win >= 0 ? product.L_win : monomial;
    const double radius =
        product_radius(magnitude, product.p, angular, grid.eps_win);
    ProductWindow window{};
    window.radius2 = radius * radius;
    for (int axis = 0; axis < 3; ++axis) {
        const double spacing = grid.L[axis] / grid.n[axis];
        const long center =
            static_cast<long>(floor(product.pc[axis] / spacing));
        const long half = static_cast<long>(ceil(radius / spacing));
        window.start[axis] = center - half;
        window.length[axis] = 2 * half + 1;
    }
    return window;
}

__device__ __forceinline__ ProductAxisPoint
product_axis_point(const CollocateGridDesc& grid, const DeviceProduct& product,
                   int axis, long index, long stride) {
    const int n = grid.n[axis];
    long wrapped = index % n;
    if (wrapped < 0)
        wrapped += n;
    const double x =
        grid.origin[axis] + grid.L[axis] * static_cast<double>(index) / n;
    const double dx = x - product.pc[axis];
    const double d2 = dx * dx;
    const double factor = exp(-product.p * d2)
                          * ipow_nonneg(x - product.ra[axis], product.la[axis])
                          * ipow_nonneg(x - product.rb[axis], product.lb[axis]);
    return {wrapped * stride, d2, factor};
}

template <class Visit>
__device__ void visit_periodic_product(const CollocateGridDesc& grid,
                                       const DeviceProduct& product,
                                       const ProductWindow& window,
                                       ProductAxisPoint* cache, Visit& visit) {
    const long stride[3] = {static_cast<long>(grid.n[1]) * grid.n[2], grid.n[2],
                            1};
    const long offset[3] = {0, window.length[0],
                            window.length[0] + window.length[1]};
    const bool cached = offset[2] + window.length[2] <= kProductAxisCacheSize;
    if (cached) {
        for (int axis = 0; axis < 3; ++axis)
            for (long i = threadIdx.x; i < window.length[axis]; i += blockDim.x)
                cache[offset[axis] + i] = product_axis_point(
                    grid, product, axis, window.start[axis] + i, stride[axis]);
        __syncthreads();
    }

    // Warps cover different rows, with consecutive lanes on the contiguous
    // grid axis. Only the one-dimensional factors are cached.
    for (long i = 0; i < window.length[0]; ++i) {
        const auto x = cached
                           ? cache[i]
                           : product_axis_point(grid, product, 0,
                                                window.start[0] + i, stride[0]);
        for (long j = threadIdx.x / warpSize; j < window.length[1];
             j += blockDim.x / warpSize) {
            const auto y =
                cached ? cache[offset[1] + j]
                       : product_axis_point(grid, product, 1,
                                            window.start[1] + j, stride[1]);
            const double dxy2 = x.distance2 + y.distance2;
            if (dxy2 > window.radius2)
                continue;
            const long row = x.offset + y.offset;
            const double xy = x.factor * y.factor;
            for (long k = threadIdx.x % warpSize; k < window.length[2];
                 k += warpSize) {
                const auto z =
                    cached ? cache[offset[2] + k]
                           : product_axis_point(grid, product, 2,
                                                window.start[2] + k, stride[2]);
                if (dxy2 + z.distance2 <= window.radius2)
                    visit(row + z.offset, xy * z.factor);
            }
        }
    }
}

} // namespace ATV::cuda::gpw::internal

/// @file cpu/scf/basis_eval.h
/// @brief Inline basis function evaluation on grid points (s/p/d type Gaussians).
///
/// Shared by scf_solver.cpp (V_xc construction), gradient.cpp (XC gradient),
/// and the cube-file writers. All coordinates in Bohr.
///
/// General Cartesian Gaussian:
///   phi_{nx,ny,nz}(r) = dx^nx dy^ny dz^nz * sum_k c_k exp(-alpha_k r^2)
///
/// with dx = rx - sh.center[0], etc., and (nx, ny, nz) decoded from
/// sh.cart via `fill_cart_comps(sh.l)` (matches libcint / PySCF order):
///
///   l=0: (0,0,0)                                   -- s
///   l=1: (1,0,0) (0,1,0) (0,0,1)                   -- px, py, pz
///   l=2: (2,0,0) (1,1,0) (1,0,1) (0,2,0) (0,1,1) (0,0,2)  -- xx xy xz yy yz zz
#pragma once

#include "atomiverse/cpu/scf/cart.h"
#include "atomiverse/cpu/scf/native_integrals.h"

#include <cmath>

namespace ATV {
namespace cpu {
namespace scf {

namespace detail {

/// @brief `value^k` via simple loop (k <= kMaxL so the loop body is tiny).
[[nodiscard]] inline double pow_int(double v, int k) {
    double r = 1.0;
    for (int i = 0; i < k; ++i)
        r *= v;
    return r;
}

/// @brief Polynomial value P = dx^nx dy^ny dz^nz.
[[nodiscard]] inline double poly_val(int nx, int ny, int nz, double dx,
                                     double dy, double dz) {
    return pow_int(dx, nx) * pow_int(dy, ny) * pow_int(dz, nz);
}

/// @brief First-derivative polynomial:
///        dP/dr_i for P = dx^nx dy^ny dz^nz, with exponents (nx,ny,nz).
///
/// The axis parameter selects: 0 = d/dx, 1 = d/dy, 2 = d/dz.
[[nodiscard]] inline double poly_grad(int axis, int nx, int ny, int nz,
                                      double dx, double dy, double dz) {
    switch (axis) {
    case 0:
        return (nx == 0) ? 0.0
                         : nx * pow_int(dx, nx - 1) * pow_int(dy, ny)
                               * pow_int(dz, nz);
    case 1:
        return (ny == 0) ? 0.0
                         : ny * pow_int(dx, nx) * pow_int(dy, ny - 1)
                               * pow_int(dz, nz);
    default:
        return (nz == 0) ? 0.0
                         : nz * pow_int(dx, nx) * pow_int(dy, ny)
                               * pow_int(dz, nz - 1);
    }
}

/// @brief Second-derivative polynomial d²P / (dr_i dr_j).
[[nodiscard]] inline double poly_hess(int i, int j, int nx, int ny, int nz,
                                      double dx, double dy, double dz) {
    int n[3] = {nx, ny, nz};
    double d[3] = {dx, dy, dz};
    if (i == j) {
        if (n[i] < 2)
            return 0.0;
        int nn[3] = {nx, ny, nz};
        nn[i] -= 2;
        return static_cast<double>(n[i]) * (n[i] - 1) * pow_int(d[0], nn[0])
               * pow_int(d[1], nn[1]) * pow_int(d[2], nn[2]);
    } else {
        if (n[i] == 0 || n[j] == 0)
            return 0.0;
        int nn[3] = {nx, ny, nz};
        nn[i] -= 1;
        nn[j] -= 1;
        return static_cast<double>(n[i]) * n[j] * pow_int(d[0], nn[0])
               * pow_int(d[1], nn[1]) * pow_int(d[2], nn[2]);
    }
}

/// @brief Decode sh.cart -> (nx, ny, nz) via the canonical libcint order.
inline void cart_exponents(const native::Shell& sh, int n[3]) {
    int table[kMaxNCart][3];
    fill_cart_comps(sh.l, table);
    n[0] = table[sh.cart][0];
    n[1] = table[sh.cart][1];
    n[2] = table[sh.cart][2];
}

} // namespace detail

/// @brief Evaluate contracted basis function at a grid point.
/// @param sh Shell descriptor (exponents, normalized coefficients, center, l, cart).
/// @param rx Grid point x [Bohr].
/// @param ry Grid point y [Bohr].
/// @param rz Grid point z [Bohr].
/// @return Basis function value phi(r).
inline double eval_basis(const native::Shell& sh, double rx, double ry,
                         double rz) {
    double dx = rx - sh.center[0];
    double dy = ry - sh.center[1];
    double dz = rz - sh.center[2];
    double r2 = dx * dx + dy * dy + dz * dz;
    double min_alpha = 1e30;
    for (int k = 0; k < sh.n_prim; ++k)
        if (sh.alpha[k] < min_alpha)
            min_alpha = sh.alpha[k];
    if (min_alpha * r2 > 36.0)
        return 0.0;
    double radial = 0;
    for (int k = 0; k < sh.n_prim; ++k)
        radial += sh.coeff[k] * std::exp(-sh.alpha[k] * r2);
    if (sh.l == 0)
        return radial;
    int n[3];
    detail::cart_exponents(sh, n);
    return detail::poly_val(n[0], n[1], n[2], dx, dy, dz) * radial;
}

/// @brief Evaluate spatial gradient of contracted basis function.
///
/// d/dr_i [P(r) * f(r)] = (dP/dr_i) * f + P * (df/dr_i)
///   where P = dx^nx dy^ny dz^nz and f = radial part.
///   df/dr_i = s-type radial gradient = sum_k c_k * (-2 alpha_k d_i) exp(-alpha_k r²).
inline void eval_basis_grad(const native::Shell& sh, double rx, double ry,
                            double rz, double& gx, double& gy, double& gz) {
    double dx = rx - sh.center[0];
    double dy = ry - sh.center[1];
    double dz = rz - sh.center[2];
    double r2 = dx * dx + dy * dy + dz * dz;
    gx = gy = gz = 0;
    double min_alpha = 1e30;
    for (int k = 0; k < sh.n_prim; ++k)
        if (sh.alpha[k] < min_alpha)
            min_alpha = sh.alpha[k];
    if (min_alpha * r2 > 36.0)
        return;
    // Radial f and df/dr_i.
    double f = 0, dfx = 0, dfy = 0, dfz = 0;
    for (int k = 0; k < sh.n_prim; ++k) {
        double e = sh.coeff[k] * std::exp(-sh.alpha[k] * r2);
        f += e;
        double fac = -2.0 * sh.alpha[k] * e;
        dfx += fac * dx;
        dfy += fac * dy;
        dfz += fac * dz;
    }
    if (sh.l == 0) {
        gx = dfx;
        gy = dfy;
        gz = dfz;
        return;
    }
    int n[3];
    detail::cart_exponents(sh, n);
    const double P = detail::poly_val(n[0], n[1], n[2], dx, dy, dz);
    const double dPx = detail::poly_grad(0, n[0], n[1], n[2], dx, dy, dz);
    const double dPy = detail::poly_grad(1, n[0], n[1], n[2], dx, dy, dz);
    const double dPz = detail::poly_grad(2, n[0], n[1], n[2], dx, dy, dz);
    gx = dPx * f + P * dfx;
    gy = dPy * f + P * dfy;
    gz = dPz * f + P * dfz;
}

/// @brief Evaluate spatial Hessian of contracted basis function.
///
/// d²/(dr_i dr_j) [P * f] = d²P/(dr_i dr_j) * f + dP/dr_i * df/dr_j
///                        + dP/dr_j * df/dr_i + P * d²f/(dr_i dr_j).
inline void eval_basis_hessian(const native::Shell& sh, double rx, double ry,
                               double rz, double& hxx, double& hxy, double& hxz,
                               double& hyy, double& hyz, double& hzz) {
    double dx = rx - sh.center[0];
    double dy = ry - sh.center[1];
    double dz = rz - sh.center[2];
    double r2 = dx * dx + dy * dy + dz * dz;
    hxx = hxy = hxz = hyy = hyz = hzz = 0;
    double min_alpha = 1e30;
    for (int k = 0; k < sh.n_prim; ++k)
        if (sh.alpha[k] < min_alpha)
            min_alpha = sh.alpha[k];
    if (min_alpha * r2 > 36.0)
        return;
    // s-type radial: f, df[3], d2f[3][3].
    double f = 0;
    double df[3] = {0, 0, 0};
    double d2f[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (int k = 0; k < sh.n_prim; ++k) {
        double a = sh.alpha[k];
        double e = sh.coeff[k] * std::exp(-a * r2);
        double a2 = a * a;
        f += e;
        double fac = -2.0 * a * e;
        df[0] += fac * dx;
        df[1] += fac * dy;
        df[2] += fac * dz;
        d2f[0][0] += e * (-2.0 * a + 4.0 * a2 * dx * dx);
        d2f[1][1] += e * (-2.0 * a + 4.0 * a2 * dy * dy);
        d2f[2][2] += e * (-2.0 * a + 4.0 * a2 * dz * dz);
        d2f[0][1] += e * 4.0 * a2 * dx * dy;
        d2f[0][2] += e * 4.0 * a2 * dx * dz;
        d2f[1][2] += e * 4.0 * a2 * dy * dz;
    }
    d2f[1][0] = d2f[0][1];
    d2f[2][0] = d2f[0][2];
    d2f[2][1] = d2f[1][2];

    if (sh.l == 0) {
        hxx = d2f[0][0];
        hxy = d2f[0][1];
        hxz = d2f[0][2];
        hyy = d2f[1][1];
        hyz = d2f[1][2];
        hzz = d2f[2][2];
        return;
    }
    int n[3];
    detail::cart_exponents(sh, n);
    const double P = detail::poly_val(n[0], n[1], n[2], dx, dy, dz);
    double dP[3] = {
        detail::poly_grad(0, n[0], n[1], n[2], dx, dy, dz),
        detail::poly_grad(1, n[0], n[1], n[2], dx, dy, dz),
        detail::poly_grad(2, n[0], n[1], n[2], dx, dy, dz),
    };
    auto h = [&](int i, int j) {
        double d2P = detail::poly_hess(i, j, n[0], n[1], n[2], dx, dy, dz);
        return d2P * f + dP[i] * df[j] + dP[j] * df[i] + P * d2f[i][j];
    };
    hxx = h(0, 0);
    hxy = h(0, 1);
    hxz = h(0, 2);
    hyy = h(1, 1);
    hyz = h(1, 2);
    hzz = h(2, 2);
}

/// @brief Fused evaluation of basis function value, gradient, and Hessian.
inline void eval_basis_all(const native::Shell& sh, double rx, double ry,
                           double rz, double& val, double& gx, double& gy,
                           double& gz, double& hxx, double& hxy, double& hxz,
                           double& hyy, double& hyz, double& hzz) {
    double dx = rx - sh.center[0];
    double dy = ry - sh.center[1];
    double dz = rz - sh.center[2];
    double r2 = dx * dx + dy * dy + dz * dz;
    val = gx = gy = gz = 0;
    hxx = hxy = hxz = hyy = hyz = hzz = 0;
    double min_alpha = 1e30;
    for (int k = 0; k < sh.n_prim; ++k)
        if (sh.alpha[k] < min_alpha)
            min_alpha = sh.alpha[k];
    if (min_alpha * r2 > 36.0)
        return;
    // Radial part and derivatives.
    double f = 0;
    double df[3] = {0, 0, 0};
    double d2f[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (int k = 0; k < sh.n_prim; ++k) {
        double a = sh.alpha[k];
        double e = sh.coeff[k] * std::exp(-a * r2);
        double a2 = a * a;
        f += e;
        double fac = -2.0 * a * e;
        df[0] += fac * dx;
        df[1] += fac * dy;
        df[2] += fac * dz;
        d2f[0][0] += e * (-2.0 * a + 4.0 * a2 * dx * dx);
        d2f[1][1] += e * (-2.0 * a + 4.0 * a2 * dy * dy);
        d2f[2][2] += e * (-2.0 * a + 4.0 * a2 * dz * dz);
        d2f[0][1] += e * 4.0 * a2 * dx * dy;
        d2f[0][2] += e * 4.0 * a2 * dx * dz;
        d2f[1][2] += e * 4.0 * a2 * dy * dz;
    }
    d2f[1][0] = d2f[0][1];
    d2f[2][0] = d2f[0][2];
    d2f[2][1] = d2f[1][2];
    if (sh.l == 0) {
        val = f;
        gx = df[0];
        gy = df[1];
        gz = df[2];
        hxx = d2f[0][0];
        hxy = d2f[0][1];
        hxz = d2f[0][2];
        hyy = d2f[1][1];
        hyz = d2f[1][2];
        hzz = d2f[2][2];
        return;
    }
    int n[3];
    detail::cart_exponents(sh, n);
    const double P = detail::poly_val(n[0], n[1], n[2], dx, dy, dz);
    double dP[3] = {
        detail::poly_grad(0, n[0], n[1], n[2], dx, dy, dz),
        detail::poly_grad(1, n[0], n[1], n[2], dx, dy, dz),
        detail::poly_grad(2, n[0], n[1], n[2], dx, dy, dz),
    };
    val = P * f;
    gx = dP[0] * f + P * df[0];
    gy = dP[1] * f + P * df[1];
    gz = dP[2] * f + P * df[2];
    auto h = [&](int i, int j) {
        double d2P = detail::poly_hess(i, j, n[0], n[1], n[2], dx, dy, dz);
        return d2P * f + dP[i] * df[j] + dP[j] * df[i] + P * d2f[i][j];
    };
    hxx = h(0, 0);
    hxy = h(0, 1);
    hxz = h(0, 2);
    hyy = h(1, 1);
    hyz = h(1, 2);
    hzz = h(2, 2);
}

/// @brief Fused evaluation of basis function value and gradient.
inline void eval_basis_val_grad(const native::Shell& sh, double rx, double ry,
                                double rz, double& val, double& gx, double& gy,
                                double& gz) {
    double dx = rx - sh.center[0];
    double dy = ry - sh.center[1];
    double dz = rz - sh.center[2];
    double r2 = dx * dx + dy * dy + dz * dz;
    val = gx = gy = gz = 0;
    double min_alpha = 1e30;
    for (int k = 0; k < sh.n_prim; ++k)
        if (sh.alpha[k] < min_alpha)
            min_alpha = sh.alpha[k];
    if (min_alpha * r2 > 36.0)
        return;
    double f = 0, dfx = 0, dfy = 0, dfz = 0;
    for (int k = 0; k < sh.n_prim; ++k) {
        double a = sh.alpha[k];
        double e = sh.coeff[k] * std::exp(-a * r2);
        f += e;
        double fac = -2.0 * a * e;
        dfx += fac * dx;
        dfy += fac * dy;
        dfz += fac * dz;
    }
    if (sh.l == 0) {
        val = f;
        gx = dfx;
        gy = dfy;
        gz = dfz;
        return;
    }
    int n[3];
    detail::cart_exponents(sh, n);
    const double P = detail::poly_val(n[0], n[1], n[2], dx, dy, dz);
    const double dPx = detail::poly_grad(0, n[0], n[1], n[2], dx, dy, dz);
    const double dPy = detail::poly_grad(1, n[0], n[1], n[2], dx, dy, dz);
    const double dPz = detail::poly_grad(2, n[0], n[1], n[2], dx, dy, dz);
    val = P * f;
    gx = dPx * f + P * dfx;
    gy = dPy * f + P * dfy;
    gz = dPz * f + P * dfz;
}

} // namespace scf
} // namespace cpu
} // namespace ATV

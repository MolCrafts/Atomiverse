/// @file core/sllod_flow.h
/// @brief LAMMPS `fix deform` ERATE + `fix nvt/sllod` remap-v math.
///
/// Host and device share these functions so the two backends cannot drift.
/// The existing `Deform` component stays remap-x; this header is only the
/// flow used by `NvtSllod`.
///
/// Citations (LAMMPS `stable`):
///   * `MathExtra::multiply_shape_shape` (`math_extra.h`)
///   * `Domain::set_global_box` h_inv, `Domain::x2lamda` / `lamda2x`
///   * `FixDeform::init` ERATE `h_rate` / `h_ratelo` (`fix_deform.cpp`)
///   * `FixNVTSllod::nve_x` (`fix_nvt_sllod.cpp`), psllod = 0
///   * `ComputeTempDeform::{remove,apply}_deform_bias`
///   * `Domain::pbc` deform_vremap image signs
///   * `nhc_half_step` is the mtchain = 1, nc_tchain = 1, drag = 0 half-step
#pragma once

#include "atomiverse/core/dynamics_math.h"

#include <cmath>

#if defined(__CUDACC__)
#define ATV_MD_HD __host__ __device__
#else
#define ATV_MD_HD
#endif

namespace ATV::md {

/// Voigt-6 upper triangle: lx, ly, lz, yz, xz, xy.
inline constexpr int kFlowX = 0;
inline constexpr int kFlowY = 1;
inline constexpr int kFlowZ = 2;
inline constexpr int kFlowYZ = 3;
inline constexpr int kFlowXZ = 4;
inline constexpr int kFlowXY = 5;

/// Row-major cell slot of one Voigt component. Lower triangle is not a slot.
inline constexpr int kFlowBoxSlot[6] = {0, 4, 8, 5, 2, 1};

struct SllodFlow {
    double h[6]{};
    double h_inv[6]{};
    double h_rate[6]{};
    double h_ratelo[6]{};
    double grad_u[6]{};
    double h0[6]{};
    double rate = 0;
    int dir = kFlowX;
};

struct SllodChain {
    double eta = 0;
    double eta_dot = 0;
    double mass = 0;
};

/// Python / binding record. Not an `MDComponentParams` alternative.
struct NvtSllodSpec {
    double temperature_K = 0;
    double tau_fs = 0;
    double dt_fs = 0;
    double rate_per_fs = 0;
    int direction = kFlowX;
};

template <typename T>
ATV_MD_HD inline T sllod_exp(T x) {
#if defined(__CUDA_ARCH__)
    return ::exp(x);
#else
    return std::exp(x);
#endif
}

template <typename T>
ATV_MD_HD inline T sllod_floor(T x) {
#if defined(__CUDA_ARCH__)
    return ::floor(x);
#else
    return std::floor(x);
#endif
}

ATV_MD_HD inline void
sllod_multiply_shape_shape(const double* one, const double* two, double* ans) {
    ans[0] = one[0] * two[0];
    ans[1] = one[1] * two[1];
    ans[2] = one[2] * two[2];
    ans[3] = one[1] * two[3] + one[3] * two[2];
    ans[4] = one[0] * two[4] + one[5] * two[3] + one[4] * two[2];
    ans[5] = one[0] * two[5] + one[5] * two[1];
}

/// False when an edge is not positive. Does not throw (device-safe).
ATV_MD_HD inline bool sllod_invert(const double* h, double* h_inv) {
    if (!(h[0] > 0.0) || !(h[1] > 0.0) || !(h[2] > 0.0))
        return false;
    h_inv[0] = 1.0 / h[0];
    h_inv[1] = 1.0 / h[1];
    h_inv[2] = 1.0 / h[2];
    h_inv[3] = -h[3] / (h[1] * h[2]);
    h_inv[4] = (h[3] * h[5] - h[1] * h[4]) / (h[0] * h[1] * h[2]);
    h_inv[5] = -h[5] / (h[0] * h[1]);
    return true;
}

/// ERATE rates are constant. Edge `dir` < 3 uses `h_ratelo = -0.5 h_rate`.
/// Shear `xy` scales by `Ly0`; `xz` and `yz` scale by `Lz0`.
ATV_MD_HD inline void sllod_erate_rates(const double* h0, double rate, int dir,
                                        double* h_rate, double* h_ratelo) {
    for (int i = 0; i < 6; ++i) {
        h_rate[i] = 0.0;
        h_ratelo[i] = 0.0;
    }
    if (dir < 3) {
        h_rate[dir] = rate * h0[dir];
        h_ratelo[dir] = -0.5 * h_rate[dir];
    } else {
        const int perp = dir == kFlowXY ? 1 : 2;
        h_rate[dir] = rate * h0[perp];
    }
}

/// Box during step `k` (before that step's end_of_step write).
/// Edge: `L = L0 (1 + R k dt)`. Shear tilt: `tilt0 + R L_perp k dt`.
ATV_MD_HD inline void sllod_box_at(const double* h0, double rate, double dt,
                                   int dir, long long step, double* h) {
    for (int i = 0; i < 6; ++i)
        h[i] = h0[i];
    const double t = rate * dt * static_cast<double>(step);
    if (dir < 3)
        h[dir] = h0[dir] * (1.0 + t);
    else {
        const int perp = dir == kFlowXY ? 1 : 2;
        h[dir] = h0[dir] + h0[perp] * t;
    }
}

/// Symmetric origin shift that keeps the cell origin at 0. Shear returns 0.
ATV_MD_HD inline double sllod_edge_shift(double length_old, double length_new,
                                         int dir) {
    if (dir >= 3)
        return 0.0;
    return 0.5 * (length_new - length_old);
}

ATV_MD_HD inline bool sllod_refresh(SllodFlow& flow) {
    if (!sllod_invert(flow.h, flow.h_inv))
        return false;
    sllod_multiply_shape_shape(flow.h_rate, flow.h_inv, flow.grad_u);
    return true;
}

ATV_MD_HD inline void sllod_deform_bias(double x, double y, double z,
                                        const double* h_inv,
                                        const double* h_rate,
                                        const double* h_ratelo, double& bx,
                                        double& by, double& bz) {
    const double lam0 = h_inv[0] * x + h_inv[5] * y + h_inv[4] * z;
    const double lam1 = h_inv[1] * y + h_inv[3] * z;
    const double lam2 = h_inv[2] * z;
    bx = h_rate[0] * lam0 + h_rate[5] * lam1 + h_rate[4] * lam2 + h_ratelo[0];
    by = h_rate[1] * lam1 + h_rate[3] * lam2 + h_ratelo[1];
    bz = h_rate[2] * lam2 + h_ratelo[2];
}

/// Streaming velocity from the current position. `dtv = 0` uses boxlo.
/// `dtv > 0` integrates the lower corner the way `apply_deform_bias_all` does.
ATV_MD_HD inline void sllod_apply_bias(double& vx, double& vy, double& vz,
                                       double x, double y, double z,
                                       const double* h, const double* grad_u,
                                       double dtv) {
    const double xmid0 = 0.5 * h[0];
    const double xmid1 = 0.5 * h[1];
    const double xmid2 = 0.5 * h[2];
    const double ylo = xmid1 + (0.0 - xmid1) * sllod_exp(grad_u[1] * dtv);
    const double zlo = xmid2 + (0.0 - xmid2) * sllod_exp(grad_u[2] * dtv);
    vx +=
        (x - xmid0) * grad_u[0] + (y - ylo) * grad_u[5] + (z - zlo) * grad_u[4];
    vy += (y - xmid1) * grad_u[1] + (z - zlo) * grad_u[3];
    vz += (z - xmid2) * grad_u[2];
}

/// One atom of `FixNVTSllod::nve_x` with psllod = 0 and boxlo = 0.
/// Velocities on entry and exit are peculiar. The caller removes the bias
/// before this call and applies it again afterwards.
ATV_MD_HD inline void sllod_nve_x(double& x, double& y, double& z, double& vx,
                                  double& vy, double& vz, const double* h,
                                  const double* grad_u, double dt) {
    const double dthalf = 0.5 * dt;
    const double xfac0 = sllod_exp(grad_u[0] * dthalf);
    const double xfac1 = sllod_exp(grad_u[1] * dthalf);
    const double xfac2 = sllod_exp(grad_u[2] * dthalf);
    const double vfac0 = sllod_exp(-grad_u[0] * dthalf);
    const double vfac1 = sllod_exp(-grad_u[1] * dthalf);
    const double vfac2 = sllod_exp(-grad_u[2] * dthalf);
    const double xmid0 = 0.5 * h[0];
    const double xmid1 = 0.5 * h[1];
    const double xmid2 = 0.5 * h[2];
    const double ylo2 = xmid1 + (0.0 - xmid1) * xfac1 * xfac1;
    const double zlo2 = xmid2 + (0.0 - xmid2) * xfac2 * xfac2;

    vy -= dthalf * grad_u[3] * vz;
    vx -= dthalf * (grad_u[5] * vy + grad_u[4] * vz);
    vx *= vfac0;
    vy *= vfac1;
    vz *= vfac2;

    y += dthalf * grad_u[3] * z;
    x += dthalf * (grad_u[5] * y + grad_u[4] * z);
    x = xmid0 + (x - xmid0) * xfac0;
    y = xmid1 + (y - xmid1) * xfac1;
    z = xmid2 + (z - xmid2) * xfac2;

    x += dt * vx;
    y += dt * vy;
    z += dt * vz;

    x = xmid0 + (x - xmid0) * xfac0;
    y = xmid1 + (y - xmid1) * xfac1;
    z = xmid2 + (z - xmid2) * xfac2;
    x += dthalf * (grad_u[5] * (y - ylo2) + grad_u[4] * (z - zlo2));
    y += dthalf * grad_u[3] * (z - zlo2);

    vx *= vfac0;
    vy *= vfac1;
    vz *= vfac2;
    vx -= dthalf * (grad_u[5] * vy + grad_u[4] * vz);
    vy -= dthalf * grad_u[3] * vz;
}

/// Lambda PBC plus the `deform_vremap` velocity jump. One path for
/// orthorhombic and triclinic cells whose lower triangle is zero.
/// `floor` of a tiny negative wraps a full image, matching LAMMPS.
ATV_MD_HD inline void sllod_pbc(double& x, double& y, double& z, double& vx,
                                double& vy, double& vz, const double* h,
                                const double* h_inv, const double* h_rate) {
    double lam0 = h_inv[0] * x + h_inv[5] * y + h_inv[4] * z;
    double lam1 = h_inv[1] * y + h_inv[3] * z;
    double lam2 = h_inv[2] * z;
    const double n0 = sllod_floor(lam0);
    const double n1 = sllod_floor(lam1);
    const double n2 = sllod_floor(lam2);
    lam0 -= n0;
    lam1 -= n1;
    lam2 -= n2;
    x = h[0] * lam0 + h[5] * lam1 + h[4] * lam2;
    y = h[1] * lam1 + h[3] * lam2;
    z = h[2] * lam2;
    vx -= n0 * h_rate[0] + n1 * h_rate[5] + n2 * h_rate[4];
    vy -= n1 * h_rate[1] + n2 * h_rate[3];
    vz -= n2 * h_rate[2];
}

/// mtchain = 1 Nosé–Hoover half-step. `kinetic` is peculiar KE [eV]
/// (`0.5 Σ m v²`). Returns the peculiar-velocity scale. `eta` does not
/// enter that scale when the chain length is 1.
ATV_MD_HD inline double sllod_nh_half(SllodChain& chain, double kinetic,
                                      double target_twice_ke, double kT,
                                      double dt) {
    double position = chain.eta;
    double velocity = chain.eta_dot;
    double mass = chain.mass;
    double ke = kinetic;
    const double scale = nhc_half_step(&position, &velocity, &mass, 1, ke,
                                       target_twice_ke, kT, 0.5 * dt, 1.0);
    chain.eta = position;
    chain.eta_dot = velocity;
    return scale;
}

} // namespace ATV::md

#undef ATV_MD_HD

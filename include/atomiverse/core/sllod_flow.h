/// @file core/sllod_flow.h
/// @brief LAMMPS `fix deform` ERATE + `fix nvt/sllod` remap-v math.
///
/// Host and device share these functions so the two backends cannot drift.
/// The existing `Deform` component stays remap-x; this header is only the
/// flow used by `NvtSllod`.
///
/// Ported from LAMMPS (https://www.lammps.org/, Sandia National
/// Laboratories), SHA e8c4c278a8599165a0321ebfcb4e4046ef79ad79. Copyright
/// (2003) Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000
/// with Sandia Corporation, the U.S. Government retains certain rights in
/// this software. Distributed under the GNU General Public License,
/// version 2.
///   * `MathExtra::multiply_shape_shape` (`src/math_extra.h`)
///   * `Domain::set_global_box` h_inv, `Domain::x2lamda` / `lamda2x`
///   * `FixDeform::init` ERATE `h_rate` / `h_ratelo` (`src/fix_deform.cpp`)
///   * `FixDeform::update_domain` tilt continuity and the `flip yes` box
///     flip decision, `FixDeform::pre_exchange` (`src/fix_deform.cpp`)
///   * `Domain::image_flip`, `Domain::remap_all` deform_vremap
///     (`src/domain.cpp`)
///   * `FixNVTSllod::nve_x` (`src/fix_nvt_sllod.cpp`), psllod = 0
///   * `ComputeTempDeform::{remove,apply}_deform_bias`
///   * `Domain::pbc` deform_vremap image signs
///   * `FixNH::nhc_temp_integrate` (`src/fix_nh.cpp`): `sllod_nhc_half` is
///     its nc_tchain = 1, drag = 0 half-step for any `tchain`
#pragma once

#include "atomiverse/core/dynamics_math.h"

#include <cmath>
#include <vector>

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

/// Flow state of one NvtSllod run. `h` is the live cell; `h0` the strain
/// origin. `flip` / `h_flip` hold a box flip decided at end_of_step and
/// executed at the next pre_exchange (LAMMPS `FixDeform::flip`,
/// `flipxy flipxz flipyz`, `set[].tilt_flip`).
struct SllodFlow {
    double h[6]{};
    double h_inv[6]{};
    double h_rate[6]{};
    double h_ratelo[6]{};
    double grad_u[6]{};
    double h0[6]{};
    double h_flip[6]{};
    double rate = 0;
    double dt = 0;
    int dir = kFlowX;
    /// LAMMPS `Domain::image_flip(m, n, p)` arguments: flipxy, flipxz, flipyz.
    int flip[3]{};
};

/// Nosé–Hoover chain of an NvtSllod run, innermost link first (LAMMPS
/// `FixNH::eta`, `eta_dot`). Diagnostic copy; the live state stays with the
/// integrator (device memory on CUDA).
struct SllodChainState {
    std::vector<double> eta;     ///< Chain positions [1].
    std::vector<double> eta_dot; ///< Chain velocities [1/fs].
};

/// Python / binding record. Not an `MDComponentParams` alternative.
struct NvtSllodSpec {
    double temperature_K = 0;
    double tau_fs = 0;
    double dt_fs = 0;
    double rate_per_fs = 0;
    int direction = kFlowX;
    int tchain = 1;
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
ATV_MD_HD inline T sllod_fabs(T x) {
#if defined(__CUDA_ARCH__)
    return ::fabs(x);
#else
    return std::fabs(x);
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
/// `shift` (nullable) receives the lattice translation n removed from the
/// position; the caller adds it to the image flags (LAMMPS `Domain::pbc`
/// image bookkeeping), so x + H·image is unchanged by the wrap.
ATV_MD_HD inline void sllod_pbc(double& x, double& y, double& z, double& vx,
                                double& vy, double& vz, const double* h,
                                const double* h_inv, const double* h_rate,
                                int* shift = nullptr) {
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
    if (shift != nullptr) {
        shift[0] = static_cast<int>(n0);
        shift[1] = static_cast<int>(n1);
        shift[2] = static_cast<int>(n2);
    }
}

/// True while a box flip decided at end_of_step waits for pre_exchange.
ATV_MD_HD inline bool sllod_flip_pending(const SllodFlow& flow) {
    return flow.flip[0] != 0 || flow.flip[1] != 0 || flow.flip[2] != 0;
}

/// `FixDeform::end_of_step` for the one ERATE direction of an NvtSllod run
/// at strain clock `step`: new box into `flow.h` (the caller refreshes
/// `h_inv` / `grad_u` with `sllod_refresh`) and the `flip yes` decision.
///
/// Undeformed components keep their current value (LAMMPS style NONE).
/// The deformed tilt follows the linear ERATE path shifted by whole box
/// lengths to the value closest to the current tilt (`update_domain`), so
/// the path continues after a flip. LAMMPS also shifts xz by xy whenever it
/// shifts a yz target; for a NONE xz that double-counts the xz change the
/// flip already made, so only the deformed tilt is shifted here.
/// The flip test then mirrors `FixDeform::update_box` (fix_deform.cpp:1181):
/// any |tilt| beyond half its box length flips, yz first (moving xz by
/// xy), the cell stays unflipped until `sllod_apply_flip`.
/// @return Symmetric edge origin shift (`sllod_edge_shift`) [Angstrom].
ATV_MD_HD inline double sllod_advance_box(SllodFlow& flow, long long step) {
    double target[6];
    for (int i = 0; i < 6; ++i)
        target[i] = flow.h[i];
    {
        double linear[6];
        sllod_box_at(flow.h0, flow.rate, flow.dt, flow.dir, step, linear);
        target[flow.dir] = linear[flow.dir];
    }
    if (flow.dir >= 3) {
        const int i = flow.dir;
        const int idenom = i == kFlowYZ ? kFlowY : kFlowX;
        const double denom = target[idenom];
        const double denom_inv = 1.0 / denom;
        const double current = flow.h[i] / flow.h[idenom];
        while (target[i] * denom_inv - current > 0.0)
            target[i] -= denom;
        while (target[i] * denom_inv - current < 0.0)
            target[i] += denom;
        if (sllod_fabs(target[i] * denom_inv - 1.0 - current)
            < sllod_fabs(target[i] * denom_inv - current))
            target[i] -= denom;
    }
    const double shift =
        sllod_edge_shift(flow.h[flow.dir], target[flow.dir], flow.dir);

    flow.flip[0] = flow.flip[1] = flow.flip[2] = 0;
    const double xprd = target[kFlowX];
    const double yprd = target[kFlowY];
    const double xprdinv = 1.0 / xprd;
    const double yprdinv = 1.0 / yprd;
    if (target[kFlowYZ] * yprdinv < -0.5 || target[kFlowYZ] * yprdinv > 0.5
        || target[kFlowXZ] * xprdinv < -0.5 || target[kFlowXZ] * xprdinv > 0.5
        || target[kFlowXY] * xprdinv < -0.5
        || target[kFlowXY] * xprdinv > 0.5) {
        for (int k = 0; k < 6; ++k)
            flow.h_flip[k] = target[k];
        double* tf = flow.h_flip;
        if (tf[kFlowYZ] * yprdinv < -0.5) {
            tf[kFlowYZ] += yprd;
            tf[kFlowXZ] += tf[kFlowXY];
            flow.flip[2] = 1;
        } else if (tf[kFlowYZ] * yprdinv > 0.5) {
            tf[kFlowYZ] -= yprd;
            tf[kFlowXZ] -= tf[kFlowXY];
            flow.flip[2] = -1;
        }
        if (tf[kFlowXZ] * xprdinv < -0.5) {
            tf[kFlowXZ] += xprd;
            flow.flip[1] = 1;
        }
        if (tf[kFlowXZ] * xprdinv > 0.5) {
            tf[kFlowXZ] -= xprd;
            flow.flip[1] = -1;
        }
        if (tf[kFlowXY] * xprdinv < -0.5) {
            tf[kFlowXY] += xprd;
            flow.flip[0] = 1;
        }
        if (tf[kFlowXY] * xprdinv > 0.5) {
            tf[kFlowXY] -= xprd;
            flow.flip[0] = -1;
        }
    }
    for (int k = 0; k < 6; ++k)
        flow.h[k] = target[k];
    return shift;
}

/// `FixDeform::pre_exchange` box half: install the flipped tilts. The
/// caller refreshes `h_inv` / `grad_u`, then applies `sllod_image_flip` and
/// `sllod_pbc` to every owned atom (`image_flip` before `remap_all`), and
/// finally clears `flow.flip`. ERATE `h_rate` is constant, so the flip
/// leaves it unchanged (only ERATERS recomputes it there).
ATV_MD_HD inline void sllod_apply_flip(SllodFlow& flow) {
    for (int k = 3; k < 6; ++k)
        flow.h[k] = flow.h_flip[k];
}

/// `Domain::image_flip(m, n, p)` for one atom, with `flip = {m, n, p}` =
/// {flipxy, flipxz, flipyz}: the unwrapped position r + H·image is the same
/// before and after the flip changes H.
ATV_MD_HD inline void sllod_image_flip(int& xbox, int& ybox, int zbox,
                                       const int* flip) {
    ybox -= flip[2] * zbox;
    xbox -= flip[0] * ybox + flip[1] * zbox;
}

/// LAMMPS `FixNH` chain masses (`eta_mass_flag`): `Q_0 = N_f kT tau^2`,
/// `Q_k = kT tau^2` for the outer links [eV fs^2].
ATV_MD_HD inline void sllod_chain_masses(double* mass, int length, double ndof,
                                         double kT, double tau) {
    for (int k = 0; k < length; ++k)
        mass[k] = (k == 0 ? ndof : 1.0) * kT * tau * tau;
}

/// `FixNH::nhc_temp_integrate` with nc_tchain = 1 and drag = 0 over a
/// chain of `length` links (`tchain`). `kinetic` is the peculiar KE [eV]
/// (`0.5 Σ m v²`, LAMMPS `t_current` as energy) and leaves scaled by the
/// returned factor squared, which is what LAMMPS carries into the next
/// half-step (`t_current *= factor_eta*factor_eta`).
/// @return The peculiar-velocity scale `factor_eta`.
ATV_MD_HD inline double sllod_nhc_half(double* eta, double* eta_dot,
                                       const double* mass, int length,
                                       double& kinetic, double target_twice_ke,
                                       double kT, double dt) {
    return nhc_half_step(eta, eta_dot, mass, length, kinetic, target_twice_ke,
                         kT, 0.5 * dt, 1.0);
}

} // namespace ATV::md

#undef ATV_MD_HD

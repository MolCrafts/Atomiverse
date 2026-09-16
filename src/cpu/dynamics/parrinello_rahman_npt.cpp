/// @file cpu/dynamics/parrinello_rahman_npt.cpp
/// @brief Full-cell MTK / Parrinello–Rahman NPT integrator.

#include "atomiverse/cpu/dynamics/parrinello_rahman_npt.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/frame/pbc.h"

#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace md {

namespace {

// Clamp |ε̇_ij · dt| so a transient barostat kick cannot collapse the cell.
constexpr double kEpsDtMaxFraction = 0.005;

/// @brief (e^{2x}−1)/(2x) with stable Taylor near 0 — MTK position drift.
inline double expm1c(double x) {
    if (std::abs(x) < 1e-6)
        return 1.0 + x + (2.0 / 3.0) * x * x;
    return std::expm1(2.0 * x) / (2.0 * x);
}

/// @brief Symmetric matrix exponential for small strain A (‖A‖ ≲ 0.01).
///        exp(A) ≈ I + A + A²/2 — exact on pure diagonals through O(A²).
inline Mat3 matrix_exp_sym(const Mat3& A) {
    Mat3 A2 = A * A;
    Mat3 R = Mat3::identity() + A + A2 * AccReal(0.5);
    // Keep result symmetric (numerical drift).
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j) {
            const AccReal s = AccReal(0.5) * (R(i, j) + R(j, i));
            R(i, j) = s;
            R(j, i) = s;
        }
    return R;
}

/// @brief Clamp each entry of A so |A_ij| ≤ lim.
inline Mat3 clamp_mat(const Mat3& A, double lim) {
    Mat3 R;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R(i, j) = std::clamp(static_cast<double>(A(i, j)), -lim, lim);
    return R;
}

} // namespace

ParrinelloRahmanNPT::ParrinelloRahmanNPT(double T_target, double P_target,
                                         double tau_T, double tau_P, double dt,
                                         double W_cell)
    : T_target_(T_target), P_target_(P_target), tau_T_(tau_T), tau_P_(tau_P),
      dt_(dt), W_override_(W_cell) {}

void ParrinelloRahmanNPT::build(const cpu::BuildContext& ctx) {
    int N = static_cast<int>(ctx.frame.nlocal());
#ifdef ATV_HAS_MPI
    if (ctx.runtime.mpi_size > 1 && ctx.runtime.md.dd
        && ctx.runtime.md.dd->cart_comm != MPI_COMM_NULL) {
        const int n_local = ctx.runtime.md.dd->n_local;
        MPI_Allreduce(&n_local, &N, 1, MPI_INT, MPI_SUM,
                      ctx.runtime.md.dd->cart_comm);
    }
#endif
    n_dof_ = ctx.runtime.md.n_dof_global;
    (void)N;
    kT_ = units::kB_ev_per_k * T_target_;

    Q_t_.assign(static_cast<std::size_t>(mtchain_), 0.0);
    Q_b_.assign(static_cast<std::size_t>(mpchain_), 0.0);
    eta_t_.assign(static_cast<std::size_t>(mtchain_), 0.0);
    eta_t_dot_.assign(static_cast<std::size_t>(mtchain_), 0.0);
    eta_b_.assign(static_cast<std::size_t>(mpchain_), 0.0);
    eta_b_dot_.assign(static_cast<std::size_t>(mpchain_), 0.0);
    eps_dot_ = Mat3::zeros();

    const double tau_T2 = tau_T_ * tau_T_;
    const double tau_P2 = tau_P_ * tau_P_;
    Q_t_[0] = static_cast<double>(n_dof_) * kT_ * tau_T2;
    for (int k = 1; k < mtchain_; ++k)
        Q_t_[static_cast<std::size_t>(k)] = kT_ * tau_T2;
    Q_b_[0] = kT_ * tau_P2;
    for (int k = 1; k < mpchain_; ++k)
        Q_b_[static_cast<std::size_t>(k)] = kT_ * tau_P2;

    if (W_override_ > 0.0)
        W_ = W_override_;
    else
        W_ = (static_cast<double>(n_dof_) + 3.0) * kT_ * tau_P2;

    tdrag_factor_ = 1.0 - drag_ * dt_ / tau_T_;
    pdrag_factor_ = 1.0 - drag_ * dt_ / tau_P_;

    KE_global_ = 0.5 * static_cast<double>(n_dof_) * kT_;
}

double
ParrinelloRahmanNPT::compute_global_KE(const cpu::LaunchContext& ctx) const {
    const auto& f = ctx.frame;
    const int n = owned_count(ctx);
    const StorageReal* __restrict sp_mass = f.span(kAtom_mass);
    const StorageReal* __restrict sp_vx = f.span(kAtom_vx);
    const StorageReal* __restrict sp_vy = f.span(kAtom_vy);
    const StorageReal* __restrict sp_vz = f.span(kAtom_vz);
    double ke_local = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : ke_local)
    for (int i = 0; i < n; ++i) {
        const double v2 =
            sp_vx[i] * sp_vx[i] + sp_vy[i] * sp_vy[i] + sp_vz[i] * sp_vz[i];
        ke_local += sp_mass[i] * v2;
    }
    ke_local *= 0.5 * units::mass_v2_to_energy;

    double ke = ke_local;
#ifdef ATV_HAS_MPI
    if (ctx.runtime.mpi_size > 1 && ctx.runtime.md.dd
        && ctx.runtime.md.dd->cart_comm != MPI_COMM_NULL) {
        MPI_Allreduce(&ke_local, &ke, 1, MPI_DOUBLE, MPI_SUM,
                      ctx.runtime.md.dd->cart_comm);
    }
#endif
    return ke;
}

std::array<double, 3>
ParrinelloRahmanNPT::compute_axis_KE(const cpu::LaunchContext& ctx) const {
    const auto& f = ctx.frame;
    const int n = owned_count(ctx);
    const StorageReal* __restrict sp_mass = f.span(kAtom_mass);
    const StorageReal* __restrict sp_vx = f.span(kAtom_vx);
    const StorageReal* __restrict sp_vy = f.span(kAtom_vy);
    const StorageReal* __restrict sp_vz = f.span(kAtom_vz);
    double kx = 0.0, ky = 0.0, kz = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : kx, ky, kz)
    for (int i = 0; i < n; ++i) {
        const double m = sp_mass[i];
        kx += m * sp_vx[i] * sp_vx[i];
        ky += m * sp_vy[i] * sp_vy[i];
        kz += m * sp_vz[i] * sp_vz[i];
    }
    double ke[3] = {0.5 * units::mass_v2_to_energy * kx,
                    0.5 * units::mass_v2_to_energy * ky,
                    0.5 * units::mass_v2_to_energy * kz};
#ifdef ATV_HAS_MPI
    if (ctx.runtime.mpi_size > 1 && ctx.runtime.md.dd
        && ctx.runtime.md.dd->cart_comm != MPI_COMM_NULL) {
        double local[3] = {ke[0], ke[1], ke[2]};
        MPI_Allreduce(local, ke, 3, MPI_DOUBLE, MPI_SUM,
                      ctx.runtime.md.dd->cart_comm);
    }
#endif
    return {ke[0], ke[1], ke[2]};
}

void ParrinelloRahmanNPT::nhc_atoms_half(const cpu::LaunchContext& ctx,
                                         double half_dt) {
    const double quarter_dt = 0.5 * half_dt;
    const double eighth_dt = 0.5 * quarter_dt;
    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    const int M = mtchain_;

    auto G_t = [&](int k) -> double {
        if (k == 0)
            return (2.0 * KE_global_ - n_dof_ * kT_) / Q_t_[0];
        return (Q_t_[static_cast<std::size_t>(k - 1)]
                    * eta_t_dot_[static_cast<std::size_t>(k - 1)]
                    * eta_t_dot_[static_cast<std::size_t>(k - 1)]
                - kT_)
               / Q_t_[static_cast<std::size_t>(k)];
    };

    for (int k = M - 1; k >= 0; --k) {
        const double expfac =
            (k < M - 1) ? std::exp(-eta_t_dot_[static_cast<std::size_t>(k + 1)]
                                   * eighth_dt)
                        : 1.0;
        eta_t_dot_[static_cast<std::size_t>(k)] *= expfac;
        eta_t_dot_[static_cast<std::size_t>(k)] += G_t(k) * quarter_dt;
        eta_t_dot_[static_cast<std::size_t>(k)] *= tdrag_factor_;
        eta_t_dot_[static_cast<std::size_t>(k)] *= expfac;
    }

    const double scale = std::exp(-eta_t_dot_[0] * half_dt);
    StorageReal* __restrict sp_vx = f.span(kAtom_vx);
    StorageReal* __restrict sp_vy = f.span(kAtom_vy);
    StorageReal* __restrict sp_vz = f.span(kAtom_vz);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        sp_vx[i] *= scale;
        sp_vy[i] *= scale;
        sp_vz[i] *= scale;
    }
    KE_global_ *= scale * scale;

    for (int k = 0; k < M; ++k)
        eta_t_[static_cast<std::size_t>(k)] +=
            eta_t_dot_[static_cast<std::size_t>(k)] * half_dt;

    for (int k = 0; k < M; ++k) {
        const double expfac =
            (k < M - 1) ? std::exp(-eta_t_dot_[static_cast<std::size_t>(k + 1)]
                                   * eighth_dt)
                        : 1.0;
        eta_t_dot_[static_cast<std::size_t>(k)] *= expfac;
        eta_t_dot_[static_cast<std::size_t>(k)] += G_t(k) * quarter_dt;
        eta_t_dot_[static_cast<std::size_t>(k)] *= expfac;
    }
}

void ParrinelloRahmanNPT::nhc_baro_half(double half_dt) {
    const double quarter_dt = 0.5 * half_dt;
    const double eighth_dt = 0.5 * quarter_dt;
    const int M = mpchain_;
    const double W_axis = W_ / 3.0;

    // Six flexible-cell DOFs (Voigt): ½ W_axis Σ_diag ε̇² + ½ W_axis Σ_off ε̇²
    // (off-diagonal counted once each for the independent shear rates).
    auto baro_2ke = [&]() -> double {
        double s = 0.0;
        for (int a = 0; a < 3; ++a)
            s += W_axis * static_cast<double>(eps_dot_(a, a))
                 * static_cast<double>(eps_dot_(a, a));
        // Off-diagonal: xy, xz, yz (symmetric matrix → one independent each).
        s += W_axis * static_cast<double>(eps_dot_(0, 1))
             * static_cast<double>(eps_dot_(0, 1));
        s += W_axis * static_cast<double>(eps_dot_(0, 2))
             * static_cast<double>(eps_dot_(0, 2));
        s += W_axis * static_cast<double>(eps_dot_(1, 2))
             * static_cast<double>(eps_dot_(1, 2));
        return s;
    };

    auto G_b = [&](int k) -> double {
        if (k == 0)
            return (baro_2ke() - kT_) / Q_b_[0];
        return (Q_b_[static_cast<std::size_t>(k - 1)]
                    * eta_b_dot_[static_cast<std::size_t>(k - 1)]
                    * eta_b_dot_[static_cast<std::size_t>(k - 1)]
                - kT_)
               / Q_b_[static_cast<std::size_t>(k)];
    };

    for (int k = M - 1; k >= 0; --k) {
        const double expfac =
            (k < M - 1) ? std::exp(-eta_b_dot_[static_cast<std::size_t>(k + 1)]
                                   * eighth_dt)
                        : 1.0;
        eta_b_dot_[static_cast<std::size_t>(k)] *= expfac;
        eta_b_dot_[static_cast<std::size_t>(k)] += G_b(k) * quarter_dt;
        eta_b_dot_[static_cast<std::size_t>(k)] *= pdrag_factor_;
        eta_b_dot_[static_cast<std::size_t>(k)] *= expfac;
    }

    const double baro_scale = std::exp(-eta_b_dot_[0] * half_dt);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            eps_dot_(i, j) *= baro_scale;

    for (int k = 0; k < M; ++k)
        eta_b_[static_cast<std::size_t>(k)] +=
            eta_b_dot_[static_cast<std::size_t>(k)] * half_dt;

    for (int k = 0; k < M; ++k) {
        const double expfac =
            (k < M - 1) ? std::exp(-eta_b_dot_[static_cast<std::size_t>(k + 1)]
                                   * eighth_dt)
                        : 1.0;
        eta_b_dot_[static_cast<std::size_t>(k)] *= expfac;
        eta_b_dot_[static_cast<std::size_t>(k)] += G_b(k) * quarter_dt;
        eta_b_dot_[static_cast<std::size_t>(k)] *= expfac;
    }
}

void ParrinelloRahmanNPT::update_eps_dot(const cpu::LaunchContext& ctx,
                                         double half_dt) {
    const auto& f = ctx.frame;
    const double V = f.box.volume();
    if (V <= 0.0)
        return;

    const std::array<double, 3> ke_axis = compute_axis_KE(ctx);

    // Voigt virial: [xx,yy,zz,xy,xz,yz] — rank-local until Reduce; allreduce.
    double vir[6] = {f.meta.get(kStress)[0], f.meta.get(kStress)[1],
                     f.meta.get(kStress)[2], f.meta.get(kStress)[3],
                     f.meta.get(kStress)[4], f.meta.get(kStress)[5]};
#ifdef ATV_HAS_MPI
    if (ctx.runtime.mpi_size > 1 && ctx.runtime.md.dd
        && ctx.runtime.md.dd->cart_comm != MPI_COMM_NULL) {
        double local[6];
        for (int c = 0; c < 6; ++c)
            local[c] = vir[c];
        MPI_Allreduce(local, vir, 6, MPI_DOUBLE, MPI_SUM,
                      ctx.runtime.md.dd->cart_comm);
    }
#endif

    // Full pressure tensor (virial + kinetic). Off-diagonal kinetic is
    // neglected at equipartition (mean zero); diagonals use 2·KE_α.
    Mat3 P_tens = Mat3::zeros();
    P_tens(0, 0) = (vir[0] + 2.0 * ke_axis[0]) / V;
    P_tens(1, 1) = (vir[1] + 2.0 * ke_axis[1]) / V;
    P_tens(2, 2) = (vir[2] + 2.0 * ke_axis[2]) / V;
    P_tens(0, 1) = P_tens(1, 0) = vir[3] / V;
    P_tens(0, 2) = P_tens(2, 0) = vir[4] / V;
    P_tens(1, 2) = P_tens(2, 1) = vir[5] / V;

    // Hydrostatic couple on the diagonal (isotropic target): average the
    // three diagonal pressures so the cubic path collapses to one volume DOF
    // like NoseHooverNPT pdim==3 / LAMMPS couple().
    {
        const double ave = (1.0 / 3.0)
                           * (static_cast<double>(P_tens(0, 0))
                              + static_cast<double>(P_tens(1, 1))
                              + static_cast<double>(P_tens(2, 2)));
        P_tens(0, 0) = P_tens(1, 1) = P_tens(2, 2) = ave;
    }

    // MTK kinetic coupling — isotropic form (pdim=3): mtk_term = 2·KE/N
    // enters each diagonal G with the historical 3·f_omega scaling.
    const double KE = ke_axis[0] + ke_axis[1] + ke_axis[2];
    const int N_atoms = (n_dof_ + 3) / 3;
    const double mtk_term =
        (N_atoms > 0) ? 2.0 * KE / static_cast<double>(N_atoms) : 0.0;

    // Per-component force. Diagonal: match NoseHooverNPT G_ε = [3V(P−P_t)+mtk]/W
    // which is (W_axis = W/3) · dε̇ = V(P−P_t) + mtk/3  →  same after *3.
    // Shear: W_axis · dε̇_ij = V · P_ij  (target shear = 0).
    const double W_axis = W_ / 3.0;
    for (int a = 0; a < 3; ++a) {
        const double G_diag =
            (3.0 * V * (static_cast<double>(P_tens(a, a)) - P_target_)
             + mtk_term)
            / W_;
        eps_dot_(a, a) += G_diag * half_dt;
    }
    // Off-diagonal (independent Voigt components). Use a stiffer shear
    // mass (3× diagonal axis mass) so hydrostatic cubic fluids do not
    // accumulate runaway tilt noise from discrete virial fluctuations.
    const double W_shear = W_; // = 3 · W_axis
    const double g_xy =
        (V * static_cast<double>(P_tens(0, 1))) / W_shear * half_dt;
    const double g_xz =
        (V * static_cast<double>(P_tens(0, 2))) / W_shear * half_dt;
    const double g_yz =
        (V * static_cast<double>(P_tens(1, 2))) / W_shear * half_dt;
    eps_dot_(0, 1) += g_xy;
    eps_dot_(1, 0) += g_xy;
    eps_dot_(0, 2) += g_xz;
    eps_dot_(2, 0) += g_xz;
    eps_dot_(1, 2) += g_yz;
    eps_dot_(2, 1) += g_yz;
    (void)W_axis;
}

void ParrinelloRahmanNPT::scale_v_by_baro(const cpu::LaunchContext& ctx,
                                          double half_dt) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);

    // MTK velocity coupling: mtk_term2 = Tr(ε̇) / (3 N)
    const double tr_eps = static_cast<double>(eps_dot_(0, 0))
                          + static_cast<double>(eps_dot_(1, 1))
                          + static_cast<double>(eps_dot_(2, 2));
    const int N_atoms = (n_dof_ + 3) / 3;
    const double mtk_term2 =
        (N_atoms > 0) ? tr_eps / (3.0 * static_cast<double>(N_atoms)) : 0.0;

    const double shear = std::abs(static_cast<double>(eps_dot_(0, 1)))
                         + std::abs(static_cast<double>(eps_dot_(0, 2)))
                         + std::abs(static_cast<double>(eps_dot_(1, 2)));

    StorageReal* __restrict sp_vx = f.span(kAtom_vx);
    StorageReal* __restrict sp_vy = f.span(kAtom_vy);
    StorageReal* __restrict sp_vz = f.span(kAtom_vz);

    if (shear < 1e-14) {
        // Exact per-axis exp scale — matches NoseHooverNPT isotropic path.
        const double sx = std::exp(
            -(static_cast<double>(eps_dot_(0, 0)) + mtk_term2) * half_dt);
        const double sy = std::exp(
            -(static_cast<double>(eps_dot_(1, 1)) + mtk_term2) * half_dt);
        const double sz = std::exp(
            -(static_cast<double>(eps_dot_(2, 2)) + mtk_term2) * half_dt);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            sp_vx[i] *= sx;
            sp_vy[i] *= sy;
            sp_vz[i] *= sz;
        }
        if (sx == sy && sy == sz)
            KE_global_ *= sx * sx;
        else
            KE_global_ = compute_global_KE(ctx);
    } else {
        Mat3 A = Mat3::zeros();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                A(i, j) = -static_cast<double>(eps_dot_(i, j)) * half_dt;
        A(0, 0) -= mtk_term2 * half_dt;
        A(1, 1) -= mtk_term2 * half_dt;
        A(2, 2) -= mtk_term2 * half_dt;
        A = clamp_mat(A, kEpsDtMaxFraction);
        const Mat3 S = matrix_exp_sym(A);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            const double vx = sp_vx[i], vy = sp_vy[i], vz = sp_vz[i];
            sp_vx[i] = static_cast<double>(S(0, 0)) * vx
                       + static_cast<double>(S(0, 1)) * vy
                       + static_cast<double>(S(0, 2)) * vz;
            sp_vy[i] = static_cast<double>(S(1, 0)) * vx
                       + static_cast<double>(S(1, 1)) * vy
                       + static_cast<double>(S(1, 2)) * vz;
            sp_vz[i] = static_cast<double>(S(2, 0)) * vx
                       + static_cast<double>(S(2, 1)) * vy
                       + static_cast<double>(S(2, 2)) * vz;
        }
        KE_global_ = compute_global_KE(ctx);
    }
}

void ParrinelloRahmanNPT::kick_v_by_force(const cpu::LaunchContext& ctx,
                                          double half_dt) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    const double dtf = half_dt * units::force_to_accel;

    const StorageReal* __restrict sp_mass = f.span(kAtom_mass);
    StorageReal* __restrict sp_vx = f.span(kAtom_vx);
    StorageReal* __restrict sp_vy = f.span(kAtom_vy);
    StorageReal* __restrict sp_vz = f.span(kAtom_vz);
    const AccReal* __restrict sp_fx = f.span(kAtom_fx);
    const AccReal* __restrict sp_fy = f.span(kAtom_fy);
    const AccReal* __restrict sp_fz = f.span(kAtom_fz);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const double dtfm = dtf / sp_mass[i];
        sp_vx[i] += dtfm * sp_fx[i];
        sp_vy[i] += dtfm * sp_fy[i];
        sp_vz[i] += dtfm * sp_fz[i];
    }
}

void ParrinelloRahmanNPT::propagate_x_and_box(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);

    StorageReal* __restrict sp_x = f.span(kAtom_x);
    StorageReal* __restrict sp_y = f.span(kAtom_y);
    StorageReal* __restrict sp_z = f.span(kAtom_z);
    const StorageReal* __restrict sp_vx = f.span(kAtom_vx);
    const StorageReal* __restrict sp_vy = f.span(kAtom_vy);
    const StorageReal* __restrict sp_vz = f.span(kAtom_vz);

    // Diagonal MTK update (exact NoseHooverNPT path) for atom coords and
    // cell edge lengths. Shear rates update the full cell matrix separately
    // so hydrostatic cubic runs stay stable while still coupling tilt DOFs.
    double eps_dt[3];
    for (int d = 0; d < 3; ++d) {
        eps_dt[d] = std::clamp(static_cast<double>(eps_dot_(d, d)) * dt_,
                               -kEpsDtMaxFraction, kEpsDtMaxFraction);
    }
    const double ax = std::exp(eps_dt[0]);
    const double ay = std::exp(eps_dt[1]);
    const double az = std::exp(eps_dt[2]);
    const double dx = expm1c(0.5 * eps_dt[0]);
    const double dy = expm1c(0.5 * eps_dt[1]);
    const double dz = expm1c(0.5 * eps_dt[2]);

    // Optional affine shear on coordinates when rates are significant.
    const double gxy = std::clamp(static_cast<double>(eps_dot_(0, 1)) * dt_,
                                  -kEpsDtMaxFraction, kEpsDtMaxFraction);
    const double gxz = std::clamp(static_cast<double>(eps_dot_(0, 2)) * dt_,
                                  -kEpsDtMaxFraction, kEpsDtMaxFraction);
    const double gyz = std::clamp(static_cast<double>(eps_dot_(1, 2)) * dt_,
                                  -kEpsDtMaxFraction, kEpsDtMaxFraction);
    const bool has_shear =
        (std::abs(gxy) + std::abs(gxz) + std::abs(gyz)) > 0.0;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double x = sp_x[i] * ax + sp_vx[i] * dt_ * dx;
        double y = sp_y[i] * ay + sp_vy[i] * dt_ * dy;
        double z = sp_z[i] * az + sp_vz[i] * dt_ * dz;
        if (has_shear) {
            // First-order symmetric strain: x' += γ·y, etc.
            const double xs = x + gxy * y + gxz * z;
            const double ys = y + gxy * x + gyz * z;
            const double zs = z + gxz * x + gyz * y;
            x = xs;
            y = ys;
            z = zs;
        }
        sp_x[i] = x;
        sp_y[i] = y;
        sp_z[i] = z;
    }

    int n_total = n;
    if (ctx.runtime.md.dd && ctx.runtime.mpi_size > 1) {
        n_total = ctx.runtime.md.dd->n_local + ctx.runtime.md.dd->n_ghost;
    }
    if (n_total > n) {
#pragma omp parallel for schedule(static)
        for (int i = n; i < n_total; ++i) {
            double x = sp_x[i] * ax;
            double y = sp_y[i] * ay;
            double z = sp_z[i] * az;
            if (has_shear) {
                const double xs = x + gxy * y + gxz * z;
                const double ys = y + gxy * x + gyz * z;
                const double zs = z + gxz * x + gyz * y;
                x = xs;
                y = ys;
                z = zs;
            }
            sp_x[i] = x;
            sp_y[i] = y;
            sp_z[i] = z;
        }
    }

    // Cell: scale edges, then first-order tilt of off-diagonal entries.
    Mat3 h = f.box.cell();
    h(0, 0) *= ax;
    h(1, 1) *= ay;
    h(2, 2) *= az;
    // Existing off-diagonals scale with the product of edge factors so
    // tilt angles are approximately preserved under pure dilation.
    h(0, 1) =
        static_cast<double>(h(0, 1)) * ax + gxy * static_cast<double>(h(1, 1));
    h(1, 0) =
        static_cast<double>(h(1, 0)) * ay + gxy * static_cast<double>(h(0, 0));
    h(0, 2) =
        static_cast<double>(h(0, 2)) * ax + gxz * static_cast<double>(h(2, 2));
    h(2, 0) =
        static_cast<double>(h(2, 0)) * az + gxz * static_cast<double>(h(0, 0));
    h(1, 2) =
        static_cast<double>(h(1, 2)) * ay + gyz * static_cast<double>(h(2, 2));
    h(2, 1) =
        static_cast<double>(h(2, 1)) * az + gyz * static_cast<double>(h(1, 1));
    // Keep the stored cell symmetric (strain-rate is symmetric).
    h(1, 0) = h(0, 1) = AccReal(0.5) * (h(0, 1) + h(1, 0));
    h(2, 0) = h(0, 2) = AccReal(0.5) * (h(0, 2) + h(2, 0));
    h(2, 1) = h(1, 2) = AccReal(0.5) * (h(1, 2) + h(2, 1));
    f.box.set_cell(h);

    if (f.box.is_orthorhombic())
        wrap_orthorhombic(f, n);

    if (ctx.runtime.md.dd && ctx.runtime.mpi_size > 1) {
        ctx.runtime.md.dd->compute_sublo_subhi(f.box);
    }
}

double
ParrinelloRahmanNPT::extended_hamiltonian(const cpu::LaunchContext& ctx) const {
    const auto& f = ctx.frame;
    double H = f.meta.get(kKinetic) + f.meta.get(kEnergy);
    const double W_axis = W_ / 3.0;
    for (int a = 0; a < 3; ++a)
        H += 0.5 * W_axis * static_cast<double>(eps_dot_(a, a))
             * static_cast<double>(eps_dot_(a, a));
    H += 0.5 * W_axis * static_cast<double>(eps_dot_(0, 1))
         * static_cast<double>(eps_dot_(0, 1));
    H += 0.5 * W_axis * static_cast<double>(eps_dot_(0, 2))
         * static_cast<double>(eps_dot_(0, 2));
    H += 0.5 * W_axis * static_cast<double>(eps_dot_(1, 2))
         * static_cast<double>(eps_dot_(1, 2));
    for (int k = 0; k < mtchain_; ++k) {
        const auto i = static_cast<std::size_t>(k);
        H += 0.5 * Q_t_[i] * eta_t_dot_[i] * eta_t_dot_[i];
    }
    for (int k = 0; k < mpchain_; ++k) {
        const auto i = static_cast<std::size_t>(k);
        H += 0.5 * Q_b_[i] * eta_b_dot_[i] * eta_b_dot_[i];
    }
    H += static_cast<double>(n_dof_) * kT_ * eta_t_[0];
    for (int k = 1; k < mtchain_; ++k)
        H += kT_ * eta_t_[static_cast<std::size_t>(k)];
    for (int k = 0; k < mpchain_; ++k)
        H += kT_ * eta_b_[static_cast<std::size_t>(k)];
    H += P_target_ * f.box.volume();
    return H;
}

void ParrinelloRahmanNPT::launch(const cpu::LaunchContext& ctx) {
    const double half_dt = 0.5 * dt_;

    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        nhc_baro_half(half_dt);
        nhc_atoms_half(ctx, half_dt);
        update_eps_dot(ctx, half_dt);
        scale_v_by_baro(ctx, half_dt);
        kick_v_by_force(ctx, half_dt);
        propagate_x_and_box(ctx);
    } else if (ctx.at(ATV::md::Phase::final_integrate)) {
        kick_v_by_force(ctx, half_dt);
    } else if (ctx.at(ATV::md::Phase::end_of_step)) {
        KE_global_ = compute_global_KE(ctx);
        scale_v_by_baro(ctx, half_dt);
        update_eps_dot(ctx, half_dt);
        nhc_atoms_half(ctx, half_dt);
        nhc_baro_half(half_dt);
        ctx.frame.meta.get(kKinetic) = KE_global_;
        ctx.frame.meta.get(kConserved) = extended_hamiltonian(ctx);
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV

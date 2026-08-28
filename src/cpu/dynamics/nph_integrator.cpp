/// @file cpu/dynamics/nph_integrator.cpp
/// @brief Thermostat-free isotropic NPH integrator.

#include "atomiverse/cpu/dynamics/nph_integrator.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/frame/pbc.h"

#include <algorithm>
#include <array>
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

constexpr double kEpsDtMaxFraction = 0.005;

inline double expm1c(double x) {
    if (std::abs(x) < 1e-6)
        return 1.0 + x + (2.0 / 3.0) * x * x;
    return std::expm1(2.0 * x) / (2.0 * x);
}

/// @brief Reference temperature used only to set the MTK barostat mass
///        when the user does not pass W_cell. Not a thermostat target.
constexpr double kTref_K = 300.0;

} // namespace

NphIntegrator::NphIntegrator(double P_target, double tau_P, double dt,
                             double W_cell)
    : P_target_(P_target), tau_P_(tau_P), dt_(dt), W_override_(W_cell) {}

void NphIntegrator::build(const cpu::BuildContext& ctx) {
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

    // Prefer instantaneous kinetic temperature from the frame when
    // velocities already exist (InitVelocity builds before integrators
    // in the usual MD composition); fall back to 300 K only if KE≈0.
    const double tau_P2 = tau_P_ * tau_P_;
    if (W_override_ > 0.0) {
        W_ = W_override_;
    } else {
        double ke = 0.0;
        const int n = owned_count(ctx);
        const StorageReal* mass = ctx.frame.span(kAtom_mass);
        const StorageReal* vx = ctx.frame.span(kAtom_vx);
        const StorageReal* vy = ctx.frame.span(kAtom_vy);
        const StorageReal* vz = ctx.frame.span(kAtom_vz);
        for (int i = 0; i < n; ++i) {
            const double v2 = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
            ke += mass[i] * v2;
        }
        ke *= 0.5 * units::mass_v2_to_energy;
        double kT = (n_dof_ > 0 && ke > 0.0)
                        ? (2.0 * ke / static_cast<double>(n_dof_))
                        : (units::kB_ev_per_k * kTref_K);
        W_ = (static_cast<double>(n_dof_) + 3.0) * kT * tau_P2;
    }

    eps_dot_ = 0.0;
    KE_global_ = 0.0;
}

double NphIntegrator::compute_global_KE(const cpu::LaunchContext& ctx) const {
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
NphIntegrator::compute_axis_KE(const cpu::LaunchContext& ctx) const {
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

void NphIntegrator::update_eps_dot(const cpu::LaunchContext& ctx,
                                   double half_dt) {
    const auto& f = ctx.frame;
    const double V = f.box.volume();
    if (V <= 0.0)
        return;

    const std::array<double, 3> ke_axis = compute_axis_KE(ctx);
    double vir[3] = {f.meta.get(kStress)[0], f.meta.get(kStress)[1],
                     f.meta.get(kStress)[2]};
#ifdef ATV_HAS_MPI
    if (ctx.runtime.mpi_size > 1 && ctx.runtime.md.dd
        && ctx.runtime.md.dd->cart_comm != MPI_COMM_NULL) {
        double local[3] = {vir[0], vir[1], vir[2]};
        MPI_Allreduce(local, vir, 3, MPI_DOUBLE, MPI_SUM,
                      ctx.runtime.md.dd->cart_comm);
    }
#endif

    // Isotropic couple: averaged diagonal pressure (LAMMPS pstyle ISO).
    const double diag_sum = (vir[0] + 2.0 * ke_axis[0])
                            + (vir[1] + 2.0 * ke_axis[1])
                            + (vir[2] + 2.0 * ke_axis[2]);
    const double P_int = diag_sum / (3.0 * V);
    const double KE = ke_axis[0] + ke_axis[1] + ke_axis[2];
    const int N_atoms = (n_dof_ + 3) / 3;
    const double mtk_term =
        (N_atoms > 0) ? 2.0 * KE / static_cast<double>(N_atoms) : 0.0;

    // Same G_ε as NoseHooverNPT isotropic (no NHC on ε̇).
    const double G_eps = (3.0 * V * (P_int - P_target_) + mtk_term) / W_;
    eps_dot_ += G_eps * half_dt;
}

void NphIntegrator::scale_v_by_baro(const cpu::LaunchContext& ctx,
                                    double half_dt) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    const int N_atoms = (n_dof_ + 3) / 3;
    // Isotropic MTK: factor exp(-(1 + 3/N_dof)·ε̇·dt/2)
    // with 3/N_dof ≈ 1/N_atoms for N_dof = 3N−3 ≈ 3N.
    const double mtk_term2 =
        (N_atoms > 0) ? eps_dot_ / static_cast<double>(N_atoms) : 0.0;
    const double scale = std::exp(-(eps_dot_ + mtk_term2) * half_dt);

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
}

void NphIntegrator::kick_v_by_force(const cpu::LaunchContext& ctx,
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

void NphIntegrator::propagate_x_and_box(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);

    const double eps_dt =
        std::clamp(eps_dot_ * dt_, -kEpsDtMaxFraction, kEpsDtMaxFraction);
    const double a = std::exp(eps_dt);
    const double drift = expm1c(eps_dt * 0.5);

    StorageReal* __restrict sp_x = f.span(kAtom_x);
    StorageReal* __restrict sp_y = f.span(kAtom_y);
    StorageReal* __restrict sp_z = f.span(kAtom_z);
    const StorageReal* __restrict sp_vx = f.span(kAtom_vx);
    const StorageReal* __restrict sp_vy = f.span(kAtom_vy);
    const StorageReal* __restrict sp_vz = f.span(kAtom_vz);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        sp_x[i] = sp_x[i] * a + sp_vx[i] * dt_ * drift;
        sp_y[i] = sp_y[i] * a + sp_vy[i] * dt_ * drift;
        sp_z[i] = sp_z[i] * a + sp_vz[i] * dt_ * drift;
    }

    int n_total = n;
    if (ctx.runtime.md.dd && ctx.runtime.mpi_size > 1) {
        n_total = ctx.runtime.md.dd->n_local + ctx.runtime.md.dd->n_ghost;
    }
    if (n_total > n) {
#pragma omp parallel for schedule(static)
        for (int i = n; i < n_total; ++i) {
            sp_x[i] *= a;
            sp_y[i] *= a;
            sp_z[i] *= a;
        }
    }

    f.box.scale_isotropic(a);
    wrap_orthorhombic(f, n);

    if (ctx.runtime.md.dd && ctx.runtime.mpi_size > 1) {
        ctx.runtime.md.dd->compute_sublo_subhi(f.box);
    }
}

double NphIntegrator::extended_enthalpy(const cpu::LaunchContext& ctx) const {
    const auto& f = ctx.frame;
    // H = KE + PE + P·V + ½ W ε̇²  (no thermostat chain terms).
    double H = f.meta.get(kKinetic) + f.meta.get(kEnergy);
    H += 0.5 * W_ * eps_dot_ * eps_dot_;
    H += P_target_ * f.box.volume();
    return H;
}

void NphIntegrator::launch(const cpu::LaunchContext& ctx) {
    const double half_dt = 0.5 * dt_;

    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        // Keep KE_global_ in sync before any scale that multiplies it.
        KE_global_ = compute_global_KE(ctx);
        // No NHC — barostat momentum, velocity scale, force kick, box.
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
        ctx.frame.meta.get(kKinetic) = KE_global_;
        ctx.frame.meta.get(kConserved) = extended_enthalpy(ctx);
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV

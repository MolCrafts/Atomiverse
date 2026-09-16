/// @file cpu/dynamics/berendsen.cpp
/// @brief CPU Berendsen weak-coupling thermostat implementation.

#include "atomiverse/cpu/dynamics/berendsen.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace md {

Berendsen::Berendsen(double T_start, double tau, double dt)
    : T_start_(T_start), T_stop_(T_start), tau_(tau), dt_(dt) {
    if (!(T_start > 0.0))
        throw std::invalid_argument("Berendsen: T_start must be > 0 K");
    if (!(tau > 0.0))
        throw std::invalid_argument("Berendsen: tau must be > 0 fs");
    if (!(dt > 0.0))
        throw std::invalid_argument("Berendsen: dt must be > 0 fs");
    T_live_ = T_start_;
}

void Berendsen::build(const cpu::BuildContext& ctx) {
    // Ndof: single owner, constraint-aware. @see Driver::publish_dof_
    n_dof_global_ = ctx.runtime.md.n_dof_global;
}

double Berendsen::compute_temp_target(std::int64_t step) const {
    double delta = 0.0;
    if (nlast_ > nfirst_) {
        delta = static_cast<double>(step - nfirst_)
                / static_cast<double>(nlast_ - nfirst_);
        delta = std::clamp(delta, 0.0, 1.0);
    }
    return T_start_ + delta * (T_stop_ - T_start_);
}

void Berendsen::launch(const cpu::LaunchContext& ctx) {
    if (!ctx.at(ATV::md::Phase::end_of_step))
        return;

    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    const int n_dof = n_dof_global_;
    if (n_dof <= 0)
        return;

    T_live_ = compute_temp_target(ctx.step);

    // Hoisted once for both loops below — never span() inside a parallel
    // region.
    const StorageReal* __restrict sp_mass = f.span(kAtom_mass);
    StorageReal* __restrict sp_vx = f.span(kAtom_vx);
    StorageReal* __restrict sp_vy = f.span(kAtom_vy);
    StorageReal* __restrict sp_vz = f.span(kAtom_vz);

    // Global kinetic energy [eV] over owned atoms (MPI-reduced).
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

    // No kinetic energy yet (velocities not initialized): T is 0 and it sits
    // in the denominator. Skip — no fudge.
    if (ke <= 0.0)
        return;

    const double T =
        2.0 * ke / (static_cast<double>(n_dof) * units::kB_ev_per_k);

    // Berendsen 1984 Eq. 11: λ² = 1 + (Δt/τ)(T₀/T − 1).
    //
    // The bracket is unbounded below when T ≫ T₀ over a long coupling step, so
    // λ² can go negative and the scaling would be imaginary. LAMMPS
    // `fix_temp_berendsen.cpp` clamps λ to [0.8, 1.25] per step; the same
    // clamp is applied here so a cold start or a bad initial geometry damps
    // instead of exploding.
    constexpr double kLambdaMin = 0.8;
    constexpr double kLambdaMax = 1.25;
    const double lambda2 = 1.0 + (dt_ / tau_) * (T_live_ / T - 1.0);
    const double lambda =
        std::clamp(lambda2 > 0.0 ? std::sqrt(lambda2) : kLambdaMin, kLambdaMin,
                   kLambdaMax);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        sp_vx[i] *= lambda;
        sp_vy[i] *= lambda;
        sp_vz[i] *= lambda;
    }

    // Cumulative thermostat work. Published to kConserved the way LAMMPS folds
    // `ecouple` into `econserve` — a diagnostic sum, not a Hamiltonian
    // invariant (see the class note).
    const double ke_after = lambda * lambda * ke;
    thermostat_energy_ += -(ke_after - ke);
    f.meta.get(kConserved) =
        f.meta.get(kEnergy) + ke_after + thermostat_energy_;
}

} // namespace md
} // namespace cpu
} // namespace ATV

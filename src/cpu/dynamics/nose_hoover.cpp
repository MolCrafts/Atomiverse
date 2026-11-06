#include "atomiverse/core/dynamics_math.h"
/// @file cpu/dynamics/nose_hoover.cpp
/// @brief CPU Nose-Hoover thermostat implementation (LAMMPS `fix nvt`).

#include "atomiverse/cpu/dynamics/nose_hoover.h"

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

NoseHoover::NoseHoover(double T_start, double tau_T, double dt)
    : T_start_(T_start), T_stop_(T_start), tau_T_(tau_T), dt_(dt) {
    // A zero or negative setpoint / coupling time makes the chain mass
    // Q = N_dof k_B T tau^2 vanish (division by zero in the chain update);
    // refused here exactly as cuda::md::NoseHoover refuses it.
    if (!(T_start > 0.0) || !std::isfinite(T_start) || !(tau_T > 0.0)
        || !std::isfinite(tau_T) || !(dt > 0.0) || !std::isfinite(dt))
        throw std::invalid_argument(
            "cpu::md::NoseHoover: T_start [K], tau_T [fs] and dt [fs] must "
            "be finite and > 0");
    T_live_ = T_start_;
}

void NoseHoover::build(const cpu::BuildContext& ctx) {
    // Driver publishes the constraint-aware global DOF count before build.
    n_dof_global_ = ctx.runtime.md.n_dof_global;
    boltz_ = ctx.runtime.units.boltz;
    // Initial chain mass at the ramp anchor `T_start_`; `launch()`
    // retunes it each step to the live setpoint (LAMMPS recomputes
    // `eta_mass[0] = tdof·boltz·t_target/t_freq²` in nhc_temp_integrate).
    Q_ = static_cast<double>(n_dof_global_) * boltz_ * T_start_ * tau_T_
         * tau_T_;
}

void NoseHoover::set_T_target(double T_target, bool reset_chain) {
    if (!(T_target > 0.0) || !std::isfinite(T_target))
        throw std::invalid_argument(
            "cpu::md::NoseHoover::set_T_target: T_target [K] must be finite "
            "and > 0");
    T_start_ = T_target;
    // Moving the anchor also collapses any configured continuous ramp —
    // a `set_T_target` call is the segmented-run idiom, distinct from
    // the `set_T_stop` + `set_run_span` continuous ramp.
    T_stop_ = T_target;
    T_live_ = T_target;
    // Rescale `Q_` to the new setpoint using the cached global DOF count.
    // Skipping this would leave the chain mass tuned to the old bath
    // temperature, producing the wrong response time after the transition.
    Q_ = static_cast<double>(n_dof_global_) * boltz_ * T_target * tau_T_
         * tau_T_;
    if (reset_chain)
        xi_ = 0.0;
}

/// @see LAMMPS `FixNH::compute_temp_target()` (`src/fix_nh.cpp:2204`):
///
///     double delta = update->ntimestep - update->beginstep;
///     if (delta != 0.0) delta /= update->endstep - update->beginstep;
///     t_target = t_start + delta * (t_stop - t_start);
///
/// Atomiverse uses the cumulative MD step index (`LaunchContext::step`,
/// continuous across chunked `run()` calls) for `ntimestep`, and the
/// `[nfirst_, nlast_]` span supplied by `set_run_span()` for
/// `beginstep`/`endstep`. `delta` is clamped to `[0,1]` so stepping
/// past `nlast_` holds the setpoint at `T_stop_` instead of
/// extrapolating. With the default span (`nlast_ <= nfirst_`) or
/// `T_stop_ == T_start_`, the result is the constant `T_start_` — the
/// no-ramp default.
double NoseHoover::compute_temp_target(std::int64_t step) const {
    return ATV::md::ramp_target(T_start_, T_stop_, step, nfirst_, nlast_);
}

void NoseHoover::launch(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    // Driver republishes this global, constraint-aware count after live edits.
    // Refresh the mass below without resetting the thermostat's chain state.
    n_dof_global_ = ctx.runtime.md.n_dof_global;
    const int n_dof = n_dof_global_;
    // A published zero-DOF system (e.g. one atom after COM removal) has no
    // thermal bath. Preserve velocities and chain state instead of dividing
    // by the zero thermostat mass.
    if (n_dof <= 0)
        return;

    if (ctx.at(ATV::md::Phase::initial_integrate)
        || ctx.at(ATV::md::Phase::final_integrate)) {

        // Hoisted once for both loops in this branch -- the
        // launch-boundary rule.
        const StorageReal* __restrict sp_mass = f.span(kAtom_mass);
        StorageReal* __restrict sp_vx = f.span(kAtom_vx);
        StorageReal* __restrict sp_vy = f.span(kAtom_vy);
        StorageReal* __restrict sp_vz = f.span(kAtom_vz);

        // Live thermostat setpoint for this step (Tstart→Tstop ramp).
        // `Q_` is retuned to the live temperature so the coupling
        // frequency `ω = 1/tau_T` stays constant across the ramp,
        // matching LAMMPS recomputing `eta_mass[0]` each NHC update.
        T_live_ = compute_temp_target(ctx.step);
        Q_ = static_cast<double>(n_dof_global_) * boltz_ * T_live_ * tau_T_
             * tau_T_;

        // Per-rank kinetic over owned atoms only.
        double ke_local = 0;
#pragma omp parallel for schedule(static) reduction(+ : ke_local)
        for (int i = 0; i < n; ++i) {
            double v2 =
                sp_vx[i] * sp_vx[i] + sp_vy[i] * sp_vy[i] + sp_vz[i] * sp_vz[i];
            ke_local += sp_mass[i] * v2;
        }
        ke_local *= double(0.5) * ctx.runtime.units.mvv2e;

        // Phase 0.5: under multi-rank reduce KE across the Cartesian comm so
        // the thermostat couples to the *global* bath rather than this
        // rank's slab. Adds two MPI_Allreduces per integration step (one in
        // initial_integrate, one in final_integrate); the cost is a single
        // double per call.
        double ke = ke_local;
#ifdef ATV_HAS_MPI
        if (ctx.runtime.mpi_size > 1 && ctx.runtime.md.dd
            && ctx.runtime.md.dd->cart_comm != MPI_COMM_NULL) {
            MPI_Allreduce(&ke_local, &ke, 1, MPI_DOUBLE, MPI_SUM,
                          ctx.runtime.md.dd->cart_comm);
        }
#endif

        // Update thermostat variable against the live target.
        xi_ +=
            (dt_ * double(0.5))
            * (double(2) * ke - static_cast<double>(n_dof) * boltz_ * T_live_)
            / Q_;

        // Rescale velocities (owners only).
        double scale = std::exp(-dt_ * double(0.5) * xi_);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            sp_vx[i] *= scale;
            sp_vy[i] *= scale;
            sp_vz[i] *= scale;
        }
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV

/// @file cpu/dynamics/nose_hoover_npt.cpp
/// @brief Single-chain Martyna-Tobias-Klein isotropic NPT.

#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
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

/// @brief Hard clamp on `|ε̇·dt|` used by the box-scaling update.
///
/// `ε̇` is the strain-rate log-volume velocity; the exact update scales
/// the box by `exp(ε̇·dt)` each half-step. A transient large `ε̇`
/// (early equilibration, large pressure error, user-chosen small
/// `τ_P`) can blow the box up or collapse it to zero in a single
/// step. Clamping the stride before `exp` matches the LAMMPS `fix
/// npt iso` guard and keeps the integrator stable without materially
/// affecting the converged NPT ensemble (the clamp only bites during
/// the first few steps of a poorly-initialised run).
// 0.5% per half-step ≈ 1% per step.  The clamp is only meant to fence
// off the first few ill-conditioned equilibration steps of an
// over-compressed start state; a well-initialised run sees `eps_dot·dt`
// bounded by the barostat natural frequency `2π/tau_P ≈ 2×10⁻³` at
// `tau_P=1000 fs`, comfortably below this fence.
constexpr double kEpsDtMaxFraction = 0.005;

/// @brief Velocity-drift factor `(e^{2x} − 1) / (2x)` for the MTK
///        position propagator, with a stable Taylor branch near 0.
///
/// The MTK position-update operator integrates `dx/dt = ε̇·x + v` over
/// one step holding `v` constant. The exact solution is
///
///     x(dt) = x(0)·e^{ε̇·dt} + v·dt·(e^{ε̇·dt} − 1)/(ε̇·dt)
///
/// so the velocity term carries the factor `(e^{ε̇·dt} − 1)/(ε̇·dt)`,
/// which equals `e^{ε̇·dt/2}·sinhc(ε̇·dt/2)`. Calling the argument
/// `x = ε̇·dt/2`, this helper returns `(e^{2x} − 1)/(2x)`.
///
/// A previous implementation used the bare `sinhc(ε̇·dt/2)` and so
/// dropped the `e^{ε̇·dt/2}` factor — a **first-order** error in
/// `ε̇·dt` (the Taylor series of `e^x·sinhc(x)` is `1 + x + 2x²/3 + …`
/// whereas `sinhc(x)` is `1 + x²/6 + …`, i.e. the linear term is
/// missing). That broke time-reversal symmetry of the integrator and
/// pumped a slow box drift over long runs (see spec
/// nose-hoover-npt-trotter-fix.md / test_lammps_npt_parity.cpp).
inline double expm1c(double x) {
    // (e^{2x} - 1) / (2x); equivalently e^x * sinh(x)/x.
    if (std::abs(x) < 1e-6) {
        // Taylor of (e^{2x}-1)/(2x) = 1 + x + 2x²/3 + x³/3 + ...
        return 1.0 + x + (2.0 / 3.0) * x * x;
    }
    return std::expm1(2.0 * x) / (2.0 * x);
}

} // namespace

NoseHooverNPT::NoseHooverNPT(double T_target, double tau_T, double P_target,
                             double tau_P, double dt, double drag, int mtchain,
                             int mpchain, std::array<bool, 3> couple_axis,
                             double T_stop)
    : T_target_(T_target),
      // `T_stop` defaults to the -1 sentinel meaning "no ramp" — fold it
      // to `T_target` so `compute_temp_target` sees `T_start == T_stop`
      // and the linear interpolation degenerates to the constant
      // setpoint. This mirrors LAMMPS `fix_nh.cpp` ctor line
      // `t_target = t_start` when only one temperature is given.
      T_stop_(T_stop < 0.0 ? T_target : T_stop), tau_T_(tau_T),
      P_target_(P_target), tau_P_(tau_P), dt_(dt), drag_(drag),
      mtchain_(mtchain), mpchain_(mpchain), couple_axis_(couple_axis) {
    // `pdim` = number of barostatted axes (LAMMPS `fix_nh.cpp`
    // `pdim = p_flag[0] + p_flag[1] + p_flag[2]`).
    pdim_ = (couple_axis_[0] ? 1 : 0) + (couple_axis_[1] ? 1 : 0)
            + (couple_axis_[2] ? 1 : 0);
    T_live_ = T_target_;
}

/// @see LAMMPS `FixNH::setup()` (`src/fix_nh.cpp`) — the barostat /
///      thermostat mass initialisation (`eta_mass`, `omega_mass`,
///      `etap_mass`). Atomiverse folds `omega_mass` (LAMMPS per-axis)
///      into the single isotropic `W_ = 3·omega_mass`.
void NoseHooverNPT::build(const cpu::BuildContext& ctx) {
    // n_dof must be the GLOBAL degree-of-freedom count: it controls Q_t_[0]
    // (thermostat coupling mass), W_ (barostat mass) and the equipartition
    // target 0.5·n_dof·kT used by compute_global_KE's NHC update. Using a
    // per-rank value blows up the barostat under multi-rank rebuilds.
    //
    // Single-rank: frame.nlocal() IS global N. Multi-rank: when build()
    // runs BEFORE partitioning (initial build), atoms.size() is global too;
    // but when build() runs AFTER a prior stage has already partitioned
    // (the polyethylene_tension per-stage driver rebuild), atoms.size() ==
    // n_local + n_ghost — wrong. The DD's `n_local` is authoritative on
    // each rank post-partition, so Allreduce-sum that to recover global N.
    int N = static_cast<int>(ctx.frame.nlocal());
#ifdef ATV_HAS_MPI
    // MUST be unconditional across all ranks: MPI_Allreduce is a collective,
    // so every rank on cart_comm has to call it or the ones that do hang
    // forever. A rank that transiently owns no atoms (n_local == 0 — common
    // at high rank counts, e.g. an empty cell in a 4x4x4 decomposition) still
    // participates, contributing 0 to the sum so N stays the correct global
    // total. Gating this on `n_local > 0` deadlocked the stage-1->stage-2 NPT
    // rebuild at 64 ranks while 32 ranks (no empty cells) ran fine.
    if (ctx.runtime.mpi_size > 1 && ctx.runtime.md.dd
        && ctx.runtime.md.dd->cart_comm != MPI_COMM_NULL) {
        const int n_local = ctx.runtime.md.dd->n_local;
        MPI_Allreduce(&n_local, &N, 1, MPI_INT, MPI_SUM,
                      ctx.runtime.md.dd->cart_comm);
    }
#endif
    // Ndof: single owner, constraint-aware. @see Driver::publish_dof_
    n_dof_ = ctx.runtime.md.n_dof_global;
    (void)N;
    kT_ = units::kB_ev_per_k * T_target_;

    // Standard MTK masses (Tuckerman 2006 §III).
    //   Q_t[0] = N_f kT tau_T^2 drives the full KE reservoir;
    //   Q_t[k] = kT tau_T^2  for k >= 1  thermostats the previous link.
    Q_t_.assign(static_cast<std::size_t>(mtchain_), 0.0);
    Q_b_.assign(static_cast<std::size_t>(mpchain_), 0.0);
    eta_t_.assign(static_cast<std::size_t>(mtchain_), 0.0);
    eta_t_dot_.assign(static_cast<std::size_t>(mtchain_), 0.0);
    eta_b_.assign(static_cast<std::size_t>(mpchain_), 0.0);
    eta_b_dot_.assign(static_cast<std::size_t>(mpchain_), 0.0);

    const double tau_T2 = tau_T_ * tau_T_;
    const double tau_P2 = tau_P_ * tau_P_;
    Q_t_[0] = static_cast<double>(n_dof_) * kT_ * tau_T2;
    for (int k = 1; k < mtchain_; ++k)
        Q_t_[static_cast<std::size_t>(k)] = kT_ * tau_T2;
    Q_b_[0] = kT_ * tau_P2; // baro-NHC driven by eps_dot KE
    for (int k = 1; k < mpchain_; ++k)
        Q_b_[static_cast<std::size_t>(k)] = kT_ * tau_P2;

    // Barostat mass: MTK-consistent `(N_f + d) kT τ_P²` (Tuckerman 2006
    // Eq. 5.6.22). Numerically equivalent to LAMMPS `(N+1) kT τ_P²`
    // once our `3V` G_ε factor is balanced against LAMMPS's `V` factor
    // (both give `d ε̇ /dt ≈ V P / (N kT τ_P²)` in the large-N limit).
    W_ = (static_cast<double>(n_dof_) + 3.0) * kT_ * tau_P2;

    // LAMMPS-style empirical drag on *every* chain link of the backward
    // sweep. Matches `FixNH::nhc_temp_integrate` (`src/fix_nh.cpp`):
    //
    //     tdrag_factor = 1 - drag * update->dt * t_freq / nc_tchain
    //                  = 1 - drag * dt / tau_T   (with nc_tchain=1)
    //
    // and is multiplied into each `eta_dot[ich]` for `ich = mtchain-1 .. 0`
    // after the accelerated-then-exp-scaled update. An early Atomiverse
    // version applied `tdrag_factor` only to the innermost link with
    // `half_dt`, which was ~6x weaker damping (fewer links × half the
    // magnitude) and let chain oscillations ring on stiff-bond systems
    // (amorphous PE Stage 2) until atoms overlapped. LAMMPS's per-link
    // drag kills that mode.
    tdrag_factor_ = 1.0 - drag_ * dt_ / tau_T_;
    pdrag_factor_ = 1.0 - drag_ * dt_ / tau_P_;

    // Later build-time velocity initializers may still change the input state.
    // Seed KE from the actual velocities at the first initial_integrate,
    // rather than assuming the input is already at the target temperature.
    KE_global_ = 0.0;
    kinetic_initialized_ = false;
    eps_dot_ = {0.0, 0.0, 0.0};
}

/// @see LAMMPS `FixNH::compute_temp_target()` (`src/fix_nh.cpp`):
///
///     double delta = update->ntimestep - update->beginstep;
///     if (delta != 0.0) delta /= update->endstep - update->beginstep;
///     t_target = t_start + delta * (t_stop - t_start);
///
/// Atomiverse uses the cumulative MD step index (`LaunchContext::step`,
/// continuous across chunked `run()` calls) for `ntimestep`, and the
/// `[nfirst_, nlast_]` span supplied by `set_run_span()` for
/// `beginstep`/`endstep`. `delta` is clamped to `[0,1]` so stepping
/// past `nlast_` (e.g. an over-long final `run()` chunk) holds the
/// setpoint at `T_stop_` instead of extrapolating. With the default
/// span (`nlast_ <= nfirst_`) or `T_stop_ == T_target_`, the live
/// setpoint stays at `T_target_` — the no-ramp default.
void NoseHooverNPT::compute_temp_target(int step) {
    double delta = 0.0;
    if (nlast_ > nfirst_) {
        delta = static_cast<double>(step - nfirst_)
                / static_cast<double>(nlast_ - nfirst_);
        delta = std::clamp(delta, 0.0, 1.0);
    }
    T_live_ = T_target_ + delta * (T_stop_ - T_target_);
    kT_ = units::kB_ev_per_k * T_live_;
}

double NoseHooverNPT::compute_global_KE(const cpu::LaunchContext& ctx) const {
    // `ctx.frame` is a reference member, so it stays mutable through a
    // const `ctx` -- bind non-const to call the (non-const) `span()`.
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

/// @see The per-axis analogue of `compute_global_KE` — `KE_α = ½Σ m·v_α²`
///      for the anisotropic barostat's per-axis driving pressure.
std::array<double, 3>
NoseHooverNPT::compute_axis_KE(const cpu::LaunchContext& ctx) const {
    // `ctx.frame` is a reference member, so it stays mutable through a
    // const `ctx` -- bind non-const to call the (non-const) `span()`.
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

/// @see LAMMPS `FixNH::nhc_temp_integrate()` (`src/fix_nh.cpp`).
void NoseHooverNPT::nhc_atoms_half(const cpu::LaunchContext& ctx,
                                   double half_dt) {
    // M-chain NHC half-step, mirroring LAMMPS `fix_nh.cpp::nhc_temp_integrate`
    // (release branch). Per-link structure inside the backward sweep:
    //
    //     eta_dot[k] *= expfac           // pre-scale by outer link
    //     eta_dot[k] += G_k * dt/4       // accelerate
    //     eta_dot[k] *= tdrag_factor     // *** drag at every link ***
    //     eta_dot[k] *= expfac           // post-scale by outer link
    //
    // with `expfac = exp(-dt/8 * eta_dot[k+1])`; for the outermost link
    // (k = M-1) the "next" link `eta_dot[M]` is a zero sentinel, so
    // `expfac = 1` and the two scalings are no-ops — drag and G still
    // apply. The forward sweep re-runs the same structure with the
    // post-scale KE, minus drag (LAMMPS does not damp on the way out).
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

    // Backward pass: M-1 → 0. Drag applied at every link.
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

    // Scale atom velocities by the innermost thermostat momentum.
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

    // Propagate the chain positions.
    for (int k = 0; k < M; ++k)
        eta_t_[static_cast<std::size_t>(k)] +=
            eta_t_dot_[static_cast<std::size_t>(k)] * half_dt;

    // Forward pass: 0 → M-1. No drag (LAMMPS pattern).
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

/// @see LAMMPS `FixNH::nhc_press_integrate()` (`src/fix_nh.cpp`). The
///      innermost-link force `(W·ε̇² - kT)/Q_b[0]` matches LAMMPS's
///      `(kecurrent - lkt_press)/etap_mass[0]` with `lkt_press = kt`
///      for `pstyle == ISO`.
void NoseHooverNPT::nhc_baro_half(double half_dt) {
    // Same M-chain pattern as `nhc_atoms_half`, but the innermost link
    // thermostats the barostat momenta `ε̇_α` instead of atom KE.
    // Drag (`pdrag_factor`) is applied at every link in the backward
    // sweep and dropped in the forward sweep, matching LAMMPS
    // `fix_nh.cpp::nhc_press_integrate`.
    //
    // The barostat NHC is a single chain over the *active* `ε̇_α` set —
    // a single collective barostat DOF (LAMMPS keeps `lkt_press = kt`
    // for `pstyle == ISO`; Atomiverse keeps the single-DOF target for
    // every coupling so the all-axes case stays bit-compatible with the
    // pre-change isotropic integrator). The innermost-link force uses
    // the summed barostat KE `2·KE_baro = Σ_{α active} W_axis·ε̇_α²`
    // with `W_axis = W_/3` the per-axis barostat mass (LAMMPS
    // `omega_mass`; the isotropic total `W_ = 3·W_axis`). For the
    // all-coupled, all-equal case `Σ_α W_axis·ε̇_α² = W_·ε̇²`, exactly
    // the legacy scalar form.
    const double quarter_dt = 0.5 * half_dt;
    const double eighth_dt = 0.5 * quarter_dt;
    const int M = mpchain_;
    const double W_axis = W_ / 3.0;

    // Summed barostat kinetic-energy term `2·KE_baro` over active axes.
    auto baro_2ke = [&]() -> double {
        double s = 0.0;
        for (int a = 0; a < 3; ++a)
            if (couple_axis_[a])
                s += W_axis * eps_dot_[a] * eps_dot_[a];
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

    // Backward pass with per-link drag.
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

    // Scale each active box-length momentum by the innermost
    // baro-thermostat (LAMMPS `factor_etap` over the active `omega_dot`).
    const double baro_scale = std::exp(-eta_b_dot_[0] * half_dt);
    for (int a = 0; a < 3; ++a)
        if (couple_axis_[a])
            eps_dot_[a] *= baro_scale;

    for (int k = 0; k < M; ++k)
        eta_b_[static_cast<std::size_t>(k)] +=
            eta_b_dot_[static_cast<std::size_t>(k)] * half_dt;

    // Forward pass, no drag.
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

/// @see LAMMPS `FixNH::nh_omega_dot()` + `FixNH::couple()` (`src/fix_nh.cpp`).
///      Atomiverse integrates a per-axis `ε̇_α` for each barostatted
///      axis, mirroring LAMMPS's per-axis `omega_dot[i]` loop guarded by
///      `p_flag[i]`, and applies LAMMPS's `couple()` pressure-averaging
///      rule.
///
/// **Pressure coupling (mirrors `FixNH::couple()`).** When all three
/// axes are barostatted (`pdim_ == 3` — LAMMPS `pstyle == ISO` /
/// `pcouple == XYZ`), `couple()` sets every `p_current[i]` to the
/// *averaged* pressure `trace(σ)/(3V)`; the three `omega_dot` are then
/// driven by the identical force and stay equal — a single isotropic
/// volume DOF. When only some axes are active, each uses its own
/// diagonal stress `σ_αα/V`. Honouring this averaging is what makes the
/// all-axes case reduce **bit-for-bit** to the legacy isotropic scalar
/// `G_ε` (msu-pe-align-01 ac-004 regression guard) — driving the three
/// axes from the slightly-unequal instantaneous per-axis stress would
/// be a different (still valid) trajectory.
///
/// **Driving force.** LAMMPS per-axis `f_omega = (p_current[i]−p_hydro)·
/// V/omega_mass[i] + mtk_term1/omega_mass[i]`. Atomiverse uses
/// `W = 3·omega_mass` and the historical `G_ε = 3·f_omega`, so
///
///     G_{ε,α}·W = 3V·(P_α − P_target) + mtk_term
///
/// with `P_α` the (possibly averaged) axis pressure.
///
/// **MTK kinetic coupling.** LAMMPS-exact: `mtk_term1 = Σ_{α active}
/// 2·KE_α / (pdim·N)` using *actual* per-axis kinetic energies
/// (`fix_nh.cpp:2541-2551`). AV uses `W = 3·omega_mass` and
/// `G_ε = 3·f_omega`, so the equation-of-motion's MTK contribution is
/// `mtk_term = 3·mtk_term1 = 6·Σ_{α active} KE_α / (pdim·N)`. For
/// `pdim=3` with equipartitioned per-axis KE this reduces to the
/// legacy `2·KE/N` (the historical isotropic integrator stays
/// bit-identical at equipartition). For `pdim<3` it uses the genuine
/// anisotropic kinetic split so the barostat tracks LAMMPS — required
/// for runs with mixed barostatted/free axes (e.g. y/z-coupled NPT
/// with x driven by an external `Deform`).
void NoseHooverNPT::update_eps_dot(const cpu::LaunchContext& ctx,
                                   double half_dt) {
    const auto& f = ctx.frame;
    const double V = f.box.volume();
    if (V <= 0.0)
        return;

    // Standard MTK sign convention (Tuckerman 2006, LAMMPS `fix_nh.cpp`):
    // pressure positive ⇔ compressive. `frame.meta.get(kStress)[0..2]` holds the
    // per-axis pair-virial diagonal; the kinetic diagonal `2·KE_α` is
    // computed fresh from the current velocities (NOT the stale
    // `frame.meta.get(kStressKinetic)` snapshot taken at `post_force` time, before
    // `kick_v_by_force` / `scale_v_by_baro` — using it pumped a runaway
    // box drift; see tests/cpu/md/test_lammps_npt_parity.cpp).
    //
    // Why a *fresh per-axis* kinetic diagonal: `scale_v_by_baro` scales
    // velocity component α by `exp(-ε̇_α·dt/2)`, removing `2·ε̇_α·KE_α`
    // from the atom KE. For the per-axis Trotter split to be symplectic,
    // the barostat must absorb exactly that — i.e. axis α's driving
    // pressure must carry its OWN `2·KE_α`, not a `2·KE/3` isotropic
    // estimate. Driving a genuinely-anisotropic barostat from the
    // isotropic KE split leaves a residual that drifts the MTK
    // Hamiltonian secularly (caught by Slow_AnisoNptConserves…).
    const std::array<double, 3> ke_axis = compute_axis_KE(ctx);

    // Both call sites consume globally published virial: initial_integrate
    // reads the setup / previous end_of_step value, and end_of_step @ -40
    // follows Reduce @ -50. Reducing it again would multiply pressure by
    // the rank count. The kinetic diagonal above is recomputed at the current
    // velocity time slice because the thermostat/barostat has changed v.
    const double vir[3] = {f.meta.get(kStress)[0], f.meta.get(kStress)[1],
                           f.meta.get(kStress)[2]};

    // Per-axis virial + kinetic diagonal, with LAMMPS `couple()` averaging:
    //   * couple_xy  → average (xx,yy) only; z independent (semi-isotropic)
    //   * pdim==3 and not couple_xy → full ISO average of all three
    //     (bit-identical to the legacy scalar G_ε)
    double diag[3] = {vir[0] + 2.0 * ke_axis[0], vir[1] + 2.0 * ke_axis[1],
                      vir[2] + 2.0 * ke_axis[2]};
    if (couple_xy_ && couple_axis_[0] && couple_axis_[1]) {
        const double ave_xy = 0.5 * (diag[0] + diag[1]);
        diag[0] = diag[1] = ave_xy;
    } else if (pdim_ == 3) {
        const double ave = (1.0 / 3.0) * (diag[0] + diag[1] + diag[2]);
        diag[0] = diag[1] = diag[2] = ave;
    }

    // Per-axis pressure targets. Default: hydrostatic P_target_ on every
    // active axis. With surface tension γ, the lateral target is
    //   P_⊥ = P_n − n_surf · γ / L_z
    // (P_n ≡ P_target_; z keeps P_n). When γ==0 this collapses to
    // semi-isotropic isobaric NPT under couple_xy.
    double P_tgt[3] = {P_target_, P_target_, P_target_};
    if (couple_xy_ && gamma_ != 0.0) {
        const double Lz = f.box.length(2);
        if (Lz > 0.0) {
            const double Plat =
                P_target_ - static_cast<double>(n_surfaces_) * gamma_ / Lz;
            P_tgt[0] = P_tgt[1] = Plat;
        }
    }

    // MTK kinetic coupling — LAMMPS-exact (`fix_nh.cpp:2541-2551`):
    //   mtk_term1 = Σ_{α active} 2·KE_α / (pdim·N)
    // AV uses W = 3·omega_mass and G_eps = 3·f_omega, so its mtk_term
    // in the per-axis G_eps drive is 3·mtk_term1 =
    //   6·Σ_{α active} KE_α / (pdim·N).
    // The kinetic contribution uses the *actual* per-axis ke_axis[α]
    // (the same per-axis split used in the diagonal above), not an
    // equipartitioned KE/3 — required to match LAMMPS for pdim<3
    // (e.g. stage-5 y/z-coupled tension). For pdim=3 with isotropic
    // KE this reduces to the legacy 2·KE/N.
    double mtk_active_ke = 0.0;
    for (int a = 0; a < 3; ++a)
        if (couple_axis_[a])
            mtk_active_ke += ke_axis[a];
    const int N_atoms = (n_dof_ + 3) / 3;
    const double mtk_term =
        (pdim_ > 0)
            ? 6.0 * mtk_active_ke / (static_cast<double>(pdim_) * N_atoms)
            : 0.0;

    if (couple_xy_ && couple_axis_[0] && couple_axis_[1]) {
        // Lateral DOF: one shared strain rate from averaged P_⊥.
        const double P_xy = diag[0] / V;
        const double G_xy = (3.0 * V * (P_xy - P_tgt[0]) + mtk_term) / W_;
        eps_dot_[0] += G_xy * half_dt;
        eps_dot_[1] = eps_dot_[0];
        if (couple_axis_[2]) {
            const double P_z = diag[2] / V;
            const double G_z = (3.0 * V * (P_z - P_tgt[2]) + mtk_term) / W_;
            eps_dot_[2] += G_z * half_dt;
        }
    } else {
        for (int a = 0; a < 3; ++a) {
            if (!couple_axis_[a])
                continue; // rigid axis: ε̇_α stays 0.
            const double P_axis = diag[a] / V;
            const double G_eps_a =
                (3.0 * V * (P_axis - P_tgt[a]) + mtk_term) / W_;
            eps_dot_[a] += G_eps_a * half_dt;
        }
    }
}

/// @see LAMMPS `FixNH::nh_v_press()` (`src/fix_nh.cpp`). LAMMPS scales
///      velocity component α by `exp(-dt/2·(ω̇_α + mtk_term2))` (two
///      `dt/4` applications); `mtk_term2 = Σ_active ω̇/(pdim·N)`.
///
/// Atomiverse generalises the legacy isotropic `exp(-(1+3/N_dof)·ε̇·dt/2)`
/// to a per-axis factor: velocity component α scales by
/// `exp(-(ε̇_α + mtk_term2)·dt/2)`. A rigid axis (`couple_axis_[α]==false`,
/// `ε̇_α == 0`) still receives the global `mtk_term2` coupling — exactly
/// LAMMPS, where `mtk_term2` is added to every component's factor.
///
/// LAMMPS-exact: `mtk_term2 = Σ_{α active} ω̇ / (pdim·N)`
/// (`fix_nh.cpp:2564-2569`). For `pdim=3` with `Σ_active ε̇ = 3·ε̇` this
/// reduces to `ε̇/N`, and combined with `n_dof ≈ 3N` the per-axis
/// factor matches the legacy `exp(-(1+3/N_dof)·ε̇·dt/2)` at
/// equipartition. For `pdim<3` it follows LAMMPS exactly, restoring
/// anisotropic-NPT alignment under partial coupling.
void NoseHooverNPT::scale_v_by_baro(const cpu::LaunchContext& ctx,
                                    double half_dt) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);

    // MTK velocity-coupling term — LAMMPS-exact (`fix_nh.cpp:2564-2569`):
    //   mtk_term2 = Σ_{α active} ω̇ / (pdim·N).
    double sum_eps_dot = 0.0;
    for (int a = 0; a < 3; ++a)
        if (couple_axis_[a])
            sum_eps_dot += eps_dot_[a];
    const int N_atoms = (n_dof_ + 3) / 3;
    const double mtk_term2 =
        (pdim_ > 0) ? sum_eps_dot / (static_cast<double>(pdim_) * N_atoms)
                    : 0.0;

    const double sx = std::exp(-(eps_dot_[0] + mtk_term2) * half_dt);
    const double sy = std::exp(-(eps_dot_[1] + mtk_term2) * half_dt);
    const double sz = std::exp(-(eps_dot_[2] + mtk_term2) * half_dt);

    StorageReal* __restrict sp_vx = f.span(kAtom_vx);
    StorageReal* __restrict sp_vy = f.span(kAtom_vy);
    StorageReal* __restrict sp_vz = f.span(kAtom_vz);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        sp_vx[i] *= sx;
        sp_vy[i] *= sy;
        sp_vz[i] *= sz;
    }

    // KE bookkeeping. Isotropic fast-path: when all three scale factors
    // are identical (the all-axes-coupled case, or any case where the
    // active `ε̇_α` happen to coincide) the cached `KE_global_` scales
    // multiplicatively — algebraically identical to the legacy
    // `KE_global_ *= scale*scale` and MPI-safe without a reduction
    // (ac-004 bit-stability). Otherwise the per-axis scaling does not
    // factor through a single scalar, so `KE_global_` is recomputed
    // from the post-scale velocities (`compute_global_KE` is itself
    // MPI-aware, summing across ranks).
    if (sx == sy && sy == sz) {
        KE_global_ *= sx * sx;
    } else {
        KE_global_ = compute_global_KE(ctx);
    }
}

/// @see LAMMPS `FixNH::nve_v()` (`src/fix_nh.cpp`) — the half-step
///      velocity kick `v += dtf·f/m` with `dtf = 0.5·dt·ftm2v`.
void NoseHooverNPT::kick_v_by_force(const cpu::LaunchContext& ctx,
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
    // KE invalidated; will be recomputed at end_of_step.
}

/// @see LAMMPS `FixNH::nve_x()` + `FixNH::remap()` (`src/fix_nh.cpp`).
///      LAMMPS brackets the plain `x += dt·v` drift between two
///      half-step `remap()` box rescalings, scaling each axis edge by
///      `exp(dt·omega_dot[i])` only when `p_flag[i]` is set. Atomiverse
///      fuses both into the closed-form per-axis
///      `x_α·e^{ε̇_α·dt} + v_α·dt·(e^{ε̇_α·dt}−1)/(ε̇_α·dt)` plus a
///      single `box.scale_diagonal(...)`.
///
/// A rigid axis (`couple_axis_[α]==false`) has `ε̇_α == 0`, so its scale
/// factor is `exp(0)=1` and its drift factor is `expm1c(0)=1`: the
/// position update on that axis degenerates to the plain `x += v·dt`
/// drift and the box edge is left untouched — exactly LAMMPS skipping
/// `remap()` for an axis absent from `p_flag`.
void NoseHooverNPT::propagate_x_and_box(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);

    // Per-axis `ε̇_α·dt`, clamped to ~±0.5%/step to fence off a
    // transient diverging `ε̇` (NaN/inf) from blowing up cell-list grid
    // sizing. A rigid axis has `ε̇_α == 0` ⇒ `eps_dt[α] == 0`.
    double eps_dt[3], a[3], drift[3];
    for (int d = 0; d < 3; ++d) {
        eps_dt[d] = std::clamp(eps_dot_[d] * dt_, -kEpsDtMaxFraction,
                               kEpsDtMaxFraction);
        a[d] = std::exp(eps_dt[d]);
        // Velocity-drift factor `(e^{ε̇·dt} − 1)/(ε̇·dt)` — the exact MTK
        // position propagator. `expm1c(0) == 1` for a rigid axis.
        drift[d] = expm1c(eps_dt[d] * 0.5);
    }
    const double ax = a[0], ay = a[1], az = a[2];
    const double dx = drift[0], dy = drift[1], dz = drift[2];

    // Hoisted once for both the owned drift loop and the ghost rescale
    // loop below.
    StorageReal* __restrict sp_x = f.span(kAtom_x);
    StorageReal* __restrict sp_y = f.span(kAtom_y);
    StorageReal* __restrict sp_z = f.span(kAtom_z);
    const StorageReal* __restrict sp_vx = f.span(kAtom_vx);
    const StorageReal* __restrict sp_vy = f.span(kAtom_vy);
    const StorageReal* __restrict sp_vz = f.span(kAtom_vz);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        sp_x[i] = sp_x[i] * ax + sp_vx[i] * dt_ * dx;
        sp_y[i] = sp_y[i] * ay + sp_vy[i] * dt_ * dy;
        sp_z[i] = sp_z[i] * az + sp_vz[i] * dt_ * dz;
    }

    // Ghost rows must also track the box scaling. Between this call and
    // the next `CommBrick`, force evaluators (bond/angle/LJ)
    // will compute owner→ghost separations against the FRESHLY-SCALED
    // box. If the ghosts are left at their pre-scale absolute positions,
    // the owner→ghost Δ is wrong by `(ax-1)*pos` per axis. With ax≈1.001
    // and pos≈40 Å, that's 0.04 Å of spurious bond stretch per step,
    // driving E_bond and per-axis stress (especially for chain-aligned
    // systems like PE) to runaway within ~100–300 steps. HaloExchange's
    // skin gate may skip the refresh entirely if atoms haven't drifted
    // far enough, so the only safe fix is to scale ghosts here.
    //
    // LAMMPS `FixNH::remap()` does the same: `domain->x2lamda(nlocal +
    // nghost)` → scale box → `domain->lamda2x(nlocal + nghost)`. The
    // velocity-drift term is intentionally omitted for ghosts — their
    // owner rank applies the v drift to its own owned copy, which the
    // next forward halo will propagate.
    // Scale ghost positions too. In multi-rank the ghost range is
    // `[dd->n_local, dd->n_local + dd->n_ghost)`; in single-rank there
    // are no ghosts. `atoms.size()` reports the Frame-ctor capacity
    // (10000) in the refactored layout, not the live nlocal+nghost
    // count — using it walks ~6000 uninitialised rows past the ghost
    // range and scales row-0 positions into a singularity that
    // catastrophically explodes LJ at the next force pass.
    int n_total = n;
    if (ctx.runtime.md.dd && ctx.runtime.mpi_size > 1) {
        n_total = ctx.runtime.md.dd->n_local + ctx.runtime.md.dd->n_ghost;
    }
    if (n_total > n) {
#pragma omp parallel for schedule(static)
        for (int i = n; i < n_total; ++i) {
            sp_x[i] *= ax;
            sp_y[i] *= ay;
            sp_z[i] *= az;
        }
    }

    // Update box edges per axis: edge_α *= exp(ε̇_α·dt). A rigid axis
    // multiplies by exp(0)=1, i.e. is left unchanged. For the all-coupled
    // case all three factors are equal so this is identical to the
    // legacy `box.scale_isotropic(a)`.
    f.box.scale_diagonal(ax, ay, az);

    // Wrap (orthorhombic) owned atoms into the freshly-scaled box.
    wrap_orthorhombic(f, n);

    // The per-rank slab bounds (sublo/subhi) were set by Backend::set_frame
    // from the box at that point. NPT just changed the box -- if we don't
    // refresh the slab, the next Migrate / HaloExchange will see
    // owned-atom positions that no longer fit the stale boundary, yielding
    // out-of-range cell-list indices and a crash. compute_sublo_subhi is
    // O(1) (three multiplies) and is called identically on every rank, so
    // the partition stays consistent.
    //
    // We deliberately do NOT bump `neighbor_generation` here. An earlier
    // revision did, to force a full halo rebuild + migration every NPT
    // step — a workaround for the MSU-PE multi-rank collapse that was in
    // fact a stale special-bonds exclusion map in `PairLJCut` (fixed
    // there). Forcing per-step reneighbor made the 8-rank run ~10×
    // slower for no correctness benefit: `Comm::exchange` (every_=10) and
    // `CommBrick` (every_=10 + position-only refresh in between,
    // which already rescales ghosts as the box breathes) keep the halo
    // valid on the standard LAMMPS-style cadence. Migration still fires
    // promptly whenever an owner crosses a slab boundary because the
    // refreshed `sublo/subhi` above feeds the next scheduled migration.
    if (ctx.runtime.md.dd && ctx.runtime.mpi_size > 1) {
        ctx.runtime.md.dd->compute_sublo_subhi(f.box);
    }
}

/// @brief MTK extended (conserved) Hamiltonian — the quantity an exact
///        symplectic NPT integrator holds constant. Atom KE + PE +
///        barostat KE + both NHC chains' KE/PE + p·V.
/// @see LAMMPS `FixNH::compute_scalar()` (`src/fix_nh.cpp`) — the same
///      thermostat+barostat energy LAMMPS adds to PE+KE for its
///      `econserve` thermo keyword.
double
NoseHooverNPT::extended_hamiltonian(const cpu::LaunchContext& ctx) const {
    const auto& f = ctx.frame;
    // Atom kinetic + potential energy.
    double H = f.meta.get(kKinetic) + f.meta.get(kEnergy);
    // Barostat kinetic energy Σ_α ½·W_axis·ε̇_α² over active axes, with
    // `W_axis = W_/3` the per-axis barostat mass. For the all-coupled,
    // all-equal case this sums to the legacy `½·W_·ε̇²`.
    const double W_axis = W_ / 3.0;
    for (int a = 0; a < 3; ++a)
        if (couple_axis_[a])
            H += 0.5 * W_axis * eps_dot_[a] * eps_dot_[a];
    // Nosé-Hoover chain kinetic energies Σ ½·Q·η̇².
    for (int k = 0; k < mtchain_; ++k) {
        const auto i = static_cast<std::size_t>(k);
        H += 0.5 * Q_t_[i] * eta_t_dot_[i] * eta_t_dot_[i];
    }
    for (int k = 0; k < mpchain_; ++k) {
        const auto i = static_cast<std::size_t>(k);
        H += 0.5 * Q_b_[i] * eta_b_dot_[i] * eta_b_dot_[i];
    }
    // Chain potential energies. The atom chain's innermost link
    // thermostats all N_dof atomic DOFs (potential N_dof·kT·η_t[0]);
    // every other link thermostats one DOF (kT·η). The isotropic
    // barostat is a single DOF, so each baro-chain link contributes
    // kT·η_b.
    H += static_cast<double>(n_dof_) * kT_ * eta_t_[0];
    for (int k = 1; k < mtchain_; ++k)
        H += kT_ * eta_t_[static_cast<std::size_t>(k)];
    for (int k = 0; k < mpchain_; ++k)
        H += kT_ * eta_b_[static_cast<std::size_t>(k)];
    // Pressure-volume term.
    H += P_target_ * f.box.volume();
    return H;
}

/// @see LAMMPS `FixNH::initial_integrate()` (front-half P1-P6) and
///      `FixNH::final_integrate()` (back-half P7-P11), `src/fix_nh.cpp`.
///      Atomiverse splits LAMMPS's `final_integrate()` across the MD
///      `final_integrate` phase (P7) and `end_of_step` (P8-P11) so the
///      `Reduce` MPI all-reduce can run in between.
void NoseHooverNPT::launch(const cpu::LaunchContext& ctx) {
    const double half_dt = 0.5 * dt_;

    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        if (!kinetic_initialized_) {
            // LAMMPS FixNH::setup, src/fix_nh.cpp,
            // a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b: read current
            // temperature before the first thermostat update.
            // Copyright (2003) Sandia Corporation. GPL-2.0-only.
            KE_global_ = compute_global_KE(ctx);
            kinetic_initialized_ = true;
        }
        // Interpolate the live thermostat setpoint for this step before
        // the chain integrates, mirroring LAMMPS `FixNH::initial_integrate`
        // calling `compute_temp_target()` ahead of `nhc_temp_integrate()`.
        // `compute_temp_target` updates `kT_`; recompute the atom-NHC
        // chain masses from it so the ramp preserves the chain
        // frequency, exactly as LAMMPS recomputes `eta_mass` each step
        // (`eta_mass_flag` is always set). With `T_stop == T_target`
        // (no ramp) `kT_` is constant ⇒ `Q_t_` is recomputed to
        // identical values, so the no-ramp path stays bit-stable.
        compute_temp_target(ctx.step);
        const double tau_T2 = tau_T_ * tau_T_;
        Q_t_[0] = static_cast<double>(n_dof_) * kT_ * tau_T2;
        for (int k = 1; k < mtchain_; ++k)
            Q_t_[static_cast<std::size_t>(k)] = kT_ * tau_T2;

        // First half of Trotter (P1 -- P6). Outer-NHC order matches
        // LAMMPS `FixNH::initial_integrate` — `nhc_press_integrate()`
        // is called before `nhc_temp_integrate()`: the baro chain
        // is integrated *before* the atom chain so the velocity scaling
        // applied by `nhc_atoms_half` sees an `eps_dot` that already
        // reflects the baro-NHC update. Swapping these two breaks
        // Trotter symmetry with `end_of_step` and pumps a sustained
        // positive drift into `eps_dot` (~3e-4 /fs) that inflates the
        // box without bound (see spec nose-hoover-npt-trotter-fix.md).
        nhc_baro_half(half_dt);       // P1
        nhc_atoms_half(ctx, half_dt); // P2
        update_eps_dot(ctx, half_dt); // uses prev step's reduced stress
        scale_v_by_baro(ctx, half_dt);
        kick_v_by_force(ctx, half_dt);
        propagate_x_and_box(ctx);

    } else if (ctx.at(ATV::md::Phase::final_integrate)) {
        // Force has been recomputed; second half-kick (P7).
        kick_v_by_force(ctx, half_dt);

    } else if (ctx.at(ATV::md::Phase::end_of_step)) {
        // Reduce has run (frame.meta.get(kStress) globally reduced).
        // Recompute KE_global_ from CURRENT v before pressure step.
        KE_global_ = compute_global_KE(ctx);

        scale_v_by_baro(ctx, half_dt); // P8
        update_eps_dot(ctx, half_dt);  // P9 -- new pressure
        // Outer-NHC order reverses to atoms-then-baro on the back-half
        // to keep the full step symmetric Trotter. Matches LAMMPS
        // `FixNH::final_integrate` — `nhc_temp_integrate()` then
        // `nhc_press_integrate()`.
        nhc_atoms_half(ctx, half_dt); // P10
        nhc_baro_half(half_dt);       // P11

        // Rewrite frame.meta.get(kKinetic) so observers bound at end_of_step @ > -40
        // see the post-NPT kinetic energy.
        ctx.frame.meta.get(kKinetic) = KE_global_;

        // Publish the conserved MTK extended Hamiltonian for `Thermo`
        // and the conservation regression gate. Computed last, once the
        // atoms, box, ε̇ and both chains hold their end-of-step values.
        ctx.frame.meta.get(kConserved) = extended_hamiltonian(ctx);
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV

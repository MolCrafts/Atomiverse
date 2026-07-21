/// @file cpu/md/fix_csvr.cpp
/// @brief CPU CSVR (Bussi) thermostat implementation.

#include "atomiverse/cpu/md/fix_csvr.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/comm/domain.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/facade/stage.h"

#include <algorithm>
#include <cmath>
#include <random>

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

/// @brief Deterministic 64-bit seed from `(seed, step)` via splitmix64.
///
/// Depends only on the global step index — NOT on the atom distribution —
/// so every MPI rank draws the identical `R₁` and `χ²` deviate and the
/// trajectory is invariant to the rank count.
std::uint64_t step_seed(std::uint32_t seed, std::int64_t step) {
    std::uint64_t s =
        static_cast<std::uint64_t>(seed) * 0x9E3779B97F4A7C15ULL
        + static_cast<std::uint64_t>(step) * 0xBF58476D1CE4E5B9ULL;
    s = (s ^ (s >> 30)) * 0xBF58476D1CE4E5B9ULL;
    s = (s ^ (s >> 27)) * 0x94D049BB133111EBULL;
    return s ^ (s >> 31);
}

} // namespace

FixCSVR::FixCSVR(double T_start, double tau, double dt, std::uint32_t seed)
    : T_start_(T_start), T_stop_(T_start), tau_(tau), dt_(dt), seed_(seed) {
    T_live_ = T_start_;
}

void FixCSVR::build(const cpu::BuildContext& ctx) {
    // Global DOF count (3N − 3). atoms.size() is global only on the initial
    // build; after a domain partition it is n_local + n_ghost, so recover the
    // global N by allreduce-summing dd->n_local (mirrors NoseHoover::build).
    int n_global = static_cast<int>(ctx.frame.nlocal());
#ifdef ATV_HAS_MPI
    if (ctx.runtime.mpi_size > 1 && ctx.runtime.md.dd
        && ctx.runtime.md.dd->cart_comm != MPI_COMM_NULL) {
        const int n_local = ctx.runtime.md.dd->n_local;
        MPI_Allreduce(&n_local, &n_global, 1, MPI_INT, MPI_SUM,
                      ctx.runtime.md.dd->cart_comm);
    }
#endif
    // Ndof: single owner, constraint-aware. @see CPUDriver::publish_dof_
    n_dof_global_ = ctx.runtime.md.n_dof_global;
    (void)n_global;
}

double FixCSVR::compute_temp_target(std::int64_t step) const {
    double delta = 0.0;
    if (nlast_ > nfirst_) {
        delta = static_cast<double>(step - nfirst_)
                / static_cast<double>(nlast_ - nfirst_);
        delta = std::clamp(delta, 0.0, 1.0);
    }
    return T_start_ + delta * (T_stop_ - T_start_);
}

void FixCSVR::launch(const cpu::LaunchContext& ctx) {
    if (!ctx.at(ATV::md::Phase::end_of_step))
        return;

    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    const int n_dof = n_dof_global_;
    if (n_dof <= 0)
        return;

    T_live_ = compute_temp_target(ctx.step);

    // Hoisted once for both the KE-accumulation and rescale loops below --
    // never span() inside #pragma omp parallel.
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

    // No kinetic energy yet (e.g. before velocities are initialized): the
    // rescale is undefined (K in the denominator). Skip — no fudge.
    if (ke <= 0.0)
        return;

    const double ke_bar =
        0.5 * static_cast<double>(n_dof) * units::kB_ev_per_k * T_live_;

    // Two GLOBAL random draws, identical on every rank (seeded from step only).
    std::mt19937_64 rng(step_seed(seed_, ctx.step));
    std::normal_distribution<double> gauss(0.0, 1.0);
    const double r1 = gauss(rng);
    // Sum of squares of the remaining N_f-1 normals = chi^2(N_f-1) =
    // Gamma((N_f-1)/2, scale 2). Drawn directly (Bussi's `sumnoises`).
    double sum_sq = 0.0;
    if (n_dof - 1 > 0) {
        std::gamma_distribution<double> chi2(
            0.5 * static_cast<double>(n_dof - 1), 2.0);
        sum_sq = chi2(rng);
    }

    // Bussi Eq. A7 propagator for alpha^2.
    const double c1 = std::exp(-dt_ / tau_);
    const double factor =
        (ke_bar / (static_cast<double>(n_dof) * ke)) * (1.0 - c1);
    double alpha2 =
        c1 + factor * (r1 * r1 + sum_sq)
        + 2.0 * std::exp(-dt_ / (2.0 * tau_)) * std::sqrt(factor) * r1;
    if (alpha2 < 0.0)
        alpha2 = 0.0; // numerical guard; analytically alpha^2 >= 0
    const double alpha = std::sqrt(alpha2);

    // Rescale owned velocities by the single global factor.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        sp_vx[i] *= alpha;
        sp_vy[i] *= alpha;
        sp_vz[i] *= alpha;
    }

    // Conserved-quantity bookkeeping (Bussi Eq. 15): the thermostat did work
    // dK = (alpha^2 - 1) K on the system; accumulate its negation so that
    // E_pot + E_kin + H~_therm is drift-free.
    const double ke_after = alpha2 * ke;
    thermostat_energy_ += -(ke_after - ke);
    f.meta.get(kConserved) =
        f.meta.get(kEnergy) + ke_after + thermostat_energy_;
}

} // namespace md
} // namespace cpu
} // namespace ATV

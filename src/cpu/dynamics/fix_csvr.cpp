/// @file cpu/dynamics/fix_csvr.cpp
/// @brief CPU CSVR (Bussi) thermostat implementation (LAMMPS
///        `fix temp/csvr`, port reference e8c4c278a8; Sandia notice in the
///        header).

#include "atomiverse/cpu/dynamics/fix_csvr.h"

#include "atomiverse/core/csvr.h"
#include "atomiverse/core/dynamics_math.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <utility>

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace md {

namespace {

void require_positive(double value, const char* what) {
    if (!std::isfinite(value) || !(value > 0.0))
        throw std::invalid_argument(std::string("FixCSVR: ") + what
                                    + " must be finite and > 0");
}

#ifdef ATV_HAS_MPI
MPI_Comm md_comm(const cpu::Context& rt) {
    if (rt.mpi_size > 1 && rt.md.dd && rt.md.dd->cart_comm != MPI_COMM_NULL)
        return rt.md.dd->cart_comm;
    return MPI_COMM_NULL;
}
#endif

} // namespace

FixCSVR::FixCSVR(double T_start, double tau, double dt, std::uint32_t seed)
    : FixCSVR(T_start, tau, dt, seed, {}, ATV::md::TemperatureBias::None) {}

FixCSVR::FixCSVR(double T_start, double tau, double dt, std::uint32_t seed,
                 std::vector<int> types, ATV::md::TemperatureBias bias)
    : T_start_(T_start), T_stop_(T_start), tau_(tau), dt_(dt), seed_(seed),
      types_(std::move(types)), bias_(bias) {
    require_positive(T_start, "T_start");
    require_positive(tau, "tau");
    require_positive(dt, "dt");
    std::sort(types_.begin(), types_.end());
    types_.erase(std::unique(types_.begin(), types_.end()), types_.end());
    for (int t : types_) {
        if (t < 0)
            throw std::invalid_argument("FixCSVR: negative atom type "
                                        + std::to_string(t) + " in the group");
        if (member_.size() <= static_cast<std::size_t>(t))
            member_.resize(static_cast<std::size_t>(t) + 1, 0);
        member_[static_cast<std::size_t>(t)] = 1;
    }
    T_live_ = T_start_;
}

void FixCSVR::build(const cpu::BuildContext& ctx) {
    const cpu::Context& rt = ctx.runtime;
    if (types_.empty()) {
        // The driver publishes the constraint-aware global DOF before build;
        // LAMMPS `compute temp` and `temp/com` on group all both give
        // 3N − 3 − constraints.
        n_dof_ = rt.md.n_dof_global;
        return;
    }
    const cpu::Frame& frame = ctx.frame;
    if (!frame.has("atoms", "type"))
        throw std::runtime_error(
            "FixCSVR: a type group needs the per-atom force-field type "
            "column (kAtom_type); provide molpy atoms.type_id");
    if (!rt.md.constraints_by_type_known)
        throw std::runtime_error(
            "FixCSVR: the driver could not attribute constraints to atom "
            "types, so the group's degrees of freedom are unknown");
    const int n = owned_count(ctx);
    const auto type = frame.span(kAtom_type);
    long long n_group = 0;
    for (int i = 0; i < n; ++i)
        n_group += in_group_(static_cast<int>(type[i]));
#ifdef ATV_HAS_MPI
    if (MPI_Comm comm = md_comm(rt); comm != MPI_COMM_NULL) {
        long long local = n_group;
        MPI_Allreduce(&local, &n_group, 1, MPI_LONG_LONG, MPI_SUM, comm);
    }
#endif
    if (n_group == 0)
        throw std::runtime_error(
            "FixCSVR: the type group selects no atoms of the frame");
    BigInt n_con = 0;
    for (int t : types_)
        if (t < rt.md.n_constraint_types)
            n_con += rt.md.n_constraints_by_type[t];
    n_dof_ = ATV::md::csvr_group_dof(n_group, n_con);
}

void FixCSVR::launch(const cpu::LaunchContext& ctx) {
    if (!ctx.at(ATV::md::Phase::end_of_step))
        return;

    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    if (types_.empty())
        n_dof_ = ctx.runtime.md.n_dof_global;
    // LAMMPS: "nothing to do, if there are no degrees of freedom".
    if (n_dof_ < 1)
        return;

    T_live_ =
        ATV::md::ramp_target(T_start_, T_stop_, ctx.step, nfirst_, nlast_);

    const StorageReal* __restrict sp_mass = f.span(kAtom_mass);
    StorageReal* __restrict sp_vx = f.span(kAtom_vx);
    StorageReal* __restrict sp_vy = f.span(kAtom_vy);
    StorageReal* __restrict sp_vz = f.span(kAtom_vz);
    const int* sp_type = types_.empty() ? nullptr : f.span(kAtom_type).data();

    // Group mass, momentum and Σ m v², plus Σ m v² of every atom (the
    // conserved quantity), over owned atoms, then MPI-summed.
    double gm = 0.0, gpx = 0.0, gpy = 0.0, gpz = 0.0, gmvv = 0.0, amvv = 0.0;
#pragma omp parallel for schedule(static)                                      \
    reduction(+ : gm, gpx, gpy, gpz, gmvv, amvv)
    for (int i = 0; i < n; ++i) {
        const double m = sp_mass[i];
        const double vx = sp_vx[i], vy = sp_vy[i], vz = sp_vz[i];
        const double mvv = m * (vx * vx + vy * vy + vz * vz);
        amvv += mvv;
        if (sp_type == nullptr || in_group_(sp_type[i])) {
            gm += m;
            gpx += m * vx;
            gpy += m * vy;
            gpz += m * vz;
            gmvv += mvv;
        }
    }
    std::array<double, 6> sums{gm, gpx, gpy, gpz, gmvv, amvv};
#ifdef ATV_HAS_MPI
    if (MPI_Comm comm = md_comm(ctx.runtime); comm != MPI_COMM_NULL) {
        std::array<double, 6> local = sums;
        MPI_Allreduce(local.data(), sums.data(), 6, MPI_DOUBLE, MPI_SUM, comm);
    }
#endif
    ATV::md::CsvrSums s;
    s.mass = sums[0];
    s.p[0] = sums[1];
    s.p[1] = sums[2];
    s.p[2] = sums[3];
    s.mvv = sums[4];
    s.mvv_all = sums[5];

    const double mvv2e = ctx.runtime.units.kinetic_energy.scale();
    const double ke = 0.5 * mvv2e * ATV::md::csvr_group_mvv(s, bias_);
    // No kinetic energy yet (velocities not initialized): the rescale is
    // undefined (K in the denominator). Skip — no fudge.
    if (!(ke > 0.0))
        return;

    const double ke_bar = 0.5 * static_cast<double>(n_dof_)
                          * ctx.runtime.units.thermal_energy(T_live_);
    const ATV::md::CsvrNoise noise =
        ATV::md::csvr_noise(seed_, ctx.step, n_dof_);
    const double alpha2 = ATV::md::csvr_alpha2(
        ke, ke_bar, static_cast<double>(n_dof_), dt_, tau_, noise);
    const double alpha = std::sqrt(alpha2);

    if (bias_ == ATV::md::TemperatureBias::Com) {
        // v ← v_cm + α (v − v_cm): remove_bias, scale, restore_bias.
        const double vcx = s.p[0] / s.mass, vcy = s.p[1] / s.mass,
                     vcz = s.p[2] / s.mass;
        const double shift = 1.0 - alpha;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            if (sp_type != nullptr && !in_group_(sp_type[i]))
                continue;
            sp_vx[i] = alpha * sp_vx[i] + shift * vcx;
            sp_vy[i] = alpha * sp_vy[i] + shift * vcy;
            sp_vz[i] = alpha * sp_vz[i] + shift * vcz;
        }
    } else {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            if (sp_type != nullptr && !in_group_(sp_type[i]))
                continue;
            sp_vx[i] *= alpha;
            sp_vy[i] *= alpha;
            sp_vz[i] *= alpha;
        }
    }

    // Bussi Eq. 15 / LAMMPS `energy += ekin_old (1 − λ²)`.
    const double dk = (alpha2 - 1.0) * ke;
    thermostat_energy_ -= dk;
    if (!f.meta.contains(kThermostatEnergy.key))
        f.meta.insert(kThermostatEnergy, 0.0);
    f.meta.get(kThermostatEnergy) -= dk;
    const double ke_all_after = 0.5 * mvv2e * s.mvv_all + dk;
    f.meta.get(kConserved) =
        f.meta.get(kEnergy) + ke_all_after + f.meta.get(kThermostatEnergy);
}

} // namespace md
} // namespace cpu
} // namespace ATV

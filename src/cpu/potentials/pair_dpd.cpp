/// @file cpu/potentials/pair_dpd.cpp
#include "atomiverse/cpu/potentials/pair_dpd.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

namespace ATV {
namespace cpu {

PairDpd::PairDpd(double a, double gamma, double temperature, double cutoff,
                 double dt, std::uint64_t seed)
    : Pair(cutoff), a_default_(a), multi_(false), gamma_(gamma),
      temperature_(temperature), dt_(dt), seed_(seed) {
    if (!(cutoff > 0.0) || !(dt > 0.0) || !(temperature > 0.0)
        || !(gamma >= 0.0))
        throw std::invalid_argument("PairDpd: invalid parameters");
    cutoff_sq_ = cutoff * cutoff;
    sigma_ = std::sqrt(2.0 * gamma_ * units::kB_ev_per_k * temperature_);
}

PairDpd::PairDpd(std::vector<std::vector<double>> a_matrix, double gamma,
                 double temperature, double cutoff, double dt,
                 std::uint64_t seed)
    : Pair(cutoff), a_mat_(std::move(a_matrix)), multi_(true), gamma_(gamma),
      temperature_(temperature), dt_(dt), seed_(seed) {
    if (!(cutoff > 0.0) || !(dt > 0.0) || !(temperature > 0.0)
        || !(gamma >= 0.0))
        throw std::invalid_argument("PairDpd: invalid parameters");
    if (a_mat_.empty())
        throw std::invalid_argument("PairDpd: empty a_matrix");
    const std::size_t n = a_mat_.size();
    for (const auto& row : a_mat_)
        if (row.size() != n)
            throw std::invalid_argument("PairDpd: a_matrix must be square");
    cutoff_sq_ = cutoff * cutoff;
    sigma_ = std::sqrt(2.0 * gamma_ * units::kB_ev_per_k * temperature_);
}

double PairDpd::a_of_(int ti, int tj) const {
    if (!multi_)
        return a_default_;
    if (ti < 0 || tj < 0 || static_cast<std::size_t>(ti) >= a_mat_.size()
        || static_cast<std::size_t>(tj) >= a_mat_.size())
        throw std::runtime_error("PairDpd: atom type out of a_matrix range");
    return a_mat_[static_cast<std::size_t>(ti)][static_cast<std::size_t>(tj)];
}

void PairDpd::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = static_cast<int>(f.nlocal());
    const double rc = pair_cutoff();
    const double inv_sqrt_dt = 1.0 / std::sqrt(dt_);
    const auto type = f.span(kAtom_type);

    double energy = 0.0;
    // Pair virial, Voigt xx yy zz xy xz yz [eV] — Σ Δr ⊗ F^C, LAMMPS sign.
    //
    // CONSERVATIVE FORCE ONLY, deliberately. F^D and F^R are thermostat
    // forces, not derivatives of the energy this style reports:
    //   * the DPD thermostat is built so the invariant measure is the Gibbs
    //     measure of the conservative potential alone (Español & Warren 1995),
    //     so the thermodynamic pressure a barostat must integrate is
    //     ρkT + (1/3V)⟨Σ r·F^C⟩ — both other terms average to zero at
    //     equilibrium (⟨r̂·v_ij⟩ = 0 by symmetry, ⟨θ⟩ = 0 by construction);
    //   * F^R ∝ 1/√dt, so its instantaneous r·F diverges as the timestep
    //     shrinks. Tallying it would feed unbounded, timestep-dependent noise
    //     into every pressure the engine reports.
    // LAMMPS `pair_style dpd` tallies the full fpair instead; that is the
    // instantaneous mechanical (Irving–Kirkwood) stress, which agrees with
    // this one on average and not term by term. The choice here is the one
    // that satisfies tr(σ) = −3V dE/dV for the energy this style writes,
    // pinned by tests/cpu/potentials/test_pair_virial.cpp
    // (`PairDpd.VirialExcludesDissipativeAndRandomForces`).
    md::Voigt6 vir{};
    const StorageReal* __restrict x = f.span(kAtom_x);
    const StorageReal* __restrict y = f.span(kAtom_y);
    const StorageReal* __restrict z = f.span(kAtom_z);
    const StorageReal* __restrict vx = f.span(kAtom_vx);
    const StorageReal* __restrict vy = f.span(kAtom_vy);
    const StorageReal* __restrict vz = f.span(kAtom_vz);
    AccReal* __restrict fx = f.span(kAtom_fx);
    AccReal* __restrict fy = f.span(kAtom_fy);
    AccReal* __restrict fz = f.span(kAtom_fz);

    // The stream index of THIS launch. `LaunchContext::step` is the only step
    // a Component sees once a Driver owns the loop, and nothing in the engine
    // calls `set_step` — so keying the stream on a stored member alone froze
    // every random force of a driven run at one draw. F^R is what injects the
    // energy F^D removes, so a frozen stream is a constant kick and the run
    // heats or cools monotonically instead of sampling `temperature_`.
    // `stream_origin_` only SHIFTS this, so a chunked run can continue the
    // stream its predecessor ended on (`set_step`).
    const std::int64_t stream_step =
        static_cast<std::int64_t>(stream_origin_) + ctx.step;

    auto theta = [&](int i, int j) {
        std::uint64_t s = seed_;
        s ^= static_cast<std::uint64_t>(stream_step) * 0x9e3779b97f4a7c15ULL;
        s ^= static_cast<std::uint64_t>(i) * 0xbf58476d1ce4e5b9ULL;
        s ^= static_cast<std::uint64_t>(j) * 0x94d049bb133111ebULL;
        s = (s ^ (s >> 30)) * 0xbf58476d1ce4e5b9ULL;
        s = (s ^ (s >> 27)) * 0x94d049bb133111ebULL;
        s = s ^ (s >> 31);
        const double u1 =
            (static_cast<double>((s & 0xffffffffULL) + 1) / 4294967297.0);
        const double u2 = (static_cast<double>(((s >> 32) & 0xffffffffULL) + 1)
                           / 4294967297.0);
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * units::pi * u2);
    };

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double dz = z[j] - z[i];
            f.box.min_image(dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= cutoff_sq_ || r2 == 0.0)
                continue;

            const double r = std::sqrt(r2);
            const double inv_r = 1.0 / r;
            const double w = 1.0 - r / rc;
            if (w <= 0.0)
                continue;

            const double aij = a_of_(type[i], type[j]);
            const double fc = aij * w;
            double fpair = -fc * inv_r;

            // F^C on i is −fc·inv_r·(r_j − r_i) = fc·inv_r·Δ with Δ = r_i − r_j,
            // so the pair term is Δ_a F_b = (a w / r)·d_a·d_b. Taken here,
            // before the dissipative and random terms join `fpair`.
            const double cvir = fc * inv_r;
            vir.v[0] += cvir * dx * dx;
            vir.v[1] += cvir * dy * dy;
            vir.v[2] += cvir * dz * dz;
            vir.v[3] += cvir * dx * dy;
            vir.v[4] += cvir * dx * dz;
            vir.v[5] += cvir * dy * dz;

            const double dvx = vx[j] - vx[i];
            const double dvy = vy[j] - vy[i];
            const double dvz = vz[j] - vz[i];
            const double rdotv = (dx * dvx + dy * dvy + dz * dvz) * inv_r;
            const double w2 = w * w;
            const double fd = -gamma_ * w2 * rdotv;
            fpair += fd * inv_r;

            const double fr = sigma_ * w * theta(i, j) * inv_sqrt_dt;
            fpair += fr * inv_r;

            fx[i] += fpair * dx;
            fy[i] += fpair * dy;
            fz[i] += fpair * dz;
            fx[j] -= fpair * dx;
            fy[j] -= fpair * dy;
            fz[j] -= fpair * dz;

            // U^C = ½ a r_c w², the potential whose gradient is F^C = a w r̂
            // (Groot–Warren 1997; LAMMPS `0.5*a0*cut*wd*wd`). The `r_c` factor
            // was missing: the energy was reported in eV/Å while `a` is a
            // force amplitude, and it disagreed with the force this same loop
            // applies by exactly that factor — visible the moment the virial
            // is checked against −3V dE/dV. Both in-tree DPD regressions use
            // r_c = 1 Å, which is why nothing caught it.
            energy += 0.5 * aij * rc * w2;
        }
    }
    f.meta.get(kEnergy) += energy;
    vir.flush_to(f.meta.get(kStress));
}

} // namespace cpu
} // namespace ATV

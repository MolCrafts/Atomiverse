/// @file cpu/potentials/pair_dpd.cpp
#include "atomiverse/cpu/potentials/pair_dpd.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/pair_loop.h"

namespace ATV {
namespace cpu {

PairDpd::PairDpd(double a, double gamma, double temperature, double cutoff,
                 double dt, std::uint64_t seed)
    : Pair(cutoff), a_default_(a), multi_(false), gamma_(gamma),
      temperature_(temperature), dt_(dt), seed_(seed) {
    if (!std::isfinite(a) || !std::isfinite(cutoff) || !std::isfinite(dt)
        || !std::isfinite(temperature) || !std::isfinite(gamma)
        || !(cutoff > 0.0) || !(dt > 0.0) || !(temperature > 0.0)
        || !(gamma >= 0.0))
        throw std::invalid_argument("PairDpd: invalid parameters");
    cutoff_sq_ = cutoff * cutoff;
    sigma_ = std::sqrt(2.0 * gamma_ * units::md_boltz * temperature_);
}

PairDpd::PairDpd(std::vector<std::vector<double>> a_matrix, double gamma,
                 double temperature, double cutoff, double dt,
                 std::uint64_t seed)
    : Pair(cutoff), a_mat_(std::move(a_matrix)), multi_(true), gamma_(gamma),
      temperature_(temperature), dt_(dt), seed_(seed) {
    if (!std::isfinite(cutoff) || !std::isfinite(dt)
        || !std::isfinite(temperature) || !std::isfinite(gamma)
        || !(cutoff > 0.0) || !(dt > 0.0) || !(temperature > 0.0)
        || !(gamma >= 0.0))
        throw std::invalid_argument("PairDpd: invalid parameters");
    if (a_mat_.empty())
        throw std::invalid_argument("PairDpd: empty a_matrix");
    const std::size_t n = a_mat_.size();
    for (const auto& row : a_mat_)
        if (row.size() != n)
            throw std::invalid_argument("PairDpd: a_matrix must be square");
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (!std::isfinite(a_mat_[i][j]) || a_mat_[i][j] != a_mat_[j][i])
                throw std::invalid_argument(
                    "PairDpd: a_matrix must be finite and symmetric");
    cutoff_sq_ = cutoff * cutoff;
    sigma_ = std::sqrt(2.0 * gamma_ * units::md_boltz * temperature_);
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
    const double rc = pair_cutoff();
    const double inv_sqrt_dt = 1.0 / std::sqrt(dt_);
    // The single-amplitude form needs no force-field type; only the
    // per-type `a_matrix` form reads (and so requires) the column.
    const int* type = multi_ ? f.span(kAtom_type).data() : nullptr;
    const auto tags = f.span(kAtom_tag);

    // Pair virial, Voigt xx yy zz xy xz yz [eV] — Σ Δr ⊗ F^C, LAMMPS sign
    // (PairTermX::fvirial below).
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
    // The stream index of THIS launch. `LaunchContext::step` is the only step
    // a Component sees once a Driver owns the loop, and nothing in the engine
    // calls `set_step` — so keying the stream on a stored member alone froze
    // every random force of a driven run at one draw. F^R is what injects the
    // energy F^D removes, so a frozen stream is a constant kick and the run
    // heats or cools monotonically instead of sampling `temperature_`.
    // `stream_origin_` only SHIFTS this, so a chunked run can continue the
    // stream its predecessor ended on (`set_step`).
    // Setup uses the preceding draw: at zero this is the reserved -1
    // stream, and live re-preparation repeats the last completed step.
    const std::int64_t stream_step =
        static_cast<std::int64_t>(stream_origin_)
        + static_cast<std::int64_t>(ctx.step)
        - (ctx.runtime.exec.initializing_forces ? 1 : 0);

    // The per-type amplitude table throws on a type outside it; surface that
    // here, serially, rather than inside the threaded walk.
    if (multi_)
        for (int i = 0; i < f.n_total(); ++i)
            (void)a_of_(type[i], type[i]);

    // F^R draws from a pair-seeded stream (seed, step, the two tags), never
    // from thread order, so a threaded run is as reproducible as the rest
    // of the chunk-owned walk.
    struct Kernel {
        const PairDpd* self;
        const int* type;
        const TagInt* tag;
        const StorageReal *vx, *vy, *vz;
        double rc_sq, rc, inv_sqrt_dt;
        std::int64_t stream_step;

        double cutoff_sq() const { return rc_sq; }
        int row(int i) const { return i; }

        double theta(int i, int j) const {
            std::uint64_t s = self->seed_;
            s ^=
                static_cast<std::uint64_t>(stream_step) * 0x9e3779b97f4a7c15ULL;
            s ^= static_cast<std::uint64_t>(std::min(tag[i], tag[j]))
                 * 0xbf58476d1ce4e5b9ULL;
            s ^= static_cast<std::uint64_t>(std::max(tag[i], tag[j]))
                 * 0x94d049bb133111ebULL;
            s = (s ^ (s >> 30)) * 0xbf58476d1ce4e5b9ULL;
            s = (s ^ (s >> 27)) * 0x94d049bb133111ebULL;
            s = s ^ (s >> 31);
            const double u1 =
                (static_cast<double>((s & 0xffffffffULL) + 1) / 4294967297.0);
            const double u2 =
                (static_cast<double>(((s >> 32) & 0xffffffffULL) + 1)
                 / 4294967297.0);
            return std::sqrt(-2.0 * std::log(u1))
                   * std::cos(2.0 * units::pi * u2);
        }

        md::PairTermX operator()(int i, const md::PairSite& p,
                                 const md::PairChunk&) const {
            const int j = p.j;
            const double r = std::sqrt(p.r2);
            const double inv_r = 1.0 / r;
            const double w = 1.0 - r / rc;
            if (w <= 0.0)
                return {};
            const double aij =
                type ? self->a_mat_[static_cast<std::size_t>(type[i])]
                                   [static_cast<std::size_t>(type[j])]
                     : self->a_default_;
            // Separation and relative velocity both r_i − r_j / v_i − v_j.
            const double dvx = vx[i] - vx[j];
            const double dvy = vy[i] - vy[j];
            const double dvz = vz[i] - vz[j];
            const double rdotv = (p.dx * dvx + p.dy * dvy + p.dz * dvz) * inv_r;
            const double w2 = w * w;
            // F^C = a w r̂: fpair·Δ with fpair = a w / r; the virial takes
            // this conservative factor alone.
            const double fc = aij * w * inv_r;
            const double fd = self->gamma_ * w2 * rdotv;
            const double fr = self->sigma_ * w * theta(i, j) * inv_sqrt_dt;
            md::PairTermX t;
            t.fpair = fc - (fd + fr) * inv_r;
            t.fvirial = fc;
            // U^C = ½ a r_c w², the potential whose gradient is F^C = a w r̂
            // (Groot–Warren 1997; LAMMPS `0.5*a0*cut*wd*wd`). The `r_c`
            // factor was missing once: the energy was reported in eV/Å
            // while `a` is a force amplitude, and it disagreed with the
            // force by exactly that factor — visible the moment the virial
            // is checked against −3V dE/dV. Both in-tree DPD regressions
            // use r_c = 1 Å, which is why nothing caught it.
            t.energy = 0.5 * aij * rc * w2;
            return t;
        }
    };
    const Kernel kernel{this,
                        type,
                        tags.data(),
                        f.span(kAtom_vx).data(),
                        f.span(kAtom_vy).data(),
                        f.span(kAtom_vz).data(),
                        cutoff_sq_,
                        rc,
                        inv_sqrt_dt,
                        stream_step};
    const md::PairTotals sums = loop_.run(ctx, kernel, "PairDpd");
    f.meta.get(kEnergy) += sums.energy;
    f.meta.get(kEnergyPair) += sums.energy;
    if (ctx.stress_requested)
        sums.virial.flush_to(f.meta.get(kStress));
}

} // namespace cpu
} // namespace ATV

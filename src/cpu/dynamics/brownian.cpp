/// @file src/cpu/dynamics/brownian.cpp
#include "atomiverse/cpu/dynamics/brownian.h"
#include "atomiverse/cpu/dynamics/internal/gauss_stream.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/owned_range.h"

namespace ATV {
namespace cpu {
namespace md {

Brownian::Brownian(double dt, double D, double temperature, std::uint64_t seed)
    : dt_(dt), D_(D), temperature_(temperature), seed_(seed) {
    if (!std::isfinite(dt) || !(dt > 0.0))
        throw std::invalid_argument("Brownian: dt must be > 0");
    if (!std::isfinite(D) || !(D > 0.0))
        throw std::invalid_argument("Brownian: D must be > 0");
    if (!std::isfinite(temperature) || !(temperature > 0.0))
        throw std::invalid_argument("Brownian: temperature must be > 0");
}

void Brownian::launch(const cpu::LaunchContext& ctx) {
    if (!ctx.at(ATV::md::Phase::initial_integrate))
        return; // final_integrate: no-op (overdamped, no half-kick)

    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    auto x = f.span(kAtom_x);
    auto y = f.span(kAtom_y);
    auto z = f.span(kAtom_z);
    const auto fx = f.span(kAtom_fx);
    const auto fy = f.span(kAtom_fy);
    const auto fz = f.span(kAtom_fz);

    const auto tags = f.span(kAtom_tag);
    // Reject before moving any atom. Every spatial rank participates so a bad
    // force cannot leave the other ranks proceeding into the next halo exchange.
    int invalid = 0;
    for (int i = 0; i < n; ++i)
        invalid |= !std::isfinite(fx[i]) || !std::isfinite(fy[i])
                   || !std::isfinite(fz[i]);
#ifdef ATV_HAS_MPI
    if (multi_rank_dd(ctx.runtime))
        MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX,
                      ctx.runtime.md.dd->cart_comm);
#endif
    if (invalid)
        throw std::runtime_error("Brownian: non-finite force on an owned atom");

    // D [A^2/fs] / kT [eV] * F [eV/A] * dt [fs] is a displacement [A].
    const double mobility_dt = D_ * dt_ / (units::md_boltz * temperature_);
    const double noise = std::sqrt(2.0 * D_ * dt_);
    const std::uint64_t base =
        seed_ * 0x9E3779B97F4A7C15ULL
        + static_cast<std::uint64_t>(ctx.step) * 0xBF58476D1CE4E5B9ULL;
    for (int i = 0; i < n; ++i) {
        internal::GaussStream random{
            base + static_cast<std::uint64_t>(tags[i]) * 0xD6E8FEB86659FD93ULL};
        x[i] += mobility_dt * fx[i] + noise * random.next();
        y[i] += mobility_dt * fy[i] + noise * random.next();
        z[i] += mobility_dt * fz[i] + noise * random.next();
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV

/// @file src/cpu/dynamics/brownian.cpp
#include "atomiverse/cpu/dynamics/brownian.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/frame/pbc.h"

namespace ATV {
namespace cpu {
namespace md {

Brownian::Brownian(double dt, double D, double temperature, std::uint64_t seed)
    : dt_(dt), D_(D), temperature_(temperature), rng_(seed) {
    if (!(dt > 0.0))
        throw std::invalid_argument("Brownian: dt must be > 0");
    if (!(D > 0.0))
        throw std::invalid_argument("Brownian: D must be > 0");
    if (!(temperature > 0.0))
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

    // Mobility μ = D / kT. Force is eV/Å; displacement = μ F dt needs unit
    // conversion: energy·time / (mass length) style — use
    // Δx = (D / kT) * F * force_to_accel * mass_ref...
    // Simpler textbook form used in reduced units: treat F already in
    // consistent units with D/kT so Δx_force = (D/(k_B T)) * F * dt
    // with k_B T in eV and F in eV/Å ⇒ (D/kT)*F*dt has units Å²/fs / eV * eV/Å * fs = Å.
    const double kT = units::kB_ev_per_k * temperature_;
    // Mobility μ = D/(kT) so μ F dt has units Å when F is eV/Å.
    const double mob = D_ / kT;
    const double noise = std::sqrt(2.0 * D_ * dt_);

    for (int i = 0; i < n; ++i) {
        // Force drift only when finite; free-particle textbook path is pure noise.
        const double fxi = static_cast<double>(fx[i]);
        const double fyi = static_cast<double>(fy[i]);
        const double fzi = static_cast<double>(fz[i]);
        if (std::isfinite(fxi) && std::isfinite(fyi) && std::isfinite(fzi)) {
            x[i] += mob * fxi * dt_;
            y[i] += mob * fyi * dt_;
            z[i] += mob * fzi * dt_;
        }
        x[i] += noise * normal_(rng_);
        y[i] += noise * normal_(rng_);
        z[i] += noise * normal_(rng_);
    }
    // Skip wrap when the box is a free-particle sandbox (L ≥ 1e5 Å).
    if (f.box.length(0) < 1.0e5)
        wrap_orthorhombic(f, n);
}

} // namespace md
} // namespace cpu
} // namespace ATV

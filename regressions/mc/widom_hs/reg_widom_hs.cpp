/// @file regressions/mc/widom_hs/reg_widom_hs.cpp
/// @brief Widom HS free-volume golden: one host sphere, fixed snapshot.
/// Public API only. See REFERENCE.md.
///
/// μ_ex = −kT ln(1 − 4πσ³/(3 L³))  for L > 2σ (exclusion ball fits in cell).

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/widom.h"

using namespace ATV;

int main() {
    const double sigma = 2.0; // Å
    const double L = 12.0;    // Å (> 2σ)
    const double T = 298.15;  // K
    const double kT = units::kB_ev_per_k * T;
    const double V_ex = 4.0 / 3.0 * units::pi * sigma * sigma * sigma;
    const double V = L * L * L;
    const double P = 1.0 - V_ex / V;
    const double mu_ref = -kT * std::log(P);

    cpu::Frame host(1);
    host.span(kAtom_x)[0] = 0.0;
    host.span(kAtom_y)[0] = 0.0;
    host.span(kAtom_z)[0] = 0.0;
    host.span(kAtom_type)[0] = 0;
    host.span(kAtom_mass)[0] = 1.0;
    host.span(kAtom_tag)[0] = 1;
    host.box = cpu::Box::cubic(L);

    // Large trial count for a tight free-volume MC estimate.
    cpu::Widom widom(T, sigma, /*n_trials=*/500000, /*seed=*/20260723,
                     cpu::WidomPotential::hard_sphere);
    widom.sample(host);

    const double avg = widom.boltzmann_avg();
    const double mu = widom.mu_ex();

    std::printf("Widom HS: P_insert=%.8f (ref=%.8f)  mu_ex=%.8e eV "
                "(ref=%.8e eV)\n",
                avg, P, mu, mu_ref);

    bool ok = true;
    if (!(std::fabs(avg - P) < 0.005)) {
        std::printf("FAIL: free-volume fraction off\n");
        ok = false;
    }
    if (!(std::fabs(mu - mu_ref) < 0.002 * kT)) {
        std::printf("FAIL: mu_ex off analytic free-volume\n");
        ok = false;
    }
    if (!ok)
        return 1;
    std::printf("PASS\n");
    return 0;
}

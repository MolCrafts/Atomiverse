/// @file regressions/md/pe_bonded_nve/reg_pe_bonded_nve.cpp
/// @brief END-TO-END REGRESSION: bonded-only PE chain NVE energy conservation.
///
/// See REFERENCE.md for the literature source of every number and for the
/// physical justification of the tolerance.
///
/// Out-of-library consumer: public headers + link atomiverse only. One flat
/// main() — no single-use helpers. Exits non-zero on failure.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"

using namespace ATV;

int main() {
    // ── OPLS-AA UA / MSU PE bonded parameters (Tschopp 2010) ────────────
    const double kcal_to_ev = 1.0 / 23.060547830619026;
    const double bond_K = 268.0 * kcal_to_ev; // [eV/Å²]
    const double bond_r0 = 1.53;              // [Å]
    const double angle_K = 60.0 * kcal_to_ev; // [eV/rad²]
    const double angle_theta0 = 109.5 * units::pi / 180.0;
    const std::vector<double> dihedral_Cn = {
        1.73 * kcal_to_ev,
        -4.49 * kcal_to_ev,
        0.776 * kcal_to_ev,
        6.99 * kcal_to_ev,
        0.0,
    };

    const int n_carbons = 20;
    const double box_len = 50.0; // [Å]
    const double dt = 0.5;       // [fs]
    const int n_steps = 1000;
    const int n_chunks = 20;

    // All-trans alkane geometry in a large cubic box.
    cpu::Frame frame(n_carbons);
    const double half_apex = (units::pi - angle_theta0) / 2.0;
    const double dx_step = bond_r0 * std::cos(half_apex);
    const double dy_step = bond_r0 * std::sin(half_apex);
    const double x0 = (box_len - (n_carbons - 1) * dx_step) / 2.0;
    const double y0 = box_len / 2.0;
    const double z0 = box_len / 2.0;
    for (int i = 0; i < n_carbons; ++i) {
        frame.span(ATV::kAtom_x)[i] = x0 + i * dx_step;
        frame.span(ATV::kAtom_y)[i] = y0 + (i % 2 == 0 ? 0.0 : dy_step);
        frame.span(ATV::kAtom_z)[i] = z0;
        frame.span(ATV::kAtom_mass)[i] = 14.027; // CH₂ [amu]
        frame.span(ATV::kAtom_type)[i] = 0;
        // Tiny alternating transverse kick — non-trivial trajectory, still
        // inside the harmonic basin of every bonded term.
        const double sign = (i % 2 == 0) ? +1.0 : -1.0;
        frame.span(ATV::kAtom_vz)[i] = sign * 1e-3; // [Å/fs]
    }
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    for (int k = 0; k < n_carbons - 1; ++k)
        frame.add_bond(k, k + 1, 0);
    for (int k = 0; k < n_carbons - 2; ++k)
        frame.add_angle(k, k + 1, k + 2, 0);
    for (int k = 0; k < n_carbons - 3; ++k)
        frame.add_dihedral(k, k + 1, k + 2, k + 3, 0);
    frame.build_special_list();
    frame.box = cpu::Box::cubic(box_len);

    // Bonded-only NVE: no pair potential, no neighbor list, no thermostat.
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.on(ATV::md::Phase::force)
        .add(cpu::BondHarmonic{{bond_K}, {bond_r0}})
        .add(cpu::AngleHarmonic{{angle_K}, {angle_theta0}})
        .add(cpu::DihedralMultiHarmonic{{dihedral_Cn}});
    sim.enable_timing(false);
    sim.build();

    sim.run(frame, 1);
    const double e0 =
        frame.meta.get(ATV::kEnergy) + frame.meta.get(ATV::kKinetic);
    if (!(std::isfinite(e0) && std::fabs(e0) > 0.0)) {
        std::printf(
            "FAIL: initial total energy is not finite/nonzero (E0=%g)\n", e0);
        return 1;
    }

    double worst_drift = 0.0;
    double drift_first_half = 0.0, drift_second_half = 0.0;
    for (int c = 1; c <= n_chunks; ++c) {
        sim.run(frame, n_steps / n_chunks);
        const double e =
            frame.meta.get(ATV::kEnergy) + frame.meta.get(ATV::kKinetic);
        const double drift = std::fabs(e - e0) / std::fabs(e0);
        if (drift > worst_drift)
            worst_drift = drift;
        if (c <= n_chunks / 2)
            drift_first_half = std::fmax(drift_first_half, drift);
        else
            drift_second_half = std::fmax(drift_second_half, drift);
    }

    // See REFERENCE.md — symplectic envelope + non-secular growth check.
    const double bound = 1.0e-4;      // relative |dE|/|E0|
    const double secular_ratio = 4.0; // 2nd half may not exceed 1st by >4×

    std::printf("PE bonded NVE, C%d UA, dt = %.1f fs, %d steps\n", n_carbons,
                dt, n_steps);
    std::printf("  E_total(0)       = %.6e eV\n", e0);
    std::printf("  worst |dE|/|E0|  = %.3e   want <= %.1e   [bounded]\n",
                worst_drift, bound);
    std::printf("  drift 1st half   = %.3e\n", drift_first_half);
    std::printf(
        "  drift 2nd half   = %.3e   want <= %.1fx the 1st [non-secular]\n",
        drift_second_half, secular_ratio);

    bool ok = true;
    if (!(worst_drift <= bound)) {
        std::printf("FAIL: energy drift %.3e exceeds bound %.1e\n", worst_drift,
                    bound);
        ok = false;
    }
    if (drift_first_half > 0.0
        && drift_second_half > secular_ratio * drift_first_half) {
        std::printf("FAIL: drift grows with time (%.3e -> %.3e) — energy is "
                    "leaking, not oscillating\n",
                    drift_first_half, drift_second_half);
        ok = false;
    }
    if (!std::isfinite(worst_drift)) {
        std::printf("FAIL: energy is not finite\n");
        ok = false;
    }

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

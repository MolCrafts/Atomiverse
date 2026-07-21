/// @file regressions/md/lj_argon/reg_lj_argon.cpp
/// @brief END-TO-END REGRESSION: NVE energy conservation of Rahman's LJ argon.
///
/// See REFERENCE.md for the literature source of every number below and for the
/// physical justification of the tolerance.
///
/// This is an OUT-OF-LIBRARY consumer: public headers + link atomiverse, nothing
/// else. It is deliberately one flat main() — setup, run, and check read top to
/// bottom, with no single-use helper functions (CLAUDE.md: "Extract on the second
/// caller, not the first").
///
/// Exits non-zero on failure.

#include <cmath>
#include <cstdio>
#include <random>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/init.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

int main() {
    // ── Rahman (1964) liquid argon ───────────────────────────────────────
    // eps/kB = 120 K, sigma = 3.4 A, number density 0.02071 A^-3, T = 94.4 K.
    const double eps = 120.0 * units::kB_ev_per_k; // 0.010341 eV
    const double sigma = 3.4;                      // [A]
    const double cutoff = 2.5 * sigma;     // 8.5 A, the textbook LJ cutoff
    const double skin = 0.5;               // [A]
    const double number_density = 0.02071; // [A^-3]
    const double temperature = 94.4;       // [K]
    const double dt = 5.0;                 // [fs]  (Rahman used 10 fs)
    const int n_side = 6;                  // 6^3 = 216 atoms
    const int n_steps = 2000;              // 10 ps
    const int n_chunks = 20;               // sample the drift 20x along the way

    // A simple-cubic lattice at the target density. The box must exceed twice
    // the cutoff so the minimum-image convention is unambiguous.
    const int n = n_side * n_side * n_side;
    const double box_len = std::cbrt(double(n) / number_density);
    if (box_len <= 2.0 * cutoff) {
        std::printf("FAIL: box %.2f A is too small for cutoff %.2f A\n",
                    box_len, cutoff);
        return 1;
    }
    const double spacing = box_len / double(n_side);

    cpu::Frame frame(n);
    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                frame.span(ATV::kAtom_x)[idx] = (double(ix) + 0.5) * spacing;
                frame.span(ATV::kAtom_y)[idx] = (double(iy) + 0.5) * spacing;
                frame.span(ATV::kAtom_z)[idx] = (double(iz) + 0.5) * spacing;
                frame.span(ATV::kAtom_mass)[idx] = 39.948; // argon [amu]
                frame.span(ATV::kAtom_type)[idx] = 0;
                ++idx;
            }
    frame.simbox = ATV::cpu::Box::cubic(box_len);

    // NVE: no thermostat. InitVelocity draws a Maxwell distribution at T once,
    // at build; from then on the total energy is the engine's to conserve.
    cpu::md::MD sim;
    sim.add_potential(cpu::PairLJCut{eps, sigma, cutoff});
    sim.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.add(cpu::md::InitVelocity{temperature, /*seed=*/12345u});
    sim.enable_timing(false); // this is a correctness check, not a benchmark
    sim.build();

    // One step to populate the energies, then take E0 as the reference.
    sim.run(frame, 1);
    const double e0 =
        frame.meta.get(ATV::kEnergy) + frame.meta.get(ATV::kKinetic); // [eV]

    // Run in chunks so we see the WHOLE drift history, not just the endpoint —
    // a symplectic integrator's energy error is bounded and oscillatory, so an
    // endpoint-only check can pass by luck at a favourable phase.
    double worst_drift = 0.0;
    double drift_first_half = 0.0, drift_second_half = 0.0;
    for (int c = 1; c <= n_chunks; ++c) {
        sim.run(frame, n_steps / n_chunks);
        const double e =
            frame.meta.get(ATV::kEnergy) + frame.meta.get(ATV::kKinetic);
        const double drift =
            std::fabs(e - e0) / (double(n) * eps); // [eps/atom]
        if (drift > worst_drift)
            worst_drift = drift;
        if (c <= n_chunks / 2)
            drift_first_half = std::fmax(drift_first_half, drift);
        else
            drift_second_half = std::fmax(drift_second_half, drift);
    }

    // ── The checks (see REFERENCE.md) ────────────────────────────────────
    // (1) Bounded: velocity Verlet is symplectic, so |dE| stays inside an
    //     O(dt^2) envelope. 1e-2 eps/atom is that envelope at dt = 5 fs.
    // (2) Non-secular: the error must NOT grow linearly with time. A symplectic
    //     integrator oscillates; a broken one ramps. Compare the two halves.
    const double bound = 1.0e-2;      // [eps/atom]
    const double secular_ratio = 4.0; // 2nd half may not exceed 1st half by >4x

    std::printf(
        "LJ argon NVE, N = %d, rho = %.5f A^-3, T0 = %.1f K, dt = %.1f fs\n", n,
        number_density, temperature, dt);
    std::printf("  E_total(0)          = %.6f eV\n", e0);
    std::printf("  worst |dE|/(N*eps)  = %.3e   want <= %.1e   [bounded]\n",
                worst_drift, bound);
    std::printf("  drift 1st half      = %.3e\n", drift_first_half);
    std::printf(
        "  drift 2nd half      = %.3e   want <= %.1fx the 1st [non-secular]\n",
        drift_second_half, secular_ratio);

    bool ok = true;
    if (!(worst_drift <= bound)) {
        std::printf("FAIL: energy drift %.3e eps/atom exceeds bound %.1e\n",
                    worst_drift, bound);
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

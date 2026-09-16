/// @file regressions/semiempirical/dftb2-md-berendsen/reg_dftb2-md-berendsen.cpp
/// @brief SCC-DFTB2 forces driving `md::MD` under a Berendsen thermostat.
///
/// Domain: Berendsen weak coupling (Berendsen et al., J. Chem. Phys. 81, 3684
/// (1984)) on SCC-DFTB2 forces (Elstner et al., Phys. Rev. B 58, 7260 (1998)).
///
/// Two things are gated, and the second is why the case exists as a *pair* of
/// runs rather than a single one:
///   1. DFTB2 binds into `md::MD` as a force source at all — until now the only
///      semiempirical-in-MD case in the tree was AM1, and no DFTB one existed.
///   2. The thermostat is what pulls the temperature to the setpoint. Asserting
///      "T ends near 300 K" on its own proves nothing: a hot Maxwell–Boltzmann
///      start equipartitions down toward 300 K on its own within tens of fs.
///      So the same trajectory is run with the thermostat off and the
///      thermostat has to beat it.
///
/// Not a DFTB+ trajectory pin: Berendsen is not time-reversible and the two
/// codes seed velocities differently, so step-by-step agreement is not a thing
/// that exists. The DFTB+-pinned quantities are energy/forces/stress
/// (`dftb2-ewald-gamma`). Public API only. <= 60 s.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/berendsen.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

// 4 waters, cubic 5.6 Å — small enough for 120 SCC-DFTB2 force evaluations
// inside the regression budget, periodic and polar so the Ewald γ is live.
constexpr int kNat = 12;
constexpr int kZnum[] = {8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1};
constexpr double kPx[] = {0.00, 0.96, -0.24, 2.80, 3.76, 2.56,
                          0.00, 0.96, -0.24, 2.80, 3.76, 2.56};
constexpr double kPy[] = {0.00, 0.00, 0.93, 0.00, 0.00, 0.93,
                          2.80, 2.80, 3.73, 2.80, 2.80, 3.73};
constexpr double kPz[] = {0.00, 0.00, 0.00, 1.40, 1.40, 1.40,
                          1.40, 1.40, 1.40, 0.00, 0.00, 0.00};

constexpr double kDt = 0.2;        // fs — the DFTB+ / Hooven protocol step
constexpr double kTau = 20.0;      // fs coupling time
constexpr double kTTarget = 300.0; // K
constexpr double kTStart = 900.0;  // K — start hot so the bath has work to do
constexpr int kSteps = 120;

cpu::Frame make_cell() {
    cpu::Frame f(kNat);
    for (int i = 0; i < kNat; ++i) {
        f.span(kAtom_type)[i] = kZnum[i];
        f.span(kAtom_x)[i] = kPx[i];
        f.span(kAtom_y)[i] = kPy[i];
        f.span(kAtom_z)[i] = kPz[i];
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_mass)[i] = (kZnum[i] == 8) ? 15.999 : 1.008;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
    }
    f.box = cpu::Box::cubic(5.6);
    f.meta.get(kEnergy) = 0;
    return f;
}

/// Instantaneous temperature [K] from the frame's velocities.
double temperature(const cpu::Frame& f, int n_dof) {
    double ke = 0;
    for (int i = 0; i < f.nlocal(); ++i) {
        const double vx = f.span(kAtom_vx)[i];
        const double vy = f.span(kAtom_vy)[i];
        const double vz = f.span(kAtom_vz)[i];
        ke += f.span(kAtom_mass)[i] * (vx * vx + vy * vy + vz * vz);
    }
    ke *= 0.5 * units::mass_v2_to_energy;
    return 2.0 * ke / (static_cast<double>(n_dof) * units::kB_ev_per_k);
}

/// Run the trajectory; returns the final temperature [K].
double run(bool thermostatted, double& final_energy) {
    cpu::Frame f = make_cell();

    cpu::semiempirical::DFTB2 dftb;
    dftb.boundary = cpu::semiempirical::DftbBoundary::lattice;
    dftb.scc_tol = 1e-7;
    dftb.max_scc = 200;
    dftb.with_forces = true;
    dftb.predictor(3); // the MD charge guess, exercised end-to-end

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    sim.set_observer(cpu::md::InitVelocity{kTStart, /*seed=*/7u});
    if (thermostatted)
        sim.set_thermostat(cpu::md::Berendsen{kTTarget, kTau, kDt});
    sim.on(md::Phase::force).add_ref(dftb);
    sim.build();
    sim.run(f, kSteps);

    final_energy = dftb.last_energy();
    return temperature(f, 3 * kNat - 3);
}

} // namespace

int main() {
    double e_free = 0, e_bath = 0;
    const double T_free = run(false, e_free);
    const double T_bath = run(true, e_bath);

    std::printf("dftb2-md-berendsen: 4 H2O, %d steps, dt %.1f fs,"
                " T0 %.0f K -> setpoint %.0f K (tau %.0f fs)\n",
                kSteps, kDt, kTStart, kTTarget, kTau);
    std::printf("  no thermostat : T_final = %8.2f K   E = %.6f eV\n", T_free,
                e_free);
    std::printf("  Berendsen     : T_final = %8.2f K   E = %.6f eV\n", T_bath,
                e_bath);

    if (!std::isfinite(T_free) || !std::isfinite(T_bath)
        || !std::isfinite(e_bath)) {
        std::printf("FAIL trajectory went non-finite\n");
        return 1;
    }

    // The thermostat has to be the thing that got us to the setpoint.
    const double err_free = std::fabs(T_free - kTTarget);
    const double err_bath = std::fabs(T_bath - kTTarget);
    if (!(err_bath < err_free)) {
        std::printf("FAIL thermostat did not beat the free trajectory"
                    " (|dT| %.2f vs %.2f K)\n",
                    err_bath, err_free);
        return 1;
    }

    // …and has to land near it. 120 steps at dt 0.2 fs is 24 fs = 1.2 tau, so
    // an ideal exponential still leaves ~30% of the initial 600 K excess;
    // the window is set from that, not from wishful thinking.
    constexpr double kWindowK = 120.0;
    if (!(err_bath < kWindowK)) {
        std::printf("FAIL Berendsen did not reach the setpoint window"
                    " (|dT| %.2f K > %.0f K)\n",
                    err_bath, kWindowK);
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}

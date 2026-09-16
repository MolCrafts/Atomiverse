/// @file regressions/md/extended-lagrangian-dipoles/reg_extended-lagrangian-dipoles.cpp
/// @brief END-TO-END: extended-Lagrangian (iEL/SCF) PIM dipole dynamics vs
///        the SCF Born–Oppenheimer twin and the literature-form two-site pin.
///
/// See REFERENCE.md for the pins and citations.
/// Public headers + link atomiverse only. Flat main(); exits non-zero on fail.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pim_induced_dipole.h"
#include "atomiverse/cpu/potentials/pim_xl_dipole.h"

using namespace ATV;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++g_fail;
    }
}

/// Two-site pin geometry of pim-induced-dipole-bulk/REFERENCE.md.
cpu::Frame two_site() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_x)[1] = 3.0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_y)[i] = f.span(kAtom_z)[i] = 0;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
    }
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_charge)[1] = -1.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 2;
    f.box = cpu::Box::cubic(50.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

/// Bulk polarizable toy: 8 ions on a perturbed 2x2x2 rocksalt-parity
/// sub-lattice (spacing 4 Å) centered in a cubic L=20 Å cell, ±1 e,
/// α = 1.0 / 1.5 Å³. All pair distances stay well inside the 10 Å cutoff
/// for the whole run, so the truncated PIM force field is strictly smooth
/// (no cutoff crossings) and KE + E_pol is a conserved energy.
cpu::Frame bulk_box() {
    cpu::Frame f(8);
    int n = 0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k, ++n) {
                const bool plus = ((i + j + k) % 2 == 0);
                f.span(kAtom_x)[n] = 8.0 + 4.0 * i + 0.15 * std::sin(1.0 + 3.0 * n);
                f.span(kAtom_y)[n] = 8.0 + 4.0 * j + 0.15 * std::sin(2.0 + 3.0 * n);
                f.span(kAtom_z)[n] = 8.0 + 4.0 * k + 0.15 * std::sin(3.0 + 3.0 * n);
                f.span(kAtom_charge)[n] = plus ? 1.0 : -1.0;
                f.span(kAtom_type)[n] = plus ? 1 : 2;
                f.span(kAtom_mass)[n] = plus ? 22.99 : 35.45;
                f.span(kAtom_tag)[n] = n + 1;
                f.span(kAtom_vx)[n] = f.span(kAtom_vy)[n] =
                    f.span(kAtom_vz)[n] = 0;
                f.span(kAtom_fx)[n] = f.span(kAtom_fy)[n] =
                    f.span(kAtom_fz)[n] = 0;
            }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

double kinetic(const cpu::Frame& f) {
    double ke = 0;
    for (int i = 0; i < f.nlocal(); ++i) {
        const double vx = f.span(kAtom_vx)[i];
        const double vy = f.span(kAtom_vy)[i];
        const double vz = f.span(kAtom_vz)[i];
        ke += 0.5 * f.span(kAtom_mass)[i] * (vx * vx + vy * vy + vz * vz);
    }
    return ke * units::mass_v2_to_energy; // amu·Å²/fs² → eV
}

/// Velocity-Verlet trajectory driven by one PIM component (SCF or XL).
/// Records μ(t) and the SCC iteration total; returns E_tot at t=0 and t=end.
template <typename Pot>
void run_traj(Pot& pot, int nsteps, double dt,
              std::vector<std::vector<double>>& mu_trace, long& iters,
              double& e_tot0, double& e_tot1) {
    cpu::Frame f = bulk_box();

    // Seed F(t=0) before the first VV half-kick (LAMMPS run-0 / fix setup).
    {
        cpu::Evaluate seed;
        seed.set_evaluator_ref(pot);
        seed.build();
        seed.run(f);
    }
    mu_trace.push_back(pot.last_mu());
    iters += pot.last_pol_iters();
    e_tot0 = f.meta.get(kEnergy) + kinetic(f);

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_evaluator_ref(pot);
    sim.enable_timing(false);
    sim.build();
    for (int s = 0; s < nsteps; ++s) {
        sim.run(f, 1);
        mu_trace.push_back(pot.last_mu());
        iters += pot.last_pol_iters();
    }
    e_tot1 = f.meta.get(kEnergy) + kinetic(f);
    // ~MD flushes continuable destroy
}

} // namespace

int main() {
    // ── Part 1: stationary two-site pin vs real MetalWalls (thole_a = 0) ─
    // Same MW-expressible gate A as pim-induced-dipole-bulk (c8ad175d):
    // E_pol = ΔCoulomb(pol − α=0), F = ΔF. MW has no Thole damping.
    // XL fixed point at a stationary geometry must BE the SCF solution.
    const double E_pin = -2.4397739572766e-01; // [eV]  −8.9660039191731e-3 Ha
    const double F_pin = 3.4926913376021e-01;  // [eV/Å] +6.7922032485569e-3 Ha/a0
    const double tol_e = 1e-7 * std::fabs(E_pin);
    const double tol_f = 1e-7 * std::fabs(F_pin);

    cpu::Frame f2 = two_site();
    cpu::PimXlDipole xl(20.0, {0.0, 1.0, 1.5}, /*thole_a=*/0.0, /*xl_order=*/5,
                        /*pol_tol=*/1e-12, /*pol_max_iter=*/200);
    // Continuable MD preserves XL aux state for warm-start checks (dt=0).
    cpu::md::MD sim_stat;
    sim_stat.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim_stat.set_evaluator_ref(xl);
    sim_stat.build();

    sim_stat.run(f2, 1); // cold
    const int iters_cold = xl.last_pol_iters();
    std::printf("stationary cold : E=%.16e iters=%d\n", xl.last_pol_energy(),
                iters_cold);
    check(std::fabs(xl.last_pol_energy() - E_pin) <= tol_e,
          "cold E_pol vs pin");
    check(std::fabs(f2.span(kAtom_fx)[0] - F_pin) <= tol_f,
          "cold F_Ax vs pin");
    check(std::fabs(f2.span(kAtom_fx)[1] + F_pin) <= tol_f,
          "cold F_Bx vs pin");
    check(!xl.last_warm(), "first launch must be cold");

    for (int rep = 1; rep <= 2; ++rep) {
        sim_stat.run(f2, 1); // warm — aux seeded stationary
        std::printf("stationary warm%d: E=%.16e iters=%d aux_err=%.3e\n", rep,
                    xl.last_pol_energy(), xl.last_pol_iters(),
                    xl.last_aux_error());
        check(xl.last_warm(), "warm launch flagged");
        check(std::fabs(xl.last_pol_energy() - E_pin) <= tol_e,
              "warm E_pol vs pin");
        check(std::fabs(f2.span(kAtom_fx)[0] - F_pin) <= tol_f,
              "warm F_Ax vs pin");
        check(xl.last_pol_iters() < iters_cold, "warm iters < cold iters");
        check(xl.last_pol_iters() <= 2, "warm iters <= 2 at fixed point");
        check(xl.last_aux_error() < 1e-10, "stationary aux error ~ 0");
    }

    // ── Part 2: dynamics — XL tracks the Born–Oppenheimer twin ─────────
    const int nsteps = 150;
    const double dt = 0.5; // [fs]

    std::vector<std::vector<double>> mu_scf, mu_xl;
    long iters_scf = 0, iters_xl = 0;
    double e0_scf = 0, e1_scf = 0, e0_xl = 0, e1_xl = 0;

    cpu::PimInducedDipole bo(10.0, {0.0, 1.0, 1.5}, 0.39, 1e-12, 200);
    run_traj(bo, nsteps, dt, mu_scf, iters_scf, e0_scf, e1_scf);

    cpu::PimXlDipole el(10.0, {0.0, 1.0, 1.5}, 0.39, 5, 1e-12, 200);
    run_traj(el, nsteps, dt, mu_xl, iters_xl, e0_xl, e1_xl);

    double mu_dev = 0;
    check(mu_scf.size() == mu_xl.size(), "trace lengths equal");
    for (std::size_t s = 0; s < mu_scf.size() && s < mu_xl.size(); ++s) {
        check(mu_scf[s].size() == mu_xl[s].size(), "mu sizes equal");
        for (std::size_t k = 0; k < mu_scf[s].size(); ++k)
            mu_dev = std::max(mu_dev, std::fabs(mu_scf[s][k] - mu_xl[s][k]));
    }

    const double drift_scf = std::fabs(e1_scf - e0_scf);
    const double drift_xl = std::fabs(e1_xl - e0_xl);
    std::printf("dynamics: steps=%d dt=%.2f fs\n", nsteps, dt);
    std::printf("  SCF twin: iters=%ld E0=%.12e E1=%.12e drift=%.3e\n",
                iters_scf, e0_scf, e1_scf, drift_scf);
    std::printf("  XL  twin: iters=%ld E0=%.12e E1=%.12e drift=%.3e\n",
                iters_xl, e0_xl, e1_xl, drift_xl);
    std::printf("  max|mu_XL - mu_SCF| = %.3e e·Å  aux_err(last)=%.3e\n",
                mu_dev, el.last_aux_error());

    check(mu_dev < 1e-6, "XL dipoles track the BO surface (< 1e-6 e·Å)");
    check(iters_xl < iters_scf, "XL warm start cheaper than cold SCC");
    check(drift_xl < 5e-6, "XL total-energy drift bounded");
    check(drift_scf < 5e-6, "SCF twin total-energy drift bounded");
    check(e0_xl < 0.0, "polarization energy negative");
    check(std::fabs(e0_xl - e0_scf) < 1e-10, "identical initial state");

    if (g_fail) {
        std::printf("FAIL: extended-lagrangian-dipoles (%d checks)\n", g_fail);
        return 1;
    }
    std::printf("PASS: extended-lagrangian-dipoles (iEL/SCF vs BO twin + "
                "two-site pin)\n");
    return 0;
}

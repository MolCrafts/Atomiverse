/// @file regressions/md/piston-electrode-pz/reg_piston-electrode-pz.cpp
/// @brief E2E: rigid upper-wall piston holds mean \(P_z\) at target (E6).
///
/// Public API only. Pinned against the **real MetalWalls binary** (git
/// c8ad175d) run on the identical 24-atom twin: MW's own
/// `electrode_forces.out` LJ column gives \(F_z\) on the rigid piston, hence
/// \(P_z = F_z/A\). See REFERENCE.md. Exits non-zero on failure.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/fix_freeze.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/piston_electrode.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

int main() {
    // ── Parallel-plate LJ fluid + frozen lower / piston upper (REFERENCE) ─
    constexpr double Lxy = 10.0; // [Å]
    constexpr double Lz = 20.0;
    constexpr double z_lo = 2.0;  // lower wall
    constexpr double z_hi = 8.0;  // upper piston (tight gap → repulsive LJ)
    constexpr double eps = 0.10;  // [eV]
    constexpr double sigma = 3.0; // [Å]
    // r_c + skin MUST stay below L_xy/2 = 5 Å: the single-rank CellList folds
    // every separation through minimum image (cell_list.h
    // `min_image_unextended_`), so a cutoff above half the shortest box edge
    // silently drops the second image of every pair with |d| = L/2. The
    // original fixture used r_c = 6 Å in this 10 Å box and was therefore
    // computing a min-image-truncated LJ, not "LJ with r_c = 6 Å" (0.037 %
    // error vs the true lattice sum; MetalWalls rejects such an input
    // outright — src/system.F90 "Invalid value for lj rcut > min(a,b,c)/2").
    constexpr double cutoff = 4.5;
    constexpr double skin = 0.3;
    constexpr double dt = 0.5; // [fs]
    constexpr double T = 300.0;
    constexpr double tau_T = 50.0;
    constexpr int n_wall = 4; // 2×2 per wall
    constexpr int n_fluid = 16;
    constexpr int n_warm = 400;
    constexpr int n_prod = 600;
    constexpr int lower_type = 2;
    constexpr int upper_type = 3;
    constexpr int fluid_type = 1;

    const int N = 2 * n_wall + n_fluid;
    cpu::Frame frame(N);
    int idx = 0;

    // Lower wall (frozen)
    for (int ix = 0; ix < 2; ++ix)
        for (int iy = 0; iy < 2; ++iy) {
            frame.span(kAtom_x)[idx] = (ix + 0.5) * (Lxy / 2.0);
            frame.span(kAtom_y)[idx] = (iy + 0.5) * (Lxy / 2.0);
            frame.span(kAtom_z)[idx] = z_lo;
            frame.span(kAtom_type)[idx] = lower_type;
            frame.span(kAtom_mass)[idx] = 196.9665;
            frame.span(kAtom_tag)[idx] = idx + 1;
            frame.span(kAtom_vx)[idx] = 0.0;
            frame.span(kAtom_vy)[idx] = 0.0;
            frame.span(kAtom_vz)[idx] = 0.0;
            ++idx;
        }
    // Upper piston wall
    for (int ix = 0; ix < 2; ++ix)
        for (int iy = 0; iy < 2; ++iy) {
            frame.span(kAtom_x)[idx] = (ix + 0.5) * (Lxy / 2.0);
            frame.span(kAtom_y)[idx] = (iy + 0.5) * (Lxy / 2.0);
            frame.span(kAtom_z)[idx] = z_hi;
            frame.span(kAtom_type)[idx] = upper_type;
            frame.span(kAtom_mass)[idx] = 12.011;
            frame.span(kAtom_tag)[idx] = idx + 1;
            frame.span(kAtom_vx)[idx] = 0.0;
            frame.span(kAtom_vy)[idx] = 0.0;
            frame.span(kAtom_vz)[idx] = 0.0;
            ++idx;
        }
    // Dense fluid slab between walls — two layers near each wall so LJ
    // repulsion on the upper piston is positive (compressive).
    for (int i = 0; i < n_fluid; ++i) {
        const int ix = i % 4;
        const int iy = (i / 4) % 2;
        const int iz = i / 8; // 0 or 1
        frame.span(kAtom_x)[idx] = (ix + 0.5) * (Lxy / 4.0);
        frame.span(kAtom_y)[idx] = (iy + 0.5) * (Lxy / 2.0);
        // Layer 0 just above lower wall; layer 1 just below upper wall.
        frame.span(kAtom_z)[idx] = (iz == 0) ? (z_lo + 2.2) : (z_hi - 2.2);
        frame.span(kAtom_type)[idx] = fluid_type;
        frame.span(kAtom_mass)[idx] = 39.948;
        frame.span(kAtom_tag)[idx] = idx + 1;
        ++idx;
    }
    frame.box = cpu::Box::orthorhombic(Lxy, Lxy, Lz);

    // Snapshot lower wall for freeze check
    std::vector<double> lo_z0;
    for (int i = 0; i < N; ++i)
        if (frame.span(kAtom_type)[i] == lower_type)
            lo_z0.push_back(frame.span(kAtom_z)[i]);

    // Probe atom forces once: piston with P_target=0 measures P_atoms.
    cpu::md::PistonElectrode probe(upper_type, /*P=*/0.0,
                                   cpu::md::PistonElectrode::Side::Upper);
    {
        cpu::md::MD probe_sim;
        probe_sim.set_evaluator(cpu::PairLJCut{eps, sigma, cutoff});
        probe_sim.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
        probe_sim.set_integrator(cpu::md::VelocityVerlet{dt});
        // PistonElectrode's three default bindings, spelled out (borrowed, so
        // `probe.last_*()` stays readable after the run).
        probe_sim.on(md::Phase::post_force).add_ref(probe, +900);
        probe_sim.on(md::Phase::post_integrate).add_ref(probe, 0);
        probe_sim.on(md::Phase::final_integrate).add_ref(probe, +200);
        probe_sim.enable_timing(false);
        probe_sim.build();
        probe_sim.run(frame, 1);
    }
    const double P_pin = probe.last_P_z();
    std::printf("Probe P_z (atoms/A) = %.6e eV/Å³  F_atoms=%.6e  A=%.3f\n",
                P_pin, probe.last_F_z_atoms(), probe.last_area());
    if (!(std::isfinite(P_pin) && std::fabs(P_pin) > 1e-8)) {
        std::printf("FAIL: probe |P_z| must be nonzero (got %.3e)\n", P_pin);
        return 1;
    }
    if (!(P_pin > 0.0)) {
        std::printf("FAIL: expected compressive probe P_z>0 on upper wall "
                    "(got %.3e); geometry not repulsive enough\n",
                    P_pin);
        return 1;
    }

    // ── MetalWalls oracle pin (REFERENCE.md) ──────────────────────────
    // MetalWalls c8ad175d, mpirun -n 1, twin input under
    // tools/metalwalls_oracle/inputs/metalwalls/piston-electrode-pz/.
    // electrode_forces.out, step 0, electrode 2 (Wup = this piston),
    // fz_lennard_jones = +1.609547904622151 Ha/Bohr.
    //   F_z = 1.609547904622151 Ha/Bohr × 27.211386245988 eV/Ha
    //                                   / 0.52917721067 Å/Bohr
    //       = 82.7662809943017 eV/Å
    //   P_z = F_z / A,  A = L_x L_y = 100 Å²
    // MW's `pressure` column is literally force_ions(3,1)/area
    // (src/output.F90:1376) — the same identity PistonElectrode reports.
    constexpr double P_z_metalwalls = 8.27662809943017e-01; // [eV/Å³]
    constexpr double P_z_tol = 1e-6;                        // relative
    const double pin_rel =
        std::fabs(P_pin - P_z_metalwalls) / P_z_metalwalls;
    std::printf("MW pin: P_z(ATV)=%.15e  P_z(MW)=%.15e  rel=%.3e\n", P_pin,
                P_z_metalwalls, pin_rel);
    if (!(pin_rel <= P_z_tol)) {
        std::printf("FAIL: probe P_z disagrees with MetalWalls c8ad175d: "
                    "rel=%.3e > %.1e\n",
                    pin_rel, P_z_tol);
        return 1;
    }

    // Reset velocities after probe (VV advanced one step)
    for (int i = 0; i < frame.nlocal(); ++i) {
        if (frame.span(kAtom_type)[i] == fluid_type)
            continue;
        frame.span(kAtom_vx)[i] = 0.0;
        frame.span(kAtom_vy)[i] = 0.0;
        frame.span(kAtom_vz)[i] = 0.0;
    }
    // Restore wall z after probe drift (piston moved under unbalanced P=0)
    for (int i = 0; i < frame.nlocal(); ++i) {
        if (frame.span(kAtom_type)[i] == upper_type)
            frame.span(kAtom_z)[i] = z_hi;
        if (frame.span(kAtom_type)[i] == lower_type)
            frame.span(kAtom_z)[i] = z_lo;
    }

    // Production: P_target = probe pin, damped piston, NVT fluid.
    cpu::md::PistonElectrode piston(upper_type, P_pin,
                                    cpu::md::PistonElectrode::Side::Upper);
    piston.tau_damp = 20.0; // [fs] textbook equilibration

    cpu::md::MD sim;
    sim.set_evaluator(cpu::PairLJCut{eps, sigma, cutoff});
    sim.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_thermostat(cpu::md::NoseHoover{T, tau_T, dt});
    sim.set_observer(cpu::md::InitVelocity{T, /*seed=*/42u});
    sim.set_observer(cpu::md::FixFreeze{lower_type});
    sim.set_observer_ref(piston);
    sim.enable_timing(false);
    sim.build();

    // Force-balance residual at first force eval with P_target=P_pin.
    // (One step: InitVelocity + integrate + force + piston post_force.)
    sim.run(frame, 1);
    const double F_atoms = piston.last_F_z_atoms();
    const double F_net = piston.last_F_net();
    const double res =
        std::fabs(F_net)
        / (std::fabs(F_atoms) + std::fabs(P_pin * piston.last_area()) + 1e-30);
    std::printf("Step1 force residual |F_net|/(|F_a|+|F_ext|)=%.3e  P_z=%.6e\n",
                res, piston.last_P_z());
    // With damping and velocity after one step, residual may include F_damp;
    // ac-005 is the static identity (unit-tested). E2E gates mean P_z.

    const auto t0 = std::chrono::steady_clock::now();
    // Warmup
    sim.run(frame, n_warm);
    // Production average of piston P_z
    double sum_P = 0.0;
    int n_samp = 0;
    constexpr int chunk = 20;
    for (int s = 0; s < n_prod; s += chunk) {
        sim.run(frame, chunk);
        sum_P += piston.last_P_z();
        ++n_samp;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();
    const double mean_P = sum_P / static_cast<double>(n_samp);

    std::printf("mean P_z = %.6e  target = %.6e  rel_err = %.3e  wall=%.4f s\n",
                mean_P, P_pin, std::fabs(mean_P - P_pin) / P_pin, wall_s);

    // ── Gates (REFERENCE.md) ──────────────────────────────────────────
    const double rel_err = std::fabs(mean_P - P_pin) / std::fabs(P_pin);
    if (!(rel_err <= 0.10)) {
        std::printf("FAIL: |<P_z>-P_target|/|P_target| = %.3e > 0.10\n",
                    rel_err);
        return 1;
    }

    // Rigid upper wall: all type-3 atoms share the same z within 1e-9
    double zmin = 1e300, zmax = -1e300;
    double max_vxy = 0.0;
    int n_up = 0;
    for (int i = 0; i < frame.nlocal(); ++i) {
        if (frame.span(kAtom_type)[i] != upper_type)
            continue;
        const double z = frame.span(kAtom_z)[i];
        zmin = std::min(zmin, z);
        zmax = std::max(zmax, z);
        max_vxy = std::max(
            max_vxy, std::fabs(static_cast<double>(frame.span(kAtom_vx)[i])));
        max_vxy = std::max(
            max_vxy, std::fabs(static_cast<double>(frame.span(kAtom_vy)[i])));
        ++n_up;
    }
    if (n_up != n_wall) {
        std::printf("FAIL: expected %d upper wall atoms, saw %d\n", n_wall,
                    n_up);
        return 1;
    }
    if (!(zmax - zmin <= 1e-9)) {
        std::printf("FAIL: upper wall not coplanar Δz=%.3e\n", zmax - zmin);
        return 1;
    }
    if (!(max_vxy <= 1e-12)) {
        std::printf("FAIL: upper wall |v_xy| max=%.3e\n", max_vxy);
        return 1;
    }

    // Lower wall frozen
    int ilo = 0;
    double max_lo_dz = 0.0;
    for (int i = 0; i < frame.nlocal(); ++i) {
        if (frame.span(kAtom_type)[i] != lower_type)
            continue;
        max_lo_dz = std::max(max_lo_dz,
                             std::fabs(frame.span(kAtom_z)[i]
                                       - lo_z0[static_cast<std::size_t>(ilo)]));
        ++ilo;
    }
    if (!(max_lo_dz <= 1e-10)) {
        std::printf("FAIL: lower wall moved Δz=%.3e\n", max_lo_dz);
        return 1;
    }

    // Finite thermo
    const double E = frame.meta.get(kEnergy);
    if (!std::isfinite(E)) {
        std::printf("FAIL: non-finite energy\n");
        return 1;
    }

    std::printf("PASS piston-electrode-pz  mean_P_z=%.6e  target=%.6e  "
                "rel_err=%.3e  wall_s=%.4f\n",
                mean_P, P_pin, rel_err, wall_s);
    return 0;
}

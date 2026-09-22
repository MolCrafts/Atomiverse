/// @file regressions/md/const-pot-md-nvt-short/reg_const-pot-md-nvt-short.cpp
/// @brief E2E: constant-potential BO + freeze + short NVT (E1 product).
///
/// Public composition only. Checks: Q polarity, electroneutrality, freeze,
/// finite E/K/charges over a short Nose–Hoover NVT trajectory.
/// See REFERENCE.md. Exits non-zero on failure. Needs ATV_BUILD_KSPACE.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/const_potential.h"
#include "atomiverse/cpu/dynamics/fix_freeze.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/pme.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

int main() {
    // ── Parallel-plate NaCl toy (see REFERENCE.md) ─────────────────────
    constexpr double Lxy = 14.0; // [Å]  (> 2× Coulomb cutoff)
    constexpr double Lz = 20.0;
    constexpr double zL = 2.0;
    constexpr double zR = 18.0;
    constexpr double psi_L = -0.5; // [V]
    constexpr double psi_R = +0.5;
    constexpr double w_gauss = 0.5; // [Å]
    constexpr double cutoff = 6.0;
    constexpr double alpha = 0.3; // [1/Å]
    constexpr double lj_eps = 0.05;
    constexpr double lj_sigma = 2.0;
    constexpr double lj_cut = 5.0;
    constexpr double neigh_cut = 6.0;
    constexpr double dt = 0.5; // [fs]
    constexpr double T = 300.0;
    constexpr double tau_T = 100.0;
    constexpr int n_steps = 400;

    // Walls 2×2 + 4 Na + 4 Cl
    constexpr int n_wall_side = 2;
    constexpr int n_per_wall = n_wall_side * n_wall_side;
    constexpr int n_ions = 8;
    const int N = 2 * n_per_wall + n_ions;

    cpu::Frame frame(N);
    int idx = 0;
    for (int wall = 0; wall < 2; ++wall) {
        const double z = (wall == 0) ? zL : zR;
        const int typ = (wall == 0) ? 2 : 3;
        for (int ix = 0; ix < n_wall_side; ++ix)
            for (int iy = 0; iy < n_wall_side; ++iy) {
                frame.span(kAtom_x)[idx] = (ix + 0.5) * (Lxy / n_wall_side);
                frame.span(kAtom_y)[idx] = (iy + 0.5) * (Lxy / n_wall_side);
                frame.span(kAtom_z)[idx] = z;
                frame.span(kAtom_type)[idx] = typ;
                frame.span(kAtom_charge)[idx] = 0.0;
                frame.span(kAtom_mass)[idx] = 12.011; // C-like wall
                frame.span(kAtom_tag)[idx] = idx + 1;
                ++idx;
            }
    }
    // Staggered mid-slab ions (charge-neutral)
    const double z_lo = 6.0, z_hi = 14.0;
    for (int i = 0; i < n_ions; ++i) {
        const bool is_na = (i % 2 == 0);
        frame.span(kAtom_x)[idx] = (0.5 + (i % 4) * 0.25) * Lxy;
        frame.span(kAtom_y)[idx] = (0.25 + (i / 4) * 0.5) * Lxy;
        frame.span(kAtom_z)[idx] = (i < 4) ? z_lo : z_hi;
        frame.span(kAtom_type)[idx] = is_na ? 0 : 1;
        frame.span(kAtom_charge)[idx] = is_na ? 1.0 : -1.0;
        frame.span(kAtom_mass)[idx] = is_na ? 22.9898 : 35.453;
        frame.span(kAtom_tag)[idx] = idx + 1;
        ++idx;
    }
    frame.box = cpu::Box::orthorhombic(Lxy, Lxy, Lz);

    // Snapshot wall positions for freeze check.
    std::vector<double> wall_x0, wall_y0, wall_z0;
    std::vector<int> wall_idx;
    for (int i = 0; i < N; ++i) {
        if (frame.span(kAtom_type)[i] == 2 || frame.span(kAtom_type)[i] == 3) {
            wall_idx.push_back(i);
            wall_x0.push_back(frame.span(kAtom_x)[i]);
            wall_y0.push_back(frame.span(kAtom_y)[i]);
            wall_z0.push_back(frame.span(kAtom_z)[i]);
        }
    }

    cpu::ConstPotentialBO cpot(std::vector<int>{2, 3},
                               std::vector<double>{psi_L, psi_R}, w_gauss);

    cpu::md::MD sim;
    sim.set_observer(cpu::md::InitVelocity{T, /*seed=*/7u});
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_thermostat(cpu::md::NoseHoover{T, tau_T, dt});
    sim.set_evaluator(cpu::PairLJCut{lj_eps, lj_sigma, lj_cut});
    sim.set_neighbor(cpu::md::NeighborList<>{neigh_cut, /*skin=*/0.5});
    sim.on(md::Phase::force).add(cpu::CoulLong{alpha, cutoff});
    sim.set_evaluator(cpu::KSpacePME{alpha, /*spacing=*/0.5, /*order=*/5});
    sim.set_observer(std::move(cpot)); // pre_force default
    // Two freeze components — one type each (left / right wall).
    sim.set_observer(cpu::md::FixFreeze{2});
    sim.set_observer(cpu::md::FixFreeze{3});
    sim.enable_timing(false);
    sim.build();

    // Keep a non-owning view is impossible after move — re-query via frame
    // charges after each segment. First step seeds BO charges.
    sim.run(frame, 1);

    auto electrode_Q = [&](int typ) {
        double q = 0.0;
        for (int i = 0; i < frame.nlocal(); ++i)
            if (frame.span(kAtom_type)[i] == typ)
                q += frame.span(kAtom_charge)[i];
        return q;
    };

    double QL = electrode_Q(2);
    double QR = electrode_Q(3);
    std::printf("After step 1: Q_L=%.6e  Q_R=%.6e  sum=%.3e\n", QL, QR,
                QL + QR);
    if (!(QL < 0.0 && QR > 0.0)) {
        std::printf("FAIL: expected Q_L<0<Q_R at Ψ_L=-0.5, Ψ_R=+0.5 "
                    "(got Q_L=%.6e Q_R=%.6e)\n",
                    QL, QR);
        return 1;
    }
    if (!(std::fabs(QL + QR) <= 1e-8)) {
        std::printf("FAIL: electrode neutrality |Q_L+Q_R|=%.3e > 1e-8\n",
                    std::fabs(QL + QR));
        return 1;
    }
    // Soft pin vs MetalWalls c8ad175d num_pbc=3 twin (total_charges.out step 0):
    // |Q| = 9.711155787094686e-2 e. ATV uses PME (3D); ~3 % is model residual.
    constexpr double kMwAbsQ = 9.711155787094686e-2; // [e]
    constexpr double kMwQRelTol = 0.05;              // 5 %
    if (!(std::fabs(std::fabs(QL) - kMwAbsQ) <= kMwAbsQ * kMwQRelTol)) {
        std::printf("FAIL: |Q_L|=%.6e vs MW %.6e (rel tol %.0f%%)\n",
                    std::fabs(QL), kMwAbsQ, 100 * kMwQRelTol);
        return 1;
    }

    // Chunked short NVT.
    const int n_chunks = 8;
    for (int c = 0; c < n_chunks; ++c) {
        sim.run(frame, n_steps / n_chunks);
        const double E = frame.meta.get(kEnergy);
        const double K = frame.meta.get(kKinetic);
        if (!(std::isfinite(E) && std::isfinite(K))) {
            std::printf("FAIL: non-finite thermo at chunk %d (E=%.3e K=%.3e)\n",
                        c, E, K);
            return 1;
        }
        for (int i = 0; i < frame.nlocal(); ++i) {
            if (!std::isfinite(frame.span(kAtom_charge)[i])) {
                std::printf("FAIL: non-finite charge on atom %d chunk %d\n", i,
                            c);
                return 1;
            }
        }
        QL = electrode_Q(2);
        QR = electrode_Q(3);
        if (!(std::fabs(QL + QR) <= 1e-6)) {
            // Slightly looser during dynamics (PME/min-image Φ mismatch noise).
            std::printf("FAIL: neutrality drifted |sumQ|=%.3e at chunk %d\n",
                        std::fabs(QL + QR), c);
            return 1;
        }
    }

    QL = electrode_Q(2);
    QR = electrode_Q(3);
    std::printf("After %d steps: E=%.6e K=%.6e  Q_L=%.6e Q_R=%.6e\n", n_steps,
                frame.meta.get(kEnergy), frame.meta.get(kKinetic), QL, QR);

    if (!(QL < 0.0 && QR > 0.0)) {
        std::printf("FAIL: polarity lost at end (Q_L=%.6e Q_R=%.6e)\n", QL, QR);
        return 1;
    }

    // Freeze check.
    double max_dx = 0.0, max_v = 0.0;
    for (std::size_t k = 0; k < wall_idx.size(); ++k) {
        const int i = wall_idx[k];
        max_dx =
            std::fmax(max_dx, std::fabs(frame.span(kAtom_x)[i] - wall_x0[k]));
        max_dx =
            std::fmax(max_dx, std::fabs(frame.span(kAtom_y)[i] - wall_y0[k]));
        max_dx =
            std::fmax(max_dx, std::fabs(frame.span(kAtom_z)[i] - wall_z0[k]));
        max_v = std::fmax(max_v, std::fabs(frame.span(kAtom_vx)[i]));
        max_v = std::fmax(max_v, std::fabs(frame.span(kAtom_vy)[i]));
        max_v = std::fmax(max_v, std::fabs(frame.span(kAtom_vz)[i]));
    }
    std::printf("Freeze: max|Δx|=%.3e Å  max|v|=%.3e Å/fs\n", max_dx, max_v);
    if (!(max_dx <= 1e-10)) {
        std::printf("FAIL: wall moved (max|Δx|=%.3e > 1e-10)\n", max_dx);
        return 1;
    }
    if (!(max_v <= 1e-12)) {
        std::printf("FAIL: wall velocity nonzero (max|v|=%.3e)\n", max_v);
        return 1;
    }

    // Fluid must have moved (selective freeze).
    bool fluid_moved = false;
    for (int i = 0; i < N; ++i) {
        if (frame.span(kAtom_type)[i] == 0 || frame.span(kAtom_type)[i] == 1) {
            // Compare against a loose bound: kinetic present implies motion
            // over 200 fs; require any ion with |v| > 1e-6.
            const double v2 =
                frame.span(kAtom_vx)[i] * frame.span(kAtom_vx)[i]
                + frame.span(kAtom_vy)[i] * frame.span(kAtom_vy)[i]
                + frame.span(kAtom_vz)[i] * frame.span(kAtom_vz)[i];
            if (v2 > 1e-12)
                fluid_moved = true;
        }
    }
    if (!fluid_moved && !(frame.meta.get(kKinetic) > 0.01)) {
        std::printf("FAIL: fluid appears frozen (K=%.3e)\n",
                    frame.meta.get(kKinetic));
        return 1;
    }

    std::printf("PASS: const-pot NVT short — finite E, Q polarity, freeze\n");
    return 0;
}

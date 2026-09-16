/// @file regressions/md/dpd_groot_warren/reg_dpd_groot_warren.cpp
/// Soft DPD fluid: conservative energy finite; kinetic T stays positive.
///
/// Full Groot–Warren equipartition needs reduced units (kB=1) and careful
/// γ–σ scaling; this regression gates framework wiring (MD + PairDpd).
#include <cmath>
#include <cstdio>
#include <random>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pair_dpd.h"

using namespace ATV;

int main() {
    constexpr int N = 32;
    const double rho = 3.0;
    const double rc = 1.0;
    const double L = std::cbrt(N / rho);
    // Physical T: 300 K — framework uses eV / Å / fs.
    const double T_K = 300.0;
    const double a = 25.0 * units::kB_ev_per_k * T_K; // soft scale ~ 25 kT
    const double gamma = 4.5;
    const double dt = 0.005;

    cpu::Frame f(N);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (int i = 0; i < N; ++i) {
        f.span(kAtom_x)[i] = u01(rng) * L;
        f.span(kAtom_y)[i] = u01(rng) * L;
        f.span(kAtom_z)[i] = u01(rng) * L;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
    }
    f.box = cpu::Box::cubic(L);

    cpu::md::MD sim;
    sim.set_evaluator(cpu::PairDpd{a, gamma, T_K, rc, dt, 7});
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_observer(cpu::md::InitVelocity{T_K, 99u});
    sim.enable_timing(false);
    sim.build();
    sim.run(f, 50);

    double ke = 0;
    for (int i = 0; i < N; ++i) {
        const double vx = f.span(kAtom_vx)[i];
        const double vy = f.span(kAtom_vy)[i];
        const double vz = f.span(kAtom_vz)[i];
        const double m = f.span(kAtom_mass)[i];
        ke += 0.5 * m * (vx * vx + vy * vy + vz * vz);
    }
    const double T = (2.0 * ke) / (3.0 * N * units::kB_ev_per_k);
    const double E = f.meta.get(kEnergy);
    std::printf("DPD: T=%.4f T0=%.1f E=%.6e N=%d\n", T, T_K, E, N);
    if (!std::isfinite(E) || !std::isfinite(T) || !(T > 1.0)) {
        std::printf("FAIL non-finite or collapsed T\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

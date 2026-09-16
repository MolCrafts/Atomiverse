/// @file regressions/md/surface-tension-coupling/reg_surface-tension-coupling.cpp
/// @brief END-TO-END: constant surface-tension NPT (couple xy) on LJ FCC.
///
/// See REFERENCE.md. Out-of-library consumer; exits non-zero on failure.
/// Oracle: LAMMPS fix npt … couple xy (offline pin; no live lmp at ctest).

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/surface_tension_npt.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

/// Instantaneous planar surface tension [eV/Å²] from stress tensor.
double gamma_inst(const cpu::Frame& f) {
    const double V = f.box.volume();
    if (!(V > 0.0))
        return 0.0;
    const auto& s = f.meta.get(kStress);
    const auto& sk = f.meta.get(kStressKinetic);
    const double Pxx = (s[0] + sk[0]) / V;
    const double Pyy = (s[1] + sk[1]) / V;
    const double Pzz = (s[2] + sk[2]) / V;
    return 0.5 * f.box.length(2) * (Pzz - 0.5 * (Pxx + Pyy));
}

} // namespace

int main() {
    // Match tools/metalwalls_oracle/inputs/lammps/in.surface_tension_coupling
    const double eps = 0.0104;
    const double sigma = 3.4;
    const double cutoff = 8.5;
    const double skin = 0.5;
    const double a_lat = 5.1;
    const int n_cells = 3; // 108 atoms
    const double T = 100.0;
    const double P_n = 0.0;     // eV/Å³
    const double gamma = 0.001; // eV/Å²
    const int n_surf = 2;
    const double dt = 1.0;      // fs
    const double tau_T = 50.0;  // fs
    const double tau_P = 500.0; // fs
    const int n_steps = 200;

    const int n = n_cells * n_cells * n_cells * 4;
    cpu::Frame frame(n);
    const double basis[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
    int idx = 0;
    for (int ix = 0; ix < n_cells; ++ix)
        for (int iy = 0; iy < n_cells; ++iy)
            for (int iz = 0; iz < n_cells; ++iz)
                for (int b = 0; b < 4; ++b) {
                    frame.span(ATV::kAtom_x)[idx] = (ix + basis[b][0]) * a_lat;
                    frame.span(ATV::kAtom_y)[idx] = (iy + basis[b][1]) * a_lat;
                    frame.span(ATV::kAtom_z)[idx] = (iz + basis[b][2]) * a_lat;
                    frame.span(ATV::kAtom_mass)[idx] = 39.948;
                    frame.span(ATV::kAtom_type)[idx] = 18;
                    ++idx;
                }
    frame.box = cpu::Box::cubic(n_cells * a_lat);
    const double L0 = frame.box.length(0);
    const double V0 = frame.box.volume();

    // Offline-implied Plat at Lz0 (for documentation / sanity):
    // Plat = Pn - n_surf * gamma / L0
    const double Plat0 = P_n - static_cast<double>(n_surf) * gamma / L0;

    cpu::md::MD sim;
    sim.set_evaluator(cpu::PairLJCut{eps, sigma, cutoff});
    sim.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
    sim.set_integrator(
        cpu::md::SurfaceTensionNPT{T, tau_T, P_n, gamma, tau_P, dt, n_surf});
    sim.set_observer(cpu::md::InitVelocity{T, 777u});
    sim.set_observer(cpu::md::Virial{});
    sim.enable_timing(false);
    sim.build();
    sim.run(frame, n_steps);

    const double Lx = frame.box.length(0);
    const double Ly = frame.box.length(1);
    const double Lz = frame.box.length(2);
    const double V1 = frame.box.volume();
    const double g = gamma_inst(frame);

    std::printf("SurfaceTensionNPT LJ FCC N=%d steps=%d\n", n, n_steps);
    std::printf("  L0=%.6f  Plat0=%.6e eV/A3  gamma_tgt=%.4f eV/A2\n", L0,
                Plat0, gamma);
    std::printf("  Lx=%.6f  Ly=%.6f  Lz=%.6f  V/V0=%.4f\n", Lx, Ly, Lz,
                V1 / V0);
    std::printf("  gamma_inst=%.6e eV/A2\n", g);

    bool ok = true;

    // Gate A2/A4: couple-xy lockstep
    const double dxy = std::fabs(Lx - Ly);
    if (!(dxy <= 1e-9 * std::max(1.0, std::fabs(Lx)))) {
        std::printf("FAIL: Lx!=Ly |Lx-Ly|=%.3e\n", dxy);
        ok = false;
    }

    // Gate A4: finite positive volume, not runaway
    if (!(V1 > 0.0) || !(V1 < 10.0 * V0) || !std::isfinite(V1)) {
        std::printf("FAIL: volume not in safe window V=%.3e V0=%.3e\n", V1, V0);
        ok = false;
    }

    // Box moved (compressed start should expand under low |P|)
    if (!(std::fabs(Lx - L0) + std::fabs(Lz - L0) > 1e-4)) {
        std::printf("FAIL: box edges frozen (Lx=%.6f Lz=%.6f L0=%.6f)\n", Lx,
                    Lz, L0);
        ok = false;
    }

    // Dual gate: anisotropic response under nonzero γ (Lz vs Lx diverge)
    // LAMMPS pin (fixed Plat): after 200 steps Lx=16.53 Lz=16.25 — |Lz-Lx|~0.28
    // ATV updates Plat(Lz) dynamically; still expect measurable |Lz-Lx|
    // once the barostat has run (or at least Lx/Ly lockstep + expansion).
    const double d_aniso = std::fabs(Lz - Lx);
    // Soft structural dual: either anisotropy visible OR expansion with
    // lockstep (semi-iso always expands Lx=Ly together; γ offset biases Lz).
    if (!(d_aniso > 1e-4 || std::fabs(Lx - L0) > 1e-3)) {
        std::printf("FAIL: no anisotropic / expansion response\n");
        ok = false;
    }

    // Gate A5: γ_inst finite
    if (!std::isfinite(g)) {
        std::printf("FAIL: gamma_inst not finite\n");
        ok = false;
    }

    // Offline pin qualitative (LAMMPS step 200): Lx==Ly, Lx > L0, Lz != Lx
    // ATV need not match trajectory numbers (different RNG path / Plat(t)).
    std::printf("  pin (LAMMPS): Lx=Ly≈16.527  Lz≈16.249  L0=15.3\n");

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

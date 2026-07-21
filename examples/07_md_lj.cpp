/// @file examples/07_md_lj.cpp
/// @brief Classical molecular dynamics on the CPU: NVE Lennard-Jones argon.
///
/// One concept: how a classical-MD run is assembled from the same
/// Driver/Component vocabulary as every other method, using the
/// user-facing categorical setters. The `cpu::md::MD` driver owns the
/// loop; `add_potential` / `set_neighbor` / `set_integrator` install the
/// force field, neighbor list, and integrator; `set_thermo` prints
/// LAMMPS-style rows. No quantum chemistry — the force provider is a
/// classical pair potential instead of an SCF.
///
/// The CUDA twin of the same lesson is examples/07_md_lj.cu
/// (target 07_md_lj_cuda).
///
/// Build:
///   cmake -B build -DATV_BUILD_EXAMPLES=ON
///   cmake --build build --target 07_md_lj
///
/// Run:
///   ./build/examples/07_md_lj
///
/// Units: eV, Å, fs, amu, K (core/units.h). Energy is conserved in NVE,
/// so the reported total-energy drift over the run should be tiny.

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

/// Simple-cubic argon lattice with zero-net-momentum random velocities.
cpu::Frame make_argon(int n_side, double density_per_A3) {
    const int n = n_side * n_side * n_side;
    const double box_len = std::cbrt(double(n) / density_per_A3);
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

    std::srand(42);
    double mvx = 0, mvy = 0, mvz = 0;
    for (int i = 0; i < n; ++i) {
        frame.span(ATV::kAtom_vx)[i] =
            0.002 * (double(std::rand()) / RAND_MAX - 0.5);
        frame.span(ATV::kAtom_vy)[i] =
            0.002 * (double(std::rand()) / RAND_MAX - 0.5);
        frame.span(ATV::kAtom_vz)[i] =
            0.002 * (double(std::rand()) / RAND_MAX - 0.5);
        mvx += frame.span(ATV::kAtom_vx)[i];
        mvy += frame.span(ATV::kAtom_vy)[i];
        mvz += frame.span(ATV::kAtom_vz)[i];
    }
    for (int i = 0; i < n; ++i) { // remove center-of-mass drift
        frame.span(ATV::kAtom_vx)[i] -= mvx / n;
        frame.span(ATV::kAtom_vy)[i] -= mvy / n;
        frame.span(ATV::kAtom_vz)[i] -= mvz / n;
    }
    return frame;
}

double total_energy(const cpu::Frame& f) {
    return f.meta.get(ATV::kEnergy) + f.meta.get(ATV::kKinetic); // PE + KE [eV]
}

} // namespace

int main() {
    // Argon LJ: epsilon = 0.0104 eV, sigma = 3.4 Å, cutoff = 8.5 Å.
    const double eps = 0.0104, sigma = 3.4, cutoff = 8.5, skin = 0.5, dt = 1.0;

    cpu::Frame frame =
        make_argon(/*n_side=*/6, /*density=*/0.0177); // 216 atoms
    std::cout << "=== CPU NVE MD: " << frame.nlocal()
              << " LJ argon atoms ===\n";

    // Build the driver from components via the categorical setters.
    cpu::md::MD sim;
    sim.add_potential(cpu::PairLJCut{eps, sigma, cutoff});
    sim.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_thermo(200); // print a thermo line every 200 steps
    sim.build();

    // One step to populate the initial energies, then the production run.
    sim.run(frame, 1);
    const double e0 = total_energy(frame);

    sim.run(frame, 1000);
    const double e1 = total_energy(frame);

    std::cout << std::fixed << std::setprecision(6)
              << "\nE_total(start): " << e0 << " eV\n"
              << "E_total(end):   " << e1 << " eV\n"
              << "drift:          " << std::scientific << std::setprecision(2)
              << std::abs(e1 - e0) << " eV  (NVE should conserve energy)\n";
    return 0;
}

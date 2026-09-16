/// @file examples/07_md_lj.cu
/// @brief Classical CUDA MD: cuda::md::MD + NeighborList + PairLJCut (NVE argon).
///
/// One concept: how a CUDA dynamics run is assembled from components.
/// The Driver owns the loop; NeighborList and PairLJCut bind to the
/// force phase; VelocityVerlet integrates; Thermo prints LAMMPS-style
/// rows. Positions start on a simple-cubic lattice with small random
/// velocities, and the run reports total-energy drift at the end.
///
/// Usage:
///   ./07_md_lj
///
/// Units: eV, Å, fs, amu, K (core/units.h).

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "atomiverse/cuda/record/thermo.cuh"

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
                frame.atoms.x[idx] = (double(ix) + 0.5) * spacing;
                frame.atoms.y[idx] = (double(iy) + 0.5) * spacing;
                frame.atoms.z[idx] = (double(iz) + 0.5) * spacing;
                frame.atoms.mass[idx] = 39.948; // argon [amu]
                frame.atoms.type[idx] = 0;
                ++idx;
            }
    frame.box = Mat3::cubic(box_len);

    std::srand(42);
    double mvx = 0, mvy = 0, mvz = 0;
    for (int i = 0; i < n; ++i) {
        frame.atoms.vx[i] = 0.002 * (double(std::rand()) / RAND_MAX - 0.5);
        frame.atoms.vy[i] = 0.002 * (double(std::rand()) / RAND_MAX - 0.5);
        frame.atoms.vz[i] = 0.002 * (double(std::rand()) / RAND_MAX - 0.5);
        mvx += frame.atoms.vx[i];
        mvy += frame.atoms.vy[i];
        mvz += frame.atoms.vz[i];
    }
    for (int i = 0; i < n; ++i) { // remove COM drift
        frame.atoms.vx[i] -= mvx / n;
        frame.atoms.vy[i] -= mvy / n;
        frame.atoms.vz[i] -= mvz / n;
    }
    return frame;
}

} // namespace

int main() {
    // Argon LJ: epsilon = 0.0104 eV, sigma = 3.4 Å, cutoff = 8.5 Å.
    const double eps = 0.0104, sigma = 3.4, cutoff = 8.5, dt = 1.0;

    cpu::Frame frame =
        make_argon(/*n_side=*/6, /*density=*/0.0177); // 216 atoms

    std::cout << "=== CUDA NVE MD: " << frame.atoms.size()
              << " LJ argon atoms ===\n";

    // Reference pass (1 step) to capture the initial total energy.
    cpu::Frame probe = frame;
    {
        cuda::md::MD sim;
        sim.set_integrator(cuda::md::VelocityVerlet{dt});
        sim.on(md::Phase::force)
            .add(cuda::NeighborList{cutoff}, -10)
            .add(cuda::PairLJCut{eps, sigma, cutoff});
        sim.build();
        sim.run(probe, 1);
    }
    const double e0 = probe.scalars.energy + probe.scalars.kinetic;

    // Production run with thermo output every 100 steps.
    cuda::md::MD sim;
    sim.set_integrator(cuda::md::VelocityVerlet{dt});
    sim.on(md::Phase::force)
        .add(cuda::NeighborList{cutoff}, -10)
        .add(cuda::PairLJCut{eps, sigma, cutoff});
    sim.on(md::Phase::end_of_step).add(cuda::Thermo{100});
    sim.build();
    sim.run(frame, 1000);

    const double e1 = frame.scalars.energy + frame.scalars.kinetic;
    std::cout << std::fixed << std::setprecision(6)
              << "\nE_total(start): " << e0 << " eV\n"
              << "E_total(end):   " << e1 << " eV\n"
              << "drift:          " << std::scientific << std::setprecision(2)
              << std::abs(e1 - e0) << " eV\n";
    return 0;
}

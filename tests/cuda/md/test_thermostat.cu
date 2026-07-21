/// @file test_thermostat.cu
/// @brief Tests for NoseHoover thermostat, Deform, cuda::Virial, Barostat.

#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/md/barostat.cuh"
#include "atomiverse/cuda/md/deform.cuh"
#include "atomiverse/cuda/md/init.cuh"
#include "atomiverse/cuda/md/md.cuh"
#include "atomiverse/cuda/md/nose_hoover.cuh"
#include "atomiverse/cuda/md/velocity_verlet.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/observer/stress.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"

using namespace ATV;

static cpu::Frame make_cubic(int n_side, double spacing) {
    int n = n_side * n_side * n_side;
    cpu::Frame frame(n);
    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                frame.atoms.x[idx] = ix * spacing;
                frame.atoms.y[idx] = iy * spacing;
                frame.atoms.z[idx] = iz * spacing;
                frame.atoms.mass[idx] = 14.0;
                ++idx;
            }
    frame.box = Mat3::cubic(n_side * spacing);
    return frame;
}

TEST(NoseHoover, TemperatureControl) {
    // Minimal: 8 atoms, well-separated, weak cuda::PairLJCut, small dt
    auto frame = make_cubic(2, 5.0); // 8 atoms

    double dt = 0.001;
    cuda::md::MD sim;
    sim.add(cuda::md::VelocityVerlet{dt});
    sim.add(cuda::md::NoseHoover{100.0, 100.0, dt});
    sim.add(cuda::md::InitVelocity{50.0, 42});
    sim.on(md::Phase::force)
        .add(cuda::NeighborList{8.0}, -10)
        .add(cuda::PairLJCut{0.01, 3.4, 8.0});
    sim.build();
    sim.run(frame, 50);

    // After 50 steps with thermostat, results should be finite
    EXPECT_TRUE(std::isfinite(frame.scalars.kinetic));
    for (int i = 0; i < frame.atoms.size(); ++i)
        EXPECT_TRUE(std::isfinite(frame.atoms.x[i]));
}

TEST(Deform, BoxGrows) {
    auto frame = make_cubic(3, 5.0);
    double L0 = frame.box(0, 0);

    double rate = 1e-4;
    double dt = 0.01;
    int n_steps = 100;

    cuda::md::MD sim;
    sim.add(cuda::md::VelocityVerlet{dt});
    sim.add(cuda::md::Deform{0, rate, dt});
    sim.on(md::Phase::force)
        .add(cuda::NeighborList{10.0}, -10)
        .add(cuda::PairLJCut{0.01, 3.4, 10.0});
    sim.build();
    sim.run(frame, n_steps);

    double expected_L = L0 * pow(1.0 + rate * dt, n_steps);
    EXPECT_NEAR(frame.box(0, 0), expected_L, expected_L * 0.01);
    EXPECT_NEAR(frame.box(1, 1), L0, 1e-6);
}

TEST(Virial, FiniteStress) {
    auto frame = make_cubic(2, 5.0);

    cuda::md::MD sim;
    sim.add(cuda::md::VelocityVerlet{0.001});
    sim.add(cuda::md::InitVelocity{50.0, 42});
    sim.on(md::Phase::force)
        .add(cuda::NeighborList{8.0}, -10)
        .add(cuda::PairLJCut{0.01, 3.4, 8.0});
    sim.on(md::Phase::post_force).add(cuda::Virial{});
    sim.build();
    sim.run(frame, 10);

    // Stress tensor should be finite
    for (int c = 0; c < 6; ++c)
        EXPECT_TRUE(std::isfinite(frame.scalars.stress[c]));
    // Kinetic contribution guarantees non-zero diagonal
    EXPECT_NE(frame.scalars.stress[0], 0.0);
}

TEST(Barostat, FiniteResults) {
    auto frame = make_cubic(2, 5.0);
    double dt = 0.001;

    cuda::md::MD sim;
    sim.add(cuda::md::VelocityVerlet{dt});
    sim.add(cuda::md::NoseHoover{100.0, 100.0, dt});
    sim.add(cuda::md::InitVelocity{50.0, 42});
    sim.add(cuda::md::Barostat{0.0, 500.0, dt});
    sim.on(md::Phase::force)
        .add(cuda::NeighborList{8.0}, -10)
        .add(cuda::PairLJCut{0.01, 3.4, 8.0});
    sim.on(md::Phase::post_force).add(cuda::Virial{});
    sim.build();
    sim.run(frame, 50);

    // All results finite
    for (int i = 0; i < frame.atoms.size(); ++i) {
        EXPECT_TRUE(std::isfinite(frame.atoms.x[i]));
        EXPECT_TRUE(std::isfinite(frame.atoms.vx[i]));
    }
    EXPECT_TRUE(std::isfinite(frame.box(0, 0)));
    EXPECT_GT(frame.box(0, 0), 0.0);
}

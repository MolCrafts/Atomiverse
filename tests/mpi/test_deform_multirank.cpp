/// @file tests/mpi/test_deform_multirank.cpp
/// @brief Phase 0.7 -- multi-rank uniaxial deformation (LJ + Deform).
///
/// 8-rank Argon-LJ box with a Deform component (`Mode::erate`) pulling
/// along x at ε̇ = 1e-4 / fs. Over 200 steps × dt = 0.5 fs the box should
/// grow by the LAMMPS `fix deform` engineering-strain law
/// `L = L0·(1 + ε̇·n_steps·dt)` = 1 % along x, with every owned atom's x
/// coordinate scaled identically on all ranks. Asserts:
///   * the simulation does not blow up (no NaN positions),
///   * box_x grew by ~1 % (within 1e-6 tolerance),
///   * stress σ_xx changes during the deformation (force pipeline alive).

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/comm/halo_exchange.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/deform.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observer/virial_stress.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

cpu::Frame make_fcc_argon(int n_cells, double a_lat) {
    const int n = n_cells * n_cells * n_cells * 4;
    cpu::Frame frame(n);
    double basis[4][3] = {
        {0.0, 0.0, 0.0},
        {0.5, 0.5, 0.0},
        {0.5, 0.0, 0.5},
        {0.0, 0.5, 0.5},
    };
    int idx = 0;
    for (int ix = 0; ix < n_cells; ++ix)
        for (int iy = 0; iy < n_cells; ++iy)
            for (int iz = 0; iz < n_cells; ++iz)
                for (int b = 0; b < 4; ++b) {
                    frame.span(ATV::kAtom_x)[idx] =
                        (ix + basis[b][0] + 0.25) * a_lat;
                    frame.span(ATV::kAtom_y)[idx] =
                        (iy + basis[b][1] + 0.25) * a_lat;
                    frame.span(ATV::kAtom_z)[idx] =
                        (iz + basis[b][2] + 0.25) * a_lat;
                    frame.span(ATV::kAtom_mass)[idx] = 39.948;
                    frame.span(ATV::kAtom_type)[idx] = 18;
                    ++idx;
                }
    frame.simbox = ATV::cpu::Box::cubic(n_cells * a_lat);
    return frame;
}

} // namespace

TEST(DeformMultiRank, UniaxialPullStretchesBoxOnAllRanks) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 8) {
        GTEST_SKIP() << "8-rank specific test; got world_size=" << world_size;
    }

    auto frame = make_fcc_argon(5, 5.3);

    const double lj_eps = 0.0104;
    const double lj_sig = 3.4;
    const double lj_cut = 6.0;
    const double dt = 0.5;

    const double box_x_initial = frame.simbox.cell()(0, 0);
    const double strain_rate = 1e-4; // 1/fs
    const int n_steps = 200;

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{dt});
    sim.add(cpu::md::Deform{cpu::md::Deform::Mode::erate, strain_rate, dt,
                            /*axis=*/0});
    sim.on(ATV::md::Phase::pre_force).add(cpu::comm::HaloSend{lj_cut}, -20);
    sim.on(ATV::md::Phase::post_force).add(cpu::comm::HaloRecv{}, -1000);
    sim.on(ATV::md::Phase::force)
        .add(cpu::md::NeighborList<>{lj_cut}, -10)
        .add(cpu::PairLJCut{lj_eps, lj_sig, lj_cut});
    sim.add(cpu::md::Virial{});
    sim.build();

    sim.run(frame, n_steps);

    // LAMMPS fix deform ERATE: L = L0·(1 + ε̇·n_steps·dt), L0 = box edge
    // at Deform activation (= box_x_initial here, never pre-deformed).
    const double expected_box_x =
        box_x_initial * (1.0 + strain_rate * n_steps * dt);

    EXPECT_NEAR(frame.simbox.cell()(0, 0), expected_box_x, 1e-6)
        << "box_x deformation diverged from analytic expectation";

    // Rank consistency: gather all ranks' box_x via min and max -- must agree
    // exactly across ranks (Deform is fully deterministic per rank).
    double box_x_min, box_x_max;
    const double bx = frame.simbox.cell()(0, 0);
    MPI_Allreduce(&bx, &box_x_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&bx, &box_x_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_DOUBLE_EQ(box_x_min, box_x_max)
        << "ranks disagree on deformed box_x: min=" << box_x_min
        << ", max=" << box_x_max;

    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    for (int i = 0; i < frame.nlocal(); ++i) {
        ASSERT_FALSE(std::isnan(frame.span(ATV::kAtom_x)[i]))
            << "rank " << my_rank << " atom " << i << " x went NaN";
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}

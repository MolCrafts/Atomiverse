/// @file tests/mpi/test_npt_multirank.cpp
/// @brief Phase 0.6 -- multi-rank NPT (LJ + Berendsen barostat).
///
/// 8-rank Argon-LJ box deliberately seeded at a non-equilibrium volume
/// (over-compressed, P >> 0). Berendsen barostat at P_target = 0 must
/// drive the volume back toward the LJ equilibrium density, with the
/// box stretching uniformly across all ranks. The test asserts:
///   * the simulation does not blow up (no NaN positions),
///   * the box volume grows over the run (over-pressure relaxes),
///   * the final pressure has dropped below the initial pressure.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/comm/halo_exchange.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/barostat.h"
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
                    const double s = 1e-4 * ((idx % 2 == 0) ? 1.0 : -1.0);
                    frame.span(ATV::kAtom_vx)[idx] = s;
                    frame.span(ATV::kAtom_vy)[idx] = s * 0.5;
                    frame.span(ATV::kAtom_vz)[idx] = s * 0.25;
                    ++idx;
                }
    frame.simbox = ATV::cpu::Box::cubic(n_cells * a_lat);
    return frame;
}

// LAMMPS-standard scalar pressure P = +trace(σ)/(3V), positive ⇔
// compressive — matches `compute pressure`, `NoseHooverNPT` and the
// post-`lammps-api-align-04` `Barostat`.
double pressure_from_frame(const cpu::Frame& frame) {
    const double V = frame.simbox.cell()(0, 0) * frame.simbox.cell()(1, 1)
                     * frame.simbox.cell()(2, 2);
    const double trace =
        (frame.meta.get(ATV::kStress)[0] + frame.meta.get(ATV::kStress)[1]
         + frame.meta.get(ATV::kStress)[2])
        + (frame.meta.get(ATV::kStressKinetic)[0]
           + frame.meta.get(ATV::kStressKinetic)[1]
           + frame.meta.get(ATV::kStressKinetic)[2]);
    return +trace / (3.0 * V);
}

} // namespace

TEST(NptMultiRank, BarostatRelaxesOverPressure) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 8) {
        GTEST_SKIP() << "8-rank specific test; got world_size=" << world_size;
    }

    // 500 atoms in a 26.5 Å box: same geometry as the 8-rank LJ NVE exit
    // gate. FCC NN ≈ 3.75 Å, σ = 3.4 Å — slightly compressed → P > 0.
    auto frame = make_fcc_argon(5, 5.3);

    const double lj_eps = 0.0104;
    const double lj_sig = 3.4;
    const double lj_cut = 6.0;
    const double dt = 0.5;

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{dt});
    sim.on(ATV::md::Phase::pre_force).add(cpu::comm::HaloSend{lj_cut}, -20);
    sim.on(ATV::md::Phase::post_force).add(cpu::comm::HaloRecv{}, -1000);
    sim.on(ATV::md::Phase::force)
        .add(cpu::md::NeighborList<>{lj_cut}, -10)
        .add(cpu::PairLJCut{lj_eps, lj_sig, lj_cut});
    sim.add(cpu::md::Virial{});
    sim.add(cpu::md::Barostat{0.0, /*tau_P=*/200.0, dt}); // P_target = 0
    sim.build();

    sim.run(frame, 1);
    const double V_initial = frame.simbox.cell()(0, 0)
                             * frame.simbox.cell()(1, 1)
                             * frame.simbox.cell()(2, 2);
    const double P_initial = pressure_from_frame(frame);

    sim.run(frame, 999);
    const double V_final = frame.simbox.cell()(0, 0) * frame.simbox.cell()(1, 1)
                           * frame.simbox.cell()(2, 2);
    const double P_final = pressure_from_frame(frame);

    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // No NaN explosion.
    for (int i = 0; i < frame.nlocal(); ++i) {
        ASSERT_FALSE(std::isnan(frame.span(ATV::kAtom_x)[i]))
            << "rank " << my_rank << " atom " << i << " x went NaN";
    }

    // Box volume should drift toward equilibrium. Initial state is above
    // LJ equilibrium density (compressed), so the barostat should grow
    // the box. Note: this is a weak assertion -- the goal here is
    // multi-rank correctness, not quantitative density convergence.
    if (P_initial > 0) {
        EXPECT_GT(V_final, V_initial)
            << "P_initial=" << P_initial << " > 0 should drive box growth, "
            << "but V_final=" << V_final << " <= V_initial=" << V_initial;
    }

    // Magnitude of pressure must shrink (barostat is doing its job).
    EXPECT_LT(std::abs(P_final), std::abs(P_initial))
        << "barostat failed to reduce |P| over 1000 steps "
        << "(P_initial=" << P_initial << ", P_final=" << P_final << ")";
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}

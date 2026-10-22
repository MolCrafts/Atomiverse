/// InitVelocity::build on one owned atom/rank must retain thermal motion.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/driver/dof.h"
#include "atomiverse/cpu/dynamics/init.h"
#include <cmath>
#include <gtest/gtest.h>
#include <mpi.h>

TEST(InitVelocity, GlobalComAndTargetTemperatureWithOneAtomPerRank) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ASSERT_GT(size, 1);
    ATV::cpu::Frame frame(1);
    frame.span(ATV::kAtom_mass)[0] = 1;
    frame.span(ATV::kAtom_tag)[0] = rank + 1;
    ATV::cpu::md::mpi::DomainDecomposition dd;
    dd.cart_comm = MPI_COMM_WORLD;
    ATV::cpu::Context context;
    context.mpi_rank = rank;
    context.mpi_size = size;
    context.md.dd = &dd;
    ATV::cpu::publish_global_dof(context, frame, 0);
    ATV::cpu::md::InitVelocity initialize(300, 42);
    initialize.build({context, frame});
    const double vx = frame.span(ATV::kAtom_vx)[0];
    const double vy = frame.span(ATV::kAtom_vy)[0];
    const double vz = frame.span(ATV::kAtom_vz)[0];
    const double local[4]{vx, vy, vz, vx * vx + vy * vy + vz * vz};
    double global[4]{};
    MPI_Allreduce(local, global, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int d = 0; d < 3; ++d)
        EXPECT_NEAR(global[d], 0, 1e-12);
    const double temperature =
        global[3] * ATV::units::mass_v2_to_energy
        / (context.md.n_dof_global * ATV::units::kB_ev_per_k);
    EXPECT_NEAR(temperature, 300, 1e-9);
}
int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}

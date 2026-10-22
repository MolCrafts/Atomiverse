/// @file tests/mpi/test_comm_brick.cpp
/// @brief MPI unit tests for `cpu::comm::CommBrick` 8-atom brick goldens.
///
/// Hard-coded: 8 atoms, cubic box L=20 Å, shell 6 Å, ATV_FORCE_DIMS=2,1,1.
/// n1: n_ghost==0. n2: n_ghost==8; reverse sum fx==8 eV/Å; exchange keeps 8
/// owners. No MD.run.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/communication/comm_brick.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/reduce.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/classical_polarization.h"

using namespace ATV;

namespace {

struct ForcedDims {
    explicit ForcedDims(const char* grid) {
        ::setenv("ATV_FORCE_DIMS", grid, 1);
    }
    ~ForcedDims() { ::unsetenv("ATV_FORCE_DIMS"); }
};

cpu::Frame make_8_atom_cube() {
    cpu::Frame frame(8);
    int idx = 0;
    for (int ix = 0; ix < 2; ++ix)
        for (int iy = 0; iy < 2; ++iy)
            for (int iz = 0; iz < 2; ++iz) {
                frame.span(kAtom_x)[idx] =
                    (static_cast<double>(ix) + 0.5) * 10.0;
                frame.span(kAtom_y)[idx] =
                    (static_cast<double>(iy) + 0.5) * 10.0;
                frame.span(kAtom_z)[idx] =
                    (static_cast<double>(iz) + 0.5) * 10.0;
                frame.span(kAtom_mass)[idx] = 1.0;
                ++idx;
            }
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

TEST(TestCommBrick, N1CutoffSixBuildsPeriodicImages) {
    int world = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1)
        GTEST_SKIP() << "n1 golden";

    auto frame = make_8_atom_cube();
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    cpu::Comm* comm = backend.context().md.comm;
    ASSERT_NE(comm, nullptr);
    comm->set_ghost_cutoff(6.0);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    comm->setup(lctx);
    comm->borders(lctx);
    EXPECT_EQ(frame.nghost(), 56);
    comm->set_ghost_cutoff(0.0);
    comm->borders(lctx);
    EXPECT_EQ(frame.nghost(), 0);
    backend.shutdown();
}

TEST(TestCommBrick, N2BrickGoldens) {
    int world = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 2)
        GTEST_SKIP() << "n2 ATV_FORCE_DIMS=2,1,1 golden";

    ForcedDims dims("2,1,1");
    auto frame = make_8_atom_cube();
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    cpu::Comm* comm = backend.context().md.comm;
    ASSERT_NE(comm, nullptr);
    auto* dd = backend.context().md.dd;
    ASSERT_NE(dd, nullptr);
    EXPECT_EQ(dd->dims[0], 2);
    EXPECT_EQ(dd->dims[1], 1);
    EXPECT_EQ(dd->dims[2], 1);

    comm->set_ghost_cutoff(6.0);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    comm->setup(lctx);
    comm->borders(lctx);
    EXPECT_EQ(frame.nghost(), 44);

    for (int i = 0; i < frame.nlocal(); ++i)
        frame.span(kAtom_fx)[i] = 0.0;
    for (int g = 0; g < frame.nghost(); ++g)
        frame.span(kAtom_fx)[frame.nlocal() + g] = 1.0;
    comm->reverse_comm(lctx);
    double sum_fx = 0.0;
    for (int i = 0; i < frame.nlocal(); ++i)
        sum_fx += static_cast<double>(frame.span(kAtom_fx)[i]);
    EXPECT_NEAR(sum_fx, 44.0, 1e-12);

    comm->exchange(lctx);
    int n_local = frame.nlocal();
    int n_owners = 0;
    MPI_Allreduce(&n_local, &n_owners, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(n_owners, 8);

    backend.shutdown();
}

TEST(TestReduce, ForceInitializationReducesBetweenProductionCadenceSteps) {
    int world = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (world < 2)
        GTEST_SKIP()
            << "requires multiple ranks to distinguish local from global";

    cpu::Frame frame(0);
    cpu::Context context;
    cpu::md::mpi::DomainDecomposition dd;
    // Borrow the test communicator; Reduce requires no topology operations.
    dd.cart_comm = MPI_COMM_WORLD;
    context.mpi_size = world;
    context.md.dd = &dd;
    cpu::md::mpi::Reduce reduce(5);
    const double local = rank + 1.0;
    const double global = world * (world + 1.0) / 2.0;
    const auto reset_partials = [&] {
        frame.meta.get(kEnergy) = local;
        frame.meta.get(kKinetic) = 2.0 * local;
        for (int k = 0; k < 6; ++k) {
            frame.meta.get(kStress)[k] = (k + 1.0) * local;
            frame.meta.get(kStressKinetic)[k] = -(k + 1.0) * local;
        }
    };
    const auto expect_totals = [&](double total) {
        EXPECT_DOUBLE_EQ(frame.meta.get(kEnergy), total);
        EXPECT_DOUBLE_EQ(frame.meta.get(kKinetic), 2.0 * total);
        for (int k = 0; k < 6; ++k) {
            EXPECT_DOUBLE_EQ(frame.meta.get(kStress)[k], (k + 1.0) * total);
            EXPECT_DOUBLE_EQ(frame.meta.get(kStressKinetic)[k],
                             -(k + 1.0) * total);
        }
    };

    reset_partials();
    reduce.launch({context, frame, 0, 7});
    expect_totals(local);

    context.exec.initializing_forces = true;
    reduce.initialize_forces({context, frame, 0, 7});
    expect_totals(global);

    context.exec.initializing_forces = false;
    reset_partials();
    reduce.launch({context, frame, 0, 10});
    expect_totals(global);
}

TEST(TestClassicalPolarization, SpatialRanksReduceOwnedMoments) {
    int world = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (world < 2)
        GTEST_SKIP() << "requires spatial ranks";
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(10);
    frame.span(kAtom_charge)[0] = 1.0;
    frame.span(kAtom_x)[0] = rank + 1.0;
    cpu::Context context;
    context.mpi_size = world;
    cpu::md::mpi::DomainDecomposition dd;
    dd.cart_comm = MPI_COMM_WORLD;
    context.md.dd = &dd;
    cpu::md::ClassicalPolarization polarization;
    polarization.launch({context, frame, 0, 0});
    EXPECT_DOUBLE_EQ(polarization.Px(), world * (world + 1.0) / 2000.0);
    // A complete replica is not a spatial shard, despite having MPI peers.
    context.md.dd = nullptr;
    polarization.launch({context, frame, 0, 0});
    EXPECT_DOUBLE_EQ(polarization.Px(), (rank + 1.0) / 1000.0);
}

TEST(TestCommBrick, TriclinicFractionalMigrationAndPeriodicSelfAxes) {
    int world = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (world != 2)
        GTEST_SKIP() << "two-rank brick";
    ForcedDims dims("2,1,1");
    cpu::Frame frame(4);
    Mat3 cell = Mat3::diagonal(20.0, 20.0, 20.0);
    cell(0, 1) = 6.0;
    cell(1, 0) = 2.0;
    cell(2, 1) = -3.0;
    frame.box.set_cell(cell);
    for (int i = 0; i < 4; ++i) {
        const Vec3 point = frame.box.to_cartesian(
            {(i / 2) == 0 ? 0.25 : 0.75, (i % 2) == 0 ? 0.05 : 0.95, 0.5});
        frame.span(kAtom_x)[i] = point[0];
        frame.span(kAtom_y)[i] = point[1];
        frame.span(kAtom_z)[i] = point[2];
        frame.span(kAtom_tag)[i] = i + 1;
    }
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    auto* dd = backend.context().md.dd;
    auto* comm = backend.context().md.comm;
    ASSERT_EQ(frame.nlocal(), 2);
    EXPECT_NEAR(dd->sublo[0], 0.5 * rank, 1e-15);
    for (int i = 0; i < frame.nlocal(); ++i)
        EXPECT_EQ(dd->owner_of(frame.span(kAtom_x)[i], frame.span(kAtom_y)[i],
                               frame.span(kAtom_z)[i], frame.box),
                  rank);
    cpu::LaunchContext ctx{backend.context(), frame, 0, 0};
    comm->set_ghost_cutoff(2.0);
    comm->setup(ctx);
    comm->borders(ctx);
    EXPECT_EQ(frame.nghost(), 2); // local y images, no x/z ghosts
    for (int g = frame.nlocal(); g < frame.nlocal() + frame.nghost(); ++g) {
        const Vec3 point = frame.box.to_fractional({frame.span(kAtom_x)[g],
                                                    frame.span(kAtom_y)[g],
                                                    frame.span(kAtom_z)[g]});
        EXPECT_NEAR(point[0], rank == 0 ? 0.25 : 0.75, 1e-13);
        EXPECT_TRUE(point[1] < 0.0 || point[1] > 1.0);
        EXPECT_NEAR(point[2], 0.5, 1e-13);
    }
    // Cross the fractional x cut while Cartesian y changes with lattice a.
    const Vec3 move =
        frame.box.to_cartesian({rank == 0 ? 0.5 : -0.5, 0.0, 0.0});
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(kAtom_x)[i] += move[0];
        frame.span(kAtom_y)[i] += move[1];
        frame.span(kAtom_z)[i] += move[2];
    }
    comm->exchange(ctx);
    ASSERT_EQ(frame.nlocal(), 2);
    for (int i = 0; i < frame.nlocal(); ++i) {
        EXPECT_EQ(dd->owner_of(frame.span(kAtom_x)[i], frame.span(kAtom_y)[i],
                               frame.span(kAtom_z)[i], frame.box),
                  rank);
        const auto tag = frame.span(kAtom_tag)[i];
        EXPECT_EQ(static_cast<int>((tag - 1) / 2), 1 - rank);
    }
    backend.shutdown();
}

TEST(TestCommBrick,
     TriclinicReplicaBuildsLocalImagesWithoutSpatialCommunicator) {
    cpu::Frame frame(1);
    Mat3 cell = Mat3::diagonal(20.0, 20.0, 20.0);
    cell(0, 1) = 4.0;
    cell(1, 0) = 2.0;
    frame.box.set_cell(cell);
    const Vec3 position = frame.box.to_cartesian({0.05, 0.05, 0.05});
    frame.span(kAtom_x)[0] = position[0];
    frame.span(kAtom_y)[0] = position[1];
    frame.span(kAtom_z)[0] = position[2];
    cpu::Runtime backend;
    backend.set_serial_on_replica();
    backend.init();
    backend.set_frame(frame);
    auto* dd = backend.context().md.dd;
    ASSERT_EQ(dd->cart_comm, MPI_COMM_NULL);
    auto* comm = backend.context().md.comm;
    comm->set_ghost_cutoff(2.0);
    cpu::LaunchContext ctx{backend.context(), frame, 0, 0};
    comm->setup(ctx);
    comm->borders(ctx);
    ASSERT_EQ(frame.nlocal(), 1);
    ASSERT_EQ(frame.nghost(), 7);
    frame.span(kAtom_x)[0] += 0.125;
    comm->forward_comm(ctx);
    for (int i = 1; i <= frame.nghost(); ++i) {
        const Vec3 image = frame.box.to_fractional(
            {frame.span(kAtom_x)[i] - frame.span(kAtom_x)[0],
             frame.span(kAtom_y)[i] - frame.span(kAtom_y)[0],
             frame.span(kAtom_z)[i] - frame.span(kAtom_z)[0]});
        for (int d = 0; d < 3; ++d)
            EXPECT_NEAR(image[d], std::nearbyint(image[d]), 1e-13);
        frame.span(kAtom_fx)[i] = 1.0;
    }
    frame.span(kAtom_fx)[0] = 0.0;
    comm->reverse_comm(ctx);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_fx)[0], 7.0);
    backend.shutdown();
}

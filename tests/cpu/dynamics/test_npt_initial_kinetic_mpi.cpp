/// @file tests/cpu/dynamics/test_npt_initial_kinetic_mpi.cpp
/// @brief MPI2 method-level NPT kinetic and published-virial contracts.
#include <gtest/gtest.h>
#include <mpi.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"

#include <cmath>
#include <cstdio>
#include <limits>

using namespace ATV;

namespace {

constexpr double kTemperature = 100.0;
constexpr double kTauT = 10.0;
constexpr double kTauP = 100.0;
constexpr double kDt = 0.1;
constexpr int kDof = 3;

cpu::Frame atoms(int n) {
    cpu::Frame frame(n);
    frame.box = cpu::Box::cubic(20.0);
    for (int i = 0; i < n; ++i) {
        frame.span(kAtom_x)[i] = 4.0 + i;
        frame.span(kAtom_y)[i] = 4.0;
        frame.span(kAtom_z)[i] = 4.0;
        frame.span(kAtom_mass)[i] = 2.0;
        frame.span(kAtom_vx)[i] = 0.0;
        frame.span(kAtom_vy)[i] = 0.0;
        frame.span(kAtom_vz)[i] = 0.0;
        frame.span(kAtom_fx)[i] = 0.0;
        frame.span(kAtom_fy)[i] = 0.0;
        frame.span(kAtom_fz)[i] = 0.0;
    }
    return frame;
}

cpu::Context context(cpu::md::mpi::DomainDecomposition& dd) {
    cpu::Context result;
    MPI_Comm_rank(MPI_COMM_WORLD, &result.mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &result.mpi_size);
    result.md.dd = &dd;
    result.md.n_dof_global = kDof;
    return result;
}

double first_nhc_scale(double kinetic) {
    // One initially stationary chain link: eta_dot = G*dt/4;
    // the first atom-thermostat half-step scales v by exp(-G*dt^2/8).
    const double kT = units::kB_ev_per_k * kTemperature;
    const double G = (2.0 * kinetic - kDof * kT) / (kDof * kT * kTauT * kTauT);
    return std::exp(-G * kDt * kDt / 8.0);
}

} // namespace

TEST(NptInitialKineticMpi, FirstHalfStepUsesActualGlobalKinetic) {
    cpu::md::mpi::DomainDecomposition dd;
    dd.cart_comm = MPI_COMM_WORLD;
    dd.n_local = 1;
    auto runtime = context(dd);
    auto frame = atoms(1);
    cpu::md::NoseHooverNPT integrator{
        kTemperature, kTauT, 0.0, kTauP, kDt, 0.0, 1, 1, {false, false, false}};
    integrator.build({runtime, frame});
    // The ranks hold different velocities, so a local KE, an equipartition
    // guess, and a duplicated global sum all give different predictions.
    frame.span(kAtom_vx)[0] = runtime.mpi_rank == 0 ? 0.05 : 0.1;
    const double local_velocity = frame.span(kAtom_vx)[0];
    const double v0 = static_cast<StorageReal>(0.05);
    const double v1 = static_cast<StorageReal>(0.1);
    const double global_kinetic =
        units::mass_v2_to_energy * (v0 * v0 + v1 * v1);
    const double expected = local_velocity * first_nhc_scale(global_kinetic);

    integrator.launch(
        {runtime, frame, slot_of(md::Phase::initial_integrate), 17});

    EXPECT_NEAR(frame.span(kAtom_vx)[0], expected,
                8 * std::numeric_limits<StorageReal>::epsilon());
}

TEST(NptInitialKineticMpi, PublishedGlobalVirialIsNotReducedAgain) {
    cpu::md::mpi::DomainDecomposition dd;
    dd.cart_comm = MPI_COMM_WORLD;
    dd.n_local = 1;
    auto runtime = context(dd);
    auto frame = atoms(1);
    constexpr double stress = 0.1;
    // This is the globally published virial visible after setup/Reduce,
    // identical on both ranks. Velocities remain zero, isolating its drive.
    for (int d = 0; d < 3; ++d)
        frame.meta.get(kStress)[d] = stress;
    const double initial_volume = frame.box.volume();
    const double kT = units::kB_ev_per_k * kTemperature;
    const double W = (kDof + 3.0) * kT * kTauP * kTauP;
    const double eps_dot = 0.5 * kDt * 3.0 * stress / W;
    const double expected_volume =
        initial_volume * std::exp(3.0 * eps_dot * kDt);
    cpu::md::NoseHooverNPT integrator{kTemperature, kTauT, 0.0, kTauP,
                                      kDt,          0.0,   1,   1};
    integrator.build({runtime, frame});

    integrator.launch(
        {runtime, frame, slot_of(md::Phase::initial_integrate), 0});

    EXPECT_NEAR(frame.box.volume(), expected_volume, 1e-12 * initial_volume);
    for (int d = 0; d < 3; ++d)
        EXPECT_DOUBLE_EQ(frame.meta.get(kStress)[d], stress);
}

TEST(NptInitialKineticMpi, SerialReplicaWithNullCartesianCommUsesWholeFrame) {
    // This is DomainRuntime::set_serial_on_replica's representation: MPI is
    // live, a DD object exists, but its cart_comm is NULL and no rows were
    // partitioned. Each rank retains its own complete two-atom Frame.
    cpu::md::mpi::DomainDecomposition dd;
    auto runtime = context(dd);
    auto frame = atoms(2);
    cpu::md::NoseHooverNPT integrator{
        kTemperature, kTauT, 0.0, kTauP, kDt, 0.0, 1, 1, {false, false, false}};
    integrator.build({runtime, frame});
    const double speed =
        static_cast<StorageReal>(0.05 * (runtime.mpi_rank + 1));
    frame.span(kAtom_vx)[0] = speed;
    frame.span(kAtom_vx)[1] = -speed;
    const double replica_kinetic =
        units::mass_v2_to_energy * 2.0 * speed * speed;
    const double expected = speed * first_nhc_scale(replica_kinetic);

    integrator.launch(
        {runtime, frame, slot_of(md::Phase::initial_integrate), 17});

    const double tolerance = 8 * std::numeric_limits<StorageReal>::epsilon();
    EXPECT_NEAR(frame.span(kAtom_vx)[0], expected, tolerance);
    EXPECT_NEAR(frame.span(kAtom_vx)[1], -expected, tolerance);
    EXPECT_NEAR(frame.span(kAtom_x)[0], 4.0 + expected * kDt, tolerance);
    EXPECT_NEAR(frame.span(kAtom_x)[1], 5.0 - expected * kDt, tolerance);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        std::fprintf(
            stderr,
            "test_npt_initial_kinetic_mpi requires exactly 2 MPI ranks\n");
        MPI_Finalize();
        return 2;
    }
    ::testing::InitGoogleTest(&argc, argv);
    const int local_result = RUN_ALL_TESTS();
    int result = 0;
    MPI_Allreduce(&local_result, &result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}

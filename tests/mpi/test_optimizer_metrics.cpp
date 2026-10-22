#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <mpi.h>

#include "atomiverse/core/stage.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/minimize/anisotropic_cell_relax.h"
#include "atomiverse/cpu/minimize/cell_relax.h"
#include "atomiverse/cpu/minimize/cg.h"
#include "atomiverse/cpu/minimize/fire.h"
#include "atomiverse/cpu/minimize/joint_cell_relax.h"
#include "atomiverse/cpu/minimize/lbfgs.h"

using namespace ATV;

TEST(MpiOptimizerMetrics, JointUsesOwnedRowsAndGlobalNormalization) {
    cpu::Context context;
    MPI_Comm_rank(MPI_COMM_WORLD, &context.mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &context.mpi_size);
    cpu::md::mpi::DomainDecomposition dd;
    dd.cart_comm = MPI_COMM_WORLD;

    context.md.dd = &dd;
    cpu::Frame frame(2);
    frame.set_nlocal(1);
    frame.set_nghost(1);
    frame.box = cpu::Box::cubic(10);
    frame.span(kAtom_fx)[0] = context.mpi_rank + 1;
    frame.span(kAtom_x)[1] = 91;
    frame.span(kAtom_fx)[1] = std::numeric_limits<double>::quiet_NaN();
    ConvergenceMetrics metrics;
    context.loop.metrics = &metrics;
    cpu::minimizer::JointIsotropicRelax joint(0, 0.1);
    joint.build({context, frame});
    joint.launch(
        {context, frame, slot_of(minimizer::Phase::search_direction), 0});
    joint.launch({context, frame, slot_of(minimizer::Phase::line_search), 0});
    joint.launch(
        {context, frame, slot_of(minimizer::Phase::convergence_check), 0});
    const double ranks = context.mpi_size;
    EXPECT_NEAR(frame.span(kAtom_x)[0], 0.1 * (context.mpi_rank + 1) / ranks,
                1e-14);
    EXPECT_EQ(frame.span(kAtom_x)[1], 91);
    EXPECT_EQ(metrics.max_force, ranks);
    EXPECT_NEAR(metrics.rms_force,
                std::sqrt((ranks + 1) * (2 * ranks + 1) / 18), 1e-14);
    EXPECT_NEAR(metrics.max_displacement, 0.1, 1e-14);
    dd.cart_comm = MPI_COMM_NULL;
}

TEST(MpiOptimizerMetrics, OneBadRankMakesEveryOptimizerFailTogether) {
    auto check = [](auto& optimizer) {
        cpu::Context context;
        MPI_Comm_rank(MPI_COMM_WORLD, &context.mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &context.mpi_size);
        cpu::md::mpi::DomainDecomposition dd;
        dd.cart_comm = MPI_COMM_WORLD;

        context.md.dd = &dd;
        cpu::Frame frame(1);
        frame.box = cpu::Box::cubic(10);
        frame.span(kAtom_x)[0] = 7;
        frame.span(kAtom_fx)[0] = context.mpi_rank == 0
                                      ? std::numeric_limits<double>::infinity()
                                      : 1.0;
        optimizer.build({context, frame});
        EXPECT_THROW(
            optimizer.launch({context, frame,
                              slot_of(minimizer::Phase::search_direction), 0}),
            std::runtime_error);
        EXPECT_EQ(frame.span(kAtom_x)[0], 7);
        dd.cart_comm = MPI_COMM_NULL;
    };
    cpu::minimizer::CG cg;
    cpu::minimizer::LBFGS lbfgs;
    cpu::minimizer::FIRE fire;
    cpu::minimizer::JointIsotropicRelax joint;
    check(cg);
    check(lbfgs);
    check(fire);
    check(joint);
}

TEST(MpiOptimizerMetrics, OneBadPressureOrParameterRejectsCellStepEverywhere) {
    auto check = [](auto& joint) {
        for (bool bad_pressure : {true, false}) {
            cpu::Context context;
            MPI_Comm_rank(MPI_COMM_WORLD, &context.mpi_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &context.mpi_size);
            cpu::md::mpi::DomainDecomposition dd;
            dd.cart_comm = MPI_COMM_WORLD;

            context.md.dd = &dd;
            cpu::Frame frame(1);
            frame.box = cpu::Box::cubic(10);
            frame.span(kAtom_x)[0] = 2;
            frame.span(kAtom_fx)[0] = 1;
            joint.build({context, frame});
            joint.launch({context, frame,
                          slot_of(minimizer::Phase::search_direction), 0});
            if (context.mpi_rank == 0) {
                if (bad_pressure)
                    frame.meta.get(kStress)[0] =
                        std::numeric_limits<double>::quiet_NaN();
                else
                    joint.bulk_mod = 0;
            }
            for (auto phase : {minimizer::Phase::search_direction,
                               minimizer::Phase::line_search,
                               minimizer::Phase::convergence_check}) {
                EXPECT_THROW(joint.launch({context, frame, slot_of(phase), 0}),
                             std::runtime_error);
                EXPECT_EQ(frame.span(kAtom_x)[0], 2);
                EXPECT_EQ(frame.box.volume(), 1000);
            }
            dd.cart_comm = MPI_COMM_NULL;
        }
    };
    cpu::minimizer::JointIsotropicRelax joint;
    cpu::minimizer::IsotropicCellRelax cell;
    cpu::minimizer::AnisotropicCellRelax anisotropic;
    check(joint);
    check(cell);
    check(anisotropic);
}

TEST(MpiOptimizerMetrics, CellScalingPreservesGhostsAndInvalidatesHalo) {
    auto check = [](auto& optimizer) {
        cpu::Context context;
        MPI_Comm_rank(MPI_COMM_WORLD, &context.mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &context.mpi_size);
        cpu::md::mpi::DomainDecomposition dd;
        dd.cart_comm = MPI_COMM_WORLD;
        dd.dims[0] = context.mpi_size;
        dd.coords[0] = context.mpi_rank;

        context.md.dd = &dd;
        context.md.reneighbor = false;
        cpu::Frame frame(2);
        frame.set_nlocal(1);
        frame.set_nghost(1);
        frame.box = cpu::Box::cubic(10);
        dd.compute_sublo_subhi(frame.box);
        const double x = 10.0 * (context.mpi_rank + 0.5) / context.mpi_size;
        frame.span(kAtom_x)[0] = x;
        frame.span(kAtom_x)[1] = 91;
        frame.span(kAtom_fx)[1] = std::numeric_limits<double>::quiet_NaN();
        frame.meta.get(kStress)[0] = 3000;
        ConvergenceMetrics metrics;
        context.loop.metrics = &metrics;
        optimizer.build({context, frame});
        optimizer.launch(
            {context, frame, slot_of(minimizer::Phase::search_direction), 0});
        optimizer.launch(
            {context, frame, slot_of(minimizer::Phase::line_search), 0});
        optimizer.launch(
            {context, frame, slot_of(minimizer::Phase::convergence_check), 0});
        EXPECT_NEAR(frame.span(kAtom_x)[0], 1.05 * x, 1e-14);
        EXPECT_EQ(frame.span(kAtom_x)[1], 91);
        EXPECT_TRUE(context.md.reneighbor);
        EXPECT_NEAR(dd.sublo[0], double(context.mpi_rank) / context.mpi_size,
                    1e-14);
        EXPECT_NEAR(dd.subhi[0],
                    double(context.mpi_rank + 1) / context.mpi_size, 1e-14);
        EXPECT_EQ(metrics.max_force, 0);
        EXPECT_NEAR(metrics.max_displacement,
                    0.5 * (context.mpi_size - 0.5) / context.mpi_size, 1e-14);
        EXPECT_TRUE(metrics.has_displacement);
        dd.cart_comm = MPI_COMM_NULL;
    };
    cpu::minimizer::JointIsotropicRelax joint;
    cpu::minimizer::IsotropicCellRelax cell;
    cpu::minimizer::AnisotropicCellRelax anisotropic;
    check(joint);
    check(cell);
    check(anisotropic);
}

TEST(MpiOptimizerMetrics, OneFullyMaskedRankFailsAnisotropicBuildEverywhere) {
    cpu::Context context;
    MPI_Comm_rank(MPI_COMM_WORLD, &context.mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &context.mpi_size);
    cpu::md::mpi::DomainDecomposition dd;
    dd.cart_comm = MPI_COMM_WORLD;
    context.md.dd = &dd;
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(10);
    cpu::minimizer::AnisotropicCellRelax optimizer;
    if (context.mpi_rank == 0)
        optimizer.relax_x = optimizer.relax_y = optimizer.relax_z = false;
    EXPECT_THROW(optimizer.build({context, frame}), std::invalid_argument);
    dd.cart_comm = MPI_COMM_NULL;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}

/// @file tests/cuda/dynamics/test_cuda_cluster_contract.cu
/// @brief OpenMM cluster contract: NeighborList publishes cl_* for force.
///
/// cuda-openmm-md-01-contract: after build+launch, pair_list has non-null
/// cl_row_ptr / cl_col_idx / cl_orig_index / cl_mask and num_clusters > 0.

#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/plan.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/driver/backend.cuh"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"

using namespace ATV;

namespace {

cpu::Frame make_lattice(int n_side, double spacing) {
    int n = n_side * n_side * n_side;
    cpu::Frame frame(n);
    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                frame.span(kAtom_x)[idx] = (ix + 0.5) * spacing;
                frame.span(kAtom_y)[idx] = (iy + 0.5) * spacing;
                frame.span(kAtom_z)[idx] = (iz + 0.5) * spacing;
                frame.span(kAtom_mass)[idx] = 1.0;
                frame.span(kAtom_type)[idx] = 0;
                frame.span(kAtom_vx)[idx] = 0.0;
                frame.span(kAtom_vy)[idx] = 0.0;
                frame.span(kAtom_vz)[idx] = 0.0;
                ++idx;
            }
    frame.box = cpu::Box::cubic(double(n_side) * spacing);
    return frame;
}

} // namespace

TEST(CudaClusterContract, NeighborListPublishesClusterCsr) {
    auto frame = make_lattice(4, 1.5);
    const double cutoff = 3.0;
    const double skin = 0.9;

    cuda::Backend backend;
    backend.init();
    backend.materialize(frame);

    cuda::NeighborList nl{cutoff, skin};
    PlanBuilder<cuda::Component> builder;
    builder.add({0, 0, &nl});
    auto plan = builder.build(ATV::slot_count<md::Phase>());
    backend.load(plan);

    cuda::BuildContext bctx{backend.context(), backend.frame()};
    nl.build(bctx);

    backend.step(0);
    backend.sync();

    const cuda::PairList* pl = backend.context().md.pair_list;
    ASSERT_NE(pl, nullptr);
    EXPECT_NE(pl->cl_row_ptr, nullptr);
    EXPECT_NE(pl->cl_col_idx, nullptr);
    EXPECT_NE(pl->cl_orig_index, nullptr);
    EXPECT_NE(pl->cl_mask, nullptr);
    EXPECT_GT(pl->num_clusters, 0);

    cuda::DestroyContext dctx{backend.context(), backend.frame()};
    nl.destroy(dctx);
    backend.shutdown();
}

TEST(CudaClusterContract, PairLjRunsOnClusterPath) {
    auto frame = make_lattice(4, 1.5);
    const double cutoff = 3.0;

    cuda::md::MD sim;
    // dt = 0: one frozen step is a force/energy pass (see
    // `test_pair_lj_tile_step0.cu`).
    sim.set_integrator(cuda::md::VelocityVerlet{0.0});
    sim.set_neighbor(cuda::NeighborList{cutoff, 0.9});
    sim.set_evaluator(cuda::PairLJCut{1.0, 1.0, cutoff});
    sim.build();
    sim.run(frame, 1);

    double e = static_cast<double>(frame.meta.get(kEnergy));
    EXPECT_TRUE(std::isfinite(e));
}

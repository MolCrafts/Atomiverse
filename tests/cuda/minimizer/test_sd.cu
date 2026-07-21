/// @file test_sd.cu
/// @brief Integration test: steepest-descent minimization of a cuda::PairLJCut
///        system through the migrated `Minimize + SteepestDescent` shape.
///
/// Post driver-taxonomy migration the CUDA minimizer is a single
/// `cuda::minimizer::Minimize` preset (a `cuda::CUDADriver<minimizer::Phase>` — the
/// `cuda::OptimizeDriver` category base is deleted; termination is a `LoopHandler`)
/// with a hot-swappable `cuda::minimizer::SteepestDescent` Optimizer
/// component — the legacy `cuda::minimizer::SD` / `CG` driver classes and
/// the old `SteepestDescentStep` component name no longer exist. This
/// file references only the new API; it is a compile-time shape guard.

#include <cmath>
#include <gtest/gtest.h>
#include <type_traits>

#include "atomiverse/cuda/driver/cuda_driver.cuh"
#include "atomiverse/cuda/minimizer/minimize.cuh"
#include "atomiverse/cuda/minimizer/sd.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "atomiverse/cuda/recorder/thermo.cuh"
#include "atomiverse/facade/component_role.h"

using namespace ATV;

namespace {

/// Create a simple cubic lattice slightly perturbed from equilibrium.
cpu::Frame make_displaced_system(int n_side, double density,
                                 double displacement) {
    int n = n_side * n_side * n_side;
    double box_len = std::cbrt(double(n) / density);
    double spacing = box_len / double(n_side);

    cpu::Frame frame(n);
    std::srand(123);
    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                frame.atoms.x[idx] =
                    (double(ix) + double(0.5)) * spacing
                    + displacement
                          * (double(std::rand()) / double(RAND_MAX)
                             - double(0.5));
                frame.atoms.y[idx] =
                    (double(iy) + double(0.5)) * spacing
                    + displacement
                          * (double(std::rand()) / double(RAND_MAX)
                             - double(0.5));
                frame.atoms.z[idx] =
                    (double(iz) + double(0.5)) * spacing
                    + displacement
                          * (double(std::rand()) / double(RAND_MAX)
                             - double(0.5));
                frame.atoms.mass[idx] = double(1.0);
                frame.atoms.type[idx] = 0;
                idx++;
            }
    frame.box = Mat3::cubic(static_cast<double>(box_len));
    return frame;
}

} // anonymous namespace

// ── Compile-time shape guard ────────────────────────────────────────
//
// `Minimize` must be a loop node over `minimizer::Phase`, and `SteepestDescent` must be
// a `cuda::Component` tagged with `role()==Optimizer`. These static checks fail to
// compile against the deleted `SD` / `CG` driver classes or the old
// `SteepestDescentStep` name.
//
// The old assertion was `is_base_of<cuda::OptimizeDriver, Minimize>` — "Minimize is in
// the Optimize category". The category is gone, so the assertion with content left is
// the stage domain: Minimize is a CUDADriver over minimizer::Phase.
static_assert(
    std::is_base_of<cuda::CUDADriver<ATV::minimizer::Phase>,
                    cuda::minimizer::Minimize>::value,
    "cuda::minimizer::Minimize must be a CUDADriver over minimizer::Phase");
static_assert(
    std::is_base_of<cuda::Component, cuda::minimizer::SteepestDescent>::value,
    "cuda::minimizer::SteepestDescent must be a cuda::Component");

TEST(Minimizer, SteepestDescentIsOptimizerRole) {
    cuda::minimizer::SteepestDescent step{double(0.01)};
    EXPECT_EQ(step.role(), ComponentRole::Optimizer);
}

TEST(Minimizer, SDEnergyDecreases) {
    auto frame = make_displaced_system(4, double(0.8), double(0.3));
    double cutoff = double(3.0);

    // Get initial energy with a single force evaluation. Even the
    // single-evaluation run needs an Optimizer component to satisfy
    // Minimize::validate_components(); at step 0 forces are zero so the
    // SteepestDescent update is a no-op.
    cpu::Frame frame_init = frame;
    {
        cuda::minimizer::Minimize min;
        min.add(cuda::minimizer::SteepestDescent{double(0.01)});
        min.on(minimizer::Phase::force)
            .add(cuda::NeighborList{cutoff}, -10)
            .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
        min.build();
        min.run(
            frame_init,
            1); // 1 step: force eval only (step 0 has zero forces → step update is no-op)
    }
    double e_after_1 = frame_init.scalars.energy;

    // Run more steps of minimization
    cpu::Frame frame_min = frame;
    {
        cuda::minimizer::Minimize min;
        min.add(cuda::minimizer::SteepestDescent{double(0.01)});
        min.on(minimizer::Phase::force)
            .add(cuda::NeighborList{cutoff}, -10)
            .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
        min.build();
        min.run(frame_min, 50);
    }
    double e_final = frame_min.scalars.energy;

    // Energy should have decreased (system relaxed)
    EXPECT_LT(static_cast<double>(e_final), static_cast<double>(e_after_1));
}

TEST(Minimizer, SDWithThermo) {
    auto frame = make_displaced_system(3, double(0.8), double(0.2));
    double cutoff = double(3.0);

    cuda::minimizer::Minimize min;
    min.add(cuda::minimizer::SteepestDescent{double(0.01)});
    min.on(minimizer::Phase::force)
        .add(cuda::NeighborList{cutoff}, -10)
        .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
    min.on(minimizer::Phase::end_of_step).add(cuda::Thermo{10});
    min.build();

    // Should print thermo without crashing
    testing::internal::CaptureStdout();
    min.run(frame, 30);
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("Step"), std::string::npos);
    EXPECT_NE(output.find("PotEng"), std::string::npos);

    // Energy should be finite
    EXPECT_TRUE(std::isfinite(static_cast<double>(frame.scalars.energy)));
}

TEST(Minimizer, BuildThrowsWithoutOptimizer) {
    auto frame = make_displaced_system(3, double(0.8), double(0.2));
    double cutoff = double(3.0);

    // A Minimize with only a force evaluator (no SteepestDescent
    // Optimizer component) must fail `Minimize::validate_components()`:
    // ≥ 1 Optimizer-role component is required.
    cuda::minimizer::Minimize min;
    min.on(minimizer::Phase::force)
        .add(cuda::NeighborList{cutoff}, -10)
        .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
    EXPECT_THROW(min.build(), std::runtime_error);
}

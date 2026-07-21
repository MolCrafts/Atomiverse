/// @file test_lbfgs.cu
/// @brief Integration tests: cuda::minimizer::LBFGS Optimizer component —
///        energy descent (two-loop direction), five-criteria early stop,
///        and the Armijo backtracking line-search path.
///
/// The five convergence tolerances used to be constructor arguments of `LBFGS`. They
/// are now the state of the `cuda::ForceConverged` LoopHandler — the component
/// measures, the handler decides — so every tolerance below is expressed with
/// `min.add_handler(cuda::ForceConverged{…})`, which also suppresses the default
/// handler `Minimize` would otherwise attach. Same numbers, same intent.

#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <type_traits>

#include "atomiverse/cuda/driver/loop_handler.cuh"
#include "atomiverse/cuda/minimizer/lbfgs.cuh"
#include "atomiverse/cuda/minimizer/minimize.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
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

/// Single force evaluation through the Minimize shape.
double initial_energy(const cpu::Frame& frame, double cutoff) {
    cpu::Frame f = frame;
    cuda::minimizer::Minimize min;
    min.add(cuda::minimizer::LBFGS{});
    min.on(minimizer::Phase::force)
        .add(cuda::NeighborList{cutoff}, -10)
        .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
    min.build();
    min.run(f, 1);
    return f.scalars.energy;
}

} // anonymous namespace

// ── Compile-time shape guards ───────────────────────────────────────
static_assert(std::is_base_of<cuda::Component, cuda::minimizer::LBFGS>::value,
              "cuda::minimizer::LBFGS must be a cuda::Component");

TEST(MinimizerLBFGS, IsOptimizerRole) {
    cuda::minimizer::LBFGS step{};
    EXPECT_EQ(step.role(), ComponentRole::Optimizer);
}

TEST(MinimizerLBFGS, BindsSearchDirectionAndConvergenceCheck) {
    cuda::minimizer::LBFGS step{};
    auto bindings = step.default_bindings();
    ASSERT_EQ(bindings.size(), 2u);
    EXPECT_EQ(bindings[0].phase,
              ATV::make_binding(minimizer::Phase::search_direction).phase);
    EXPECT_EQ(bindings[1].phase,
              ATV::make_binding(minimizer::Phase::convergence_check).phase);
}

TEST(MinimizerLBFGS, EnergyDecreases) {
    auto frame = make_displaced_system(4, double(0.8), double(0.3));
    double cutoff = double(3.0);
    double e0 = initial_energy(frame, cutoff);

    cuda::minimizer::Minimize min;
    // Tight tolerances: exercise the two-loop direction over the full
    // budget (history fills past the default memory of 5 pairs).
    min.add(cuda::minimizer::LBFGS{5, double(0.01), 10});
    min.add_handler(cuda::ForceConverged{/*ftol=*/1e-8, /*rms_ftol=*/1e-8,
                                         /*max_dtol=*/1e-10, /*rms_dtol=*/1e-10,
                                         /*etol=*/1e-12});
    min.on(minimizer::Phase::force)
        .add(cuda::NeighborList{cutoff}, -10)
        .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
    min.build();
    min.run(frame, 50);

    EXPECT_LT(static_cast<double>(frame.scalars.energy), e0);
    EXPECT_TRUE(std::isfinite(static_cast<double>(frame.scalars.energy)));
}

TEST(MinimizerLBFGS, ConvergesAndStopsEarly) {
    auto frame = make_displaced_system(3, double(0.8), double(0.2));
    double cutoff = double(3.0);

    cuda::minimizer::Minimize min;
    // Loose tolerances trip the five-criteria rule well inside the budget; lvalue
    // registration keeps the component inspectable. The verdict belongs to the LOOP
    // (`min.converged()`) — `LBFGS::converged()` is deleted, because an optimizer that
    // only publishes metrics does not know the verdict.
    cuda::minimizer::LBFGS lbfgs{5, double(0.01), 10};
    min.add(lbfgs);
    min.add_handler(cuda::ForceConverged{/*ftol=*/10.0, /*rms_ftol=*/10.0,
                                         /*max_dtol=*/1.0, /*rms_dtol=*/1.0,
                                         /*etol=*/10.0});
    min.on(minimizer::Phase::force)
        .add(cuda::NeighborList{cutoff}, -10)
        .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
    min.build();
    int iters = min.run(frame, 200);

    EXPECT_LT(iters, 200) << "loose tolerances must stop the loop early";
    EXPECT_TRUE(min.converged());
    EXPECT_EQ(min.stop_reason(), "force_converged");
    EXPECT_FALSE(lbfgs.trace().empty());
    EXPECT_EQ(lbfgs.trace().back().step, iters);
}

TEST(MinimizerLBFGS, ArmijoLineSearchEnergyDecreases) {
    auto frame = make_displaced_system(4, double(0.8), double(0.3));
    double cutoff = double(3.0);
    double e0 = initial_energy(frame, cutoff);

    cuda::minimizer::LBFGS step{5, double(0.05), 20};
    step.line_search(); // Armijo default constants.

    cuda::minimizer::Minimize min;
    min.add(std::move(step));
    min.add_handler(cuda::ForceConverged{/*ftol=*/1e-8, /*rms_ftol=*/1e-8,
                                         /*max_dtol=*/1e-10, /*rms_dtol=*/1e-10,
                                         /*etol=*/1e-12});
    min.on(minimizer::Phase::force)
        .add(cuda::NeighborList{cutoff}, -10)
        .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
    min.build();
    min.run(frame, 80);

    EXPECT_LT(static_cast<double>(frame.scalars.energy), e0);
    EXPECT_TRUE(std::isfinite(static_cast<double>(frame.scalars.energy)));
}

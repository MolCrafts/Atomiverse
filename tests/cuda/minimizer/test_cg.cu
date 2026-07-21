/// @file test_cg.cu
/// @brief Integration tests: cuda::minimizer::CG Optimizer component —
///        energy descent, five-criteria early stop, and optimizer
///        hot-swap parity inside the single `Minimize` driver (the CUDA
///        mirror of tests/cpu/test_minimize_driver_swap_optimizer.cpp).
///
/// The five convergence tolerances used to be constructor arguments of `CG`. They are
/// now the state of the `cuda::ForceConverged` LoopHandler — the component measures,
/// the handler decides — so every tolerance below is expressed with
/// `min.add_handler(cuda::ForceConverged{…})`, which also suppresses the default
/// handler `Minimize` would otherwise attach. Same numbers, same intent.

#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <type_traits>

#include "atomiverse/cuda/driver/loop_handler.cuh"
#include "atomiverse/cuda/minimizer/cg.cuh"
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

/// CG applies a FIXED step x += α₀·d with no displacement cap (CPU
/// parity — only LBFGS caps at 0.1 Å), so the tests below run in its
/// stable regime: gentle perturbations + α₀ = 1e-3, keeping α₀·|F|
/// well below the LJ core length scale. The CPU CG test does the same
/// with a soft-force H₂/SCF system.
constexpr double kStableAlpha = 0.001;

/// Single force evaluation through the Minimize shape (step 0 is a no-op
/// move for every Optimizer: forces are zero until the force phase runs).
double initial_energy(const cpu::Frame& frame, double cutoff) {
    cpu::Frame f = frame;
    cuda::minimizer::Minimize min;
    min.add(cuda::minimizer::CG{kStableAlpha});
    min.on(minimizer::Phase::force)
        .add(cuda::NeighborList{cutoff}, -10)
        .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
    min.build();
    min.run(f, 1);
    return f.scalars.energy;
}

} // anonymous namespace

// ── Compile-time shape guards ───────────────────────────────────────
static_assert(std::is_base_of<cuda::Component, cuda::minimizer::CG>::value,
              "cuda::minimizer::CG must be a cuda::Component");

TEST(MinimizerCG, IsOptimizerRole) {
    cuda::minimizer::CG step{double(0.01)};
    EXPECT_EQ(step.role(), ComponentRole::Optimizer);
}

TEST(MinimizerCG, BindsSearchDirectionAndConvergenceCheck) {
    cuda::minimizer::CG step{double(0.01)};
    auto bindings = step.default_bindings();
    ASSERT_EQ(bindings.size(), 2u);
    EXPECT_EQ(bindings[0].phase,
              ATV::make_binding(minimizer::Phase::search_direction).phase);
    EXPECT_EQ(bindings[1].phase,
              ATV::make_binding(minimizer::Phase::convergence_check).phase);
}

TEST(MinimizerCG, EnergyDecreases) {
    auto frame = make_displaced_system(4, double(0.8), double(0.1));
    double cutoff = double(3.0);
    double e0 = initial_energy(frame, cutoff);

    cuda::minimizer::Minimize min;
    // Tight tolerances: exercise the full 50-iteration budget.
    min.add(cuda::minimizer::CG{kStableAlpha});
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

TEST(MinimizerCG, ConvergesAndStopsEarly) {
    auto frame = make_displaced_system(3, double(0.8), double(0.05));
    double cutoff = double(3.0);

    cuda::minimizer::Minimize min;
    // Loose tolerances so the five-criteria rule trips well inside the
    // budget — verifies that CG's published metrics reach the handler, that the
    // handler's Converged verdict reaches the loop, and that Minimize::run breaks
    // early. The verdict is the LOOP's now (`min.converged()`); `CG::converged()` is
    // deleted, because an optimizer that only measures has no business claiming to know
    // it. Lvalue registration stores a reference, so the component remains inspectable
    // after run() (the documented add() lvalue contract).
    cuda::minimizer::CG cg{kStableAlpha};
    min.add(cg);
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
    EXPECT_FALSE(cg.trace().empty());
    EXPECT_EQ(cg.trace().back().step, iters);
}

TEST(MinimizerCG, SwapOptimizerCGvsLBFGS) {
    // The CUDA mirror of the CPU optimizer hot-swap test: the same
    // Minimize driver shape accepts CG and LBFGS interchangeably, and
    // both relax the perturbed lattice (energy strictly below the
    // single-evaluation baseline).
    auto frame = make_displaced_system(4, double(0.8), double(0.1));
    double cutoff = double(3.0);
    double e0 = initial_energy(frame, cutoff);

    cpu::Frame f_cg = frame;
    {
        cuda::minimizer::Minimize min;
        min.add(cuda::minimizer::CG{kStableAlpha});
        min.on(minimizer::Phase::force)
            .add(cuda::NeighborList{cutoff}, -10)
            .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
        min.build();
        min.run(f_cg, 100);
    }

    cpu::Frame f_lbfgs = frame;
    {
        cuda::minimizer::Minimize min;
        min.add(cuda::minimizer::LBFGS{});
        min.on(minimizer::Phase::force)
            .add(cuda::NeighborList{cutoff}, -10)
            .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
        min.build();
        min.run(f_lbfgs, 100);
    }

    EXPECT_LT(static_cast<double>(f_cg.scalars.energy), e0);
    EXPECT_LT(static_cast<double>(f_lbfgs.scalars.energy), e0);
}

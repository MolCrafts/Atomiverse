/// @file tests/cpu/dynamics/test_const_pot_maze_symplectic.cpp
/// @brief Unit tests for the true mass-zero constrained ConstPotMaZe path
///        (single methods; no e2e MD — that lives in
///        regressions/md/maze-symplectic-charge-dynamics).

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/const_pot_maze.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// nx×ny sites per wall (types 1/2) + two electrolyte ions (type 3).
cpu::Frame plate_frame(int nx, int ny) {
    const int n_wall = nx * ny;
    const int n = 2 * n_wall + 2;
    cpu::Frame f(n);
    const double Lx = static_cast<double>(nx);
    const double Ly = static_cast<double>(ny);
    int a = 0;
    for (int wall = 0; wall < 2; ++wall) {
        const double z = (wall == 0) ? 6.0 : 14.0;
        const int typ = (wall == 0) ? 1 : 2;
        for (int iy = 0; iy < ny; ++iy)
            for (int ix = 0; ix < nx; ++ix) {
                f.span(kAtom_x)[a] = (ix + 0.5) * (Lx / nx);
                f.span(kAtom_y)[a] = (iy + 0.5) * (Ly / ny);
                f.span(kAtom_z)[a] = z;
                f.span(kAtom_charge)[a] = 0.0;
                f.span(kAtom_type)[a] = typ;
                f.span(kAtom_mass)[a] = 196.97;
                f.span(kAtom_tag)[a] = a + 1;
                ++a;
            }
    }
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_x)[a] = 0.5 * Lx;
        f.span(kAtom_y)[a] = 0.5 * Ly;
        f.span(kAtom_z)[a] = (i == 0) ? 9.0 : 11.0;
        f.span(kAtom_charge)[a] = (i == 0) ? 0.5 : -0.5;
        f.span(kAtom_type)[a] = 3;
        f.span(kAtom_mass)[a] = 1.0;
        f.span(kAtom_tag)[a] = a + 1;
        ++a;
    }
    f.box = cpu::Box::orthorhombic(Lx, Ly, 20.0);
    return f;
}

cpu::md::ConstPotMaZe make_maze() {
    cpu::md::ConstPotMaZe maze(2.0);
    maze.add_electrode(1, 0.0);
    maze.add_electrode(2, 1.0);
    return maze;
}

} // namespace

TEST(ConstPotMaZeSymplectic, MatchesDenseBOOnStaticFrame) {
    auto f_m = plate_frame(2, 2);
    auto f_d = plate_frame(2, 2);
    auto maze = make_maze();
    ASSERT_TRUE(maze.symplectic); // true MaZe is the default path
    cpu::md::ConstPotBO dense(2.0);
    dense.add_electrode(1, 0.0).add_electrode(2, 1.0);
    dense.use_cg = false;

    cpu::Context ctx;
    maze.build({ctx, f_m});
    maze.launch({ctx, f_m, 0, 0});
    dense.build({ctx, f_d});
    dense.launch({ctx, f_d, 0, 0});

    const auto& Qm = maze.solver().charges();
    const auto& Qd = dense.charges();
    ASSERT_EQ(Qm.size(), Qd.size());
    for (std::size_t i = 0; i < Qm.size(); ++i)
        EXPECT_NEAR(Qm[i], Qd[i], 1e-8) << "site " << i;
    EXPECT_LE(maze.constraint_residual(), maze.maze_tol);
}

TEST(ConstPotMaZeSymplectic, NeutralityHeldEachStep) {
    auto f = plate_frame(2, 2);
    auto maze = make_maze();
    cpu::Context ctx;
    maze.build({ctx, f});
    const int ion = f.nlocal() - 2;
    for (int s = 0; s < 5; ++s) {
        f.span(kAtom_z)[ion] = 9.0 + 0.05 * s;
        maze.launch({ctx, f, 0, s});
        EXPECT_LE(maze.solver().neutrality_residual(), 1e-10) << "step " << s;
        EXPECT_LE(maze.constraint_residual(), maze.maze_tol) << "step " << s;
    }
}

TEST(ConstPotMaZeSymplectic, GlobalNeutralityTargetsElectrolyte) {
    auto f = plate_frame(2, 2);
    // Net +0.3 e electrolyte: ΣQ_electrode must become −0.3 e.
    f.span(kAtom_charge)[f.nlocal() - 2] = 0.5;
    f.span(kAtom_charge)[f.nlocal() - 1] = -0.2;
    auto maze = make_maze();
    maze.solver().global_electroneutrality = true;
    cpu::Context ctx;
    maze.build({ctx, f});
    maze.launch({ctx, f, 0, 0});
    double sum_q = 0.0;
    for (double q : maze.solver().charges())
        sum_q += q;
    EXPECT_NEAR(sum_q, -0.3, 1e-10);
}

TEST(ConstPotMaZeSymplectic, StaticFrameReSolveTakesZeroIterations) {
    auto f = plate_frame(2, 2);
    auto maze = make_maze();
    cpu::Context ctx;
    maze.build({ctx, f});
    maze.launch({ctx, f, 0, 0});
    EXPECT_GT(maze.last_maze_iters(), 0); // cold bootstrap does real work
    maze.launch({ctx, f, 0, 1});
    EXPECT_EQ(maze.last_maze_iters(), 0); // guess = previous solution
    maze.launch({ctx, f, 0, 2});
    EXPECT_EQ(maze.last_maze_iters(), 0); // Verlet extrapolation is exact
    EXPECT_LE(maze.max_dQ(), 1e-12);      // Q(t) continuity on a static frame
}

TEST(ConstPotMaZeSymplectic, VerletWarmStartBeatsColdStart) {
    auto f_w = plate_frame(3, 3);
    auto f_c = plate_frame(3, 3);
    auto warm = make_maze();
    auto cold = make_maze();
    cpu::Context ctx;
    warm.build({ctx, f_w});
    cold.build({ctx, f_c});
    const int ion = f_w.nlocal() - 2;
    long it_warm = 0;
    long it_cold = 0;
    for (int s = 0; s < 8; ++s) {
        const double z = 9.0 + 0.04 * s;
        f_w.span(kAtom_z)[ion] = z;
        f_c.span(kAtom_z)[ion] = z;
        warm.launch({ctx, f_w, 0, s});
        cold.clear_history(); // force a cold bootstrap every step
        cold.launch({ctx, f_c, 0, s});
        if (s >= 2) { // both extrapolation slots filled
            it_warm += warm.last_maze_iters();
            it_cold += cold.last_maze_iters();
        }
    }
    EXPECT_LT(it_warm, it_cold);
}

TEST(ConstPotMaZeSymplectic, FootholdTwinStillRunsAspcCg) {
    auto f = plate_frame(2, 2);
    auto maze = make_maze();
    maze.symplectic = false; // wave-4 adiabatic CG+ASPC twin
    cpu::Context ctx;
    maze.build({ctx, f});
    for (int s = 0; s < 4; ++s)
        maze.launch({ctx, f, 0, s});
    EXPECT_TRUE(maze.solver().last_used_predictor());
    EXPECT_LE(maze.solver().neutrality_residual(), 1e-8);
    EXPECT_LE(maze.max_dQ(), 1e-8); // static frame → continuous charges
}

TEST(ConstPotMaZeSymplectic, BuildValidatesParameters) {
    auto f = plate_frame(2, 2);
    cpu::Context ctx;
    {
        auto maze = make_maze();
        maze.maze_tol = 0.0;
        EXPECT_THROW(maze.build({ctx, f}), std::invalid_argument);
    }
    {
        auto maze = make_maze();
        maze.max_maze_iters = 0;
        EXPECT_THROW(maze.build({ctx, f}), std::invalid_argument);
    }
}

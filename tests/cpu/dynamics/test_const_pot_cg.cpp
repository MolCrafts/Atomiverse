/// @file tests/cpu/dynamics/test_const_pot_cg.cpp
/// @brief Unit tests for ConstPotCG (single methods; no e2e MD).

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/const_pot_cg.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

cpu::Frame madelung_primary() {
    cpu::Frame f(4);
    const double z[4] = {8.0, 12.0, 9.0, 11.0};
    const double q0[4] = {0.0, 0.0, 0.5, -0.5};
    const int typ[4] = {1, 2, 3, 3};
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = 0.0;
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_charge)[i] = q0[i];
        f.span(kAtom_type)[i] = typ[i];
        f.span(kAtom_mass)[i] = (typ[i] < 3) ? 196.97 : 1.0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::orthorhombic(1.0, 1.0, 20.0);
    return f;
}

/// nx×ny sites per wall, two electrolyte ions on axis.
cpu::Frame multi_site_plate(int nx, int ny) {
    const int n_wall = nx * ny;
    const int n = 2 * n_wall + 2;
    cpu::Frame f(n);
    const double Lx = static_cast<double>(nx);
    const double Ly = static_cast<double>(ny);
    const double Lz = 20.0;
    const double z_bot = 8.0;
    const double z_top = 12.0;
    int a = 0;
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            const double xx = (ix + 0.5) * (Lx / nx);
            const double yy = (iy + 0.5) * (Ly / ny);
            f.span(kAtom_x)[a] = xx;
            f.span(kAtom_y)[a] = yy;
            f.span(kAtom_z)[a] = z_bot;
            f.span(kAtom_charge)[a] = 0.0;
            f.span(kAtom_type)[a] = 1;
            f.span(kAtom_mass)[a] = 196.97;
            f.span(kAtom_tag)[a] = a + 1;
            ++a;
        }
    }
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            const double xx = (ix + 0.5) * (Lx / nx);
            const double yy = (iy + 0.5) * (Ly / ny);
            f.span(kAtom_x)[a] = xx;
            f.span(kAtom_y)[a] = yy;
            f.span(kAtom_z)[a] = z_top;
            f.span(kAtom_charge)[a] = 0.0;
            f.span(kAtom_type)[a] = 2;
            f.span(kAtom_mass)[a] = 196.97;
            f.span(kAtom_tag)[a] = a + 1;
            ++a;
        }
    }
    // Electrolyte ±0.5 near mid-gap.
    f.span(kAtom_x)[a] = 0.5 * Lx;
    f.span(kAtom_y)[a] = 0.5 * Ly;
    f.span(kAtom_z)[a] = 9.0;
    f.span(kAtom_charge)[a] = 0.5;
    f.span(kAtom_type)[a] = 3;
    f.span(kAtom_mass)[a] = 1.0;
    f.span(kAtom_tag)[a] = a + 1;
    ++a;
    f.span(kAtom_x)[a] = 0.5 * Lx;
    f.span(kAtom_y)[a] = 0.5 * Ly;
    f.span(kAtom_z)[a] = 11.0;
    f.span(kAtom_charge)[a] = -0.5;
    f.span(kAtom_type)[a] = 3;
    f.span(kAtom_mass)[a] = 1.0;
    f.span(kAtom_tag)[a] = a + 1;

    f.box = cpu::Box::orthorhombic(Lx, Ly, Lz);
    return f;
}

void run_once_cg(cpu::md::ConstPotCG& conp, cpu::Frame& f) {
    cpu::Context ctx;
    conp.build({ctx, f});
    conp.launch({ctx, f, /*stage=*/0, /*step=*/0});
}

void run_once_bo(cpu::md::ConstPotBO& conp, cpu::Frame& f) {
    cpu::Context ctx;
    conp.build({ctx, f});
    conp.launch({ctx, f, /*stage=*/0, /*step=*/0});
}

} // namespace

TEST(ConstPotCG, NameAndNoConfigFactory) {
    cpu::md::ConstPotCG conp(2.0);
    EXPECT_EQ(conp.name(), "const_pot_cg");
    conp.add_electrode(1, 0.0).add_electrode(2, 1.0);
    EXPECT_DOUBLE_EQ(conp.delta_psi(), 1.0);
}

TEST(ConstPotCG, MatchesDenseBoMadelungPrimary) {
    constexpr double kQTopBo = 0.27175356;
    auto f = madelung_primary();
    cpu::md::ConstPotCG conp(2.0);
    conp.add_electrode(1, 0.0);
    conp.add_electrode(2, 1.0);
    conp.electroneutrality = true;
    conp.xy_image_range = 0;
    conp.cg_tol = 1e-14;
    run_once_cg(conp, f);
    EXPECT_LE(conp.neutrality_residual(), 1e-10);
    EXPECT_NEAR(conp.total_charge(1), kQTopBo, 1e-6);
    EXPECT_NEAR(conp.total_charge(0), -kQTopBo, 1e-6);
    EXPECT_GT(conp.cg_iterations(), 0);
}

TEST(ConstPotCG, MatchesDenseBoSitewise) {
    auto f_cg = madelung_primary();
    auto f_bo = madelung_primary();

    cpu::md::ConstPotCG cg(2.0);
    cg.add_electrode(1, 0.0).add_electrode(2, 1.0);
    cg.xy_image_range = 0;
    cg.cg_tol = 1e-14;
    run_once_cg(cg, f_cg);

    cpu::md::ConstPotBO bo(2.0);
    bo.add_electrode(1, 0.0).add_electrode(2, 1.0);
    bo.xy_image_range = 0;
    run_once_bo(bo, f_bo);

    EXPECT_NEAR(cg.total_charge(0), bo.total_charge(0), 1e-9);
    EXPECT_NEAR(cg.total_charge(1), bo.total_charge(1), 1e-9);
    EXPECT_NEAR(static_cast<double>(f_cg.span(kAtom_charge)[0]),
                static_cast<double>(f_bo.span(kAtom_charge)[0]), 1e-9);
    EXPECT_NEAR(static_cast<double>(f_cg.span(kAtom_charge)[1]),
                static_cast<double>(f_bo.span(kAtom_charge)[1]), 1e-9);
}

TEST(ConstPotCG, LargePlateMatchesDenseBo) {
    constexpr int nx = 3, ny = 3; // 18 electrode sites
    auto f_cg = multi_site_plate(nx, ny);
    auto f_bo = multi_site_plate(nx, ny);

    cpu::md::ConstPotCG cg(2.0);
    cg.add_electrode(1, 0.0).add_electrode(2, 1.0);
    cg.xy_image_range = 0;
    cg.cg_tol = 1e-14;
    run_once_cg(cg, f_cg);

    cpu::md::ConstPotBO bo(2.0);
    bo.add_electrode(1, 0.0).add_electrode(2, 1.0);
    bo.xy_image_range = 0;
    run_once_bo(bo, f_bo);

    EXPECT_EQ(cg.n_electrode(), 18);
    EXPECT_LE(cg.neutrality_residual(), 1e-10);
    EXPECT_NEAR(cg.total_charge(0), bo.total_charge(0), 1e-8);
    EXPECT_NEAR(cg.total_charge(1), bo.total_charge(1), 1e-8);

    double max_dq = 0.0;
    const int n_wall = nx * ny;
    for (int i = 0; i < 2 * n_wall; ++i) {
        max_dq = std::max(
            max_dq,
            std::abs(static_cast<double>(f_cg.span(kAtom_charge)[i])
                     - static_cast<double>(f_bo.span(kAtom_charge)[i])));
    }
    EXPECT_LE(max_dq, 1e-8);
}

TEST(ConstPotCG, ElectrolyteChargesUntouched) {
    auto f = madelung_primary();
    cpu::md::ConstPotCG conp(2.0);
    conp.add_electrode(1, 0.0).add_electrode(2, 1.0);
    run_once_cg(conp, f);
    EXPECT_DOUBLE_EQ(static_cast<double>(f.span(kAtom_charge)[2]), 0.5);
    EXPECT_DOUBLE_EQ(static_cast<double>(f.span(kAtom_charge)[3]), -0.5);
}

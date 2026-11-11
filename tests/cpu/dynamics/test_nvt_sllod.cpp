/// @file tests/cpu/dynamics/test_nvt_sllod.cpp
/// @brief `cpu::md::NvtSllod`: LAMMPS FixNH chain order with the carried
///        peculiar temperature and the `flip yes` box flip (component level;
///        the LJ run across a flip is `test_nvt_sllod_flip_lj.cpp`).
#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/dynamics_math.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/sllod_flow.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/nvt_sllod.h"
#include "atomiverse/cpu/frame/frame.h"
#include "cpu/dynamics/lammps_nhc_reference.h"

using namespace ATV;

namespace {

constexpr int kN = 6;

cpu::Frame small_input(const Mat3& cell) {
    cpu::Frame f(kN);
    const double x[kN] = {0.5, 3.0, 9.9, 5.0, 7.5, 2.0};
    const double y[kN] = {8.95, 2.0, 4.0, 0.05, 6.0, 7.0};
    const double z[kN] = {1.0, 7.9, 3.0, 4.0, 0.1, 5.5};
    const double vx[kN] = {0.01, -0.02, 0.15, 0.0, 0.03, -0.01};
    const double vy[kN] = {0.12, 0.01, -0.02, -0.1, 0.0, 0.02};
    const double vz[kN] = {0.0, 0.14, 0.01, -0.03, -0.12, 0.0};
    for (int i = 0; i < kN; ++i) {
        f.span(kAtom_x)[i] = x[i];
        f.span(kAtom_y)[i] = y[i];
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_vx)[i] = vx[i];
        f.span(kAtom_vy)[i] = vy[i];
        f.span(kAtom_vz)[i] = vz[i];
        f.span(kAtom_mass)[i] = 1.0 + 0.5 * i;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box(cell);
    return f;
}

/// Flow of the live cell (h_rate is constant for ERATE).
md::SllodFlow live_flow(const cpu::Frame& f, const double* h0, double rate,
                        int dir) {
    md::SllodFlow flow{};
    const auto& c = f.box.cell();
    const double h[6] = {c(0, 0), c(1, 1), c(2, 2), c(1, 2), c(0, 2), c(0, 1)};
    for (int i = 0; i < 6; ++i)
        flow.h[i] = h[i];
    md::sllod_erate_rates(h0, rate, dir, flow.h_rate, flow.h_ratelo);
    EXPECT_TRUE(md::sllod_refresh(flow));
    return flow;
}

double peculiar_ke(const cpu::Frame& f, const md::SllodFlow& flow) {
    double ke = 0;
    for (int i = 0; i < kN; ++i) {
        double bx, by, bz;
        md::sllod_deform_bias(f.span(kAtom_x)[i], f.span(kAtom_y)[i],
                              f.span(kAtom_z)[i], flow.h_inv, flow.h_rate,
                              flow.h_ratelo, bx, by, bz);
        const double px = f.span(kAtom_vx)[i] - bx;
        const double py = f.span(kAtom_vy)[i] - by;
        const double pz = f.span(kAtom_vz)[i] - bz;
        ke += f.span(kAtom_mass)[i] * (px * px + py * py + pz * pz);
    }
    return 0.5 * units::UnitSystem::lj().mvv2e * ke;
}

void launch(cpu::md::NvtSllod& sllod, cpu::Context& ctx, cpu::Frame& f,
            md::Phase phase, BigInt step) {
    sllod.launch({ctx, f, slot_of(phase), step});
}

/// r + H image, one axis.
double unwrapped(const cpu::Frame& f, int i, int a) {
    double u = a == 0   ? f.span(kAtom_x)[i]
               : a == 1 ? f.span(kAtom_y)[i]
                        : f.span(kAtom_z)[i];
    for (int b = 0; b < 3; ++b)
        u +=
            f.box.cell()(a, b) * static_cast<double>(f.span(kAtom_image)(i, b));
    return u;
}

} // namespace

TEST(CpuNvtSllod, RejectsAnEmptyChain) {
    EXPECT_THROW(cpu::md::NvtSllod(300, 100, 1, 0.01, 5, 0),
                 std::invalid_argument);
    EXPECT_NO_THROW(cpu::md::NvtSllod(300, 100, 1, 0.01, 5));
}

TEST(CpuNvtSllod, ChainFollowsFixNhOrderWithTheCarriedTemperature) {
    // Edge flow along x, so the end_of_step cell change makes a fresh
    // initial_integrate measurement differ from LAMMPS's carried t_current.
    constexpr int kChain = 3;
    const double T = 300, tau = 20, dt = 1, rate = 0.02;
    const int dir = md::kFlowX;
    cpu::Frame f = small_input(Mat3::diagonal(10.0, 9.0, 8.0));
    const double h0[6] = {10, 9, 8, 0, 0, 0};
    cpu::Context ctx;
    ctx.md.n_dof_global = md::degrees_of_freedom(kN);
    const double ndof = ctx.md.n_dof_global;
    const double kT = units::UnitSystem::lj().boltz * T;
    cpu::md::NvtSllod sllod(T, tau, dt, rate, dir, kChain);
    sllod.build({ctx, f});
    atv_test::LammpsNhc ref(kChain, ndof, kT, tau, dt);

    // FixNH::setup measures after the kick, so the input velocities are
    // the peculiar ones.
    double ke = 0;
    for (int i = 0; i < kN; ++i)
        ke += 0.5 * units::UnitSystem::lj().mvv2e * f.span(kAtom_mass)[i]
              * (std::pow(f.span(kAtom_vx)[i], 2)
                 + std::pow(f.span(kAtom_vy)[i], 2)
                 + std::pow(f.span(kAtom_vz)[i], 2));
    double max_gap = 0;
    for (BigInt step = 0; step < 6; ++step) {
        if (step > 0) {
            const double fresh = peculiar_ke(f, live_flow(f, h0, rate, dir));
            max_gap = std::max(max_gap, std::abs(fresh - ke) / ke);
        }
        ref.integrate(ke); // initial_integrate: carried (setup on step 0)
        launch(sllod, ctx, f, md::Phase::initial_integrate, step);
        launch(sllod, ctx, f, md::Phase::pre_exchange, step);
        ke = peculiar_ke(f, live_flow(f, h0, rate, dir)); // zero force kick
        ref.integrate(ke); // final_integrate: fresh compute_scalar
        launch(sllod, ctx, f, md::Phase::final_integrate, step);
        launch(sllod, ctx, f, md::Phase::end_of_step, step);
    }
    EXPECT_GT(max_gap, 1e-6)
        << "the edge flow must make the carry differ from a fresh measure";
    const auto chain = sllod.thermostat_chain();
    ASSERT_EQ(chain.eta.size(), static_cast<std::size_t>(kChain));
    for (int k = 0; k < kChain; ++k) {
        EXPECT_NEAR(chain.eta[k], ref.eta[k],
                    1e-12 * (1 + std::abs(ref.eta[k])))
            << "link " << k;
        EXPECT_NEAR(chain.eta_dot[k], ref.eta_dot[k],
                    1e-12 * (1e-6 + std::abs(ref.eta_dot[k])))
            << "link " << k;
    }
    EXPECT_NE(chain.eta_dot[2], 0.0);
}

TEST(CpuNvtSllod, FlipRequestsReneighborAndKeepsUnwrappedState) {
    // xy shear from 4.8 on lx = 10: tilt + 0.09 per step passes 5 at step 3.
    Mat3 cell = Mat3::diagonal(10.0, 9.0, 8.0);
    cell(0, 1) = 4.8;
    cpu::Frame f = small_input(cell);
    const double rate = 0.01;
    const double h0[6] = {10, 9, 8, 0, 0, 4.8};
    cpu::Context ctx;
    ctx.md.n_dof_global = md::degrees_of_freedom(kN);
    cpu::md::NvtSllod sllod(300.0, 100.0, 1.0, rate, md::kFlowXY);
    sllod.build({ctx, f});
    int flipped_step = -1;
    for (BigInt step = 0; step < 6 && flipped_step < 0; ++step) {
        launch(sllod, ctx, f, md::Phase::initial_integrate, step);
        if (ctx.md.reneighbor_requested) {
            // Runtime consumes the request before pre_exchange.
            ctx.md.reneighbor_requested = false;
            double u[kN][3], w[kN];
            md::SllodFlow flow = live_flow(f, h0, rate, md::kFlowXY);
            for (int i = 0; i < kN; ++i) {
                for (int a = 0; a < 3; ++a)
                    u[i][a] = unwrapped(f, i, a);
                // Unwrapped lab velocity: v + Hdot image (xy shear: x only).
                w[i] = f.span(kAtom_vx)[i]
                       + flow.h_rate[md::kFlowXY]
                             * static_cast<double>(f.span(kAtom_image)(i, 1));
            }
            const double before = f.box.cell()(0, 1);
            EXPECT_GT(before, 5.0);
            launch(sllod, ctx, f, md::Phase::pre_exchange, step);
            EXPECT_NEAR(f.box.cell()(0, 1), before - 10.0, 1e-12);
            for (int i = 0; i < kN; ++i) {
                for (int a = 0; a < 3; ++a)
                    EXPECT_NEAR(unwrapped(f, i, a), u[i][a], 1e-12)
                        << "row " << i << " axis " << a;
                EXPECT_NEAR(
                    f.span(kAtom_vx)[i]
                        + flow.h_rate[md::kFlowXY]
                              * static_cast<double>(f.span(kAtom_image)(i, 1)),
                    w[i], 1e-12);
                // In the flipped cell.
                const auto s =
                    f.box.to_fractional({f.span(kAtom_x)[i], f.span(kAtom_y)[i],
                                         f.span(kAtom_z)[i]});
                for (int a = 0; a < 3; ++a) {
                    EXPECT_GE(s[a], 0.0) << "row " << i;
                    EXPECT_LT(s[a], 1.0) << "row " << i;
                }
            }
            flipped_step = static_cast<int>(step);
        }
        launch(sllod, ctx, f, md::Phase::final_integrate, step);
        launch(sllod, ctx, f, md::Phase::end_of_step, step);
    }
    // Decided at the end of step 2 (third end_of_step), run at step 3.
    EXPECT_EQ(flipped_step, 3);
    // Later steps continue the linear tilt shifted by one box length.
    EXPECT_NEAR(f.box.cell()(0, 1), 4.8 + 0.09 * 4 - 10.0, 1e-12);
    EXPECT_FALSE(ctx.md.reneighbor_requested);
}

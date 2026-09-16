/// @file tests/cpu/electrodes/test_charge_aspc.cpp
/// @brief Unit tests for ChargeASPC + ConstPotBO CG warm-start (single methods).

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/electrodes/charge_aspc.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// 2 electrode sites + 2 electrolyte (legacy madelung unit tests).
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

/// Larger walls so CG iteration counts can differ (cold vs ASPC).
cpu::Frame multi_site_plate(int n_side = 3) {
    const int n_per = n_side * n_side;
    const int n_ions = 4;
    const int N = 2 * n_per + n_ions;
    cpu::Frame f(N);
    const double Lxy = 6.0;
    int idx = 0;
    for (int wall = 0; wall < 2; ++wall) {
        const double z = (wall == 0) ? 4.0 : 16.0;
        const int typ = (wall == 0) ? 1 : 2;
        for (int ix = 0; ix < n_side; ++ix)
            for (int iy = 0; iy < n_side; ++iy) {
                f.span(kAtom_x)[idx] = (ix + 0.5) * (Lxy / n_side);
                f.span(kAtom_y)[idx] = (iy + 0.5) * (Lxy / n_side);
                f.span(kAtom_z)[idx] = z;
                f.span(kAtom_type)[idx] = typ;
                f.span(kAtom_charge)[idx] = 0.0;
                f.span(kAtom_mass)[idx] = 12.0;
                f.span(kAtom_tag)[idx] = idx + 1;
                ++idx;
            }
    }
    // 4 ions staggered
    for (int i = 0; i < n_ions; ++i) {
        const bool pos = (i % 2 == 0);
        f.span(kAtom_x)[idx] = (0.5 + (i % 2) * 0.4) * Lxy;
        f.span(kAtom_y)[idx] = (0.3 + (i / 2) * 0.4) * Lxy;
        f.span(kAtom_z)[idx] = pos ? 8.0 : 12.0;
        f.span(kAtom_type)[idx] = 3;
        f.span(kAtom_charge)[idx] = pos ? 0.5 : -0.5;
        f.span(kAtom_mass)[idx] = 1.0;
        f.span(kAtom_tag)[idx] = idx + 1;
        ++idx;
    }
    f.box = cpu::Box::orthorhombic(Lxy, Lxy, 20.0);
    return f;
}

void run_once(cpu::md::ConstPotBO& conp, cpu::Frame& f) {
    cpu::Context ctx;
    conp.build({ctx, f});
    conp.launch({ctx, f, /*stage=*/0, /*step=*/0});
}

} // namespace

TEST(ChargeASPC, Order2CoefficientsKolafa) {
    // Kolafa ASPC K=2: B = {5/2, -2, 1/2} newest → oldest.
    cpu::electrodes::ChargeASPC aspc(2);
    EXPECT_EQ(aspc.order(), 2);
    EXPECT_DOUBLE_EQ(aspc.coefficient(0), 0.5);
    EXPECT_DOUBLE_EQ(aspc.coefficient(1), -2.0);
    EXPECT_DOUBLE_EQ(aspc.coefficient(2), 2.5);
}

TEST(ChargeASPC, Order5CoefficientsMatchMetalWalls) {
    // MetalWalls c8ad175d src/electrode_charge.F90:572-577 hard-codes exactly
    // these six constants as the electrode-charge CG warm start.
    cpu::electrodes::ChargeASPC aspc(5);
    EXPECT_DOUBLE_EQ(aspc.coefficient(5), 22.0 / 7.0);   // B1
    EXPECT_DOUBLE_EQ(aspc.coefficient(4), -55.0 / 14.0); // B2
    EXPECT_DOUBLE_EQ(aspc.coefficient(3), 55.0 / 21.0);  // B3
    EXPECT_DOUBLE_EQ(aspc.coefficient(2), -22.0 / 21.0); // B4
    EXPECT_DOUBLE_EQ(aspc.coefficient(1), 5.0 / 21.0);   // B5
    EXPECT_DOUBLE_EQ(aspc.coefficient(0), -1.0 / 42.0);  // B6
}

TEST(ChargeASPC, CoefficientsSumToOne) {
    for (int K = 0; K <= 6; ++K) {
        cpu::electrodes::ChargeASPC aspc(K);
        double s = 0.0;
        for (int k = 0; k <= K; ++k)
            s += aspc.coefficient(k);
        EXPECT_NEAR(s, 1.0, 1e-14) << "K=" << K;
    }
}

TEST(ChargeASPC, LinearHistoryPredictsNext) {
    // ASPC is exact through degree 1 (sum B_j = 1, sum j B_j = 0).
    cpu::electrodes::ChargeASPC aspc(2);
    for (int n = 0; n <= 2; ++n)
        aspc.record(std::vector<double>{static_cast<double>(n)});
    std::vector<double> pred;
    ASSERT_TRUE(aspc.predict(pred));
    ASSERT_EQ(pred.size(), 1u);
    EXPECT_NEAR(pred[0], 3.0, 1e-12);
}

TEST(ChargeASPC, QuadraticHistoryIsDampedNotExact) {
    // Kolafa's damping: on Q_k = k^2 the ASPC K=2 predictor returns 8, not the
    // Lagrange value 9. This is the discriminator between ASPC and plain
    // polynomial extrapolation — the scheme Kolafa 2004 shows to be unstable.
    cpu::electrodes::ChargeASPC aspc(2);
    for (int n = 0; n <= 2; ++n)
        aspc.record(std::vector<double>{static_cast<double>(n * n)});
    std::vector<double> pred;
    ASSERT_TRUE(aspc.predict(pred));
    EXPECT_NEAR(pred[0], 8.0, 1e-12);
}

TEST(ChargeASPC, InsufficientHistoryReturnsFalse) {
    cpu::electrodes::ChargeASPC aspc(2);
    aspc.record(std::vector<double>{1.0, 2.0});
    std::vector<double> pred;
    EXPECT_FALSE(aspc.predict(pred));
}

TEST(ChargeASPC, CorrectorBlend) {
    cpu::electrodes::ChargeASPC aspc(2);
    std::vector<double> scf{1.0}, pred{0.0}, corr;
    aspc.correct(scf, pred, corr);
    EXPECT_NEAR(corr[0], 4.0 / 7.0, 1e-15);
}

TEST(ConstPotBO, CgMatchesDense) {
    auto f_dense = madelung_primary();
    auto f_cg = madelung_primary();

    cpu::md::ConstPotBO dense(2.0);
    dense.add_electrode(1, 0.0).add_electrode(2, 1.0);
    dense.electroneutrality = true;
    dense.xy_image_range = 0;
    dense.use_cg = false;
    run_once(dense, f_dense);

    cpu::md::ConstPotBO cg(2.0);
    cg.add_electrode(1, 0.0).add_electrode(2, 1.0);
    cg.electroneutrality = true;
    cg.xy_image_range = 0;
    cg.use_cg = true;
    cg.cg_tol = 1e-12;
    cg.predictor_order = 0;
    run_once(cg, f_cg);

    EXPECT_NEAR(cg.total_charge(0), dense.total_charge(0), 1e-8);
    EXPECT_NEAR(cg.total_charge(1), dense.total_charge(1), 1e-8);
    EXPECT_LE(cg.neutrality_residual(), 1e-8);
    EXPECT_GT(cg.last_iters(), 0);
}

TEST(ConstPotBO, PredictorReducesCgItersOnMovingElectrolyte) {
    // Multi-site walls: cold CG restarts from 0; ASPC K=2 warm-starts.
    auto make = []() {
        cpu::md::ConstPotBO c(2.0);
        c.add_electrode(1, 0.0);
        c.add_electrode(2, 1.0);
        c.electroneutrality = true;
        c.xy_image_range = 0;
        c.use_cg = true;
        c.cg_tol = 1e-10;
        c.max_cg_iters = 500;
        return c;
    };

    auto cold = make();
    cold.predictor_order = 0;
    auto warm = make();
    warm.predictor_order = 2;

    auto f_cold = multi_site_plate(3);
    auto f_warm = multi_site_plate(3);
    cpu::Context ctx;
    cold.build({ctx, f_cold});
    warm.build({ctx, f_warm});

    // Ion local rows: last 4 atoms
    const int n = f_cold.nlocal();
    const int ion0 = n - 4;

    long sum_cold = 0, sum_warm = 0;
    int n_post = 0;
    constexpr int n_steps = 16;
    for (int s = 0; s < n_steps; ++s) {
        const double dz = 0.03 * (s + 1);
        for (auto* f : {&f_cold, &f_warm}) {
            for (int k = 0; k < 4; ++k) {
                const double z0 = (k % 2 == 0) ? 8.0 : 12.0;
                const double sign = (k % 2 == 0) ? 1.0 : -0.5;
                f->span(kAtom_z)[ion0 + k] = z0 + sign * dz;
            }
        }
        cold.launch({ctx, f_cold, 0, s});
        warm.launch({ctx, f_warm, 0, s});
        if (s >= 3) {
            sum_cold += cold.last_iters();
            sum_warm += warm.last_iters();
            ++n_post;
            EXPECT_TRUE(warm.last_used_predictor());
        }
    }
    ASSERT_GT(n_post, 0);
    const double mean_cold = static_cast<double>(sum_cold) / n_post;
    const double mean_warm = static_cast<double>(sum_warm) / n_post;
    EXPECT_LT(mean_warm, mean_cold)
        << "mean_warm=" << mean_warm << " mean_cold=" << mean_cold;
    EXPECT_LE(warm.neutrality_residual(), 1e-8);
}

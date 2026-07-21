/// @file tests/cpu/pw/test_pw_forces.cpp
/// @brief Unit gates for plane-wave ionic forces (FD oracle + analytic).

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/pw/pw_evaluator.h"
#include "atomiverse/cpu/pw/pw_solver.h"
#include "atomiverse/facade/stage.h"

using namespace ATV;
using namespace ATV::cpu::pw;

namespace {

/// Single H atom in a cubic cell — force must vanish by lattice symmetry.
cpu::Frame make_h_cubic(double L) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.simbox = cpu::Box::cubic(L);
    return f;
}

PWConfig cheap_cfg() {
    PWConfig cfg;
    cfg.ecut_ry = 5.0;
    cfg.k_mesh = 1;
    cfg.tol = 1e-4;
    cfg.max_iter = 6;
    cfg.mix_beta = 0.5;
    return cfg;
}

} // namespace

TEST(PWForces, RejectsBadStep) {
    auto f = make_h_cubic(6.0);
    EXPECT_THROW(pw_forces_fd(f, cheap_cfg(), 0.0), std::invalid_argument);
    EXPECT_THROW(pw_forces_fd(f, cheap_cfg(), -1e-3), std::invalid_argument);
}

TEST(PWForces, AnalyticRejectsEmptyFrame) {
    cpu::Frame f(0);
    EXPECT_THROW(pw_forces_analytic(f, cheap_cfg()), std::invalid_argument);
}

TEST(PWForces, SingleAtomForceVanishesFd) {
    // 6 SCF solves; cheap settings keep this under ~30 s on a laptop.
    auto f = make_h_cubic(6.0);
    const auto F = pw_forces_fd(f, cheap_cfg(), 5e-3);
    ASSERT_EQ(F.size(), 1u);
    const double mag =
        std::sqrt(F[0][0] * F[0][0] + F[0][1] * F[0][1] + F[0][2] * F[0][2]);
    // Toy PP + loose SCF: force is numerical noise around zero.
    EXPECT_LT(mag, 0.5) << " |F|=" << mag;
}

TEST(PWForces, AnalyticSingleAtomForceVanishes) {
    auto f = make_h_cubic(6.0);
    const auto F = pw_forces_analytic(f, cheap_cfg());
    ASSERT_EQ(F.size(), 1u);
    const double mag =
        std::sqrt(F[0][0] * F[0][0] + F[0][1] * F[0][1] + F[0][2] * F[0][2]);
    EXPECT_LT(mag, 0.5) << " |F_analytic|=" << mag;
}

TEST(PWForces, EvaluatorWritesAnalyticForcesWhenRequested) {
    auto frame = make_h_cubic(6.0);
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(kAtom_fx)[i] = 99.0;
        frame.span(kAtom_fy)[i] = 99.0;
        frame.span(kAtom_fz)[i] = 99.0;
    }

    PWEvaluator pwe{cheap_cfg()};
    pwe.with_gradient(true);

    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(pwe);
    eval.enable_timing(false);
    eval.build();
    eval.run(frame);

    EXPECT_TRUE(std::isfinite(frame.meta.get(kEnergy)));
    EXPECT_NE(frame.span(kAtom_fx)[0], AccReal(99.0));
    const double mag = std::sqrt(static_cast<double>(
        frame.span(kAtom_fx)[0] * frame.span(kAtom_fx)[0]
        + frame.span(kAtom_fy)[0] * frame.span(kAtom_fy)[0]
        + frame.span(kAtom_fz)[0] * frame.span(kAtom_fz)[0]));
    EXPECT_LT(mag, 0.5);
}

TEST(PWForces, EvaluatorEnergyOnlySkipsForces) {
    auto frame = make_h_cubic(6.0);
    PWEvaluator pwe{cheap_cfg()};
    EXPECT_FALSE(pwe.compute_forces());

    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(pwe);
    eval.enable_timing(false);
    eval.build();
    eval.run(frame);
    EXPECT_TRUE(std::isfinite(frame.meta.get(kEnergy)));
}

TEST(PWForces, AnalyticUpfSiDimerFiniteForces) {
    // Si UPF path exercises local + KB nonlocal. Cheap ecut/k-mesh.
    // Pin: forces finite, |sum F| soft, and |F| not identically zero when
    // atoms are slightly off the ideal high-symmetry relative placement.
    namespace fs = std::filesystem;
    const char* candidates[] = {
        "benchmarks/pw/si/validation/reference/Si_lda_nc.upf",
        "python/examples/pseudopotentials/Si_nc.upf",
    };
    std::string upf;
    for (const char* c : candidates) {
        if (fs::exists(c)) {
            upf = c;
            break;
        }
    }
    if (upf.empty()) {
        GTEST_SKIP() << "no Si UPF found in tree";
    }

    const double a = 5.43;
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 14;
    f.span(kAtom_type)[1] = 14;
    f.span(kAtom_mass)[0] = 28.0855;
    f.span(kAtom_mass)[1] = 28.0855;
    // Slightly stretched bond along body diagonal of FCC primitive cell.
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = a / 4.0 + 0.05;
    f.span(kAtom_y)[1] = a / 4.0;
    f.span(kAtom_z)[1] = a / 4.0;
    double h = a / 2.0;
    Mat3 cell;
    cell(0, 0) = 0;
    cell(0, 1) = h;
    cell(0, 2) = h;
    cell(1, 0) = h;
    cell(1, 1) = 0;
    cell(1, 2) = h;
    cell(2, 0) = h;
    cell(2, 1) = h;
    cell(2, 2) = 0;
    f.simbox.set_cell(cell);

    PWConfig cfg;
    cfg.upf_path = upf;
    cfg.ecut_ry = 8.0;
    cfg.k_mesh = 1;
    cfg.tol = 1e-4;
    cfg.max_iter = 20;
    cfg.mix_beta = 0.5;

    const auto Fa = pw_forces_analytic(f, cfg);
    ASSERT_EQ(Fa.size(), 2u);
    double max_abs = 0;
    for (int ia = 0; ia < 2; ++ia) {
        for (int k = 0; k < 3; ++k) {
            const double v =
                Fa[static_cast<std::size_t>(ia)][static_cast<std::size_t>(k)];
            EXPECT_TRUE(std::isfinite(v));
            max_abs = std::max(max_abs, std::abs(v));
        }
    }
    const double sum_fx = Fa[0][0] + Fa[1][0];
    const double sum_fy = Fa[0][1] + Fa[1][1];
    const double sum_fz = Fa[0][2] + Fa[1][2];
    const double sum_mag =
        std::sqrt(sum_fx * sum_fx + sum_fy * sum_fy + sum_fz * sum_fz);
    EXPECT_GT(max_abs, 1e-6) << "expected nonzero force on stretched Si dimer";
    EXPECT_LT(sum_mag, 2.0) << " |sum F|=" << sum_mag;
}

TEST(PWForces, AnalyticMatchesFdToyDimer) {
    // Two H in a large box — toy PP, cheap ecut. Compare analytic vs FD.
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_x)[0] = -0.4;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.5;
    f.span(kAtom_y)[1] = 0.1;
    f.span(kAtom_z)[1] = -0.05;
    f.simbox = cpu::Box::cubic(8.0);

    PWConfig cfg = cheap_cfg();
    cfg.max_iter = 12;
    cfg.tol = 1e-5;

    const auto Fa = pw_forces_analytic(f, cfg);
    const auto Ff = pw_forces_fd(f, cfg, 2e-3);
    ASSERT_EQ(Fa.size(), 2u);
    ASSERT_EQ(Ff.size(), 2u);

    double max_diff = 0.0;
    double sum_fx = 0.0, sum_fy = 0.0, sum_fz = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            max_diff = std::max(
                max_diff,
                std::abs(
                    Fa[static_cast<std::size_t>(a)][static_cast<std::size_t>(k)]
                    - Ff[static_cast<std::size_t>(a)]
                        [static_cast<std::size_t>(k)]));
        }
        sum_fx += Fa[static_cast<std::size_t>(a)][0];
        sum_fy += Fa[static_cast<std::size_t>(a)][1];
        sum_fz += Fa[static_cast<std::size_t>(a)][2];
    }
    // Toy + short SCF: loose absolute gate (eV/Å).
    EXPECT_LT(max_diff, 0.8) << " max |F_a - F_fd|=" << max_diff;
    // Translational soft sum rule
    const double sum_mag =
        std::sqrt(sum_fx * sum_fx + sum_fy * sum_fy + sum_fz * sum_fz);
    EXPECT_LT(sum_mag, 0.5) << " |sum F|=" << sum_mag;
}

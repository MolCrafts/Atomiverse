/// @file tests/cpu/pw/test_pw_forces.cpp
/// @brief Unit gates for plane-wave ionic forces (FD oracle + analytic).

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_evaluator.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

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
    f.box = cpu::Box::cubic(L);
    return f;
}

PW cheap_cfg() {
    PW cfg;
    cfg.gth_name = "GTH-PBE-q1"; // analytic H, no file dependency
    cfg.ecut_ry = 4.0;
    cfg.k_mesh = 1;
    cfg.tol = 1e-4;
    cfg.max_iter = 6;
    cfg.mix_beta = 0.5;
    return cfg;
}

} // namespace

TEST(PWForces, RejectsBadStep) {
    auto f = make_h_cubic(4.5);
    EXPECT_THROW(pw_forces_fd(f, cheap_cfg(), 0.0), std::invalid_argument);
    EXPECT_THROW(pw_forces_fd(f, cheap_cfg(), -1e-3), std::invalid_argument);
}

TEST(PWForces, AnalyticRejectsEmptyFrame) {
    cpu::Frame f(0);
    EXPECT_THROW(pw_forces_analytic(f, cheap_cfg()), std::invalid_argument);
}

TEST(PWForces, SingleAtomForceVanishesFd) {
    // 6 SCF solves. A single ion at a cubic high-symmetry site has F = 0 by
    // lattice symmetry, exactly — the gate is the FD/SCF noise floor, not a
    // physical magnitude. With a real GTH potential the residual is ~1e-7
    // eV/Å; the old 0.5 eV/Å bound dated from the toy potential and was so
    // loose it would have accepted a completely wrong local force.
    auto f = make_h_cubic(4.5);
    const auto F = pw_forces_fd(f, cheap_cfg(), 5e-3);
    ASSERT_EQ(F.size(), 1u);
    const double mag =
        std::sqrt(F[0][0] * F[0][0] + F[0][1] * F[0][1] + F[0][2] * F[0][2]);
    EXPECT_LT(mag, 1e-5) << " |F|=" << mag;
}

TEST(PWForces, AnalyticSingleAtomForceVanishes) {
    // Same exact symmetry statement as the FD test above, on the analytic
    // path: the Hellmann–Feynman sum over G must cancel shell by shell.
    auto f = make_h_cubic(4.5);
    const auto F = pw_forces_analytic(f, cheap_cfg());
    ASSERT_EQ(F.size(), 1u);
    const double mag =
        std::sqrt(F[0][0] * F[0][0] + F[0][1] * F[0][1] + F[0][2] * F[0][2]);
    EXPECT_LT(mag, 1e-8) << " |F_analytic|=" << mag;
}

TEST(PWForces, EvaluatorWritesAnalyticForcesWhenRequested) {
    auto frame = make_h_cubic(4.5);
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
    EXPECT_LT(mag, 1e-8) << " |F_frame|=" << mag; // same symmetry zero
}

TEST(PWForces, EvaluatorEnergyOnlySkipsForces) {
    auto frame = make_h_cubic(4.5);
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
#ifdef ATV_SOURCE_DIR
        // Absolute: ctest's working directory is the build tree, so a
        // relative path here skipped this test instead of running it.
        ATV_SOURCE_DIR "/tests/data/pw/Si_lda_nc.upf",
#endif
        "tests/data/pw/Si_lda_nc.upf",
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
    f.box.set_cell(cell);

    PW cfg;
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

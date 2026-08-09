/// @file test_pw_si.cpp
/// @brief Integration tests for the explicit PW lifecycle.
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace ATV;
using namespace ATV::cpu::pw;

namespace {
cpu::Frame make_si_fcc(double a) {
    cpu::Frame f(2);
    f.span(ATV::kAtom_type)[0] = 14;
    f.span(ATV::kAtom_type)[1] = 14;
    f.span(ATV::kAtom_mass)[0] = 28.0855;
    f.span(ATV::kAtom_mass)[1] = 28.0855;
    f.span(ATV::kAtom_x)[0] = 0;
    f.span(ATV::kAtom_y)[0] = 0;
    f.span(ATV::kAtom_z)[0] = 0;
    f.span(ATV::kAtom_x)[1] = a / 4.0;
    f.span(ATV::kAtom_y)[1] = a / 4.0;
    f.span(ATV::kAtom_z)[1] = a / 4.0;
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
    return f;
}
} // namespace

TEST(PW, CompilesAndRuns) {
    auto frame = make_si_fcc(5.43);
    PW cfg;
    cfg.gth_name = "GTH-PBE-q4";
    cfg.ecut_ry = 8.0;
    cfg.k_mesh = 1;
    cfg.tol = 1e-4;
    cfg.max_iter = 6;
    PW solver = std::move(cfg);
    solver.build(frame);
    const auto& r = solver.run(frame);
    EXPECT_TRUE(std::isfinite(r.energy))
        << "Energy should be finite, got: " << r.energy;
}

TEST(PW, RunsWithoutCrash) {
    auto frame = make_si_fcc(5.43);
    PW cfg;
    cfg.gth_name = "GTH-PBE-q4";
    cfg.ecut_ry = 8.0;
    cfg.k_mesh = 1;
    cfg.tol = 1e-4;
    cfg.max_iter = 8;
    PW solver = std::move(cfg);
    solver.build(frame);
    const auto& r = solver.run(frame);
    EXPECT_TRUE(std::isfinite(r.energy))
        << "Energy should be finite, got: " << r.energy;
}

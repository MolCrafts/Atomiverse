/// @file tests/cpu/gpw/test_cpu_gpw_one_electron.cpp
/// @brief GPW one-electron integrals: McMurchie-Davidson S and T vs closed forms.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/one_electron.h"

using ATV::cpu::gpw::GridBasisFunction;
using ATV::cpu::gpw::GridPrimitive;
using ATV::cpu::gpw::kinetic;
using ATV::cpu::gpw::overlap;
using ATV::cpu::gpw::prim_kinetic;
using ATV::cpu::gpw::prim_overlap;

namespace {

GridPrimitive prim(double alpha, double cx, double cy, double cz, int lx,
                   int ly, int lz, double coeff = 1.0) {
    GridPrimitive p;
    p.alpha = alpha;
    p.center[0] = cx;
    p.center[1] = cy;
    p.center[2] = cz;
    p.l[0] = lx;
    p.l[1] = ly;
    p.l[2] = lz;
    p.coeff = coeff;
    return p;
}

double dist2(const GridPrimitive& a, const GridPrimitive& b) {
    double d = 0.0;
    for (int k = 0; k < 3; ++k)
        d += (a.center[k] - b.center[k]) * (a.center[k] - b.center[k]);
    return d;
}

} // namespace

// s-s overlap closed form.
TEST(GpwOneElectron, SsOverlap) {
    GridPrimitive a = prim(0.8, 0, 0, 0, 0, 0, 0);
    GridPrimitive b = prim(1.3, 1.1, -0.4, 0.6, 0, 0, 0);
    const double p = a.alpha + b.alpha;
    const double mu = a.alpha * b.alpha / p;
    const double expected =
        std::pow(M_PI / p, 1.5) * std::exp(-mu * dist2(a, b));
    EXPECT_NEAR(prim_overlap(a, b), expected, 1e-13 * std::abs(expected));
}

// s-s kinetic closed form: T = mu(3 - 2 mu R^2)(pi/p)^{3/2} exp(-mu R^2).
TEST(GpwOneElectron, SsKinetic) {
    GridPrimitive a = prim(0.8, 0, 0, 0, 0, 0, 0);
    GridPrimitive b = prim(1.3, 1.1, -0.4, 0.6, 0, 0, 0);
    const double p = a.alpha + b.alpha;
    const double mu = a.alpha * b.alpha / p;
    const double R2 = dist2(a, b);
    const double expected = mu * (3.0 - 2.0 * mu * R2) * std::pow(M_PI / p, 1.5)
                            * std::exp(-mu * R2);
    EXPECT_NEAR(prim_kinetic(a, b), expected, 1e-13 * std::abs(expected));
}

// Same-center p_x self-overlap: <x|x> = (1/(4 alpha)) (pi/(2 alpha))^{3/2}.
// (The physically correct second moment; a factor 1/(4 alpha), not 1/(2 alpha).)
TEST(GpwOneElectron, PxSelfOverlap) {
    const double alpha = 0.9;
    GridPrimitive px = prim(alpha, 0, 0, 0, 1, 0, 0);
    const double expected =
        (1.0 / (4.0 * alpha)) * std::pow(M_PI / (2.0 * alpha), 1.5);
    EXPECT_NEAR(prim_overlap(px, px), expected, 1e-13 * std::abs(expected));
    // Orthogonality: p_x and p_y on the same center do not overlap.
    GridPrimitive py = prim(alpha, 0, 0, 0, 0, 1, 0);
    EXPECT_NEAR(prim_overlap(px, py), 0.0, 1e-14);
}

// Symmetry of S and T.
TEST(GpwOneElectron, Symmetry) {
    GridPrimitive a = prim(0.7, 0.2, 0.0, 0.0, 1, 0, 0);
    GridPrimitive b = prim(1.1, -0.5, 0.3, 0.1, 0, 1, 1);
    EXPECT_NEAR(prim_overlap(a, b), prim_overlap(b, a), 1e-13);
    EXPECT_NEAR(prim_kinetic(a, b), prim_kinetic(b, a), 1e-13);
}

// Contracted overlap is the double sum over primitive pairs.
TEST(GpwOneElectron, ContractedIsSumOfPrimitives) {
    GridBasisFunction a = {prim(0.6, 0, 0, 0, 0, 0, 0, 0.4),
                           prim(1.8, 0, 0, 0, 0, 0, 0, 0.7)};
    GridBasisFunction b = {prim(0.9, 0.8, 0, 0, 0, 0, 0, 0.5)};
    double manual = 0.0;
    for (const auto& pa : a)
        for (const auto& pb : b)
            manual += prim_overlap(pa, pb);
    EXPECT_NEAR(overlap(a, b), manual, 1e-14 * std::max(1.0, std::abs(manual)));
    EXPECT_GT(overlap(a, a), 0.0);
    EXPECT_GT(kinetic(a, a), 0.0); // kinetic energy is positive
}

// d functions: finite and symmetric.
TEST(GpwOneElectron, DFunctionsFinite) {
    GridPrimitive dxx = prim(0.85, 0, 0, 0, 2, 0, 0);
    GridPrimitive dxy = prim(0.85, 0.3, 0.2, 0.0, 1, 1, 0);
    EXPECT_TRUE(std::isfinite(prim_overlap(dxx, dxy)));
    EXPECT_TRUE(std::isfinite(prim_kinetic(dxx, dxy)));
    EXPECT_NEAR(prim_overlap(dxx, dxy), prim_overlap(dxy, dxx), 1e-13);
    EXPECT_GT(prim_overlap(dxx, dxx), 0.0);
}

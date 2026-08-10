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

// Multi-prim pure-s contracted S/T (SIMD monom-pair path when na·nb ≥ batch).
// Closed form: S = Σ_pq c_p c_q (π/p)^{3/2} exp(-μ R²), T = μ(3−2μR²)S_pq.
TEST(GpwOneElectron, ContractedSsMatchesClosedForm) {
    // 4×4 monoms → 16 pairs (fills ≥1 AVX2/AVX-512 lane of doubles).
    GridBasisFunction a = {
        prim(0.30, 0, 0, 0, 0, 0, 0, 0.21), prim(0.80, 0, 0, 0, 0, 0, 0, 0.35),
        prim(1.50, 0, 0, 0, 0, 0, 0, 0.40), prim(3.20, 0, 0, 0, 0, 0, 0, 0.28)};
    GridBasisFunction b = {
        prim(0.40, 1.2, -0.3, 0.5, 0, 0, 0, 0.18),
        prim(0.95, 1.2, -0.3, 0.5, 0, 0, 0, 0.33),
        prim(1.90, 1.2, -0.3, 0.5, 0, 0, 0, 0.37),
        prim(4.00, 1.2, -0.3, 0.5, 0, 0, 0, 0.25)};
    double s_ref = 0.0, t_ref = 0.0;
    for (const auto& pa : a) {
        for (const auto& pb : b) {
            const double p = pa.alpha + pb.alpha;
            const double mu = pa.alpha * pb.alpha / p;
            const double R2 = dist2(pa, pb);
            const double S = pa.coeff * pb.coeff * std::pow(M_PI / p, 1.5)
                             * std::exp(-mu * R2);
            s_ref += S;
            t_ref += mu * (3.0 - 2.0 * mu * R2) * S;
        }
    }
    const double s = overlap(a, b);
    const double t = kinetic(a, b);
    EXPECT_NEAR(s, s_ref, 1e-13 * std::max(1.0, std::abs(s_ref)));
    EXPECT_NEAR(t, t_ref, 1e-13 * std::max(1.0, std::abs(t_ref)));
}

// Fused Γ periodic S+T matches the lattice sum of closed-form pure-s pairs.
// m large enough that live images ≥ native SIMD width on typical hosts.
TEST(GpwOneElectron, ContractedSTPeriodicFusedMatchesImageSum) {
    using ATV::cpu::gpw::contracted_ST_periodic;
    GridBasisFunction a = {
        prim(0.50, 0, 0, 0, 0, 0, 0, 0.40), prim(1.20, 0, 0, 0, 0, 0, 0, 0.60)};
    GridBasisFunction b = {prim(0.70, 0.4, 0.1, -0.2, 0, 0, 0, 0.55),
                           prim(1.80, 0.4, 0.1, -0.2, 0, 0, 0, 0.45)};
    // Slab-class: many images along z (m_z=4 → 9 live z slots × planar).
    const double L[3] = {6.0, 6.0, 4.0};
    const int m[3] = {2, 2, 4}; // 5×5×9 = 225 images before reach screen
    const double reach = 40.0;  // keep essentially all images live

    double s_fused = 0.0, t_fused = 0.0;
    contracted_ST_periodic(a, b, L, m, reach, s_fused, t_fused);

    double s_ref = 0.0, t_ref = 0.0;
    for (int tx = -m[0]; tx <= m[0]; ++tx)
        for (int ty = -m[1]; ty <= m[1]; ++ty)
            for (int tz = -m[2]; tz <= m[2]; ++tz) {
                for (const auto& pa : a) {
                    for (const auto& pb : b) {
                        GridPrimitive bT = pb;
                        bT.center[0] = pb.center[0] + tx * L[0];
                        bT.center[1] = pb.center[1] + ty * L[1];
                        bT.center[2] = pb.center[2] + tz * L[2];
                        const double p = pa.alpha + pb.alpha;
                        const double mu = pa.alpha * pb.alpha / p;
                        const double R2 = dist2(pa, bT);
                        const double S = pa.coeff * pb.coeff
                                         * std::pow(M_PI / p, 1.5)
                                         * std::exp(-mu * R2);
                        s_ref += S;
                        t_ref += mu * (3.0 - 2.0 * mu * R2) * S;
                    }
                }
            }

    EXPECT_NEAR(s_fused, s_ref, 1e-12 * std::max(1.0, std::abs(s_ref)));
    EXPECT_NEAR(t_fused, t_ref, 1e-12 * std::max(1.0, std::abs(t_ref)));
    // Public periodic overlap/kinetic share the fused image sweep.
    EXPECT_NEAR(overlap(a, b, L, m, reach), s_fused, 1e-14);
    EXPECT_NEAR(kinetic(a, b, L, m, reach), t_fused, 1e-14);
}

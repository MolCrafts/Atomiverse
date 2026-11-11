/// @file tests/cpu/gpw/test_cpu_gpw_one_electron.cpp
/// @brief GPW one-electron integrals: McMurchie-Davidson S and T vs closed forms.

#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>
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
    GridBasisFunction b = {prim(0.40, 1.2, -0.3, 0.5, 0, 0, 0, 0.18),
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
    GridBasisFunction a = {prim(0.50, 0, 0, 0, 0, 0, 0, 0.40),
                           prim(1.20, 0, 0, 0, 0, 0, 0, 0.60)};
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

// Mixed-angular (p/d) fused Γ periodic S+T vs independent prim_ST image sum.
// Exercises image-axis Hermite xsimd (slab-class image count).
TEST(GpwOneElectron, ContractedSTPeriodicAngularMatchesPrimImageSum) {
    using ATV::cpu::gpw::contracted_ST_periodic;
    using ATV::cpu::gpw::prim_kinetic;
    using ATV::cpu::gpw::prim_overlap;
    GridBasisFunction a = {
        prim(0.55, 0.0, 0.0, 0.0, 1, 0, 0, 0.40),
        prim(0.55, 0.0, 0.0, 0.0, 0, 1, 0, 0.35),
        prim(0.55, 0.0, 0.0, 0.0, 0, 0, 1, 0.30),
        prim(1.40, 0.0, 0.0, 0.0, 1, 0, 0, 0.25),
    };
    GridBasisFunction b = {
        prim(0.70, 0.8, -0.2, 0.3, 1, 0, 0, 0.45),
        prim(0.70, 0.8, -0.2, 0.3, 0, 1, 0, 0.40),
        prim(0.70, 0.8, -0.2, 0.3, 0, 0, 1, 0.35),
        prim(1.90, 0.8, -0.2, 0.3, 0, 1, 1, 0.20),
    };
    const double L[3] = {6.0, 6.0, 4.0};
    const int m[3] = {2, 2, 4};
    const double reach = 40.0;

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
                        s_ref += prim_overlap(pa, bT);
                        t_ref += prim_kinetic(pa, bT);
                    }
                }
            }

    EXPECT_NEAR(s_fused, s_ref, 1e-11 * std::max(1.0, std::abs(s_ref)));
    EXPECT_NEAR(t_fused, t_ref, 1e-11 * std::max(1.0, std::abs(t_ref)));
    EXPECT_NEAR(overlap(a, b, L, m, reach), s_fused, 1e-14);
    EXPECT_NEAR(kinetic(a, b, L, m, reach), t_fused, 1e-14);
}

// Non-periodic mixed-angular contracted ST still matches prim double-sum.
TEST(GpwOneElectron, ContractedAngularMatchesPrimSum) {
    using ATV::cpu::gpw::contracted_ST_shift;
    GridBasisFunction a = {prim(0.80, 0, 0, 0, 1, 0, 0, 0.5),
                           prim(0.80, 0, 0, 0, 0, 1, 0, 0.4)};
    GridBasisFunction b = {prim(1.10, 0.6, 0.1, -0.2, 0, 0, 1, 0.6),
                           prim(1.10, 0.6, 0.1, -0.2, 1, 0, 0, 0.3)};
    const double T0[3] = {0.0, 0.0, 0.0};
    double s = 0.0, t = 0.0;
    contracted_ST_shift(a, b, T0, s, t);
    double s_ref = 0.0, t_ref = 0.0;
    for (const auto& pa : a) {
        for (const auto& pb : b) {
            s_ref += prim_overlap(pa, pb);
            t_ref += prim_kinetic(pa, pb);
        }
    }
    EXPECT_NEAR(s, s_ref, 1e-13 * std::max(1.0, std::abs(s_ref)));
    EXPECT_NEAR(t, t_ref, 1e-13 * std::max(1.0, std::abs(t_ref)));
}

// The Gaussian characteristic function is
// F_ss(q) = c_a c_b (π/p)^(3/2) exp[−μ|A−B|²−|q|²/(4p)−iq·P].
// These closed-form tests differentiate that function analytically; they do
// not duplicate the Hermite-table implementation.
TEST(GpwOneElectron, FourierSsSeparatedCenters) {
    using ATV::cpu::gpw::prim_overlap_fourier;
    const auto a = prim(0.8, 0.2, -0.3, 0.5, 0, 0, 0, -0.7);
    const auto b = prim(1.3, -0.6, 0.4, 0.1, 0, 0, 0, 1.2);
    const double q[3] = {0.7, -0.45, 0.3};
    const double p = a.alpha + b.alpha;
    const double mu = a.alpha * b.alpha / p;
    double phase = 0.0, q2 = 0.0;
    for (int d = 0; d < 3; ++d) {
        phase += q[d] * (a.alpha * a.center[d] + b.alpha * b.center[d]) / p;
        q2 += q[d] * q[d];
    }
    const auto want = a.coeff * b.coeff * std::pow(M_PI / p, 1.5)
                      * std::exp(-mu * dist2(a, b) - q2 / (4.0 * p))
                      * std::complex<double>{std::cos(phase), -std::sin(phase)};
    EXPECT_NEAR(std::abs(prim_overlap_fourier(a, b, q) - want), 0.0, 2e-13);
}

// Odd products can have zero overlap at q=0 and a nonzero Fourier overlap.
// ∫ x exp(−p r²−iq·r) d³r = −iq_x/(2p) F_ss(q).
// ∫ x³ exp(...) d³r = i[q_x³/(8p³)−3q_x/(4p²)] F_ss(q).
TEST(GpwOneElectron, FourierOddProductsRetainNonzeroImaginaryPart) {
    using ATV::cpu::gpw::prim_overlap_fourier;
    const double alpha = 0.7, beta = 1.1, p = alpha + beta;
    const double q[3] = {0.9, -0.6, 0.35};
    const double q2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
    const double fss = std::pow(M_PI / p, 1.5) * std::exp(-q2 / (4.0 * p));
    const auto s = prim(alpha, 0, 0, 0, 0, 0, 0);
    const auto px = prim(beta, 0, 0, 0, 1, 0, 0);
    const auto dxx = prim(alpha, 0, 0, 0, 2, 0, 0);
    EXPECT_DOUBLE_EQ(prim_overlap(s, px), 0.0);
    EXPECT_DOUBLE_EQ(prim_overlap(dxx, px), 0.0);
    const std::complex<double> want_sp{0.0, -q[0] / (2.0 * p) * fss};
    const std::complex<double> want_dp{
        0.0,
        (q[0] * q[0] * q[0] / (8.0 * p * p * p) - 3.0 * q[0] / (4.0 * p * p))
            * fss};
    EXPECT_NEAR(std::abs(prim_overlap_fourier(s, px, q) - want_sp), 0.0, 2e-13);
    EXPECT_NEAR(std::abs(prim_overlap_fourier(dxx, px, q) - want_dp), 0.0,
                2e-13);
}

// Same-center even moments, with nonzero q in every Cartesian direction:
// m2(q)=1/(2p)−q²/(4p²),
// m4(q)=3/(4p²)−3q²/(4p³)+q⁴/(16p⁴).
// The old S(q=0)*form-factor approximation misses the q-dependent terms.
TEST(GpwOneElectron, FourierPAndDClosedMoments) {
    using ATV::cpu::gpw::prim_overlap_fourier;
    const double alpha = 0.7, beta = 1.1, p = alpha + beta;
    const double center[3] = {0.3, -0.2, 0.4};
    const double q[3] = {0.9, -0.6, 0.35};
    const double q2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
    const double phase = q[0] * center[0] + q[1] * center[1] + q[2] * center[2];
    const auto fss = -0.7 * 0.4 * std::pow(M_PI / p, 1.5)
                     * std::exp(-q2 / (4.0 * p))
                     * std::complex<double>{std::cos(phase), -std::sin(phase)};
    const double m2x = 1.0 / (2.0 * p) - q[0] * q[0] / (4.0 * p * p);
    const double m2y = 1.0 / (2.0 * p) - q[1] * q[1] / (4.0 * p * p);
    const double m4x = 3.0 / (4.0 * p * p)
                       - 3.0 * q[0] * q[0] / (4.0 * p * p * p)
                       + std::pow(q[0], 4) / (16.0 * p * p * p * p);
    const auto s = prim(alpha, center[0], center[1], center[2], 0, 0, 0, -0.7);
    const auto px = prim(alpha, center[0], center[1], center[2], 1, 0, 0, -0.7);
    const auto px_b = prim(beta, center[0], center[1], center[2], 1, 0, 0, 0.4);
    const auto dxx =
        prim(alpha, center[0], center[1], center[2], 2, 0, 0, -0.7);
    const auto dxx_b =
        prim(beta, center[0], center[1], center[2], 2, 0, 0, 0.4);
    const auto dxy =
        prim(alpha, center[0], center[1], center[2], 1, 1, 0, -0.7);
    const auto dxy_b =
        prim(beta, center[0], center[1], center[2], 1, 1, 0, 0.4);
    EXPECT_NEAR(std::abs(prim_overlap_fourier(s, dxx_b, q) - fss * m2x), 0.0,
                2e-13);
    EXPECT_NEAR(std::abs(prim_overlap_fourier(px, px_b, q) - fss * m2x), 0.0,
                2e-13);
    EXPECT_NEAR(std::abs(prim_overlap_fourier(dxx, dxx_b, q) - fss * m4x), 0.0,
                2e-13);
    EXPECT_NEAR(std::abs(prim_overlap_fourier(dxy, dxy_b, q) - fss * m2x * m2y),
                0.0, 2e-13);
}

// For displaced centers, completing the square gives a complex mean
// z=P−iq/(2p) and variance v=1/(2p). The x-only p(A)*d(B) polynomial
// has expectation (z−A)(z−B)²+v[(z−A)+2(z−B)].
TEST(GpwOneElectron, FourierSeparatedPdClosedForm) {
    using ATV::cpu::gpw::prim_overlap_fourier;
    const auto a = prim(0.8, 0.2, -0.3, 0.5, 1, 0, 0, -0.7);
    const auto b = prim(1.3, -0.6, 0.4, 0.1, 2, 0, 0, 1.2);
    const double q[3] = {0.7, -0.45, 0.3};
    const double p = a.alpha + b.alpha;
    const double mu = a.alpha * b.alpha / p;
    double phase = 0.0, q2 = 0.0;
    for (int d = 0; d < 3; ++d) {
        phase += q[d] * (a.alpha * a.center[d] + b.alpha * b.center[d]) / p;
        q2 += q[d] * q[d];
    }
    const auto fss = a.coeff * b.coeff * std::pow(M_PI / p, 1.5)
                     * std::exp(-mu * dist2(a, b) - q2 / (4.0 * p))
                     * std::complex<double>{std::cos(phase), -std::sin(phase)};
    const double P = (a.alpha * a.center[0] + b.alpha * b.center[0]) / p;
    const std::complex<double> za{P - a.center[0], -q[0] / (2.0 * p)};
    const std::complex<double> zb{P - b.center[0], -q[0] / (2.0 * p)};
    const auto want = fss * (za * zb * zb + (za + 2.0 * zb) / (2.0 * p));
    EXPECT_NEAR(std::abs(prim_overlap_fourier(a, b, q) - want), 0.0, 2e-13);
}

TEST(GpwOneElectron, FourierZeroWavevectorAndConjugation) {
    using ATV::cpu::gpw::prim_overlap_fourier;
    const std::array<std::array<int, 3>, 10> powers{{{0, 0, 0},
                                                     {1, 0, 0},
                                                     {0, 1, 0},
                                                     {0, 0, 1},
                                                     {2, 0, 0},
                                                     {1, 1, 0},
                                                     {1, 0, 1},
                                                     {0, 2, 0},
                                                     {0, 1, 1},
                                                     {0, 0, 2}}};
    const double zero[3] = {0, 0, 0};
    const double q[3] = {0.9, -0.6, 0.35};
    const double minus_q[3] = {-q[0], -q[1], -q[2]};
    for (const auto& la : powers)
        for (const auto& lb : powers) {
            const auto a = prim(0.8, 0.2, -0.3, 0.5, la[0], la[1], la[2], -0.7);
            const auto b = prim(1.3, -0.6, 0.4, 0.1, lb[0], lb[1], lb[2], 1.2);
            const auto f0 = prim_overlap_fourier(a, b, zero);
            EXPECT_NEAR(f0.real(), prim_overlap(a, b), 2e-13);
            EXPECT_DOUBLE_EQ(f0.imag(), 0.0);
            const auto fq = prim_overlap_fourier(a, b, q);
            EXPECT_NEAR(
                std::abs(prim_overlap_fourier(a, b, minus_q) - std::conj(fq)),
                0.0, 2e-13);
            EXPECT_NEAR(std::abs(prim_overlap_fourier(b, a, q) - fq), 0.0,
                        2e-13);
        }
}

TEST(GpwOneElectron, FourierRejectsInvalidPrimitiveDomain) {
    using ATV::cpu::gpw::prim_overlap_fourier;
    const double q[3] = {0.2, 0.3, -0.4};
    auto a = prim(0.8, 0, 0, 0, 0, 0, 0);
    const auto b = prim(1.1, 0, 0, 0, 0, 0, 0);
    a.alpha = -0.8;
    EXPECT_THROW(prim_overlap_fourier(a, b, q), std::invalid_argument);
    a.alpha = 0.8;
    a.l[0] = 9;
    EXPECT_THROW(prim_overlap_fourier(a, b, q), std::invalid_argument);
    a.l[0] = -1;
    EXPECT_THROW(prim_overlap_fourier(a, b, q), std::invalid_argument);
}

// Separate rows call the shared Fourier kernel concurrently during Berry /
// Wannier assembly. Different exponents, centers and angular powers exercise
// the per-thread Hermite geometry keys against closed characteristic moments.
TEST(GpwOneElectron, FourierConcurrentHermiteGeometry) {
    using ATV::cpu::gpw::prim_overlap_fourier;
    constexpr int count = 96;
    std::array<GridPrimitive, count> a, b;
    std::array<std::array<double, 3>, count> q;
    std::array<std::complex<double>, count> got, want;
    for (int n = 0; n < count; ++n) {
        const double alpha = 0.5 + 0.01 * n, beta = 0.8 + 0.005 * n;
        const double p = alpha + beta;
        const double center[3] = {0.01 * n, -0.02 * n, 0.03 * n};
        const int lx = n % 3 == 0 ? 0 : 1;
        const int ly = n % 3 == 2 ? 1 : 0;
        a[n] = prim(alpha, center[0], center[1], center[2], lx, ly, 0, 0.7);
        b[n] = prim(beta, center[0], center[1], center[2], lx, ly, 0, -0.4);
        q[n] = {0.2 + 0.001 * n, -0.3 + 0.003 * n, 0.4};
        const double phase =
            q[n][0] * center[0] + q[n][1] * center[1] + q[n][2] * center[2];
        const double q2 =
            q[n][0] * q[n][0] + q[n][1] * q[n][1] + q[n][2] * q[n][2];
        want[n] = -0.28 * std::pow(M_PI / p, 1.5) * std::exp(-q2 / (4.0 * p))
                  * std::complex<double>{std::cos(phase), -std::sin(phase)};
        if (lx)
            want[n] *= 1.0 / (2.0 * p) - q[n][0] * q[n][0] / (4.0 * p * p);
        if (ly)
            want[n] *= 1.0 / (2.0 * p) - q[n][1] * q[n][1] / (4.0 * p * p);
    }
#pragma omp parallel for num_threads(4) schedule(dynamic, 1)
    for (int n = 0; n < count; ++n)
        got[n] = prim_overlap_fourier(a[n], b[n], q[n].data());
    for (int n = 0; n < count; ++n)
        EXPECT_NEAR(std::abs(got[n] - want[n]), 0.0, 2e-13) << "row " << n;
}

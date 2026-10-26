/// @file tests/cpu/planewave/test_gth_upf.cpp
/// @brief `internal::upf_from_gth` — analytic GTH evaluated onto UPF radial
///        tables, checked against the closed-form G-space expression.
///
/// The two routes are genuinely independent: `upf_from_gth` builds V_loc(r)
/// on a log mesh and `upf_vloc_to_gspace` transforms it by radial quadrature,
/// while `gth_vloc_gspace` is Goedecker et al. (1996) Eq. 5 evaluated
/// directly. Agreement therefore tests the radial synthesis, the unit
/// conventions (v_loc in Ry, d_ij in Ry, beta = r p(r)) and the adequacy of
/// the mesh at once.
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/planewave/internal/g_vector.h"
#include "atomiverse/cpu/planewave/internal/gth_upf.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

using namespace ATV;
using namespace ATV::cpu::pw::internal;

namespace {

/// Cubic cell of side `a` [Bohr].
cpu::Box cubic_bohr(double a) {
    cpu::Box b;
    Mat3 c;
    c(0, 0) = a;
    c(0, 1) = 0;
    c(0, 2) = 0;
    c(1, 0) = 0;
    c(1, 1) = a;
    c(1, 2) = 0;
    c(2, 0) = 0;
    c(2, 1) = 0;
    c(2, 2) = a;
    b.set_cell(c);
    return b;
}

/// Compare the quadrature route against the closed form for one potential.
void expect_vloc_matches_closed_form(molrs::Element el, const char* name,
                                     double tol_abs) {
    const cpu::gpw::GthPotential pot(el, name);
    const UPFData upf = upf_from_gth(pot);

    const double a = 12.0; // Bohr
    const auto box = cubic_bohr(a);
    const double omega = a * a * a;
    const auto gv = enumerate_g_vectors(box, /*ecut_ht=*/8.0, Vec3::zeros());

    const auto vg = upf_vloc_to_gspace(upf, gv, omega);
    ASSERT_EQ(static_cast<int>(vg.size()), gv.count());

    double worst = 0.0;
    for (int ig = 0; ig < gv.count(); ++ig) {
        const double g = std::sqrt(gv.g[ig].norm_squared());
        const double ref = gth_vloc_gspace(pot, g, omega);
        worst = std::max(worst, std::abs(vg[ig] - ref));
    }
    EXPECT_LT(worst, tol_abs)
        << name << ": radial-quadrature V_loc(G) departs from the closed form "
        << "by " << worst << " Hartree";
}

} // namespace

TEST(GthUpf, LocalPotentialMatchesClosedFormHydrogen) {
    // H GTH-PBE-q1 has no nonlocal channels: pure local test.
    expect_vloc_matches_closed_form(molrs::Element::H, "GTH-PBE-q1", 1e-4);
}

TEST(GthUpf, LocalPotentialMatchesClosedFormSilicon) {
    expect_vloc_matches_closed_form(molrs::Element::Si, "GTH-PBE-q4", 1e-4);
}

TEST(GthUpf, LocalPotentialMatchesClosedFormOxygen) {
    expect_vloc_matches_closed_form(molrs::Element::O, "GTH-PBE-q6", 1e-4);
}

TEST(GthUpf, ProjectorsAreNormalized) {
    // p_i^l is normalized as \int p^2 r^2 dr = 1; beta holds r p(r), so the
    // same integral is \int beta^2 dr. Independent of the local part.
    const cpu::gpw::GthPotential pot(molrs::Element::Si, "GTH-PBE-q4");
    const UPFData upf = upf_from_gth(pot);
    ASSERT_GT(upf.n_proj(), 0);

    for (const auto& kb : upf.kb) {
        std::vector<AccReal> density(upf.r.size());
        for (std::size_t ir = 0; ir < upf.r.size(); ++ir)
            density[ir] = kb.beta[ir] * kb.beta[ir];
        const double norm = upf.integrate_radial(density);
        EXPECT_NEAR(norm, 1.0, 1e-4)
            << "GTH projector l=" << kb.l << " is not unit-normalized";
    }
}

TEST(GthUpf, CarriesValenceChargeAndNormConservingTag) {
    const cpu::gpw::GthPotential pot(molrs::Element::O, "GTH-PBE-q6");
    const UPFData upf = upf_from_gth(pot);
    EXPECT_EQ(upf.z_val, 6);
    EXPECT_EQ(upf.pseudo_type, "NC");
    EXPECT_TRUE(upf.rho_atom.empty()); // GTH supplies no atomic density
}

TEST(GthUpf, UnknownParameterizationThrows) {
    EXPECT_THROW((void)cpu::gpw::GthPotential(molrs::Element::Si, "GTH-NOPE"),
                 std::runtime_error);
}

namespace {
struct GaussianKb {
    UPFData upf;
    std::vector<double> alpha{0.6, 0.4, 1.2, 0.9, 1.7};
};

GaussianKb gaussian_kb() {
    GaussianKb data;
    constexpr int angular[] = {1, 0, 1, 0, 0}; // interleaved 2x2 p / 3x3 s
    constexpr int nr = 10001;
    constexpr double dr =
        0.001; // Bohr; Gaussian exponential tails at r=10 are below 5e-18.
    data.upf.r.resize(nr);
    data.upf.rab.assign(nr, dr);
    data.upf.kb.resize(5);
    for (int ir = 0; ir < nr; ++ir)
        data.upf.r[ir] = ir * dr;
    for (int ip = 0; ip < 5; ++ip) {
        auto& projector = data.upf.kb[ip];
        projector.l = angular[ip];
        projector.beta.resize(nr);
        projector.radial_points = nr;
        for (int ir = 0; ir < nr; ++ir) {
            const double r = data.upf.r[ir];
            // UPF beta=r*p(r), p(r)=r^l exp(-alpha*r^2).
            projector.beta[ir] = std::pow(r, angular[ip] + 1)
                                 * std::exp(-data.alpha[ip] * r * r);
        }
    }
    data.upf.d_ij = {
        2, 0, 0.7, 0,   0, 0,   3,   0, 0.4,  -0.2, 0.7, 0,  5,
        0, 0, 0,   0.4, 0, 1.5, 0.6, 0, -0.2, 0,    0.6, 2.1}; // Ry, symmetric blocks
    return data;
}

// Closed Gaussian/Bessel integral, independent of the production quadrature:
// integral r^(l+2) exp(-a*r^2) j_l(q*r) dr
// = sqrt(pi)*q^l*exp(-q^2/(4a)) / (2^(l+2)*a^(l+3/2)).
double gaussian_radial(int l, double alpha, double q) {
    return std::sqrt(M_PI) * std::pow(q, l) * std::exp(-q * q / (4 * alpha))
           / (std::pow(2.0, l + 2) * std::pow(alpha, l + 1.5));
}
} // namespace

TEST(KbProjectors, RotatedBlocksPreserveAnalyticGaussianOperator) {
    auto data = gaussian_kb();
    GVectorList gv;
    gv.g = {{0, 0, 0}, {0.3, -0.2, 0.1}, {1.1, 0.4, -0.3}, {2, -0.5, 0.8}};
    gv.kvec = {0.17, -0.1, 0.07}; // must use |k+G|, not |G|
    const std::vector<Vec3> positions{{0, 0, 0}};
    constexpr double omega = 64.0; // Bohr^3, fixes full operator normalization
    constexpr double tolerance = 2e-6; // Ha, radial quadrature at dr=.001
    for (bool diagonal : {false, true}) {
        SCOPED_TRACE(diagonal);
        if (diagonal)
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 5; ++j)
                    if (i != j)
                        data.upf.d_ij[5 * i + j] = 0;
        std::vector<std::vector<PWWorkspace::KBProjG>> result;
        setup_kb_projectors(result, gv, data.upf, positions);
        ASSERT_EQ(result.size(), 1u);
        ASSERT_EQ(result[0].size(), 5u);
        for (int ip = 0; ip < 5; ++ip) {
            EXPECT_EQ(result[0][ip].l, ip < 3 ? 0 : 1);
            ASSERT_EQ(result[0][ip].f_G.size(), gv.g.size());
        }
        double largest_error = 0;
        for (int g = 0; g < gv.count(); ++g)
            for (int h = 0; h < gv.count(); ++h) {
                const Vec3 q = gv.g[g] + gv.kvec, p = gv.g[h] + gv.kvec;
                const double qn = q.norm(), pn = p.norm();
                double cosine = 0;
                for (int axis = 0; axis < 3; ++axis)
                    cosine += q[axis] * p[axis] / (qn * pn);
                double got = 0, want = 0;
                for (int l = 0; l <= 1; ++l) {
                    // Addition theorem sums all m analytically. At atom R=0,
                    // (-i)^l phase cancels its conjugate in the KB operator.
                    const double factor = 4 * M_PI / omega * (2 * l + 1)
                                          * (l == 0 ? 1.0 : cosine);
                    for (const auto& projector : result[0])
                        if (projector.l == l)
                            got += factor * projector.d * projector.f_G[g]
                                   * projector.f_G[h];
                    for (int i = 0; i < 5; ++i)
                        for (int j = 0; j < 5; ++j)
                            if (data.upf.kb[i].l == l && data.upf.kb[j].l == l)
                                want += factor * 0.5 * data.upf.d_ij[5 * i + j]
                                        * gaussian_radial(l, data.alpha[i], qn)
                                        * gaussian_radial(l, data.alpha[j], pn);
                }
                ASSERT_TRUE(std::isfinite(got));
                largest_error = std::max(largest_error, std::abs(got - want));
            }
        EXPECT_LT(largest_error, tolerance);
    }
}

TEST(KbProjectors, InvalidCouplingMatricesAndAngularMomentumThrow) {
    const auto original = gaussian_kb();
    GVectorList gv;
    gv.g = {{0, 0, 0}};
    const std::vector<Vec3> positions{{0, 0, 0}};
    std::vector<std::vector<PWWorkspace::KBProjG>> result;
    auto invalid = original.upf;
    invalid.d_ij.pop_back();
    EXPECT_THROW(setup_kb_projectors(result, gv, invalid, positions),
                 std::invalid_argument);
    invalid.d_ij.clear();
    EXPECT_THROW(setup_kb_projectors(result, gv, invalid, positions),
                 std::invalid_argument);
    invalid = original.upf;
    invalid.d_ij[2] = 0.8; // asymmetric p-block
    EXPECT_THROW(setup_kb_projectors(result, gv, invalid, positions),
                 std::invalid_argument);
    invalid = original.upf;
    invalid.d_ij[1] = invalid.d_ij[5] = 0.25; // couples s and p
    EXPECT_THROW(setup_kb_projectors(result, gv, invalid, positions),
                 std::invalid_argument);
    invalid = original.upf;
    invalid.d_ij[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(setup_kb_projectors(result, gv, invalid, positions),
                 std::invalid_argument);
    invalid = original.upf;
    invalid.kb[0].l = 3; // no f-channel spherical Bessel/angular implementation
    EXPECT_THROW(setup_kb_projectors(result, gv, invalid, positions),
                 std::invalid_argument);
}

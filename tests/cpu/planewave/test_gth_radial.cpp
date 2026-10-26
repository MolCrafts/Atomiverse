/// @file tests/cpu/planewave/test_gth_radial.cpp
/// @brief `internal::sample_gth` — analytic GTH evaluated onto UPF radial
///        tables, checked against the closed-form G-space expression.
///
/// The two routes are genuinely independent: `sample_gth` builds V_loc(r)
/// on a log mesh and `nc_vloc_to_gspace` transforms it by radial quadrature,
/// while `gth_vloc_gspace` is Goedecker et al. (1996) Eq. 5 evaluated
/// directly. Agreement therefore tests the radial synthesis, the unit
/// conventions (v_loc in Ha, d_ij in Ha, beta = r p(r)) and the adequacy of
/// the mesh at once.
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/g_vector.h"
#include "atomiverse/cpu/planewave/internal/gth_radial.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <xtensor/generators/xbuilder.hpp>

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
    const NcPotentialData upf = sample_gth(pot);

    const double a = 12.0; // Bohr
    const auto box = cubic_bohr(a);
    const double omega = a * a * a;
    const auto gv = enumerate_g_vectors(box, /*ecut_ht=*/8.0, Vec3::zeros());

    const auto vg = nc_vloc_to_gspace(upf, gv, omega);
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
    const NcPotentialData upf = sample_gth(pot);
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

TEST(GthUpf, CarriesChemicalIdentityAndValence) {
    const cpu::gpw::GthPotential pot(molrs::Element::O, "GTH-PBE-q6");
    const NcPotentialData upf = sample_gth(pot);
    EXPECT_EQ(upf.z_val, 6);
    EXPECT_EQ(upf.atomic_number, 8);
    EXPECT_TRUE(upf.rho_atom.empty()); // GTH supplies no atomic density
}

TEST(GthUpf, UnknownParameterizationThrows) {
    EXPECT_THROW((void)cpu::gpw::GthPotential(molrs::Element::Si, "GTH-NOPE"),
                 std::runtime_error);
}

namespace {
struct GaussianKb {
    NcPotentialData upf;
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
    data.upf.d_ij = {1,    0,    0.35, 0,    0, 0,   1.5, 0,   0.2,
                     -0.1, 0.35, 0,    2.5,  0, 0,   0,   0.2, 0,
                     0.75, 0.3,  0,    -0.1, 0, 0.3, 1.05}; // Ha, symmetric blocks
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
        setup_kb_projectors(result, gv, data.upf, positions, omega);
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
                                want += factor * data.upf.d_ij[5 * i + j]
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
    EXPECT_THROW(setup_kb_projectors(result, gv, original.upf, positions, 0),
                 std::invalid_argument);
    EXPECT_THROW(setup_kb_projectors(result, gv, original.upf, positions,
                                     std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
    auto invalid = original.upf;
    invalid.d_ij.pop_back();
    EXPECT_THROW(setup_kb_projectors(result, gv, invalid, positions, 64),
                 std::invalid_argument);
    invalid.d_ij.clear();
    EXPECT_THROW(setup_kb_projectors(result, gv, invalid, positions, 64),
                 std::invalid_argument);
    invalid = original.upf;
    invalid.d_ij[2] = 0.8; // asymmetric p-block
    EXPECT_THROW(setup_kb_projectors(result, gv, invalid, positions, 64),
                 std::invalid_argument);
    invalid = original.upf;
    invalid.d_ij[1] = invalid.d_ij[5] = 0.25; // couples s and p
    EXPECT_THROW(setup_kb_projectors(result, gv, invalid, positions, 64),
                 std::invalid_argument);
    invalid = original.upf;
    invalid.d_ij[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(setup_kb_projectors(result, gv, invalid, positions, 64),
                 std::invalid_argument);
    invalid = original.upf;
    invalid.kb[0].l = 3; // no f-channel spherical Bessel/angular implementation
    EXPECT_THROW(setup_kb_projectors(result, gv, invalid, positions, 64),
                 std::invalid_argument);
}

TEST(KbProjectors, EachCompactPrefixUsesItsOwnSimpsonEndpoint) {
    NcPotentialData data;
    data.r = {0, .5, 1, 1.5, 2, 2.5, 3};
    data.rab.assign(data.r.size(), .5);
    data.kb = {{0, {0, .5, 1, 100, 100, 100, 100}, 3},
               {0, {0, .5, 1, 1.5, 100, 100, 100}, 4},
               {0, data.r, 7}};
    data.d_ij = {1, 0, 0, 0, 2, 0, 0, 0, 3};
    GVectorList gv;
    gv.g = {{0, 0, 0}};
    std::vector<std::vector<PWWorkspace::KBProjG>> result;
    setup_kb_projectors(result, gv, data, {{0, 0, 0}}, 64);
    double operator_value = 0;
    for (const auto& projector : result[0])
        operator_value += projector.d * projector.f_G[0] * projector.f_G[0];
    // beta=r gives integral r*beta dr = R^3/3 exactly. The short
    // projectors' trailing storage is outside their integration domain.
    const double want =
        std::pow(1.0 / 3.0, 2) + 2 * std::pow(1.125, 2) + 3 * 81;
    EXPECT_NEAR(operator_value, want, 1e-12);
}

TEST(GthUpf, RejectsMalformedExplicitAnalyticData) {
    const cpu::gpw::GthPotential original(molrs::Element::Si, "GTH-PBE-q4");
    auto malformed = original;
    malformed.channels[0].h.pop_back();
    EXPECT_THROW(sample_gth(malformed), std::invalid_argument);
    malformed = original;
    malformed.rloc = std::numeric_limits<double>::infinity();
    EXPECT_THROW(sample_gth(malformed), std::invalid_argument);
    malformed = original;
    malformed.channels[0].nproj = -1;
    EXPECT_THROW(sample_gth(malformed), std::invalid_argument);
    malformed = original;
    malformed.cloc[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(sample_gth(malformed), std::invalid_argument);
    malformed = original;
    malformed.channels[0].l = 3;
    EXPECT_THROW(sample_gth(malformed), std::invalid_argument);
}

TEST(KbProjectors, PreparedComplexColumnsMatchAnalyticGaussianHamiltonian) {
    using Cx = std::complex<AccReal>;
    const auto original = gaussian_kb();
    for (int l = 0; l <= 2; ++l) {
        auto data = original.upf;
        data.kb.resize(1);
        data.kb[0].l = l;
        constexpr double alpha = 0.6, strength = -0.7;
        for (std::size_t ir = 0; ir < data.r.size(); ++ir)
            data.kb[0].beta[ir] = std::pow(data.r[ir], l + 1)
                                  * std::exp(-alpha * data.r[ir] * data.r[ir]);
        data.d_ij = {strength};
        PWWorkspace ws;
        ws.n_g = 4;
        ws.nr = 64;
        ws.fft_nx = ws.fft_ny = 8;
        ws.fft_nz = 1;
        ws.fft = std::make_unique<FFTPlan3D>(8, 8, 1);
        ws.gvec = &ws.gvec0;
        ws.gvec0.h = {0, 1, -1, 0};
        ws.gvec0.k = {0, 0, 0, 1};
        ws.gvec0.l = {0, 0, 0, 0};
        ws.V_eff_r = xt::zeros<AccReal>({64});
        ws.T_G = xt::zeros<AccReal>({4});
        const std::vector<Cx> psi{{.3, -.2}, {.4, .1}, {-.1, .7}, {.2, -.3}};
        std::vector<Cx> result(4);
        FFTWBuffer<fftw_complex> work(ws.nr);
        // Same shapes, different cell metric, Bloch k and atomic geometry.
        for (int geometry = 0; geometry < 2; ++geometry) {
            SCOPED_TRACE(l);
            SCOPED_TRACE(geometry);
            ws.omega = geometry == 0 ? 64 : 125;
            const double scale = geometry == 0 ? 1 : .8;
            ws.gvec0.g = {
                {0, 0, 0}, {scale, 0, 0}, {-scale, 0, 0}, {0, scale, 0}};
            ws.gvec0.kvec =
                geometry == 0 ? Vec3{.17, -.1, .07} : Vec3{-.11, .23, .13};
            ws.atom_pos_bohr =
                geometry == 0
                    ? std::vector<Vec3>{{.2, -.4, .7}, {1.3, .6, -.2}}
                    : std::vector<Vec3>{{-.5, .3, 1.1}, {.8, -1.2, .4}};
            for (int g = 0; g < 4; ++g)
                ws.T_G[g] = .5 * (ws.gvec0.g[g] + ws.gvec0.kvec).norm_squared();
            setup_kb_projectors(ws.kb_proj, ws.gvec0, data, ws.atom_pos_bohr,
                                ws.omega);
            for (const auto& atom : ws.kb_proj) {
                ASSERT_EQ(atom[0].beta_G.dimension(), 2u);
                EXPECT_EQ(atom[0].beta_G.shape()[0], std::size_t(2 * l + 1));
                EXPECT_EQ(atom[0].beta_G.shape()[1], 4u);
            }
            pw_apply_h(ws, psi.data(), result.data(), work.data());
            // Independent dense H(G,H) = T + Σ D β(G) conj(β(H));
            // checks the complex BLAS conjugation/layout at roundoff scale.
            for (int g = 0; g < 4; ++g) {
                Cx dense_action{};
                for (int h = 0; h < 4; ++h) {
                    Cx matrix_element = g == h ? Cx(ws.T_G[g], 0) : Cx{};
                    for (const auto& atom : ws.kb_proj)
                        for (const auto& projector : atom)
                            for (int m = 0; m < 2 * projector.l + 1; ++m)
                                matrix_element +=
                                    projector.d * projector.beta_G(m, g)
                                    * std::conj(projector.beta_G(m, h));
                    dense_action += matrix_element * psi[h];
                }
                EXPECT_NEAR(result[g].real(), dense_action.real(), 1e-12);
                EXPECT_NEAR(result[g].imag(), dense_action.imag(), 1e-12);
            }
            for (int g = 0; g < 4; ++g) {
                const Vec3 q = ws.gvec0.g[g] + ws.gvec0.kvec;
                Cx expected = ws.T_G[g] * psi[g];
                for (int h = 0; h < 4; ++h) {
                    const Vec3 p = ws.gvec0.g[h] + ws.gvec0.kvec;
                    const double cosine =
                        (q[0] * p[0] + q[1] * p[1] + q[2] * p[2])
                        / (q.norm() * p.norm());
                    const double legendre =
                        l == 0   ? 1
                        : l == 1 ? cosine
                                 : .5 * (3 * cosine * cosine - 1);
                    const double radial = strength * 4 * M_PI / ws.omega
                                          * (2 * l + 1) * legendre
                                          * gaussian_radial(l, alpha, q.norm())
                                          * gaussian_radial(l, alpha, p.norm());
                    for (const auto& position : ws.atom_pos_bohr) {
                        const Vec3 delta = q - p;
                        const double phase = delta[0] * position[0]
                                             + delta[1] * position[1]
                                             + delta[2] * position[2];
                        expected += radial
                                    * Cx(std::cos(phase), -std::sin(phase))
                                    * psi[h];
                    }
                }
                EXPECT_NEAR(result[g].real(), expected.real(), 2e-6);
                EXPECT_NEAR(result[g].imag(), expected.imag(), 2e-6);
            }
            ws.kb_proj[0][0].beta_G = xt::xarray<Cx>{};
            EXPECT_THROW(pw_apply_h(ws, psi.data(), result.data(), work.data()),
                         std::invalid_argument);
        }
    }
}

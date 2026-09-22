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

#include <cmath>
#include <gtest/gtest.h>

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

/// Worst |V_loc(G) quadrature - closed form| over the G-sphere [Hartree].
double vloc_worst_error(molrs::Element el, const char* name, int n_r) {
    const cpu::gpw::GthPotential pot(el, name);
    const UPFData upf = upf_from_gth(pot, /*r_max=*/12.0, n_r);

    const double a = 12.0; // Bohr
    const auto box = cubic_bohr(a);
    const double omega = a * a * a;
    const auto gv = enumerate_g_vectors(box, /*ecut_ht=*/8.0, Vec3::zeros());
    const auto vg = upf_vloc_to_gspace(upf, gv, omega);

    double worst = 0.0;
    for (int ig = 0; ig < gv.count(); ++ig) {
        const double g = std::sqrt(gv.g[ig].norm_squared());
        worst =
            std::max(worst, std::abs(vg[ig] - gth_vloc_gspace(pot, g, omega)));
    }
    return worst;
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
        double norm = 0.0;
        for (std::size_t ir = 1; ir < upf.r.size(); ++ir) {
            const double dr = upf.r[ir] - upf.r[ir - 1];
            const double b = 0.5 * (kb.beta[ir] + kb.beta[ir - 1]);
            norm += b * b * dr;
        }
        EXPECT_NEAR(norm, 1.0, 1e-4)
            << "GTH projector l=" << kb.l << " is not unit-normalized";
    }
}

TEST(GthUpf, LocalPotentialConvergesSecondOrderInMeshSpacing) {
    // The residual against the closed form is trapezoid discretization, not a
    // wrong formula: on a log mesh the error is O(dx^2), so each doubling of
    // the point count must cut it by ~4. A formula error would be
    // mesh-independent instead.
    const double e1 = vloc_worst_error(molrs::Element::O, "GTH-PBE-q6", 1000);
    const double e2 = vloc_worst_error(molrs::Element::O, "GTH-PBE-q6", 2000);
    const double e4 = vloc_worst_error(molrs::Element::O, "GTH-PBE-q6", 4000);
    EXPECT_GT(e1 / e2, 3.0) << "e1=" << e1 << " e2=" << e2;
    EXPECT_LT(e1 / e2, 5.0) << "e1=" << e1 << " e2=" << e2;
    EXPECT_GT(e2 / e4, 3.0) << "e2=" << e2 << " e4=" << e4;
    EXPECT_LT(e2 / e4, 5.0) << "e2=" << e2 << " e4=" << e4;
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

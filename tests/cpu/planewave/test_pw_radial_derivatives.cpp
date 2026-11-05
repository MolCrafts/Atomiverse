#include "atomiverse/cpu/planewave/internal/pw_radial_derivatives.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/pw_subspace.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

using namespace ATV;
using namespace ATV::cpu::pw::internal;
namespace {
constexpr double pi = 3.1415926535897932384626433832795;

// Independent spherical Bessel reference j_l(x) / x^l for l <= 3, so the
// test does not lean on std::sph_bessel (absent from Apple libc++) or on the
// in-tree regularized_bessel_jet it checks. Small x uses the ascending
// series (Abramowitz & Stegun 10.1.2), which has no cancellation; larger x
// uses the closed forms (A&S 10.1.11), which are accurate once x >= 2.
double reference_bessel_over_xl(int l, double x) {
    if (std::abs(x) < 2) {
        double odd = 1; // (2l+1)!!
        for (int k = 3; k <= 2 * l + 1; k += 2)
            odd *= k;
        double term = 1 / odd, sum = term;
        for (int k = 1; k < 40; ++k) {
            term *= -x * x / (2. * k * (2. * l + 2. * k + 1.));
            sum += term;
        }
        return sum;
    }
    const double s = std::sin(x), c = std::cos(x), x2 = x * x;
    double j = 0;
    switch (l) {
    case 0:
        j = s / x;
        break;
    case 1:
        j = s / x2 - c / x;
        break;
    case 2:
        j = (3 / x2 - 1) * s / x - 3 * c / x2;
        break;
    case 3:
        j = (15 / (x2 * x) - 6 / x) * s / x - (15 / x2 - 1) * c / x;
        break;
    default:
        ADD_FAILURE() << "reference Bessel implemented for l <= 3 only";
    }
    return j / std::pow(x, l);
}
double reference_sph_bessel(int l, double x) {
    return reference_bessel_over_xl(l, x) * std::pow(x, l);
}
constexpr int aa[6] = {0, 1, 2, 0, 0, 1}, bb[6] = {0, 1, 2, 1, 2, 2};
NcPotentialData gaussian_data() {
    NcPotentialData data;
    data.atomic_number = 1;
    data.z_val = 1;
    data.l_max = 2;
    constexpr int n = 4001;
    constexpr double dr = .005, sigma = .4;
    data.r.resize(n);
    data.rab.assign(n, dr);
    data.v_loc.resize(n);
    for (int i = 0; i < n; ++i) {
        const double r = i * dr;
        data.r[i] = r;
        data.v_loc[i] = i ? -std::erf(r / (std::sqrt(2.) * sigma)) / r
                          : -std::sqrt(2. / pi) / sigma;
    }
    for (int l = 0; l <= 2; ++l)
        for (double alpha : {.7, 1.1}) {
            NcPotentialData::KBProjector p;
            p.l = l;
            p.radial_points = n;
            p.beta.resize(n);
            for (int i = 0; i < n; ++i)
                p.beta[i] = std::pow(data.r[i], l + 1)
                            * std::exp(-alpha * data.r[i] * data.r[i]);
            data.kb.push_back(std::move(p));
        }
    data.d_ij.assign(36, 0);
    for (int l = 0; l <= 2; ++l) {
        const int p = 2 * l;
        data.d_ij[6 * p + p] = .7;
        data.d_ij[6 * (p + 1) + p + 1] = 1.1;
        data.d_ij[6 * p + p + 1] = data.d_ij[6 * (p + 1) + p] = -.21;
    }
    data.validate();
    return data;
}
double gaussian_regular(int l, double alpha, double q) {
    return std::sqrt(pi) / (std::pow(2., l + 2) * std::pow(alpha, l + 1.5))
           * std::exp(-q * q / (4 * alpha));
}
std::array<double, 5> exact_projector(int l, double alpha, const Vec3& q) {
    std::array<double, 5> out{};
    const double norm = q.norm();
    if (norm == 0) {
        if (l == 0)
            out[0] = gaussian_regular(l, alpha, 0) / std::sqrt(4 * pi);
        return out;
    }
    compute_ylm(q[0], q[1], q[2], norm, l, out.data());
    for (int m = 0; m < 2 * l + 1; ++m)
        out[m] *= gaussian_regular(l, alpha, norm) * std::pow(norm, l);
    return out;
}
TEST(PwRadialDerivatives, RegularizedBesselMatchesIndependentSpecialFunctions) {
    for (int l = 0; l <= 2; ++l) {
        const double leading = l == 0 ? 1. : l == 1 ? 1. / 3 : 1. / 15;
        auto zero = regularized_bessel_jet(l, 0);
        EXPECT_DOUBLE_EQ(zero.value, leading);
        EXPECT_DOUBLE_EQ(zero.slope_over_x, -leading / (2 * l + 3));
        for (double x : {1e-12, 1e-7, 1e-3, .3, .9999, 1.0001, 4., 12.}) {
            SCOPED_TRACE(::testing::Message() << "l=" << l << " x=" << x);
            auto got = regularized_bessel_jet(l, x);
            EXPECT_NEAR(got.value, reference_bessel_over_xl(l, x), 3e-13);
            EXPECT_NEAR(got.slope_over_x, -reference_bessel_over_xl(l + 1, x),
                        3e-13);
        }
    }
}
TEST(PwRadialDerivatives, GaussianRadialValueAndSlopeIncludeOrigin) {
    const auto data = gaussian_data();
    for (int l = 0; l <= 2; ++l)
        for (int j = 0; j < 2; ++j)
            for (double q : {0., 1e-12, 1e-7, 1e-3, .3, 1.7, 4.1}) {
                const double alpha = j ? 1.1 : .7;
                const auto got = nc_projector_radial_jet(data, 2 * l + j, q);
                const double want = gaussian_regular(l, alpha, q);
                EXPECT_NEAR(got.value, want, 1e-11);
                EXPECT_NEAR(got.slope_over_q, -want / (2 * alpha), 1e-11);
            }
}
TEST(PwRadialDerivatives, ExactEvenAndOddProjectorPrefixesIgnoreUnusedSuffix) {
    auto data = gaussian_data();
    for (std::size_t points : {1000u, 1001u}) {
        data.kb[3].radial_points = points;
        const auto before = nc_projector_radial_jet(data, 3, .71);
        for (std::size_t i = points; i < data.kb[3].beta.size(); ++i)
            data.kb[3].beta[i] = 1e20;
        const auto after = nc_projector_radial_jet(data, 3, .71);
        EXPECT_DOUBLE_EQ(after.value, before.value);
        EXPECT_DOUBLE_EQ(after.slope_over_q, before.slope_over_q);
        std::vector<AccReal> val(points), der(points);
        for (std::size_t i = 0; i < points; ++i) {
            const double r = data.r[i], x = .71 * r;
            val[i] = r * data.kb[3].beta[i] * reference_sph_bessel(1, x) / .71;
            // d/dq[f_1(q)/q]/q = -integral r^3 beta j_2(qr)/(qr)^2 dr.
            der[i] = x == 0 ? 0
                            : -std::pow(r, 4) * data.kb[3].beta[i]
                                  * reference_sph_bessel(2, x) / (x * x);
        }
        EXPECT_NEAR(after.value, data.integrate_radial(val), 1e-12);
        EXPECT_NEAR(after.slope_over_q, data.integrate_radial(der), 1e-12);
        data = gaussian_data();
    }
}
TEST(PwRadialDerivatives,
     CartesianJetsMatchCurrentAngularOrderAndAllDerivatives) {
    const auto data = gaussian_data();
    for (int l = 0; l <= 2; ++l)
        for (const Vec3 q : {Vec3{0, 0, 0}, Vec3{1e-12, -2e-12, 3e-12},
                             Vec3{.3, -.2, .4}, Vec3{1.1, .4, -.7}}) {
            const auto got = nc_projector_cartesian_jet(data, 2 * l, q);
            const auto expected = exact_projector(l, .7, q);
            for (int m = 0; m < 2 * l + 1; ++m)
                EXPECT_NEAR(got.value[m], expected[m], 2e-12);
            for (double h : {3e-4, 3e-5, 3e-6})
                for (int k = 0; k < 3; ++k) {
                    auto qp = q, qm = q;
                    qp[k] += h;
                    qm[k] -= h;
                    auto plus = exact_projector(l, .7, qp),
                         minus = exact_projector(l, .7, qm);
                    qp[k] += h;
                    qm[k] -= h;
                    auto plus2 = exact_projector(l, .7, qp),
                         minus2 = exact_projector(l, .7, qm);
                    for (int m = 0; m < 2 * l + 1; ++m) {
                        const double fd =
                            (8 * (plus[m] - minus[m]) - (plus2[m] - minus2[m]))
                            / (12 * h);
                        EXPECT_NEAR(got.gradient[m][k], fd, 2e-10);
                    }
                }
        }
}

using C = std::complex<double>;
const std::array<Vec3, 5> modes{
    {{0, 0, 0}, {.3, -.2, .4}, {1.1, .4, -.7}, {-.7, .8, .2}, {.1, .9, -.3}}};
const std::array<C, 5> psi{
    {{.3, .2}, {-.1, .4}, {.2, -.3}, {.5, .1}, {-.2, .3}}};
const std::array<Vec3, 2> atoms{{{.2, .4, .7}, {1.1, -.3, .5}}};
Vec3 strain_vector(const Vec3& v, int k, double h, bool reciprocal) {
    Vec3 out = v;
    int a = aa[k], b = bb[k];
    if (a == b)
        out[a] = reciprocal ? v[a] / (1 + h) : v[a] * (1 + h);
    else if (reciprocal) {
        out[a] = (v[a] - h * v[b]) / (1 - h * h);
        out[b] = (v[b] - h * v[a]) / (1 - h * h);
    } else {
        out[a] = v[a] + h * v[b];
        out[b] = v[b] + h * v[a];
    }
    return out;
}
double exact_nonlocal_energy(int strain = -1, double h = 0) {
    const double det = strain < 0                 ? 1
                       : aa[strain] == bb[strain] ? 1 + h
                                                  : 1 - h * h;
    const double pref = 4 * pi / std::sqrt(64 * det);
    double e = 0;
    for (auto r : atoms) {
        if (strain >= 0)
            r = strain_vector(r, strain, h, false);
        for (int l = 0; l <= 2; ++l) {
            std::array<std::array<C, 5>, 2> c{};
            const C il = l == 0 ? C{1, 0} : l == 1 ? C{0, -1} : C{-1, 0};
            for (int i = 0; i < 2; ++i)
                for (int g = 0; g < 5; ++g) {
                    auto q = modes[g];
                    if (strain >= 0)
                        q = strain_vector(q, strain, h, true);
                    const auto b = exact_projector(l, i ? 1.1 : .7, q);
                    const C phase = std::polar(1., -q.dot(r));
                    for (int m = 0; m < 2 * l + 1; ++m)
                        c[i][m] += std::conj(pref * il * phase * b[m]) * psi[g];
                }
            for (int m = 0; m < 2 * l + 1; ++m)
                e += .7 * std::norm(c[0][m]) + 1.1 * std::norm(c[1][m])
                     - .42 * std::real(std::conj(c[0][m]) * c[1][m]);
        }
    }
    return e;
}
TEST(PwRadialDerivatives,
     CompleteNonlocalTensorMatchesAffineEnergyWithFullCoupling) {
    const auto data = gaussian_data();
    const double pref = 4 * pi / std::sqrt(64.);
    std::array<double, 6> w{};
    double energy = 0;
    for (const auto& r : atoms)
        for (int l = 0; l <= 2; ++l) {
            std::array<std::array<C, 5>, 2> c{};
            std::array<std::array<std::array<C, 5>, 6>, 2> eta{};
            const C il = l == 0 ? C{1, 0} : l == 1 ? C{0, -1} : C{-1, 0};
            for (int i = 0; i < 2; ++i)
                for (int g = 0; g < 5; ++g) {
                    const auto q = modes[g];
                    const auto jet =
                        nc_projector_cartesian_jet(data, 2 * l + i, q);
                    const C phase = std::polar(1., -q.dot(r));
                    for (int m = 0; m < 2 * l + 1; ++m) {
                        c[i][m] += std::conj(pref * il * phase * jet.value[m])
                                   * psi[g];
                        for (int k = 0; k < 6; ++k)
                            eta[i][k][m] +=
                                q[aa[k]]
                                * std::conj(pref * il * phase
                                            * jet.gradient[m][bb[k]])
                                * psi[g];
                    }
                }
            for (int m = 0; m < 2 * l + 1; ++m) {
                const double em =
                    .7 * std::norm(c[0][m]) + 1.1 * std::norm(c[1][m])
                    - .42 * std::real(std::conj(c[0][m]) * c[1][m]);
                energy += em;
                for (int k = 0; k < 6; ++k)
                    w[k] += (k < 3 ? em : 0)
                            + 2
                                  * std::real(std::conj(c[0][m])
                                                  * (.7 * eta[0][k][m]
                                                     - .21 * eta[1][k][m])
                                              + std::conj(c[1][m])
                                                    * (-.21 * eta[0][k][m]
                                                       + 1.1 * eta[1][k][m]));
            }
        }
    EXPECT_NEAR(energy, exact_nonlocal_energy(), 3e-11);
    for (double h : {1e-3, 3e-4, 1e-4})
        for (int k = 0; k < 6; ++k) {
            const double factor = k < 3 ? 1 : 2;
            const double fd =
                -(exact_nonlocal_energy(k, h) - exact_nonlocal_energy(k, -h))
                / (2 * h * factor);
            const double fd2 = -(exact_nonlocal_energy(k, 2 * h)
                                 - exact_nonlocal_energy(k, -2 * h))
                               / (4 * h * factor);
            EXPECT_NEAR(w[k], (4 * fd - fd2) / 3, 1e-8)
                << "component=" << k << " h=" << h;
        }
}

TEST(PwRadialDerivatives, LocalValueAndSlopeMatchAnalyticGaussianPotential) {
    const auto data = gaussian_data();
    constexpr double omega = 64, sigma = .4;
    auto zero = nc_vloc_fourier_jet(data, 0, omega);
    EXPECT_NEAR(zero.value, 2 * pi * sigma * sigma / omega, 1e-12);
    EXPECT_DOUBLE_EQ(zero.slope, 0);
    GVectorList gv;
    gv.g = {{0, 0, 0},  {.03, 0, 0}, {.3, 0, 0},
            {.7, 0, 0}, {1.3, 0, 0}, {3.1, 0, 0}};
    const auto normal = nc_vloc_to_gspace(data, gv, omega);
    for (int g = 0; g < gv.count(); ++g)
        EXPECT_NEAR(normal[g],
                    nc_vloc_fourier_jet(data, gv.g[g].norm(), omega).value,
                    1e-12);
    for (double g : {.03, .3, .7, 1.3, 3.1}) {
        const auto got = nc_vloc_fourier_jet(data, g, omega);
        const double screen =
            4 * pi / omega * std::exp(-.5 * sigma * sigma * g * g);
        EXPECT_NEAR(got.value, -screen / (g * g), 1e-10);
        EXPECT_NEAR(got.slope, screen * (sigma * sigma / g + 2 / (g * g * g)),
                    1e-9);
    }
}

TEST(PwRadialDerivatives,
     NormalProjectorSetupMatchesJetsWithRotatedFullCoupling) {
    const auto data = gaussian_data();
    GVectorList gv;
    gv.g.assign(modes.begin(), modes.end());
    const std::vector<Vec3> positions(atoms.begin(), atoms.end());
    constexpr double pref = 4 * pi / 8;
    for (const Vec3 k : {Vec3{0, 0, 0}, Vec3{.17, .05, -.08}}) {
        gv.kvec = k;
        std::vector<std::vector<PWWorkspace::KBProjG>> normal;
        setup_kb_projectors(normal, gv, data, positions, 64);
        ASSERT_EQ(normal.size(), positions.size());
        for (std::size_t a = 0; a < positions.size(); ++a) {
            std::array<std::array<std::array<C, 5>, 5>, 6> raw{};
            for (int i = 0; i < 6; ++i) {
                const int l = data.kb[i].l;
                const C il = l == 0 ? C{1, 0} : l == 1 ? C{0, -1} : C{-1, 0};
                for (int g = 0; g < 5; ++g) {
                    const auto q = gv.g[g] + k;
                    const auto jet = nc_projector_cartesian_jet(data, i, q);
                    const C phase = std::polar(1., -q.dot(positions[a]));
                    for (int m = 0; m < jet.components; ++m)
                        raw[i][m][g] = pref * il * phase * jet.value[m];
                }
            }
            for (int g = 0; g < 5; ++g)
                for (int h = 0; h < 5; ++h) {
                    C got{}, want{};
                    for (const auto& projector : normal[a])
                        for (int m = 0; m < 2 * projector.l + 1; ++m)
                            got += projector.d * projector.beta_G(m, g)
                                   * std::conj(projector.beta_G(m, h));
                    for (int i = 0; i < 6; ++i)
                        for (int j = 0; j < 6; ++j) {
                            if (data.kb[i].l != data.kb[j].l)
                                continue;
                            for (int m = 0; m < 2 * data.kb[i].l + 1; ++m)
                                want += raw[i][m][g] * data.d_ij[6 * i + j]
                                        * std::conj(raw[j][m][h]);
                        }
                    EXPECT_NEAR(std::abs(got - want), 0, 3e-11);
                }
        }
    }
}
double exact_local_energy(int strain = -1, double h = 0) {
    const double det = strain < 0                 ? 1
                       : aa[strain] == bb[strain] ? 1 + h
                                                  : 1 - h * h;
    const double omega = 64 * det;
    double e = 0;
    for (int g = 0; g < 5; ++g)
        for (int sign : {-1, 1}) {
            if (g == 0 && sign == -1)
                continue;
            auto q = modes[g] * double(sign);
            if (strain >= 0)
                q = strain_vector(q, strain, h, true);
            C structure{0, 0};
            for (auto r : atoms) {
                if (strain >= 0)
                    r = strain_vector(r, strain, h, false);
                structure += std::polar(1., -q.dot(r));
            }
            C rho = g == 0 ? C{2. / 64, 0} : C{.002 * g, -.003};
            if (sign == -1)
                rho = std::conj(rho);
            rho /= det;
            const double g2 = q.norm_squared();
            const double v = g == 0
                                 ? 2 * pi * .16 / omega
                                 : -4 * pi / omega * std::exp(-.08 * g2) / g2;
            e += omega * std::real(std::conj(rho) * structure) * v;
        }
    return e;
}
TEST(PwRadialDerivatives, CompleteLocalTensorIncludesFiniteZeroModeAndVolume) {
    const auto data = gaussian_data();
    std::array<double, 6> w{};
    double energy = 0;
    for (int g = 0; g < 5; ++g)
        for (int sign : {-1, 1}) {
            if (g == 0 && sign == -1)
                continue;
            const auto q = modes[g] * double(sign);
            C structure{0, 0};
            for (const auto& r : atoms)
                structure += std::polar(1., -q.dot(r));
            C rho = g == 0 ? C{2. / 64, 0} : C{.002 * g, -.003};
            if (sign == -1)
                rho = std::conj(rho);
            const auto jet = nc_vloc_fourier_jet(data, q.norm(), 64);
            const double factor = 64 * std::real(std::conj(rho) * structure);
            const double eg = factor * jet.value;
            energy += eg;
            for (int k = 0; k < 6; ++k)
                w[k] += (k < 3 ? eg : 0)
                        + (g == 0 ? 0
                                  : factor * jet.slope * q[aa[k]] * q[bb[k]]
                                        / q.norm());
        }
    EXPECT_NEAR(energy, exact_local_energy(), 1e-11);
    for (double h : {1e-3, 3e-4, 1e-4})
        for (int k = 0; k < 6; ++k) {
            const double factor = k < 3 ? 1 : 2;
            const double fd =
                -(exact_local_energy(k, h) - exact_local_energy(k, -h))
                / (2 * h * factor);
            const double fd2 =
                -(exact_local_energy(k, 2 * h) - exact_local_energy(k, -2 * h))
                / (4 * h * factor);
            EXPECT_NEAR(w[k], (4 * fd - fd2) / 3, 1e-9)
                << "component=" << k << " h=" << h;
        }
}
} // namespace

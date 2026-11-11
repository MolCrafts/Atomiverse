#include "atomiverse/cpu/planewave/internal/frozen_operator_virial.h"
#include "atomiverse/cpu/planewave/internal/poisson.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <tuple>

using namespace ATV;
using namespace ATV::cpu::pw::internal;
namespace {
using C = std::complex<AccReal>;
constexpr AccReal pi = 3.14159265358979323846;
void close(AccReal actual, AccReal expected, const char* why) {
    EXPECT_TRUE(std::isfinite(actual)) << why;
    EXPECT_NEAR(actual, expected, 1e-8 + 1e-7 * std::abs(expected)) << why;
}
NcPotentialData data() {
    NcPotentialData d;
    d.atomic_number = 1;
    d.z_val = 1;
    d.l_max = 2;
    const int n = 2001;
    const double dr = .004, s = .4;
    d.r.resize(n);
    d.rab.assign(n, dr);
    d.v_loc.resize(n);
    for (int i = 0; i < n; ++i) {
        const double r = i * dr;
        d.r[i] = r;
        d.v_loc[i] = i ? -std::erf(r / (std::sqrt(2.) * s)) / r
                       : -std::sqrt(2. / pi) / s;
    }
    for (int ell = 0; ell <= 2; ++ell)
        for (double alpha : {.7, 1.1}) {
            NcPotentialData::KBProjector p;
            p.l = ell;
            p.radial_points = n;
            p.beta.resize(n);
            for (int i = 0; i < n; ++i)
                p.beta[i] = std::pow(d.r[i], ell + 1)
                            * std::exp(-alpha * d.r[i] * d.r[i]);
            d.kb.push_back(p);
        }
    d.d_ij.assign(36, 0);
    for (int ell = 0; ell <= 2; ++ell) {
        const int p = 2 * ell;
        d.d_ij[6 * p + p] = .7;
        d.d_ij[6 * (p + 1) + p + 1] = -.4;
        d.d_ij[6 * p + p + 1] = d.d_ij[6 * (p + 1) + p] = -.21;
    }
    d.validate();
    return d;
}
GVectorList modes() {
    GVectorList g;
    g.g = {{0, 0, 0},      {.3, -.2, .4},          {-.3, .2, -.4},
           {1.1, .4, -.7}, {-1.1, -.4, .7},        {-.7, .8, .2},
           {.7, -.8, -.2}, {1e-12, -2e-12, 3e-12}, {-1e-12, 2e-12, -3e-12}};
    g.nx = g.ny = g.nz = 8;
    for (int i = 0; i < g.count(); ++i) {
        g.h.push_back(i);
        g.k.push_back(0);
        g.l.push_back(0);
        g.ekin.push_back(.5 * g.g[i].norm_squared());
    }
    return g;
}
std::vector<C> wave(const GVectorList& g) {
    std::vector<C> c(3 * g.count());
    for (int b = 0; b < 3; ++b) {
        double norm = 0;
        for (int i = 0; i < g.count(); ++i) {
            c[b * g.count() + i] = {std::sin(.37 * (i + 1) * (b + 1)),
                                    std::cos(.29 * (i + 2) * (b + 2))};
            norm += std::norm(c[b * g.count() + i]);
        }
        for (int i = 0; i < g.count(); ++i)
            c[b * g.count() + i] /= std::sqrt(norm);
    }
    return c;
}
std::vector<C> density() {
    return {{.019, 0},      {.001, .002},    {.001, -.002},
            {-.002, .0007}, {-.002, -.0007}, {.0005, -.001},
            {.0005, .001},  {7e-28, 1e-28},  {7e-28, -1e-28}};
}
Mat3 identity() {
    Mat3 f;
    for (int a = 0; a < 3; ++a)
        f(a, a) = 1;
    return f;
}
GVectorList transform(const GVectorList& g, const Mat3& f) {
    auto out = g;
    const auto reciprocal = f.inverse().transpose();
    for (auto& q : out.g)
        q = reciprocal * q;
    out.kvec = reciprocal * g.kvec;
    for (int i = 0; i < out.count(); ++i)
        out.ekin[i] = .5 * (out.g[i] + out.kvec).norm_squared();
    return out;
}
std::vector<Vec3> transform(std::span<const Vec3> r, const Mat3& f) {
    std::vector<Vec3> out;
    for (const auto& q : r)
        out.push_back(f * q);
    return out;
}
inline double value_local(const NcPotentialData& d, const GVectorList& g,
                          std::span<const Vec3> r, std::span<const C> rho,
                          double volume) {
    const auto vg = nc_vloc_to_gspace(d, g, volume);
    double e = 0;
    for (int i = 0; i < g.count(); ++i) {
        C structure{};
        for (const auto& a : r)
            structure += std::polar(1., -g.g[i].dot(a));
        e += volume * std::real(std::conj(rho[i]) * structure) * vg[i];
    }
    return e;
}
inline double value_nl(const NcPotentialData& d, const GVectorList& g,
                       std::span<const Vec3> r, FrozenBandView o,
                       double volume) {
    std::vector<Vec3> atoms(r.begin(), r.end());
    std::vector<std::vector<KBProjector>> proj;
    setup_kb_projectors(proj, g, d, atoms, volume);
    double e = 0;
    for (std::size_t b = 0; b < o.occupations.size(); ++b)
        for (const auto& site : proj)
            for (const auto& p : site)
                for (int m = 0; m < 2 * p.l + 1; ++m) {
                    C overlap{};
                    for (int i = 0; i < g.count(); ++i)
                        overlap += std::conj(p.beta_G(m, i))
                                   * o.coefficients[b * g.count() + i];
                    e += o.k_weight * o.occupations[b] * p.d
                         * std::norm(overlap);
                }
    return e;
}
inline double value_h(const GVectorList& g, std::span<const C> rho,
                      double volume) {
    PWWorkspace ws;
    ws.gvec = &g;
    ws.n_g = g.count();
    ws.rho_G.resize({rho.size()});
    ws.V_H_G.resize({rho.size()});
    std::copy(rho.begin(), rho.end(), ws.rho_G.begin());
    PoissonSolver poisson;
    poisson.step(ws);
    double e = 0;
    for (int i = 0; i < g.count(); ++i)
        e += .5 * volume * std::real(std::conj(rho[i]) * ws.V_H_G[i]);
    return e;
}
inline double value_t(const GVectorList& g, FrozenBandView o) {
    double e = 0;
    for (std::size_t b = 0; b < o.occupations.size(); ++b)
        for (int i = 0; i < g.count(); ++i)
            e += o.k_weight * o.occupations[b]
                 * std::norm(o.coefficients[b * g.count() + i]) * .5
                 * (g.g[i] + g.kvec).norm_squared();
    return e;
}
struct Model {
    NcPotentialData d = data();
    GVectorList g = modes();
    std::vector<C> c;
    std::vector<C> rho = density();
    const std::vector<Vec3> atoms = {{.2, .4, .7}, {1.1, -.3, .5}};
    const AccReal volume = 125;
    const std::array<AccReal, 3> occupations = {2., -.17, .43};
    Model(bool unequal, bool shifted) {
        if (unequal) {
            d.kb[1].radial_points = 578;
            d.kb[3].radial_points = 733;
            d.kb[5].radial_points = 888;
        }
        if (shifted)
            g.kvec = {.11, -.07, .03};
        c = wave(g);
    }
    FrozenBandView orbitals() const { return {c, occupations, .73}; }
    std::array<FrozenOperatorVirial, 4> terms() const {
        return {kinetic_frozen_virial(g, orbitals()),
                hartree_frozen_virial(g, rho, volume),
                local_potential_frozen_virial(d, g, atoms, rho, volume),
                nonlocal_frozen_virial(d, g, atoms, orbitals(), volume)};
    }
};
class FrozenOperatorVirialTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {};
TEST_P(FrozenOperatorVirialTest,
       IndependentEnergiesAndAllNineAffineDerivatives) {
    const auto [unequal, shifted] = GetParam();
    const Model model(unequal, shifted);
    const auto terms = model.terms();
    const auto& g = model.g;
    const auto& d = model.d;
    const auto& r = model.atoms;
    const auto& rho = model.rho;
    const auto o = model.orbitals();
    const auto volume = model.volume;
    const std::array<AccReal, 4> values = {
        value_t(g, o), value_h(g, rho, volume),
        value_local(d, g, r, rho, volume), value_nl(d, g, r, o, volume)};
    for (int t = 0; t < 4; ++t)
        close(terms[t].energy_ha, values[t],
              "independent production value transform");
    for (AccReal h : {1e-5, 3e-6})
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                SCOPED_TRACE(::testing::Message()
                             << "h=" << h << " a=" << a << " b=" << b);
                std::array<std::array<AccReal, 4>, 2> e{};
                for (int sign = 0; sign < 2; ++sign) {
                    auto f = identity();
                    f(a, b) += sign == 0 ? h : -h;
                    auto gs = transform(g, f);
                    auto rs = transform(r, f);
                    auto rh = rho;
                    for (auto& v : rh)
                        v /= f.det();
                    const auto vs = volume * f.det();
                    e[sign] = {value_t(gs, o), value_h(gs, rh, vs),
                               value_local(d, gs, rs, rh, vs),
                               value_nl(d, gs, rs, o, vs)};
                    for (int ig = 0; ig < g.count(); ++ig)
                        for (std::size_t site = 0; site < r.size(); ++site)
                            close((gs.g[ig] + gs.kvec).dot(rs[site]),
                                  (g.g[ig] + g.kvec).dot(r[site]),
                                  "affine phase cancellation");
                }
                for (int t = 0; t < 4; ++t)
                    close(-(e[0][t] - e[1][t]) / (2 * h),
                          terms[t].virial_ha[3 * a + b],
                          "frozen operator strain derivative");
            }
}
TEST_P(FrozenOperatorVirialTest, RotationCovarianceAndUnprojectedAntisymmetry) {
    const auto [unequal, shifted] = GetParam();
    const Model model(unequal, shifted);
    const auto terms = model.terms();
    auto rot = identity();
    const AccReal angle = .31;
    rot(0, 0) = rot(1, 1) = std::cos(angle);
    rot(0, 1) = -std::sin(angle);
    rot(1, 0) = std::sin(angle);
    const auto gr = transform(model.g, rot);
    const auto rr = transform(model.atoms, rot);
    const auto o = model.orbitals();
    const std::array<FrozenOperatorVirial, 4> rotated = {
        kinetic_frozen_virial(gr, o),
        hartree_frozen_virial(gr, model.rho, model.volume),
        local_potential_frozen_virial(model.d, gr, rr, model.rho, model.volume),
        nonlocal_frozen_virial(model.d, gr, rr, o, model.volume)};
    for (int t = 0; t < 4; ++t) {
        close(rotated[t].energy_ha, terms[t].energy_ha, "rotated energy");
        Mat3 tensor;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                tensor(a, b) = terms[t].virial_ha[3 * a + b];
                close(terms[t].virial_ha[3 * a + b],
                      terms[t].virial_ha[3 * b + a],
                      "unprojected antisymmetry");
            }
        const auto want = rot * tensor * rot.transpose();
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                close(rotated[t].virial_ha[3 * a + b], want(a, b),
                      "rotated tensor");
    }
}
TEST_P(FrozenOperatorVirialTest, CompleteDAndSignedOccupationsAreNontrivial) {
    const auto [unequal, shifted] = GetParam();
    const Model model(unequal, shifted);
    const auto terms = model.terms();
    auto dropped = model.d;
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            if (i != j)
                dropped.d_ij[6 * i + j] = 0;
    EXPECT_GT(std::abs(nonlocal_frozen_virial(dropped, model.g, model.atoms,
                                              model.orbitals(), model.volume)
                           .energy_ha
                       - terms[3].energy_ha),
              1e-5);
    auto clipped = model.occupations;
    for (auto& f : clipped)
        f = std::max(f, AccReal(0));
    const FrozenBandView positive{model.c, clipped, .73};
    EXPECT_GT(std::abs(kinetic_frozen_virial(model.g, positive).energy_ha
                       - terms[0].energy_ha),
              1e-4);
    EXPECT_GT(std::abs(nonlocal_frozen_virial(model.d, model.g, model.atoms,
                                              positive, model.volume)
                           .energy_ha
                       - terms[3].energy_ha),
              1e-5);
}
INSTANTIATE_TEST_SUITE_P(FullAndUnequalPrefixesAtGammaAndNonzeroK,
                         FrozenOperatorVirialTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));
TEST(FrozenOperatorVirial, ProjectorPrefixesIgnoreUnusedSuffix) {
    Model model(true, false);
    const auto before = model.terms()[3];
    for (auto& p : model.d.kb)
        for (std::size_t i = p.radial_points; i < p.beta.size(); ++i)
            p.beta[i] = 1e20;
    const auto after = model.terms()[3];
    EXPECT_DOUBLE_EQ(before.energy_ha, after.energy_ha);
    for (int ab = 0; ab < 9; ++ab)
        EXPECT_DOUBLE_EQ(before.virial_ha[ab], after.virial_ha[ab]);
}
TEST(FrozenOperatorVirial, UniformModeHasOnlyLocalVolumeContribution) {
    const Model model(false, false);
    GVectorList zero;
    zero.g = {{0, 0, 0}};
    zero.h = {0};
    zero.k = {0};
    zero.l = {0};
    zero.ekin = {0};
    const std::array<C, 1> rho = {C{.019, 0}};
    const auto h = hartree_frozen_virial(zero, rho, model.volume),
               local = local_potential_frozen_virial(model.d, zero, model.atoms,
                                                     rho, model.volume);
    EXPECT_EQ(h.energy_ha, 0);
    EXPECT_GT(std::abs(local.energy_ha), 1e-4);
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            EXPECT_EQ(h.virial_ha[3 * a + b], 0);
            EXPECT_DOUBLE_EQ(local.virial_ha[3 * a + b],
                             a == b ? local.energy_ha : 0);
        }
}
TEST(FrozenOperatorVirial, RejectsMismatchedShapesAndNonpositiveVolume) {
    const Model model(false, false);
    const std::array<C, 1> wrong = {C{0, 0}};
    const FrozenBandView bad{wrong, model.occupations, .73};
    EXPECT_THROW(kinetic_frozen_virial(model.g, bad), std::invalid_argument);
    EXPECT_THROW(nonlocal_frozen_virial(model.d, model.g, model.atoms, bad,
                                        model.volume),
                 std::invalid_argument);
    EXPECT_THROW(hartree_frozen_virial(model.g, wrong, model.volume),
                 std::invalid_argument);
    EXPECT_THROW(local_potential_frozen_virial(model.d, model.g, model.atoms,
                                               wrong, model.volume),
                 std::invalid_argument);
    for (AccReal volume : {0., -1.}) {
        EXPECT_THROW(hartree_frozen_virial(model.g, model.rho, volume),
                     std::invalid_argument);
        EXPECT_THROW(local_potential_frozen_virial(
                         model.d, model.g, model.atoms, model.rho, volume),
                     std::invalid_argument);
        EXPECT_THROW(nonlocal_frozen_virial(model.d, model.g, model.atoms,
                                            model.orbitals(), volume),
                     std::invalid_argument);
    }
}
} // namespace

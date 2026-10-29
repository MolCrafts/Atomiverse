// One XC quadrature component, no SCF or public stress calls.
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include "atomiverse/cpu/planewave/internal/xc_quadrature.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>

using namespace ATV;
using namespace ATV::cpu::pw::internal;
using Complex = std::complex<AccReal>;
namespace {
constexpr double pi = 3.1415926535897932384626433832795;
constexpr int aa[6] = {0, 1, 2, 0, 0, 1}, bb[6] = {0, 1, 2, 1, 2, 2};
struct Case {
    cpu::xc::XCType xc;
    int spin;
    bool clipped;
    const char* name;
};
Mat3 cell() {
    Mat3 h;
    h(0, 0) = 6;
    h(1, 1) = 5;
    h(2, 2) = 4;
    h(0, 1) = .4;
    h(0, 2) = .2;
    h(1, 2) = .3;
    return h;
}
void configure(PWWorkspace& ws, int spin, bool clipped, bool uniform = false) {
    ws.fft_nx = ws.fft_ny = ws.fft_nz = 6;
    ws.nr = 216;
    ws.nc = 4 * 6 * 6;
    ws.fft = std::make_unique<FFTPlan3D>(6, 6, 6);
    ws.nspin = spin;
    ws.omega = cell().det();
    const auto reciprocal = cell().inverse().transpose() * (2 * pi);
    constexpr int modes[5][3] = {
        {0, 0, 0}, {1, 1, 0}, {0, 1, 1}, {1, 0, 1}, {2, -1, 1}};
    constexpr Complex alpha[5] = {
        {.04, 0}, {.006, .002}, {-.004, .001}, {.002, -.001}, {.0015, .0007}};
    constexpr Complex beta[5] = {
        {.03, 0}, {-.004, -.002}, {.003, -.001}, {.002, .001}, {-.001, .0005}};
    std::vector<Complex> ca, cb;
    for (int m = 0; m < 5; ++m)
        for (int sign : {1, -1}) {
            if (m == 0 && sign < 0)
                continue;
            const auto& n = modes[m];
            ws.gvec0.h.push_back(sign * n[0]);
            ws.gvec0.k.push_back(sign * n[1]);
            ws.gvec0.l.push_back(sign * n[2]);
            ws.gvec0.g.push_back(reciprocal
                                 * Vec3{double(sign * n[0]),
                                        double(sign * n[1]),
                                        double(sign * n[2])});
            ca.push_back(sign > 0 ? alpha[m] : std::conj(alpha[m]));
            cb.push_back(sign > 0 ? beta[m] : std::conj(beta[m]));
            if (m == 0 && clipped)
                cb.back() = .003;
            if (uniform && m != 0) {
                ca.back() = cb.back() = 0;
            }
        }
    ws.n_g = static_cast<int>(ca.size());
    ws.gvec = &ws.gvec0;
    ws.rho_G = xt::zeros<Complex>({ca.size()});
    ws.V_xc_G = xt::zeros<Complex>({ca.size()});
    for (std::size_t g = 0; g < ca.size(); ++g)
        ws.rho_G[g] = ca[g] + cb[g];
    if (spin == 2) {
        ws.rho_spin_G.resize(2);
        for (auto& c : ws.rho_spin_G)
            c = xt::zeros<Complex>({ca.size()});
        for (std::size_t g = 0; g < ca.size(); ++g) {
            ws.rho_spin_G[0][g] = ca[g];
            ws.rho_spin_G[1][g] = cb[g];
        }
    }
}
void evaluate(PWWorkspace& ws, XCBuilder& builder, cpu::xc::XCType xc) {
    if (ws.nspin == 2)
        builder.step_spin(ws, xc);
    else
        builder.step(ws, xc);
}
struct Oracle {
    double energy = 0, min_cross = 0, max_clipped_flux = 0;
    int clipped_points = 0;
};
// Independent direct Fourier quadrature at the exact existing padded shape.
// Only the existing point energy/derivatives are called; no builder FFT map.
Oracle direct(const PWWorkspace& ws, cpu::xc::XCType xc) {
    const bool gga = cpu::xc::is_gga(xc);
    const int nx = xc_quadrature_dimension(ws.fft_nx, gga),
              ny = xc_quadrature_dimension(ws.fft_ny, gga),
              nz = xc_quadrature_dimension(ws.fft_nz, gga);
    Oracle out;
    const auto& gv = *ws.gvec;
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                double rho[2]{}, grad[2][3]{};
                for (int s = 0; s < 2; ++s)
                    for (int g = 0; g < ws.n_g; ++g) {
                        const auto c = ws.nspin == 2 ? ws.rho_spin_G[s][g]
                                                     : .5 * ws.rho_G[g];
                        const double phase = 2 * pi
                                             * (double(gv.h[g]) * x / nx
                                                + double(gv.k[g]) * y / ny
                                                + double(gv.l[g]) * z / nz);
                        const auto value = c * std::exp(Complex(0, phase));
                        rho[s] += value.real();
                        if (gga)
                            for (int d = 0; d < 3; ++d)
                                grad[s][d] +=
                                    (Complex(0, gv.g[g][d]) * value).real();
                    }
                double sigma[3]{};
                for (int d = 0; d < 3; ++d) {
                    sigma[0] += grad[0][d] * grad[0][d];
                    sigma[1] += grad[0][d] * grad[1][d];
                    sigma[2] += grad[1][d] * grad[1][d];
                }
                out.min_cross = std::min(out.min_cross, sigma[1]);
                const double ra = std::max(rho[0], 0.),
                             rb = std::max(rho[1], 0.);
                if (ra + rb < 1e-12)
                    continue;
                const auto point = cpu::xc::evaluate_xc(xc, ra, rb, sigma[0],
                                                        sigma[1], sigma[2]);
                out.energy += point.energy_density;
                if (rho[1] < 0) {
                    ++out.clipped_points;
                    double norm = 0;
                    for (int d = 0; d < 3; ++d) {
                        const double f = 2 * point.vsigma_bb * grad[1][d]
                                         + point.vsigma_ab * grad[0][d];
                        norm += f * f;
                    }
                    out.max_clipped_flux =
                        std::max(out.max_clipped_flux, std::sqrt(norm));
                }
            }
    out.energy *= ws.omega / (nx * ny * nz);
    return out;
}
struct Frozen {
    AccReal omega;
    std::vector<Vec3> metric;
    xt::xarray<Complex> rho;
    std::vector<xt::xarray<Complex>> spins;
    explicit Frozen(const PWWorkspace& ws)
        : omega(ws.omega), metric(ws.gvec->g), rho(ws.rho_G),
          spins(ws.rho_spin_G) {}
    void apply(PWWorkspace& ws, int lane, double h) const {
        Mat3 f;
        for (int k = 0; k < 3; ++k)
            f(k, k) = 1;
        f(aa[lane], bb[lane]) += h;
        if (aa[lane] != bb[lane])
            f(bb[lane], aa[lane]) += h;
        const double j = f.det();
        ws.omega = omega * j;
        const auto inverse = f.inverse().transpose();
        for (std::size_t g = 0; g < metric.size(); ++g)
            ws.gvec0.g[g] = inverse * metric[g];
        ws.rho_G = rho / j;
        if (ws.nspin == 2)
            for (int s = 0; s < 2; ++s)
                ws.rho_spin_G[s] = spins[s] / j;
    }
};
class PwXcVirial : public ::testing::TestWithParam<Case> {};
TEST_P(PwXcVirial, ExactExistingDiscreteEnergyMatchesAllSixAffineDerivatives) {
    const auto c = GetParam();
    PWWorkspace ws;
    configure(ws, c.spin, c.clipped);
    XCBuilder builder;
    builder.prepare(ws);
    evaluate(ws, builder, c.xc);
    const auto virial = builder.virial_ha();
    const auto baseline = direct(ws, c.xc);
    EXPECT_NEAR(ws.E_xc, baseline.energy, 2e-11);
    if (cpu::xc::is_gga(c.xc) && c.spin == 2) {
        EXPECT_LT(baseline.min_cross, -1e-5);
        for (int k = 3; k < 6; ++k)
            EXPECT_GT(std::abs(virial[k]), 1e-6);
    }
    if (c.clipped) {
        EXPECT_GT(baseline.clipped_points, 0);
        if (cpu::xc::is_gga(c.xc))
            EXPECT_GT(baseline.max_clipped_flux, 1e-8);
    }
    const Frozen reference(ws);
    double largest = 0;
    for (double h : {1e-3, 3e-4, 1e-4})
        for (int lane = 0; lane < 6; ++lane) {
            double energies[4];
            const double offsets[4] = {h, -h, 2 * h, -2 * h};
            for (int s = 0; s < 4; ++s) {
                reference.apply(ws, lane, offsets[s]);
                evaluate(ws, builder, c.xc);
                energies[s] = ws.E_xc;
            }
            const double fac = lane < 3 ? 1 : 2;
            const double fd =
                -(8 * (energies[0] - energies[1]) - (energies[2] - energies[3]))
                / (12 * h * fac);
            largest = std::max(largest, std::abs(fd - virial[lane]));
            EXPECT_NEAR(fd, virial[lane], 1e-8)
                << "h=" << h << " lane=" << lane;
        }
    reference.apply(ws, 0, 0);
    evaluate(ws, builder, c.xc);
    for (int k = 0; k < 6; ++k)
        EXPECT_DOUBLE_EQ(builder.virial_ha()[k], virial[k]);
    std::printf("XC_VIRIAL %s max_five_point_error_Ha=%.12g\n", c.name,
                largest);
}
std::vector<Case> cases() {
    std::vector<Case> out;
    static const char* names[3][5] = {
        {"RKS_LSDA", "RKS_PBE", "RKS_PBEsol", "RKS_revPBE", "RKS_RPBE"},
        {"UKS_LSDA", "UKS_PBE", "UKS_PBEsol", "UKS_revPBE", "UKS_RPBE"},
        {"Clipped_LSDA", "Clipped_PBE", "Clipped_PBEsol", "Clipped_revPBE",
         "Clipped_RPBE"}};
    constexpr cpu::xc::XCType types[] = {
        cpu::xc::XCType::LSDA, cpu::xc::XCType::PBE, cpu::xc::XCType::PBEsol,
        cpu::xc::XCType::RevPBE, cpu::xc::XCType::RPBE};
    for (int style = 0; style < 3; ++style)
        for (int x = 0; x < 5; ++x)
            out.push_back(
                {types[x], style == 0 ? 1 : 2, style == 2, names[style][x]});
    return out;
}
INSTANTIATE_TEST_SUITE_P(FixedQuadrature, PwXcVirial,
                         ::testing::ValuesIn(cases()),
                         [](const auto& p) { return p.param.name; });

TEST(PwXcVirialStandalone, UniformLdaPressureAndRestrictedSpinCollapse) {
    for (auto xc :
         {cpu::xc::XCType::LSDA, cpu::xc::XCType::PBE, cpu::xc::XCType::PBEsol,
          cpu::xc::XCType::RevPBE, cpu::xc::XCType::RPBE}) {
        PWWorkspace r, s;
        configure(r, 1, false, true);
        configure(s, 2, false, true);
        s.rho_spin_G[0] = .5 * r.rho_G;
        s.rho_spin_G[1] = .5 * r.rho_G;
        s.rho_G = r.rho_G;
        XCBuilder br, bs;
        br.prepare(r);
        bs.prepare(s);
        evaluate(r, br, xc);
        evaluate(s, bs, xc);
        const auto point = cpu::xc::evaluate_xc(xc, .035, .035);
        const double pressure =
            r.omega
            * (.035 * point.vrho_alpha + .035 * point.vrho_beta
               - point.energy_density);
        EXPECT_NEAR(r.E_xc, s.E_xc, 2e-12);
        for (int k = 0; k < 6; ++k) {
            EXPECT_NEAR(br.virial_ha()[k], bs.virial_ha()[k], 2e-12);
            EXPECT_NEAR(br.virial_ha()[k], k < 3 ? pressure : 0, 2e-11);
        }
    }
}
TEST(PwXcVirialStandalone, ResponsePreservesLastCompletedVirial) {
    PWWorkspace ws;
    configure(ws, 2, false);
    XCBuilder builder;
    builder.prepare(ws);
    evaluate(ws, builder, cpu::xc::XCType::PBE);
    const auto before = builder.virial_ha();
    std::vector<Complex> variation(ws.n_g, Complex{});
    std::array<std::vector<AccReal>, 2> output;
    for (auto& v : output)
        v.resize(ws.nr);
    std::array<XCBuilder::ResponseChannel, 2> channels;
    for (int s = 0; s < 2; ++s)
        channels[s] = {{ws.rho_spin_G[s].data(), std::size_t(ws.n_g)},
                       variation,
                       output[s]};
    builder.response(ws, cpu::xc::XCType::PBE, channels, 1e-4);
    EXPECT_EQ(builder.virial_ha(), before);
}
TEST(PwXcVirialStandalone,
     InvalidOrIncompleteEvaluationPreservesCompletedVirial) {
    PWWorkspace ws;
    configure(ws, 2, false);
    XCBuilder builder;
    builder.prepare(ws);
    evaluate(ws, builder, cpu::xc::XCType::PBE);
    const auto before = builder.virial_ha();
    const auto channel = ws.rho_spin_G[1];
    ws.rho_spin_G[1] = xt::zeros<Complex>({1});
    EXPECT_THROW(builder.step_spin(ws, cpu::xc::XCType::PBE),
                 std::invalid_argument);
    EXPECT_EQ(builder.virial_ha(), before);
    ws.rho_spin_G[1] = channel;
    ws.fft.reset();
    builder.step_spin(ws, cpu::xc::XCType::PBE);
    EXPECT_EQ(builder.virial_ha(), before);
}
} // namespace

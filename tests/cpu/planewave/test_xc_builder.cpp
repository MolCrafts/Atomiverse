/// @file CPU PW XC quadrature tests: Fourier fields only, no SCF.
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

using namespace ATV;
using namespace ATV::cpu::pw::internal;
using Complex = std::complex<AccReal>;

namespace {
void configure(PWWorkspace& ws, int spin, double scale = 1.0) {
    ws.fft_nx = 8;
    ws.fft_ny = 6;
    ws.fft_nz = 4;
    ws.nr = 8 * 6 * 4;
    ws.nc = 5 * 6 * 4;
    ws.fft = std::make_unique<FFTPlan3D>(8, 6, 4);
    ws.omega = 6 * 7 * 8 * scale * scale * scale;
    ws.nspin = spin;
    ws.gvec0 = GVectorList{};
    constexpr int modes[5][3] = {
        {0, 0, 0}, {3, 0, 0}, {-3, 0, 0}, {0, 2, 1}, {0, -2, -1}};
    const double lengths[3] = {6 * scale, 7 * scale, 8 * scale};
    for (const auto& mode : modes) {
        ws.gvec0.h.push_back(mode[0]);
        ws.gvec0.k.push_back(mode[1]);
        ws.gvec0.l.push_back(mode[2]);
        ws.gvec0.g.push_back({2 * M_PI * mode[0] / lengths[0],
                              2 * M_PI * mode[1] / lengths[1],
                              2 * M_PI * mode[2] / lengths[2]});
    }
    ws.n_g = 5;
    ws.gvec = &ws.gvec0;
    ws.rho_G =
        xt::xarray<Complex>{.04, Complex(.003, .001), Complex(.003, -.001),
                            Complex(.002, -.0007), Complex(.002, .0007)};
    ws.V_xc_G = xt::zeros<Complex>({5});
    if (spin == 2) {
        ws.rho_spin_G.resize(2);
        ws.rho_spin_G[0] = .6 * ws.rho_G;
        ws.rho_spin_G[1] = .4 * ws.rho_G;
    }
}

// Independent analytic Fourier interpolation/gradient and direct trapezoid
// integration. Uses current Cartesian G vectors, not the builder's FFT map.
double quadrature_energy(const PWWorkspace& ws, cpu::xc::XCType type) {
    const auto pad = [type](int n) {
        return cpu::xc::is_gga(type) ? 3 * n : 2 * ((3 * n + 3) / 4);
    };
    const int nx = pad(ws.fft_nx), ny = pad(ws.fft_ny), nz = pad(ws.fft_nz);
    const auto& gv = *ws.gvec;
    double energy = 0;
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                double rho[2] = {0, 0};
                double grad[2][3]{};
                for (int spin = 0; spin < 2; ++spin)
                    for (int g = 0; g < ws.n_g; ++g) {
                        const auto coefficient = ws.nspin == 2
                                                     ? ws.rho_spin_G[spin][g]
                                                     : .5 * ws.rho_G[g];
                        const double phase = 2 * M_PI
                                             * (double(gv.h[g]) * x / nx
                                                + double(gv.k[g]) * y / ny
                                                + double(gv.l[g]) * z / nz);
                        const auto value =
                            coefficient * std::exp(Complex(0, phase));
                        rho[spin] += value.real();
                        for (int axis = 0; axis < 3; ++axis)
                            grad[spin][axis] +=
                                (Complex(0, gv.g[g][axis]) * value).real();
                    }
                double aa = 0, ab = 0, bb = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    aa += grad[0][axis] * grad[0][axis];
                    ab += grad[0][axis] * grad[1][axis];
                    bb += grad[1][axis] * grad[1][axis];
                }
                rho[0] = std::max(rho[0], 0.0);
                rho[1] = std::max(rho[1], 0.0);
                if (rho[0] + rho[1] >= 1e-12)
                    energy +=
                        cpu::xc::evaluate_xc(type, rho[0], rho[1], aa, ab, bb)
                            .energy_density;
            }
    return energy * ws.omega / (nx * ny * nz);
}

void expect_base_interpolation(const PWWorkspace& ws,
                               const xt::xarray<Complex>& coefficients,
                               const xt::xarray<AccReal>& actual) {
    ASSERT_EQ(actual.size(), static_cast<std::size_t>(ws.nr));
    const auto& gv = *ws.gvec;
    for (int z = 0; z < ws.fft_nz; ++z)
        for (int y = 0; y < ws.fft_ny; ++y)
            for (int x = 0; x < ws.fft_nx; ++x) {
                Complex value{};
                for (int g = 0; g < ws.n_g; ++g) {
                    const double phase = 2 * M_PI
                                         * (double(gv.h[g]) * x / ws.fft_nx
                                            + double(gv.k[g]) * y / ws.fft_ny
                                            + double(gv.l[g]) * z / ws.fft_nz);
                    value += coefficients[g] * std::exp(Complex(0, phase));
                }
                const int at = x + ws.fft_nx * (y + ws.fft_ny * z);
                EXPECT_NEAR(actual[at], value.real(), 2e-12);
            }
}

void expect_energy_adjoint(PWWorkspace& ws, XCBuilder& builder,
                           cpu::xc::XCType type = cpu::xc::XCType::PBE) {
    xt::xarray<Complex> direction =
        xt::zeros<Complex>({static_cast<std::size_t>(ws.n_g)});
    for (int g = 0; g < ws.n_g; ++g) {
        const auto& gv = *ws.gvec;
        const int magnitude =
            std::abs(gv.h[g]) + std::abs(gv.k[g]) + std::abs(gv.l[g]);
        const int signed_index = gv.h[g] + gv.k[g] + gv.l[g];
        direction[g] =
            magnitude == 0
                ? Complex(.13, 0)
                : Complex(.2 / magnitude, -.11 * signed_index / magnitude);
    }
    const auto base = ws.rho_G;
    const auto spins = ws.rho_spin_G;
    if (ws.nspin == 2)
        builder.step_spin(ws, type);
    else
        builder.step(ws, type);
    EXPECT_NEAR(ws.E_xc, quadrature_energy(ws, type), 2e-11);
    double expected = 0, spin_integral = 0;
    for (int g = 0; g < ws.n_g; ++g) {
        if (ws.nspin == 1)
            expected += (std::conj(direction[g]) * ws.V_xc_G[g]).real();
        else
            for (int spin = 0; spin < 2; ++spin) {
                const double weight = spin == 0 ? .7 : -.3;
                expected +=
                    weight
                    * (std::conj(direction[g]) * ws.V_xc_spin_G[spin][g])
                          .real();
                spin_integral +=
                    (std::conj(spins[spin][g]) * ws.V_xc_spin_G[spin][g])
                        .real();
            }
    }
    if (ws.nspin == 2) {
        EXPECT_NEAR(ws.e_xc_int_spin, ws.omega * spin_integral, 2e-11);
        for (int spin = 0; spin < 2; ++spin)
            expect_base_interpolation(ws, ws.V_xc_spin_G[spin],
                                      ws.V_xc_spin_r[spin]);
    } else
        expect_base_interpolation(ws, ws.V_xc_G, ws.V_xc_r);
    constexpr double h = 1e-6;
    double energies[2];
    for (int sign = 0; sign < 2; ++sign) {
        const double delta = sign == 0 ? h : -h;
        if (ws.nspin == 1) {
            ws.rho_G = base + delta * direction;
            builder.step(ws, type);
        } else {
            ws.rho_spin_G[0] = spins[0] + .7 * delta * direction;
            ws.rho_spin_G[1] = spins[1] - .3 * delta * direction;
            ws.rho_G = ws.rho_spin_G[0] + ws.rho_spin_G[1];
            builder.step_spin(ws, type);
        }
        energies[sign] = ws.E_xc;
    }
    EXPECT_NEAR((energies[0] - energies[1]) / (2 * h), ws.omega * expected,
                2e-7);
    ws.rho_G = base;
    ws.rho_spin_G = spins;
}
} // namespace

TEST(PwXcQuadrature, UniformDensityPreservesChargeAndEnergyNormalization) {
    for (auto type : {cpu::xc::XCType::LSDA, cpu::xc::XCType::PBE}) {
        PWWorkspace ws;
        configure(ws, 1);
        ws.rho_G.fill(Complex(0));
        ws.rho_G[0] = .04;
        XCBuilder builder;
        builder.prepare(ws);
        builder.step(ws, type);
        const auto reference = cpu::xc::evaluate_xc(type, .02, .02);
        EXPECT_NEAR(ws.E_xc, ws.omega * reference.energy_density, 2e-11);
        for (double value : ws.V_xc_r)
            EXPECT_NEAR(value, reference.vrho_alpha, 2e-12);
        EXPECT_NEAR(ws.V_xc_G[0].real(), reference.vrho_alpha, 2e-12);
        for (int g = 1; g < ws.n_g; ++g)
            EXPECT_LT(std::abs(ws.V_xc_G[g]), 2e-12);
    }
}

TEST(PwXcQuadrature, RestrictedAndSpinWeightedEnergyAdjointsNearNyquist) {
    for (int spin : {1, 2})
        for (auto type : {cpu::xc::XCType::LSDA, cpu::xc::XCType::PBE}) {
            PWWorkspace ws;
            configure(ws, spin);
            XCBuilder builder;
            builder.prepare(ws);
            expect_energy_adjoint(ws, builder, type);
        }
}

TEST(PwXcQuadrature, SameShapeChangedCellAndModeOrderingUseCurrentMetric) {
    PWWorkspace ws;
    configure(ws, 2);
    XCBuilder builder;
    builder.prepare(ws);
    expect_energy_adjoint(ws, builder);
    configure(ws, 2, 1.37);
    ws.gvec0.h[1] = 2;
    ws.gvec0.h[2] = -2;
    ws.gvec0.g[1][0] *= 2.0 / 3;
    ws.gvec0.g[2][0] *= 2.0 / 3;
    // Change list order, including G=0, at the same FFT shape.
    for (int g = 0; g < 2; ++g) {
        const int other = 4 - g;
        std::swap(ws.gvec0.h[g], ws.gvec0.h[other]);
        std::swap(ws.gvec0.k[g], ws.gvec0.k[other]);
        std::swap(ws.gvec0.l[g], ws.gvec0.l[other]);
        std::swap(ws.gvec0.g[g], ws.gvec0.g[other]);
        std::swap(ws.rho_G[g], ws.rho_G[other]);
        for (int spin = 0; spin < 2; ++spin)
            std::swap(ws.rho_spin_G[spin][g], ws.rho_spin_G[spin][other]);
    }
    // The cached plan depends only on shape; mapping and metric must be
    // consumed from this new cell/list even without another allocation.
    expect_energy_adjoint(ws, builder);
}

TEST(PwXcQuadrature, PhysicalDensitySpaceRejectsAliasedNyquistModes) {
    PWWorkspace ws;
    configure(ws, 1);
    XCBuilder builder;
    builder.prepare(ws);
    ws.gvec0.h[1] = 4;
    ws.gvec0.h[2] = -4;
    EXPECT_THROW(builder.step(ws, cpu::xc::XCType::PBE), std::invalid_argument);
}

TEST(PwXcQuadrature,
     ClippedSpinRingingPreservesEnergyVariationAndDoubleCounting) {
    for (auto type : {cpu::xc::XCType::LSDA, cpu::xc::XCType::PBE}) {
        PWWorkspace ws;
        configure(ws, 2);
        // The alpha Fourier field is negative at some quadrature points,
        // away from clipping boundaries. Beta keeps the total density finite.
        ws.rho_spin_G[0].fill(Complex(0));
        ws.rho_spin_G[0][0] = .001;
        ws.rho_spin_G[0][1] = Complex(.002, .0007);
        ws.rho_spin_G[0][2] = std::conj(ws.rho_spin_G[0][1]);
        ws.rho_G = ws.rho_spin_G[0] + ws.rho_spin_G[1];
        XCBuilder builder;
        builder.prepare(ws);
        expect_energy_adjoint(ws, builder, type);
    }
}

namespace {
void expect_fused_response(PWWorkspace& ws, XCBuilder& response,
                           cpu::xc::XCType type,
                           bool cross_density_cutoff = false) {
    const int channels = ws.nspin;
    const auto total = ws.rho_G;
    const auto spin = ws.rho_spin_G;
    std::array<std::vector<Complex>, 2> reference, variation;
    for (int s = 0; s < channels; ++s) {
        const auto& rho = channels == 1 ? total : spin[s];
        reference[s].assign(rho.begin(), rho.end());
        variation[s].resize(rho.size());
        for (int g = 0; g < ws.n_g; ++g) {
            const auto phase = ws.gvec->h[g] + ws.gvec->k[g] - ws.gvec->l[g];
            variation[s][g] =
                rho[g] * Complex(s == 0 ? .11 : -.23, .07 * phase);
            if (cross_density_cutoff)
                variation[s][g] =
                    phase == 0 && ws.gvec->g[g].norm_squared() == 0
                        ? Complex(2e-10, 0)
                        : Complex(0, 0);
        }
    }
    constexpr double epsilon = 1e-3;
    XCBuilder full;
    full.prepare(ws);
    std::array<std::array<std::vector<AccReal>, 2>, 2> complete;
    for (int sign = 0; sign < 2; ++sign) {
        const double displacement = sign == 0 ? epsilon : -epsilon;
        for (int s = 0; s < channels; ++s) {
            auto& rho = channels == 1 ? ws.rho_G : ws.rho_spin_G[s];
            for (int g = 0; g < ws.n_g; ++g)
                rho[g] = reference[s][g] + displacement * variation[s][g];
        }
        if (channels == 1)
            full.step(ws, type);
        else {
            ws.rho_G = ws.rho_spin_G[0] + ws.rho_spin_G[1];
            full.step_spin(ws, type);
        }
        for (int s = 0; s < channels; ++s) {
            const auto& v = channels == 1 ? ws.V_xc_r : ws.V_xc_spin_r[s];
            complete[sign][s].assign(v.begin(), v.end());
        }
    }
    ws.rho_G = total;
    ws.rho_spin_G = spin;
    if (channels == 1)
        full.step(ws, type);
    else
        full.step_spin(ws, type);
    const auto potential = ws.V_xc_G;
    const auto spin_potential = ws.V_xc_spin_G;
    const auto coarse_potential = ws.V_xc_r;
    const auto coarse_spin_potential = ws.V_xc_spin_r;
    const auto energy = ws.E_xc, double_counting = ws.e_xc_int_spin;
    std::array<std::vector<AccReal>, 2> actual;
    std::array<XCBuilder::ResponseChannel, 2> fields;
    for (int s = 0; s < channels; ++s) {
        actual[s].resize(ws.nr);
        fields[s] = {reference[s], variation[s], actual[s]};
    }
    response.response(
        ws, type,
        std::span<const XCBuilder::ResponseChannel>(fields).first(channels),
        epsilon);
    for (int s = 0; s < channels; ++s)
        for (int i = 0; i < ws.nr; ++i)
            EXPECT_NEAR(actual[s][i],
                        (complete[0][s][i] - complete[1][s][i]) / (2 * epsilon),
                        5e-10);
    EXPECT_DOUBLE_EQ(ws.E_xc, energy);
    EXPECT_DOUBLE_EQ(ws.e_xc_int_spin, double_counting);
    EXPECT_TRUE(std::equal(total.begin(), total.end(), ws.rho_G.begin()));
    EXPECT_TRUE(
        std::equal(potential.begin(), potential.end(), ws.V_xc_G.begin()));
    EXPECT_TRUE(std::equal(coarse_potential.begin(), coarse_potential.end(),
                           ws.V_xc_r.begin()));
    for (int s = 0; s < (channels == 2 ? 2 : 0); ++s) {
        EXPECT_TRUE(std::equal(spin[s].begin(), spin[s].end(),
                               ws.rho_spin_G[s].begin()));
        EXPECT_TRUE(std::equal(spin_potential[s].begin(),
                               spin_potential[s].end(),
                               ws.V_xc_spin_G[s].begin()));
        EXPECT_TRUE(std::equal(coarse_spin_potential[s].begin(),
                               coarse_spin_potential[s].end(),
                               ws.V_xc_spin_r[s].begin()));
    }
}
} // namespace

TEST(PwXcResponse,
     MatchesCompleteFiniteDifferencesForRestrictedAndUnequalSpins) {
    for (int spin : {1, 2})
        for (auto type : {cpu::xc::XCType::LSDA, cpu::xc::XCType::PBE}) {
            PWWorkspace ws;
            configure(ws, spin);
            XCBuilder response;
            response.prepare(ws);
            expect_fused_response(ws, response, type);
            // Same grid shape, changed density and Cartesian metric: a cached
            // reference must follow the physical input rather than its address.
            ws.rho_G *= 1.03;
            for (auto& rho : ws.rho_spin_G)
                rho *= 1.03;
            for (auto& g : ws.gvec0.g)
                for (int d = 0; d < 3; ++d)
                    g[d] *= 1.1;
            expect_fused_response(ws, response, type);
        }
}

TEST(PwXcResponse, MatchesFiniteDifferencesAcrossClippingAndLowDensityCutoffs) {
    for (auto type : {cpu::xc::XCType::LSDA, cpu::xc::XCType::PBE}) {
        PWWorkspace ws;
        configure(ws, 2);
        ws.rho_spin_G[0].fill(Complex(0));
        ws.rho_spin_G[0][0] = .001;
        ws.rho_spin_G[0][1] = Complex(.002, .0007);
        ws.rho_spin_G[0][2] = std::conj(ws.rho_spin_G[0][1]);
        ws.rho_G = ws.rho_spin_G[0] + ws.rho_spin_G[1];
        XCBuilder response;
        response.prepare(ws);
        expect_fused_response(ws, response, type);
        for (auto& rho : ws.rho_spin_G)
            rho.fill(Complex(0));
        ws.rho_spin_G[0][0] = 4.2e-13;
        ws.rho_spin_G[1][0] = 6.5e-13;
        ws.rho_G = ws.rho_spin_G[0] + ws.rho_spin_G[1];
        expect_fused_response(ws, response, type, true);
    }
}

TEST(PwXcResponse, RejectsInvalidCallsBeforeChangingOutput) {
    PWWorkspace ws;
    configure(ws, 1);
    XCBuilder response;
    response.prepare(ws);
    std::vector<AccReal> output(ws.nr, 7);
    const XCBuilder::ResponseChannel channels[] = {
        {{ws.rho_G.data(), ws.rho_G.size()},
         {ws.rho_G.data(), ws.rho_G.size()},
         output}};
    EXPECT_THROW(response.response(ws, cpu::xc::XCType::PBE, channels, 0),
                 std::invalid_argument);
    for (const auto value : output)
        EXPECT_EQ(value, 7);
}

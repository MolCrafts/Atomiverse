#include "atomiverse/cpu/planewave/internal/density_builder.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <xtensor/generators/xbuilder.hpp>

TEST(PwDensityBuilder, RestrictedResidualCannotBeHiddenByMixing) {
    using namespace ATV;
    using namespace cpu::pw::internal;
    using Complex = std::complex<AccReal>;
    for (const AccReal damping : {AccReal(0), AccReal(.25), AccReal(.8)}) {
        PWWorkspace ws;
        ws.fft_nx = 8;
        ws.fft_ny = ws.fft_nz = 4;
        ws.nr = 8 * 4 * 4;
        ws.nc = 5 * 4 * 4;
        ws.fft = std::make_unique<FFTPlan3D>(8, 4, 4);
        ws.gvec0.h = {0, 1, -1};
        ws.gvec0.k = ws.gvec0.l = {0, 0, 0};
        ws.gvec = &ws.gvec0;
        ws.n_g = 3;
        ws.mix_beta = damping;
        ws.rho_G = xt::xarray<Complex>{1., .02, .02};
        const auto input = ws.rho_G;
        const xt::xarray<Complex> output = {1., Complex(.3, .2),
                                            Complex(.3, -.2)};
        DensityBuilder builder;
        builder.prepare(ws);
        builder.begin(ws);
        // An analytic real Fourier field, independent of the FFT extractor.
        for (int z = 0; z < 4; ++z)
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 8; ++x) {
                    const double phase = 2 * M_PI * x / 8;
                    builder.accumulation()[x + 8 * (y + 4 * z)] =
                        1 + .6 * std::cos(phase) - .4 * std::sin(phase);
                }
        EXPECT_NEAR(builder.finish(ws), std::sqrt(2 * (.28 * .28 + .2 * .2)),
                    1e-14);
        for (int g = 0; g < 3; ++g) {
            EXPECT_EQ(ws.rho_G_old[g], input[g]);
            EXPECT_NEAR(
                std::abs(ws.rho_G[g]
                         - (input[g] + damping * (output[g] - input[g]))),
                0, 1e-14);
        }
    }
}

TEST(PwDensityBuilder, OppositeSpinErrorsCannotCancelOrBeHiddenByDamping) {
    using namespace ATV;
    using namespace cpu::pw::internal;
    using Complex = std::complex<AccReal>;
    for (double damping : {0.25, 0.8}) {
        PWWorkspace ws;
        ws.n_g = 3;
        ws.nspin = 2;
        ws.mix_beta = damping;
        ws.rho_G = xt::xarray<Complex>{2., 0., 0.};
        ws.rho_G_old = ws.rho_G;
        ws.rho_spin_G = {xt::xarray<Complex>{1., 0., 0.},
                         xt::xarray<Complex>{1., 0., 0.}};
        ws.rho_spin_G_old = ws.rho_spin_G;
        const xt::xarray<Complex> alpha = {1., Complex(.3, .2),
                                           Complex(.3, -.2)};
        const xt::xarray<Complex> beta = {1., Complex(-.3, -.2),
                                          Complex(-.3, .2)};
        DensityBuilder builder;
        builder.prepare(ws);
        EXPECT_NEAR(builder.mix_spin(ws, alpha, beta), std::sqrt(.52), 1e-14);
        for (int g = 0; g < 3; ++g)
            EXPECT_EQ(ws.rho_G[g], g == 0 ? Complex(2.) : Complex(0.));
        EXPECT_EQ(ws.rho_spin_G[0][1], damping * alpha[1]);
        EXPECT_EQ(ws.rho_spin_G[1][1], damping * beta[1]);
        EXPECT_EQ(ws.rho_spin_G_old[0][1], Complex(0.));
        const auto before = ws.rho_spin_G;
        const auto history_size = ws.pulay_res.size();
        const xt::xarray<Complex> wrong = {1., 0.};
        EXPECT_THROW(builder.mix_spin(ws, alpha, wrong), std::invalid_argument);
        for (int spin = 0; spin < 2; ++spin)
            for (int g = 0; g < 3; ++g)
                EXPECT_EQ(ws.rho_spin_G[spin][g], before[spin][g]);
        EXPECT_EQ(ws.pulay_res.size(), history_size);
        EXPECT_EQ(ws.pulay_rho.size(), history_size);
    }
}

TEST(PwDensityBuilder, JointPulayMixHandlesStiffLinearMapsAndSingularHistory) {
    using namespace ATV;
    using namespace cpu::pw::internal;
    using Complex = std::complex<AccReal>;
    PWWorkspace ws;
    ws.n_g = 3;
    ws.nspin = 2;
    ws.mix_beta = AccReal(.7);
    const auto seed = [&](AccReal amplitude = AccReal(.01)) {
        const xt::xarray<Complex> initial = {1., amplitude, amplitude};
        ws.rho_spin_G = {initial, initial};
        ws.rho_spin_G_old = ws.rho_spin_G;
        ws.rho_G = xt::xarray<Complex>{2., 2 * amplitude, 2 * amplitude};
        ws.rho_G_old = ws.rho_G;
    };
    DensityBuilder builder;
    seed();
    // Preparation may see a current wave sphere smaller than the physical
    // density sphere; joint scratch follows rho_G, not the transient n_g.
    ws.n_g = 1;
    builder.prepare(ws);
    ws.n_g = 3;
    // Fixed linear beta=.7 is unstable for F(x)=-3x and F(x)=-100x.
    // Joint Pulay cancels either affine error from two samples; later calls
    // exercise dependent and zero residual histories without losing charge.
    // These are fixed sequences of method calls, with no SCF or stop rule.
    for (const AccReal amplitude : {AccReal(.01), AccReal(1e-12)})
        for (const AccReal slope : {AccReal(-3), AccReal(-100)}) {
            seed(amplitude);
            builder.reset(ws);
            for (int call = 0; call < 8; ++call) {
                auto alpha = ws.rho_spin_G[0];
                auto beta = ws.rho_spin_G[1];
                AccReal state2 = 0;
                for (int g = 1; g < 3; ++g) {
                    alpha[g] *= slope;
                    beta[g] *= slope;
                    state2 += std::norm(ws.rho_spin_G[0][g])
                              + std::norm(ws.rho_spin_G[1][g]);
                }
                const AccReal expected =
                    (AccReal(1) - slope) * std::sqrt(state2);
                EXPECT_NEAR(builder.mix_spin(ws, alpha, beta), expected,
                            AccReal(1e-14) * (AccReal(1) + expected));
                EXPECT_NEAR(ws.rho_G[0].real(), AccReal(2), 1e-13);
                EXPECT_EQ(ws.rho_G[0].imag(), AccReal(0));
                EXPECT_EQ(ws.mix_beta, AccReal(.7));
            }
            for (int spin = 0; spin < 2; ++spin)
                for (int g = 1; g < 3; ++g)
                    EXPECT_LT(std::abs(ws.rho_spin_G[spin][g]),
                              amplitude * 1e-8);
        }

    // Reset discards the previous physical density history. The first update
    // reads the live fraction and is ordinary linear mixing.
    for (AccReal fraction : {AccReal(.7), AccReal(.25)}) {
        seed();
        ws.mix_beta = fraction;
        builder.reset(ws);
        const xt::xarray<Complex> output = {1., -.03, -.03};
        builder.mix_spin(ws, output, output);
        EXPECT_NEAR(ws.rho_spin_G[0][1].real(), .01 - .04 * fraction, 1e-15);
        EXPECT_EQ(ws.mix_beta, fraction);
        EXPECT_EQ(ws.pulay_res.size(), 1u);
        EXPECT_EQ(ws.pulay_rho.size(), 1u);
    }
}

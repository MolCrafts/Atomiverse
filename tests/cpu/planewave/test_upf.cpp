/// @file test_upf.cpp
#include "atomiverse/cpu/planewave/internal/upf_reader.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

using namespace ATV;
using namespace ATV::cpu::pw::internal;

TEST(UPF, ParseMinimal) {
    // Write a minimal valid UPF v1 file
    const char* tmpfile = "/tmp/test_minimal.upf";
    {
        std::ofstream o(tmpfile);
        o << "<UPF version=\"1.0\">\n";
        o << "<PP_HEADER z_valence=\"4\" l_max=\"1\" />\n";
        o << "<PP_MESH>\n  <PP_R>\n";
        o << "0.0 0.5 1.0 1.5 2.0\n";
        o << "  </PP_R>\n  <PP_RAB>\n";
        o << "0.5 0.5 0.5 0.5 0.5\n";
        o << "  </PP_RAB>\n</PP_MESH>\n";
        o << "<PP_LOCAL>\n";
        o << "-2.0 -2.0 -2.0 -2.0 -2.0\n";
        o << "</PP_LOCAL>\n";
        o << "<PP_NLCC>\n";
        o << "0.1 0.2 0.3 0.4 0.5\n";
        o << "</PP_NLCC>\n";
        o << "<PP_NONLOCAL>\n";
        o << "  <PP_BETA angular_momentum=\"0\" cutoff_radius_index=\"3\">\n";
        o << "  0.1 0.2 0.3 0.4 0.5\n";
        o << "  </PP_BETA>\n";
        o << "  <PP_DIJ>\n  0.5\n  </PP_DIJ>\n";
        o << "</PP_NONLOCAL>\n";
        o << "</UPF>\n";
    }

    auto upf = read_upf_v1(tmpfile);
    EXPECT_EQ(upf.z_val, 4);
    EXPECT_EQ(upf.l_max, 1);
    EXPECT_EQ(upf.r.size(), 5u);
    EXPECT_EQ(upf.rab, std::vector<AccReal>(5, 0.5));
    EXPECT_EQ(upf.v_loc.size(), 5u);
    EXPECT_EQ(upf.rho_atom.size(), 5u);
    EXPECT_GE(upf.kb.size(), 1u);
    EXPECT_EQ(upf.kb[0].l, 0);
    EXPECT_EQ(upf.kb[0].beta.size(), 5u);
    EXPECT_EQ(upf.kb[0].radial_points, 3u);

    std::remove(tmpfile);
}

TEST(UPF, RadialIntegralUsesJacobianAndEvenEndpointCorrection) {
    for (int n : {9, 10}) {
        UPFData upf;
        upf.r.resize(n);
        upf.rab.assign(n, 0.125);
        std::vector<AccReal> quadratic(n);
        for (int i = 0; i < n; ++i) {
            upf.r[i] = 0.125 * i;
            quadratic[i] = upf.r[i] * upf.r[i];
        }
        const double end = upf.r.back();
        EXPECT_NEAR(upf.integrate_radial(quadratic), end * end * end / 3,
                    1e-13);
        // A declared compact projector domain uses its own last-point rule.
        const double prefix_end = upf.r[n - 2];
        EXPECT_NEAR(upf.integrate_radial(
                        std::span<const AccReal>(quadratic).first(n - 1)),
                    prefix_end * prefix_end * prefix_end / 3, 1e-13);
    }
}

TEST(UPF, RadialIntegralIsFourthOrderOnLogMesh) {
    // r=exp(t), r in [1,2]. Integral r^4 dr = 31/5. The mapped
    // integrand exp(5t) has nonvanishing endpoint derivatives, so the
    // composite Simpson error is O(dt^4), unlike the old midpoint rule.
    double errors[3];
    int index = 0;
    for (int n : {33, 65, 129}) {
        UPFData upf;
        upf.r.resize(n);
        upf.rab.resize(n);
        std::vector<AccReal> values(n);
        const double dt = std::log(2.0) / (n - 1);
        for (int i = 0; i < n; ++i) {
            const double r = std::exp(i * dt);
            upf.r[i] = r;
            upf.rab[i] = r * dt;
            values[i] = std::pow(r, 4);
        }
        errors[index++] = std::abs(upf.integrate_radial(values) - 31.0 / 5);
    }
    for (int i = 0; i < 2; ++i) {
        EXPECT_GT(errors[i] / errors[i + 1], 12.0);
        EXPECT_LT(errors[i] / errors[i + 1], 20.0);
    }
}

TEST(UPF, RadialIntegralRejectsMissingInvalidOrUnorderedData) {
    UPFData upf;
    upf.r = {0, 0.5, 1};
    std::vector<AccReal> values{1, 1, 1};
    EXPECT_THROW(upf.integrate_radial(values), std::invalid_argument);
    upf.rab = {0.5, 0.5, 0.5};
    EXPECT_THROW(
        upf.integrate_radial(std::span<const AccReal>(values).first(2)),
        std::invalid_argument);
    values[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(upf.integrate_radial(values), std::invalid_argument);
    values[1] = 1;
    upf.rab[1] = 0;
    EXPECT_THROW(upf.integrate_radial(values), std::invalid_argument);
    upf.rab[1] = 0.5;
    upf.r[2] = 0.5;
    EXPECT_THROW(upf.integrate_radial(values), std::invalid_argument);
}

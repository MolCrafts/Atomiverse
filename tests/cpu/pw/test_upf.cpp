/// @file test_upf.cpp
#include "atomiverse/cpu/pw/internal/upf_reader.h"
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>

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
        o << "  </PP_R>\n</PP_MESH>\n";
        o << "<PP_LOCAL>\n";
        o << "-2.0 -2.0 -2.0 -2.0 -2.0\n";
        o << "</PP_LOCAL>\n";
        o << "<PP_NLCC>\n";
        o << "0.1 0.2 0.3 0.4 0.5\n";
        o << "</PP_NLCC>\n";
        o << "<PP_NONLOCAL>\n";
        o << "  <PP_BETA angular_momentum=\"0\">\n";
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
    EXPECT_EQ(upf.v_loc.size(), 5u);
    EXPECT_EQ(upf.rho_atom.size(), 5u);
    EXPECT_GE(upf.kb.size(), 1u);
    EXPECT_EQ(upf.kb[0].l, 0);
    EXPECT_EQ(upf.kb[0].beta.size(), 5u);

    std::remove(tmpfile);
}

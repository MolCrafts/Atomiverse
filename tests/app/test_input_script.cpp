/// @file tests/app/test_input_script.cpp
/// @brief Unit: native input tokenizer/parser (no MD run).
#include "cli_args.h"
#include "input_script.h"

#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

TEST(InputScript, ParsesLjDeck) {
    std::istringstream in("units metal\n"
                          "atom_style atomic\n"
                          "lattice sc 4.0\n"
                          "region box block 0 16 0 16 0 16\n"
                          "create_box 1 box\n"
                          "create_atoms 1 box\n"
                          "mass 1 39.948\n"
                          "pair_style lj/cut 6.8\n"
                          "pair_coeff * * 0.0104 3.4\n"
                          "neighbor 0.5 bin\n"
                          "timestep 1.0\n"
                          "thermo 50\n"
                          "velocity all create 94.4 42\n"
                          "run 20\n");
    const auto d = ATV::app::parse_input_script(in);
    EXPECT_DOUBLE_EQ(d.lattice_a, 4.0);
    EXPECT_TRUE(d.have_region);
    EXPECT_TRUE(d.created_atoms);
    EXPECT_TRUE(d.have_pair);
    EXPECT_DOUBLE_EQ(d.pair_cut, 6.8);
    EXPECT_DOUBLE_EQ(d.pair_eps, 0.0104);
    EXPECT_DOUBLE_EQ(d.timestep, 1.0);
    EXPECT_EQ(d.run_steps, 20);
    EXPECT_DOUBLE_EQ(d.velocity_T, 94.4);
}

TEST(InputScript, RejectsUnknownCommand) {
    std::istringstream in("pair_style eam 1.0\n");
    EXPECT_THROW(ATV::app::parse_input_script(in), std::invalid_argument);
}

TEST(InputScript, RejectsBadUnits) {
    std::istringstream in("units real\n");
    EXPECT_THROW(ATV::app::parse_input_script(in), std::invalid_argument);
}

TEST(CliArgs, InFlag) {
    char a0[] = "atomiverse";
    char a1[] = "-in";
    char a2[] = "in.lj";
    char* argv[] = {a0, a1, a2};
    const auto c = ATV::app::parse_cli_args(3, argv);
    EXPECT_EQ(c.in_path, "in.lj");
    EXPECT_TRUE(c.py_script.empty());
}

TEST(CliArgs, PyScript) {
    char a0[] = "atomiverse";
    char a1[] = "run.py";
    char* argv[] = {a0, a1};
    const auto c = ATV::app::parse_cli_args(2, argv);
    EXPECT_EQ(c.py_script, "run.py");
    EXPECT_TRUE(c.in_path.empty());
}

TEST(CliArgs, Help) {
    char a0[] = "atomiverse";
    char a1[] = "-h";
    char* argv[] = {a0, a1};
    EXPECT_TRUE(ATV::app::parse_cli_args(2, argv).help);
}

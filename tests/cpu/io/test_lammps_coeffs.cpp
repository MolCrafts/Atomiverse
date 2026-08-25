/// @file tests/cpu/io/test_lammps_coeffs.cpp
#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include "atomiverse/cpu/io/lammps_coeffs.h"
#include "atomiverse/cpu/io/lammps_real.h"

using namespace ATV::cpu;

namespace {

/// Write `text` to a unique temp file, return its path.
std::string write_tmp(const std::string& text) {
    static int counter = 0;
    std::string path = ::testing::TempDir() + "atv_coeffs_"
                       + std::to_string(counter++) + ".ff";
    std::ofstream out(path);
    out << text;
    return path;
}

} // namespace

TEST(LammpsCoeffs, ParsesHybridGaffFile) {
    const std::string path =
        write_tmp("# atom_style full\n"
                  "bond_style harmonic \n"
                  "bond_coeff 1 4.611e+02 1.3984\n"
                  "bond_coeff 3 3.458e+02 1.086\n"
                  "angle_style harmonic\n"
                  "angle_coeff 2 66.0 120.02\n"
                  "dihedral_style hybrid charmm multi/harmonic\n"
                  "dihedral_coeff 1 charmm 3.625 2 180 0\n"
                  "dihedral_coeff 2 multi/harmonic 0.6 -1.8 0.0 2.4 0.0\n"
                  "special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.83333333\n"
                  "pair_style lj/cut/coul/long 9.0 9.0\n"
                  "pair_modify tail yes mix arithmetic\n"
                  "pair_coeff 1 1 0.086 3.3996695\n"
                  "kspace_style pppm 1e-4\n");
    const LammpsCoeffs c = load_lammps_coeffs(path);
    std::remove(path.c_str());

    EXPECT_EQ(c.pair_style, "lj/cut/coul/long");
    EXPECT_DOUBLE_EQ(c.pair_cut_lj, 9.0);
    EXPECT_DOUBLE_EQ(c.pair_cut_coul, 9.0);
    ASSERT_EQ(c.lj.size(), 1u);
    EXPECT_EQ(c.lj[0].type_i, 0);
    EXPECT_EQ(c.lj[0].type_j, 0);
    EXPECT_NEAR(c.lj[0].epsilon, kcal_per_mol_to_ev(0.086), 1e-15);
    EXPECT_DOUBLE_EQ(c.lj[0].sigma, 3.3996695);

    // bond type 2 (index 1) was never given → zero-filled
    ASSERT_EQ(c.bond_K.size(), 3u);
    EXPECT_NEAR(c.bond_K[0], kcal_per_mol_to_ev(461.1), 1e-12);
    EXPECT_DOUBLE_EQ(c.bond_r0[0], 1.3984);
    EXPECT_DOUBLE_EQ(c.bond_K[1], 0.0);
    EXPECT_NEAR(c.bond_K[2], kcal_per_mol_to_ev(345.8), 1e-12);

    ASSERT_EQ(c.angle_K.size(), 2u);
    EXPECT_NEAR(c.angle_K[1], kcal_per_mol_to_ev(66.0), 1e-12);
    EXPECT_NEAR(c.angle_theta0[1], degree_to_rad(120.02), 1e-12);

    // hybrid split: both tables cover both types
    EXPECT_TRUE(c.has_charmm);
    EXPECT_TRUE(c.has_multi);
    ASSERT_EQ(c.dihedral_charmm_K.size(), 2u);
    ASSERT_EQ(c.dihedral_multi_Cn.size(), 2u);
    EXPECT_NEAR(c.dihedral_charmm_K[0], kcal_per_mol_to_ev(3.625), 1e-12);
    EXPECT_EQ(c.dihedral_charmm_n[0], 2);
    EXPECT_NEAR(c.dihedral_charmm_delta[0], degree_to_rad(180.0), 1e-12);
    EXPECT_DOUBLE_EQ(c.dihedral_charmm_K[1], 0.0); // multi type → zero charmm
    ASSERT_EQ(c.dihedral_multi_Cn[0].size(), 1u);  // charmm type → {0}
    EXPECT_DOUBLE_EQ(c.dihedral_multi_Cn[0][0], 0.0);
    ASSERT_EQ(c.dihedral_multi_Cn[1].size(), 5u);
    EXPECT_NEAR(c.dihedral_multi_Cn[1][1], kcal_per_mol_to_ev(-1.8), 1e-12);

    EXPECT_TRUE(c.has_special);
    EXPECT_DOUBLE_EQ(c.special_lj[2], 0.5);
    EXPECT_NEAR(c.special_coul[2], 0.83333333, 1e-12);

    EXPECT_EQ(c.kspace_style, "pppm");
    EXPECT_DOUBLE_EQ(c.kspace_accuracy, 1e-4);
    EXPECT_TRUE(c.tail_correction);
}

TEST(LammpsCoeffs, AmberKeywordExpands) {
    const std::string path = write_tmp("special_bonds amber\n");
    const LammpsCoeffs c = load_lammps_coeffs(path);
    std::remove(path.c_str());
    EXPECT_DOUBLE_EQ(c.special_lj[2], 0.5);
    EXPECT_NEAR(c.special_coul[2], 5.0 / 6.0, 1e-15);
    EXPECT_DOUBLE_EQ(c.special_lj[0], 0.0);
}

TEST(LammpsCoeffs, NonHybridDihedralStyle) {
    const std::string path = write_tmp("dihedral_style charmm\n"
                                       "dihedral_coeff 1 1.1 3 0 0\n");
    const LammpsCoeffs c = load_lammps_coeffs(path);
    std::remove(path.c_str());
    EXPECT_TRUE(c.has_charmm);
    EXPECT_FALSE(c.has_multi);
    EXPECT_NEAR(c.dihedral_charmm_K[0], kcal_per_mol_to_ev(1.1), 1e-12);
    EXPECT_EQ(c.dihedral_charmm_n[0], 3);
}

TEST(LammpsCoeffs, RejectsNonzeroCharmmWeight) {
    const std::string path = write_tmp("dihedral_style charmm\n"
                                       "dihedral_coeff 1 1.1 3 0 0.5\n");
    EXPECT_THROW(load_lammps_coeffs(path), std::invalid_argument);
    std::remove(path.c_str());
}

TEST(LammpsCoeffs, RejectsUnsupportedStyles) {
    const std::string p1 = write_tmp("pair_style eam/alloy\n");
    EXPECT_THROW(load_lammps_coeffs(p1), std::invalid_argument);
    std::remove(p1.c_str());
    const std::string p2 = write_tmp("dihedral_style opls\n");
    EXPECT_THROW(load_lammps_coeffs(p2), std::invalid_argument);
    std::remove(p2.c_str());
    const std::string p3 = write_tmp("bond_style fene\n");
    EXPECT_THROW(load_lammps_coeffs(p3), std::invalid_argument);
    std::remove(p3.c_str());
}

TEST(LammpsCoeffs, MalformedRowThrows) {
    const std::string path = write_tmp("bond_coeff 1 abc 1.5\n");
    EXPECT_THROW(load_lammps_coeffs(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LammpsCoeffs, UnknownCommandsCollected) {
    const std::string path = write_tmp("group litfsi molecule > 20\n"
                                       "neighbor 2.0 bin\n");
    const LammpsCoeffs c = load_lammps_coeffs(path);
    std::remove(path.c_str());
    ASSERT_EQ(c.skipped.size(), 2u);
    EXPECT_EQ(c.skipped[0], "group");
    EXPECT_EQ(c.skipped[1], "neighbor");
}

TEST(LammpsCoeffs, MissingFileThrows) {
    EXPECT_THROW(load_lammps_coeffs("/nonexistent/x.ff"), std::runtime_error);
}

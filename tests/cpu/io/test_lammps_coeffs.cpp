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
    // ctest runs each test in its own process: key by test name too.
    const std::string test =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::string path = ::testing::TempDir() + "atv_coeffs_" + test + "_"
                       + std::to_string(counter++) + ".ff";
    std::ofstream out(path);
    out << text;
    return path;
}

} // namespace

TEST(LammpsCoeffs, PairCoeffCarriesAnOptionalPerPairCutoff) {
    const std::string path = write_tmp("pair_style lj/cut/coul/long 9.0\n"
                                       "pair_coeff 1 1 0.086 3.4\n"
                                       "pair_coeff 1 2 0.05 3.2 7.5\n");
    const LammpsCoeffs c = load_lammps_coeffs(path);
    std::remove(path.c_str());
    ASSERT_EQ(c.lj.size(), 2u);
    EXPECT_FALSE(c.lj[0].cutoff.has_value());
    ASSERT_TRUE(c.lj[1].cutoff.has_value());
    EXPECT_DOUBLE_EQ(*c.lj[1].cutoff, 7.5);

    const std::string coul = write_tmp("pair_coeff 1 1 0.086 3.4 8.0 9.0\n");
    EXPECT_THROW(load_lammps_coeffs(coul), std::runtime_error);
    std::remove(coul.c_str());
}

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

TEST(LammpsCoeffs, ParsesFourierAndCvff) {
    const std::string path =
        write_tmp("dihedral_style fourier\n"
                  "dihedral_coeff 2 2 0.80 1 180.0 0.383 3 0.0\n"
                  "dihedral_coeff 1 1 0.15 3 0.0\n"
                  "improper_style cvff\n"
                  "improper_coeff 1 1.1 -1 2\n"
                  "improper_coeff 2 10.5 1 0\n"
                  "pair_modify mix arithmetic shift no tail no\n");
    const LammpsCoeffs c = load_lammps_coeffs(path);
    std::remove(path.c_str());
    ASSERT_TRUE(c.has_fourier);
    EXPECT_FALSE(c.has_charmm);
    ASSERT_EQ(c.dihedral_fourier.size(), 2u);
    ASSERT_EQ(c.dihedral_fourier[1].size(), 2u);
    EXPECT_DOUBLE_EQ(c.dihedral_fourier[1][0].K, kcal_per_mol_to_ev(0.80));
    EXPECT_EQ(c.dihedral_fourier[1][0].n, 1);
    EXPECT_DOUBLE_EQ(c.dihedral_fourier[1][0].delta, degree_to_rad(180.0));
    EXPECT_EQ(c.dihedral_fourier[1][1].n, 3);
    ASSERT_EQ(c.dihedral_fourier[0].size(), 1u);
    ASSERT_TRUE(c.has_improper_cvff);
    ASSERT_EQ(c.improper_K.size(), 2u);
    EXPECT_DOUBLE_EQ(c.improper_K[0], kcal_per_mol_to_ev(1.1));
    EXPECT_EQ(c.improper_d[0], -1);
    EXPECT_EQ(c.improper_n[0], 2);
    EXPECT_EQ(c.improper_d[1], 1);
    EXPECT_EQ(c.improper_n[1], 0);
    EXPECT_TRUE(c.has_shift);
    EXPECT_FALSE(c.energy_shift);
    EXPECT_FALSE(c.tail_correction);
}

TEST(LammpsCoeffs, HybridFourierLeavesOtherTypesEmpty) {
    const std::string path =
        write_tmp("dihedral_style hybrid charmm fourier\n"
                  "dihedral_coeff 1 charmm 3.625 2 180 0\n"
                  "dihedral_coeff 2 fourier 1 0.15 3 0.0\n");
    const LammpsCoeffs c = load_lammps_coeffs(path);
    std::remove(path.c_str());
    ASSERT_EQ(c.dihedral_fourier.size(), 2u);
    EXPECT_TRUE(c.dihedral_fourier[0].empty());
    EXPECT_EQ(c.dihedral_charmm_K[1], 0.0);
}

TEST(LammpsCoeffs, RejectsMalformedFourierAndCvff) {
    for (const char* text :
         {"dihedral_style fourier\ndihedral_coeff 1 2 0.8 1 180.0\n",
          "improper_style cvff\nimproper_coeff 1 1.1 0 2\n",
          "improper_style cvff\nimproper_coeff 1 1.1 1 7\n",
          "improper_coeff 1 1.1 1 2\n"}) {
        const std::string path = write_tmp(text);
        EXPECT_THROW(load_lammps_coeffs(path), std::runtime_error) << text;
        std::remove(path.c_str());
    }
    const std::string p = write_tmp("improper_style harmonic\n");
    EXPECT_THROW(load_lammps_coeffs(p), std::invalid_argument);
    std::remove(p.c_str());
}

TEST(LammpsCoeffs, RejectsNonArithmeticMixing) {
    const std::string path = write_tmp("pair_modify mix geometric\n");
    EXPECT_THROW(load_lammps_coeffs(path), std::invalid_argument);
    std::remove(path.c_str());
}

/// @file tests/cpu/io/test_lammps_data_full.cpp
/// @brief load_lammps_data: atom_style full (charges) + molecular regression.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/io/lammps_data.h"

using ATV::cpu::Frame;
using ATV::cpu::load_lammps_data;
using ATV::kAtom_charge;
using ATV::kAtom_nspecial;
using ATV::kAtom_special;
using ATV::kAtom_type;
using ATV::kAtom_x;
using ATV::kAtom_y;
using ATV::kAtom_z;

namespace {

std::string write_temp(const std::string& tag, const std::string& content) {
    const auto path = (std::filesystem::temp_directory_path()
                       / ("atv_lmpdata_" + tag + ".data"))
                          .string();
    std::ofstream out(path);
    out << content;
    out.close();
    return path;
}

// Two atoms, one bond, full style with non-zero charges.
const char* kMiniFull = R"(LAMMPS data file mini full

2 atoms
1 bonds
0 angles
0 dihedrals

2 atom types
1 bond types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 12.01
2 1.008

Atoms

1 1 1 -0.25 1.0 2.0 3.0
2 1 2  0.25 1.5 2.0 3.0

Bonds

1 1 1 2
)";

// Same geometry, molecular (no charge column).
const char* kMiniMolecular = R"(LAMMPS data file mini molecular

2 atoms
1 bonds
0 angles
0 dihedrals

2 atom types
1 bond types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 12.01
2 1.008

Atoms

1 1 1 1.0 2.0 3.0
2 1 2 1.5 2.0 3.0

Bonds

1 1 1 2
)";

const char* kMalformedAtoms = R"(LAMMPS data file bad

2 atoms
0 bonds

1 atom types

0.0 5.0 xlo xhi
0.0 5.0 ylo yhi
0.0 5.0 zlo zhi

Masses

1 1.0

Atoms

1 1 1
2 1 1 0.0 0.0 0.0
)";

} // namespace

TEST(LammpsDataFull, LoadsChargeTypeCoords) {
    const std::string path = write_temp("full", kMiniFull);
    Frame f = load_lammps_data(path);
    std::remove(path.c_str());

    ASSERT_EQ(f.nlocal(), 2);
    EXPECT_EQ(f.span(kAtom_type)[0], 0);
    EXPECT_EQ(f.span(kAtom_type)[1], 1);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_charge)[0]), -0.25, 1e-12);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_charge)[1]), 0.25, 1e-12);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_x)[0]), 1.0, 1e-12);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_y)[0]), 2.0, 1e-12);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_z)[0]), 3.0, 1e-12);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_x)[1]), 1.5, 1e-12);
}

TEST(LammpsDataFull, MolecularPathStillWorks) {
    const std::string path = write_temp("mol", kMiniMolecular);
    Frame f = load_lammps_data(path);
    std::remove(path.c_str());

    ASSERT_EQ(f.nlocal(), 2);
    EXPECT_EQ(f.span(kAtom_type)[0], 0);
    EXPECT_EQ(f.span(kAtom_type)[1], 1);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_x)[0]), 1.0, 1e-12);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_y)[0]), 2.0, 1e-12);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_z)[0]), 3.0, 1e-12);
    // Charges default to 0 for molecular.
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_charge)[0]), 0.0, 1e-15);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_charge)[1]), 0.0, 1e-15);
}

TEST(LammpsDataFull, BondsBuildSpecialList) {
    const std::string path = write_temp("spec", kMiniFull);
    Frame f = load_lammps_data(path);
    std::remove(path.c_str());

    const auto nspecial = f.span(kAtom_nspecial);
    bool any_12 = false;
    for (int i = 0; i < f.nlocal(); ++i) {
        if (nspecial(i, 0) > 0)
            any_12 = true;
    }
    EXPECT_TRUE(any_12) << "expected non-zero 1-2 special counts after bond";

    // At least one partner tag should appear on atom 0 or 1.
    const auto special = f.span(kAtom_special);
    bool partner_ok = false;
    for (int i = 0; i < f.nlocal(); ++i) {
        const int n12 = nspecial(i, 0);
        for (int s = 0; s < n12; ++s) {
            const auto p = special(i, s);
            if (p == 0 || p == 1)
                partner_ok = true;
        }
    }
    EXPECT_TRUE(partner_ok);
}

TEST(LammpsDataFull, MalformedAtomsThrows) {
    const std::string path = write_temp("bad", kMalformedAtoms);
    try {
        Frame f = load_lammps_data(path);
        (void)f;
        FAIL() << "expected throw";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_TRUE(msg.find("malformed") != std::string::npos
                    || msg.find(path) != std::string::npos)
            << msg;
    }
    std::remove(path.c_str());
}

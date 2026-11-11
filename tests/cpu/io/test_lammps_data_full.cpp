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

using ATV::kAtom_charge;
using ATV::kAtom_nspecial;
using ATV::kAtom_special;
using ATV::kAtom_type;
using ATV::kAtom_x;
using ATV::kAtom_y;
using ATV::kAtom_z;
using ATV::cpu::Frame;
using ATV::cpu::load_lammps_data;

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
    // A data file names types, not elements: no species is guessed.
    EXPECT_FALSE(f.has("atoms", "atomic_number"));
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

namespace {

// Non-zero box origin + Velocities section (write_data restart shape).
const char* kOriginVel = R"(LAMMPS data file origin+velocities

2 atoms
0 bonds
0 angles
0 dihedrals

1 atom types

2.0 12.0 xlo xhi
3.0 13.0 ylo yhi
4.0 14.0 zlo zhi

Masses

1 12.01

Atoms # full

1 1 1 0.1 2.5 3.5 4.5 0 0 0
2 1 1 -0.1 6.0 7.0 8.0 0 0 0

Velocities

1 0.01 -0.02 0.03
2 -0.04 0.05 -0.06
)";

} // namespace

TEST(LammpsDataFull, ShiftsPositionsByBoxOrigin) {
    const std::string path = write_temp("origin", kOriginVel);
    Frame f = load_lammps_data(path);
    std::remove(path.c_str());

    // Positions land in the [0, L) frame: shifted by (xlo, ylo, zlo).
    EXPECT_NEAR(f.span(kAtom_x)[0], 0.5, 1e-12);
    EXPECT_NEAR(f.span(kAtom_y)[0], 0.5, 1e-12);
    EXPECT_NEAR(f.span(kAtom_z)[0], 0.5, 1e-12);
    EXPECT_NEAR(f.span(kAtom_x)[1], 4.0, 1e-12);
    // Box edge lengths are hi - lo.
    EXPECT_NEAR(f.box.length(0), 10.0, 1e-12);
    EXPECT_NEAR(f.box.length(1), 10.0, 1e-12);
}

TEST(LammpsDataFull, ReadsVelocities) {
    const std::string path = write_temp("vel", kOriginVel);
    Frame f = load_lammps_data(path);
    std::remove(path.c_str());

    EXPECT_NEAR(f.span(ATV::kAtom_vx)[0], 0.01, 1e-12);
    EXPECT_NEAR(f.span(ATV::kAtom_vy)[0], -0.02, 1e-12);
    EXPECT_NEAR(f.span(ATV::kAtom_vz)[0], 0.03, 1e-12);
    EXPECT_NEAR(f.span(ATV::kAtom_vx)[1], -0.04, 1e-12);
    EXPECT_NEAR(f.span(ATV::kAtom_vz)[1], -0.06, 1e-12);
}

namespace {

// write_data shape: Type Labels + Coeffs sections around Masses, impropers.
const char* kSections = R"(LAMMPS data file written by write_data

4 atoms
3 bonds
2 angles
1 dihedrals
1 impropers

2 atom types
1 bond types
1 angle types
1 dihedral types
1 improper types

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Atom Type Labels

1 c
2 o

Masses

1 12.011
2 15.999

Pair Coeffs # lj/cut/coul/long

1 0.086 3.4
2 0.21 2.96

Bond Coeffs # harmonic

1 300.0 1.5

Atoms # full

1 1 1 0.1 1.0 1.0 1.0
2 1 1 0.0 2.5 1.0 1.0
3 1 2 -0.1 3.0 2.3 1.0
4 1 1 0.0 4.0 2.3 2.0

Bonds

1 1 1 2
2 1 2 3
3 1 3 4

Angles

1 1 1 2 3
2 1 2 3 4

Dihedrals

1 1 1 2 3 4

Impropers

1 1 1 2 3 4
)";

} // namespace

TEST(LammpsDataFull, CoeffSectionsDoNotOverwriteMasses) {
    const std::string path = write_temp("sections", kSections);
    Frame f = load_lammps_data(path);
    std::remove(path.c_str());
    const auto mass = f.span(ATV::kAtom_mass);
    EXPECT_DOUBLE_EQ(mass[0], 12.011);
    EXPECT_DOUBLE_EQ(mass[2], 15.999);
    EXPECT_DOUBLE_EQ(mass[3], 12.011);
}

TEST(LammpsDataFull, ReadsImpropersOnCentralSecondAtom) {
    const std::string path = write_temp("impropers", kSections);
    Frame f = load_lammps_data(path);
    std::remove(path.c_str());
    const auto num = f.span(ATV::kAtom_num_improper);
    const auto atom = f.span(ATV::kAtom_improper_atom);
    const auto type = f.span(ATV::kAtom_improper_type);
    ASSERT_EQ(num[1], 1); // owner = atom 2 of `1 2 3 4`
    EXPECT_EQ(num[0] + num[2] + num[3], 0);
    EXPECT_EQ(atom(1, 0), 0);
    EXPECT_EQ(atom(1, 1), 2);
    EXPECT_EQ(atom(1, 2), 3);
    EXPECT_EQ(type(1, 0), 0);
}

TEST(LammpsDataFull, UnsupportedSectionThrows) {
    std::string text = kSections;
    text += "\nEllipsoids\n\n1 1 1 1\n";
    const std::string path = write_temp("unsupported", text);
    EXPECT_THROW(load_lammps_data(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LammpsDataFull, ExtraRowsThrow) {
    std::string text = kSections;
    text += "2 1 4 3 2 1\n"; // a second improper the header did not declare
    const std::string path = write_temp("extra", text);
    EXPECT_THROW(load_lammps_data(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LammpsDataFull, TriclinicTiltThrows) {
    std::string text = kSections;
    text.replace(text.find("0 20 zlo zhi"), 12,
                 "0 20 zlo zhi\n1.0 0 0 xy xz yz");
    const std::string path = write_temp("tilt", text);
    EXPECT_THROW(load_lammps_data(path), std::runtime_error);
    std::remove(path.c_str());
}

// Trailing `ix iy iz` land in kAtom_image (no silent drop), beyond the old
// int8 range; a non-integer flag throws.
TEST(LammpsDataFull, ReadsTrailingImageFlags) {
    const std::string row1 = "1 1 1 -0.25 1.0 2.0 3.0";
    const std::string row2 = "2 1 2  0.25 1.5 2.0 3.0";
    std::string text = kMiniFull;
    text.replace(text.find(row1), row1.size(), row1 + " 300 -2 0");
    text.replace(text.find(row2), row2.size(), row2 + " 0 1 -40000");
    const std::string path = write_temp("image", text);
    const Frame f = load_lammps_data(path);
    std::remove(path.c_str());
    const auto image = f.span(ATV::kAtom_image);
    EXPECT_EQ(image(0, 0), 300);
    EXPECT_EQ(image(0, 1), -2);
    EXPECT_EQ(image(0, 2), 0);
    EXPECT_EQ(image(1, 0), 0);
    EXPECT_EQ(image(1, 1), 1);
    EXPECT_EQ(image(1, 2), -40000);

    std::string bad = kMiniFull;
    bad.replace(bad.find(row1), row1.size(), row1 + " 0.5 0 0");
    bad.replace(bad.find(row2), row2.size(), row2 + " 0 0 0");
    const std::string bad_path = write_temp("image_bad", bad);
    EXPECT_THROW(load_lammps_data(bad_path), std::runtime_error);
    std::remove(bad_path.c_str());
}

/// @file tests/cpu/test_materialize.cpp
/// @brief Round-trip + boundary tests for cpu::materialize / cpu::write_back.
///
/// Covers the rust-frame-lowering-02 acceptance criteria:
///   - ac-002: molrs.Frame -> cpu::Frame -> molrs.Frame round-trips
///             positions / velocities / forces / charge / topology / box /
///             typed system metadata element-wise.
///   - ac-003: required fields fail loud (named exception); optional fields
///             default to zero without throwing.
///   - ac-004: a cpu::Frame produced by cpu::materialize is runnable under
///             cpu::md::MD and yields a finite energy.
///
/// Plus the arch-refactor-06-frame Design §2 write_back field contract
/// (golden round-trip): image (int8 -> i32 ix/iy/iz widen-back) and tag
/// (TagInt -> i32 checked narrowing, overflow throws) ARE written back;
/// type / mass / topology / box are designated non-write-back and must keep
/// their pre-materialize molrs values.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/materialize.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/molrs.hpp"

using ATV::cpu::Frame;
using ATV::cpu::materialize;
using ATV::cpu::write_back;

namespace {

// Build a fully-populated molrs.Frame with three atoms, one bond, one
// angle, one dihedral, and a cubic box.
::rust::Box<molrs::FrameRef> make_full_frame() {
    auto fr = molrs::frame_new();

    const std::vector<double> x = {0.0, 1.5, 3.0};
    const std::vector<double> y = {0.1, 1.6, 3.1};
    const std::vector<double> z = {0.2, 1.7, 3.2};
    const std::vector<double> vx = {0.01, 0.02, 0.03};
    const std::vector<double> vy = {-0.01, -0.02, -0.03};
    const std::vector<double> vz = {0.04, 0.05, 0.06};
    const std::vector<double> fx = {1.0, 2.0, 3.0};
    const std::vector<double> fy = {-1.0, -2.0, -3.0};
    const std::vector<double> fz = {0.5, 0.6, 0.7};
    const std::vector<double> mass = {12.011, 1.008, 15.999};
    const std::vector<double> charge = {-0.5, 0.25, 0.25};
    const std::vector<std::int32_t> type = {1, 2, 3};
    const std::vector<std::int32_t> tag = {10, 11, 12};

    molrs::frame_set_column_f64(
        *fr, "atoms", "x", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "y", ::rust::Slice<const double>(y.data(), y.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "z", ::rust::Slice<const double>(z.data(), z.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "vx", ::rust::Slice<const double>(vx.data(), vx.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "vy", ::rust::Slice<const double>(vy.data(), vy.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "vz", ::rust::Slice<const double>(vz.data(), vz.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "fx", ::rust::Slice<const double>(fx.data(), fx.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "fy", ::rust::Slice<const double>(fy.data(), fy.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "fz", ::rust::Slice<const double>(fz.data(), fz.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "mass",
        ::rust::Slice<const double>(mass.data(), mass.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "charge",
        ::rust::Slice<const double>(charge.data(), charge.size()));
    molrs::frame_set_column_i32(
        *fr, "atoms", "type",
        ::rust::Slice<const std::int32_t>(type.data(), type.size()));
    molrs::frame_set_column_i32(
        *fr, "atoms", "tag",
        ::rust::Slice<const std::int32_t>(tag.data(), tag.size()));

    // Topology: one bond, one angle, one dihedral. molrs topology columns
    // reference atoms by tag (materialize scatters them via the tag->local
    // map), so use the atom tags {10,11,12} above -- not row indices.
    const std::vector<std::int32_t> bi = {10}, bj = {11}, bt = {1};
    molrs::frame_set_column_i32(
        *fr, "bonds", "atomi",
        ::rust::Slice<const std::int32_t>(bi.data(), bi.size()));
    molrs::frame_set_column_i32(
        *fr, "bonds", "atomj",
        ::rust::Slice<const std::int32_t>(bj.data(), bj.size()));
    molrs::frame_set_column_i32(
        *fr, "bonds", "type",
        ::rust::Slice<const std::int32_t>(bt.data(), bt.size()));

    const std::vector<std::int32_t> ai = {10}, aj = {11}, ak = {12}, at = {1};
    molrs::frame_set_column_i32(
        *fr, "angles", "atomi",
        ::rust::Slice<const std::int32_t>(ai.data(), ai.size()));
    molrs::frame_set_column_i32(
        *fr, "angles", "atomj",
        ::rust::Slice<const std::int32_t>(aj.data(), aj.size()));
    molrs::frame_set_column_i32(
        *fr, "angles", "atomk",
        ::rust::Slice<const std::int32_t>(ak.data(), ak.size()));
    molrs::frame_set_column_i32(
        *fr, "angles", "type",
        ::rust::Slice<const std::int32_t>(at.data(), at.size()));

    const std::vector<std::int32_t> di = {10}, dj = {11}, dk = {12}, dl = {12},
                                    dt = {1};
    molrs::frame_set_column_i32(
        *fr, "dihedrals", "atomi",
        ::rust::Slice<const std::int32_t>(di.data(), di.size()));
    molrs::frame_set_column_i32(
        *fr, "dihedrals", "atomj",
        ::rust::Slice<const std::int32_t>(dj.data(), dj.size()));
    molrs::frame_set_column_i32(
        *fr, "dihedrals", "atomk",
        ::rust::Slice<const std::int32_t>(dk.data(), dk.size()));
    molrs::frame_set_column_i32(
        *fr, "dihedrals", "atoml",
        ::rust::Slice<const std::int32_t>(dl.data(), dl.size()));
    molrs::frame_set_column_i32(
        *fr, "dihedrals", "type",
        ::rust::Slice<const std::int32_t>(dt.data(), dt.size()));

    // Cubic box, row-major 3x3 H.
    const std::vector<double> h = {20.0, 0.0, 0.0, 0.0, 20.0,
                                   0.0,  0.0, 0.0, 20.0};
    molrs::frame_set_simbox(*fr,
                            ::rust::Slice<const double>(h.data(), h.size()));

    return fr;
}

// Extend the fully-populated frame with per-axis PBC image columns
// (i32 in molrs, int8 [n,3] in cpu::Frame) for the write_back contract
// tests. Values must fit int8 (|image| <= 127).
::rust::Box<molrs::FrameRef> make_contract_frame() {
    auto fr = make_full_frame();

    const std::vector<std::int32_t> ix = {1, -1, 2};
    const std::vector<std::int32_t> iy = {0, 3, -2};
    const std::vector<std::int32_t> iz = {-1, 0, 4};
    molrs::frame_set_column_i32(
        *fr, "atoms", "ix",
        ::rust::Slice<const std::int32_t>(ix.data(), ix.size()));
    molrs::frame_set_column_i32(
        *fr, "atoms", "iy",
        ::rust::Slice<const std::int32_t>(iy.data(), iy.size()));
    molrs::frame_set_column_i32(
        *fr, "atoms", "iz",
        ::rust::Slice<const std::int32_t>(iz.data(), iz.size()));

    return fr;
}

// Minimal molrs frame: required columns only (x/y/z/type), no box, no
// topology, no tag/image columns.
::rust::Box<molrs::FrameRef> make_minimal_frame() {
    auto fr = molrs::frame_new();
    const std::vector<double> x = {0.0, 1.0};
    const std::vector<std::int32_t> type = {1, 1};
    molrs::frame_set_column_f64(
        *fr, "atoms", "x", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "y", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "z", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_i32(
        *fr, "atoms", "type",
        ::rust::Slice<const std::int32_t>(type.data(), type.size()));
    return fr;
}

} // namespace

// ── ac-002: round-trip materialize -> write_back -> read back ───────────

TEST(Materialize, RoundTrip) {
    auto fr = make_full_frame();

    Frame cf = materialize(*fr);

    ASSERT_EQ(cf.nlocal(), 3);
    int n_bond = 0, n_angle = 0, n_dihedral = 0;
    for (int i = 0; i < cf.nlocal(); ++i) {
        n_bond += cf.span(ATV::kAtom_num_bond)[i];
        n_angle += cf.span(ATV::kAtom_num_angle)[i];
        n_dihedral += cf.span(ATV::kAtom_num_dihedral)[i];
    }
    ASSERT_EQ(n_bond, 1);
    ASSERT_EQ(n_angle, 1);
    ASSERT_EQ(n_dihedral, 1);

    // Mutate the cpu::Frame so write-back has something distinct to carry.
    cf.span(ATV::kAtom_x)[1] = 7.25;
    cf.span(ATV::kAtom_vy)[2] = -9.5;
    cf.span(ATV::kAtom_fz)[0] = 42.0;
    cf.span(ATV::kAtom_charge)[0] = -0.875;
    cf.meta.get(ATV::kEnergy) = -123.5;
    cf.meta.get(ATV::kKinetic) = 64.25;
    cf.meta.get(ATV::kConserved) = -59.25;
    for (int k = 0; k < 6; ++k) {
        cf.meta.get(ATV::kStress)[k] = 1.0 + k;
        cf.meta.get(ATV::kStressKinetic)[k] = 10.0 + k;
    }

    write_back(cf, *fr);

    // Read every per-atom column back out of the molrs.Frame.
    auto rx = molrs::frame_column_f64(*fr, "atoms", "x");
    auto ry = molrs::frame_column_f64(*fr, "atoms", "y");
    auto rz = molrs::frame_column_f64(*fr, "atoms", "z");
    auto rvx = molrs::frame_column_f64(*fr, "atoms", "vx");
    auto rvy = molrs::frame_column_f64(*fr, "atoms", "vy");
    auto rvz = molrs::frame_column_f64(*fr, "atoms", "vz");
    auto rfx = molrs::frame_column_f64(*fr, "atoms", "fx");
    auto rfy = molrs::frame_column_f64(*fr, "atoms", "fy");
    auto rfz = molrs::frame_column_f64(*fr, "atoms", "fz");
    auto rq = molrs::frame_column_f64(*fr, "atoms", "charge");

    ASSERT_EQ(rx.size(), 3u);
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(rx[i], cf.span(ATV::kAtom_x)[i]);
        EXPECT_DOUBLE_EQ(ry[i], cf.span(ATV::kAtom_y)[i]);
        EXPECT_DOUBLE_EQ(rz[i], cf.span(ATV::kAtom_z)[i]);
        EXPECT_DOUBLE_EQ(rvx[i], cf.span(ATV::kAtom_vx)[i]);
        EXPECT_DOUBLE_EQ(rvy[i], cf.span(ATV::kAtom_vy)[i]);
        EXPECT_DOUBLE_EQ(rvz[i], cf.span(ATV::kAtom_vz)[i]);
        EXPECT_DOUBLE_EQ(rfx[i], cf.span(ATV::kAtom_fx)[i]);
        EXPECT_DOUBLE_EQ(rfy[i], cf.span(ATV::kAtom_fy)[i]);
        EXPECT_DOUBLE_EQ(rfz[i], cf.span(ATV::kAtom_fz)[i]);
        EXPECT_DOUBLE_EQ(rq[i], cf.span(ATV::kAtom_charge)[i]);
    }

    // Topology survives the materialize (write_back leaves it untouched).
    auto bi = molrs::frame_column_i32(*fr, "bonds", "atomi");
    auto bj = molrs::frame_column_i32(*fr, "bonds", "atomj");
    ASSERT_EQ(bi.size(), 1u);
    EXPECT_EQ(bi[0], 10);
    EXPECT_EQ(bj[0], 11);

    // cpu side: topology scattered into the per-atom layout, keyed by tag.
    // bond (tags 10-11): smaller tag 10 -> row 0 owns; partner tag 11.
    ASSERT_EQ(cf.span(ATV::kAtom_num_bond)[0], 1);
    EXPECT_EQ(cf.span(ATV::kAtom_bond_atom)(0, 0), 11);
    EXPECT_EQ(cf.span(ATV::kAtom_bond_type)(0, 0), 1);
    // angle (10-11-12): center tag 11 -> row 1 owns; k-partner tag is 12.
    ASSERT_EQ(cf.span(ATV::kAtom_num_angle)[1], 1);
    EXPECT_EQ(cf.span(ATV::kAtom_angle_atom)(1, 1), 12);
    // dihedral (10-11-12-12): atom-j tag 11 -> row 1 owns; l-partner tag 12.
    ASSERT_EQ(cf.span(ATV::kAtom_num_dihedral)[1], 1);
    EXPECT_EQ(cf.span(ATV::kAtom_dihedral_atom)(1, 2), 12);

    // Box round-trips.
    auto h = molrs::frame_simbox(*fr);
    ASSERT_EQ(h.size(), 9u);
    EXPECT_DOUBLE_EQ(h[0], 20.0);
    EXPECT_DOUBLE_EQ(h[4], 20.0);
    EXPECT_DOUBLE_EQ(h[8], 20.0);
    EXPECT_DOUBLE_EQ(cf.simbox.length(0), 20.0);
    EXPECT_DOUBLE_EQ(cf.simbox.length(1), 20.0);
    EXPECT_DOUBLE_EQ(cf.simbox.length(2), 20.0);

    // System results cross the boundary only as exact typed metadata. No
    // synthetic result block exists in schema v2.
    const auto meta = molrs::frame_meta_entries(*fr);
    auto find_meta = [&](const char* key) -> const molrs::MetaEntry* {
        for (const auto& entry : meta)
            if (std::string(entry.key.data(), entry.key.size()) == key)
                return &entry;
        return nullptr;
    };
    const auto* energy = find_meta("energy");
    const auto* kinetic = find_meta("kinetic");
    const auto* conserved = find_meta("conserved");
    ASSERT_NE(energy, nullptr);
    ASSERT_NE(kinetic, nullptr);
    ASSERT_NE(conserved, nullptr);
    EXPECT_EQ(energy->dtype, molrs::MetaType::F64);
    EXPECT_DOUBLE_EQ(energy->f64_value, -123.5);
    EXPECT_DOUBLE_EQ(kinetic->f64_value, 64.25);
    EXPECT_DOUBLE_EQ(conserved->f64_value, -59.25);

    const auto* stress = find_meta("stress");
    const auto* stress_kinetic = find_meta("stress_kinetic");
    ASSERT_NE(stress, nullptr);
    ASSERT_NE(stress_kinetic, nullptr);
    EXPECT_EQ(stress->dtype, molrs::MetaType::F64x6);
    EXPECT_EQ(stress_kinetic->dtype, molrs::MetaType::F64x6);
    ASSERT_EQ(stress->f64_values.size(), 6u);
    ASSERT_EQ(stress_kinetic->f64_values.size(), 6u);
    for (int k = 0; k < 6; ++k) {
        EXPECT_DOUBLE_EQ(stress->f64_values[k], 1.0 + k);
        EXPECT_DOUBLE_EQ(stress_kinetic->f64_values[k], 10.0 + k);
    }
    EXPECT_FALSE(molrs::frame_has_block(*fr, "scalars"));
    EXPECT_FALSE(molrs::frame_has_block(*fr, "stress"));
    EXPECT_FALSE(molrs::frame_has_block(*fr, "stress_kinetic"));
}

// ── ac-003: required fields fail loud; optional fields default to zero ──

TEST(Materialize, MissingPositionsThrows) {
    auto fr = molrs::frame_new();
    // type present, x/y/z absent.
    const std::vector<std::int32_t> type = {1, 2};
    molrs::frame_set_column_i32(
        *fr, "atoms", "type",
        ::rust::Slice<const std::int32_t>(type.data(), type.size()));

    try {
        Frame cf = materialize(*fr);
        FAIL() << "materialize must throw when atoms.x is absent";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("x"), std::string::npos)
            << "exception message must name the missing field: " << e.what();
    }
}

TEST(Materialize, MissingTypeThrows) {
    auto fr = molrs::frame_new();
    // positions present, type absent.
    const std::vector<double> x = {0.0, 1.0};
    molrs::frame_set_column_f64(
        *fr, "atoms", "x", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "y", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "z", ::rust::Slice<const double>(x.data(), x.size()));

    try {
        Frame cf = materialize(*fr);
        FAIL() << "materialize must throw when atoms.type is absent";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("type"), std::string::npos)
            << "exception message must name the missing field: " << e.what();
    }
}

TEST(Materialize, MissingOptionalFieldsDefaultToZero) {
    auto fr = molrs::frame_new();
    // Only required fields: x, y, z, type. No velocities, no charge.
    const std::vector<double> x = {0.0, 1.0, 2.0};
    const std::vector<std::int32_t> type = {1, 1, 1};
    molrs::frame_set_column_f64(
        *fr, "atoms", "x", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "y", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "z", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_i32(
        *fr, "atoms", "type",
        ::rust::Slice<const std::int32_t>(type.data(), type.size()));

    Frame cf = materialize(*fr); // must NOT throw

    ASSERT_EQ(cf.nlocal(), 3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(cf.span(ATV::kAtom_vx)[i], 0.0);
        EXPECT_DOUBLE_EQ(cf.span(ATV::kAtom_vy)[i], 0.0);
        EXPECT_DOUBLE_EQ(cf.span(ATV::kAtom_vz)[i], 0.0);
        EXPECT_DOUBLE_EQ(cf.span(ATV::kAtom_fx)[i], 0.0);
        EXPECT_DOUBLE_EQ(cf.span(ATV::kAtom_charge)[i], 0.0);
    }
    // No topology -> per-atom counts all zero.
    for (int i = 0; i < cf.nlocal(); ++i) {
        EXPECT_EQ(cf.span(ATV::kAtom_num_bond)[i], 0);
        EXPECT_EQ(cf.span(ATV::kAtom_num_angle)[i], 0);
        EXPECT_EQ(cf.span(ATV::kAtom_num_dihedral)[i], 0);
    }
}

// ── ac-004: a materialized cpu::Frame is runnable under cpu::md::MD ─────

TEST(Materialize, MaterializedFrameIsRunnable) {
    auto fr = molrs::frame_new();

    // Two argon-like atoms separated by ~sigma in a roomy cubic box.
    const std::vector<double> x = {0.0, 3.4};
    const std::vector<double> y = {0.0, 0.0};
    const std::vector<double> z = {0.0, 0.0};
    const std::vector<double> mass = {39.948, 39.948};
    const std::vector<std::int32_t> type = {1, 1};
    molrs::frame_set_column_f64(
        *fr, "atoms", "x", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "y", ::rust::Slice<const double>(y.data(), y.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "z", ::rust::Slice<const double>(z.data(), z.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "mass",
        ::rust::Slice<const double>(mass.data(), mass.size()));
    molrs::frame_set_column_i32(
        *fr, "atoms", "type",
        ::rust::Slice<const std::int32_t>(type.data(), type.size()));
    const std::vector<double> h = {30.0, 0.0, 0.0, 0.0, 30.0,
                                   0.0,  0.0, 0.0, 30.0};
    molrs::frame_set_simbox(*fr,
                            ::rust::Slice<const double>(h.data(), h.size()));

    Frame cf = materialize(*fr);
    ASSERT_EQ(cf.nlocal(), 2);

    ATV::cpu::md::MD sim;
    sim.set_neighbor(ATV::cpu::md::NeighborList<>{6.8, /*skin=*/0.5});
    sim.add_potential(ATV::cpu::PairLJCut{0.0104, 3.4, 6.8});
    sim.set_integrator(ATV::cpu::md::VelocityVerlet{/*dt=*/0.5});
    sim.build();
    sim.run(cf, 10);

    EXPECT_TRUE(std::isfinite(cf.meta.get(ATV::kEnergy)))
        << "MD run on a materialized frame must leave a finite energy";
}

// ── arch-refactor-06-frame §2: write_back field contract (golden) ────────

// Contract fields (x/v/f/charge, image, tag, typed metadata) mutated on the
// cpu::Frame must reach the molrs.Frame through write_back. image goes back
// through the same ix/iy/iz i32 columns materialize reads (int8 -> i32
// widen); tag presents the permuted values (TagInt -> i32 narrowing).
TEST(WriteBackContract, GoldenRoundTripContractFields) {
    auto fr = make_contract_frame();

    Frame cf = materialize(*fr);
    ASSERT_EQ(cf.nlocal(), 3);

    // Anchor the materialize read side of the symmetric format first:
    // ix/iy/iz i32 columns landed in the int8 [n,3] image array.
    ASSERT_EQ(cf.span(ATV::kAtom_image)(0, 0), 1);
    ASSERT_EQ(cf.span(ATV::kAtom_image)(1, 1), 3);
    ASSERT_EQ(cf.span(ATV::kAtom_image)(2, 2), 4);
    ASSERT_EQ(cf.span(ATV::kAtom_tag)[0], 10);
    ASSERT_EQ(cf.span(ATV::kAtom_tag)[2], 12);

    // Mutate every contract field so write_back has distinct values to carry.
    cf.span(ATV::kAtom_x)[1] = 6.75;
    cf.span(ATV::kAtom_vy)[0] = -3.5;
    cf.span(ATV::kAtom_fz)[2] = 17.0;
    cf.span(ATV::kAtom_charge)[1] = 0.125;
    cf.span(ATV::kAtom_image)(0, 0) = 3;  // ix[0]: 1 -> 3
    cf.span(ATV::kAtom_image)(1, 1) = -7; // iy[1]: 3 -> -7
    cf.span(ATV::kAtom_image)(2, 2) = 90; // iz[2]: 4 -> 90
    // Permute the tags (reverse): {10, 11, 12} -> {12, 11, 10}.
    cf.span(ATV::kAtom_tag)[0] = 12;
    cf.span(ATV::kAtom_tag)[2] = 10;
    cf.meta.get(ATV::kEnergy) = -77.25;
    cf.meta.get(ATV::kKinetic) = 5.5;

    write_back(cf, *fr);

    // Established write set still carries the mutations.
    auto rx = molrs::frame_column_f64(*fr, "atoms", "x");
    auto rvy = molrs::frame_column_f64(*fr, "atoms", "vy");
    auto rfz = molrs::frame_column_f64(*fr, "atoms", "fz");
    auto rq = molrs::frame_column_f64(*fr, "atoms", "charge");
    ASSERT_EQ(rx.size(), 3u);
    EXPECT_DOUBLE_EQ(rx[1], 6.75);
    EXPECT_DOUBLE_EQ(rvy[0], -3.5);
    EXPECT_DOUBLE_EQ(rfz[2], 17.0);
    EXPECT_DOUBLE_EQ(rq[1], 0.125);

    // NEW contract: image writes back through ix/iy/iz (int8 -> i32 widen,
    // symmetric with the materialize read format).
    auto rix = molrs::frame_column_i32(*fr, "atoms", "ix");
    auto riy = molrs::frame_column_i32(*fr, "atoms", "iy");
    auto riz = molrs::frame_column_i32(*fr, "atoms", "iz");
    ASSERT_EQ(rix.size(), 3u);
    ASSERT_EQ(riy.size(), 3u);
    ASSERT_EQ(riz.size(), 3u);
    EXPECT_EQ(rix[0], 3) << "mutated image(0,0) must reach molrs atoms.ix";
    EXPECT_EQ(rix[1], -1);
    EXPECT_EQ(rix[2], 2);
    EXPECT_EQ(riy[0], 0);
    EXPECT_EQ(riy[1], -7) << "mutated image(1,1) must reach molrs atoms.iy";
    EXPECT_EQ(riy[2], -2);
    EXPECT_EQ(riz[0], -1);
    EXPECT_EQ(riz[1], 0);
    EXPECT_EQ(riz[2], 90) << "mutated image(2,2) must reach molrs atoms.iz";

    // NEW contract: tag writes back (TagInt -> i32 checked narrowing) and
    // presents the permutation.
    auto rtag = molrs::frame_column_i32(*fr, "atoms", "tag");
    ASSERT_EQ(rtag.size(), 3u);
    EXPECT_EQ(rtag[0], 12) << "permuted tag must reach molrs atoms.tag";
    EXPECT_EQ(rtag[1], 11);
    EXPECT_EQ(rtag[2], 10) << "permuted tag must reach molrs atoms.tag";

    // System results stay a typed-metadata write channel. Synthetic scalar
    // blocks are forbidden by the v2 exchange schema.
    const auto meta = molrs::frame_meta_entries(*fr);
    auto find_meta = [&](const char* key) -> const molrs::MetaEntry* {
        for (const auto& entry : meta)
            if (std::string(entry.key.data(), entry.key.size()) == key)
                return &entry;
        return nullptr;
    };
    const auto* energy = find_meta("energy");
    const auto* kinetic = find_meta("kinetic");
    ASSERT_NE(energy, nullptr);
    ASSERT_NE(kinetic, nullptr);
    EXPECT_EQ(energy->dtype, molrs::MetaType::F64);
    EXPECT_EQ(kinetic->dtype, molrs::MetaType::F64);
    EXPECT_DOUBLE_EQ(energy->f64_value, -77.25);
    EXPECT_DOUBLE_EQ(kinetic->f64_value, 5.5);
    EXPECT_FALSE(molrs::frame_has_block(*fr, "scalars"));
}

// Non-contract fields (type / mass / topology blocks / box) are designated
// non-write-back: mutating them on the cpu::Frame must NOT leak into the
// molrs.Frame — every such column keeps its pre-materialize value.
TEST(WriteBackContract, NonContractFieldsUntouched) {
    auto fr = make_contract_frame();

    Frame cf = materialize(*fr);
    ASSERT_EQ(cf.nlocal(), 3);

    // Mutate NON-contract fields...
    cf.span(ATV::kAtom_type)[0] = 99;
    cf.span(ATV::kAtom_mass)[1] = 999.9;
    cf.simbox.set_cell(ATV::Mat3::cubic(42.0));
    // ...plus one contract sentinel proving write_back actually ran.
    cf.span(ATV::kAtom_x)[0] = 8.5;

    write_back(cf, *fr);

    auto rx = molrs::frame_column_f64(*fr, "atoms", "x");
    ASSERT_EQ(rx.size(), 3u);
    ASSERT_DOUBLE_EQ(rx[0], 8.5) << "sentinel: write_back must have run";

    // type / mass keep the make_full_frame originals.
    auto rtype = molrs::frame_column_i32(*fr, "atoms", "type");
    ASSERT_EQ(rtype.size(), 3u);
    EXPECT_EQ(rtype[0], 1) << "atoms.type is non-write-back";
    EXPECT_EQ(rtype[1], 2);
    EXPECT_EQ(rtype[2], 3);
    auto rmass = molrs::frame_column_f64(*fr, "atoms", "mass");
    ASSERT_EQ(rmass.size(), 3u);
    EXPECT_DOUBLE_EQ(rmass[0], 12.011);
    EXPECT_DOUBLE_EQ(rmass[1], 1.008) << "atoms.mass is non-write-back";
    EXPECT_DOUBLE_EQ(rmass[2], 15.999);

    // Topology blocks stay exactly as materialized.
    EXPECT_EQ(static_cast<int>(molrs::frame_block_nrows(*fr, "bonds")), 1);
    EXPECT_EQ(static_cast<int>(molrs::frame_block_nrows(*fr, "angles")), 1);
    EXPECT_EQ(static_cast<int>(molrs::frame_block_nrows(*fr, "dihedrals")), 1);
    auto bi = molrs::frame_column_i32(*fr, "bonds", "atomi");
    auto bj = molrs::frame_column_i32(*fr, "bonds", "atomj");
    auto bt = molrs::frame_column_i32(*fr, "bonds", "type");
    ASSERT_EQ(bi.size(), 1u);
    EXPECT_EQ(bi[0], 10);
    EXPECT_EQ(bj[0], 11);
    EXPECT_EQ(bt[0], 1);
    auto ak = molrs::frame_column_i32(*fr, "angles", "atomk");
    ASSERT_EQ(ak.size(), 1u);
    EXPECT_EQ(ak[0], 12);
    auto dl = molrs::frame_column_i32(*fr, "dihedrals", "atoml");
    ASSERT_EQ(dl.size(), 1u);
    EXPECT_EQ(dl[0], 12);

    // Box keeps the pre-materialize 20 A cube despite the cpu-side 42 A cell.
    auto h = molrs::frame_simbox(*fr);
    ASSERT_EQ(h.size(), 9u);
    EXPECT_DOUBLE_EQ(h[0], 20.0) << "box is non-write-back";
    EXPECT_DOUBLE_EQ(h[4], 20.0);
    EXPECT_DOUBLE_EQ(h[8], 20.0);
}

// A tag that does not fit i32 must fail loud (std::runtime_error naming the
// field), never narrow silently.
TEST(WriteBackContract, TagOverflowThrows) {
    if (sizeof(ATV::TagInt) < 8) {
        GTEST_SKIP() << "TagInt is 32-bit under -DATV_ID_SIZE=smallbig; "
                        "a tag > INT32_MAX is unrepresentable";
    }

    auto fr = make_minimal_frame();
    Frame cf = materialize(*fr);
    ASSERT_EQ(cf.nlocal(), 2);

    const std::int64_t over =
        std::int64_t{std::numeric_limits<std::int32_t>::max()} + 1;
    cf.span(ATV::kAtom_tag)[0] = static_cast<ATV::TagInt>(over);

    try {
        write_back(cf, *fr);
        FAIL() << "write_back must throw std::runtime_error when a tag "
                  "exceeds INT32_MAX instead of silently truncating";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("tag"), std::string::npos)
            << "exception message must name the tag field: " << e.what();
    }
}

// Boundary: a minimal frame (required columns only, no box / topology /
// tag / image columns) survives write_back without throwing.
TEST(WriteBackContract, MinimalFrameWriteBackDoesNotThrow) {
    auto fr = make_minimal_frame();
    Frame cf = materialize(*fr);
    ASSERT_EQ(cf.nlocal(), 2);

    EXPECT_NO_THROW(write_back(cf, *fr));
}

// ── The molrs boundary vs the frame-block mapping facade ────────────────────
//
// frame-block-01 gave cpu::Frame two things molrs has no place for: registry-
// owned EXTENSION COLUMNS (`insert("rho", …)` — a per-atom payload a component
// attaches at runtime). Such columns are engine-local method scratch. Typed,
// serializable metadata is different: it is the current molrs exchange
// channel and crosses exactly, without a synthetic block.

TEST(WriteBackContract, ExtensionColumnsStayLocalAndTypedMetaCrosses) {
    auto fr = make_contract_frame();
    Frame cf = materialize(*fr);
    ASSERT_EQ(cf.nlocal(), 3);

    // Attach an extension column and an owned meta entry, then write back.
    const ATV::ColKey<ATV::StorageReal> rho_key{
        "atoms", "rho", ATV::kDynamicFieldId, ATV::ColDType::Storage};
    cf.block("atoms").insert("rho", ATV::cpu::ColumnSpec{ATV::ColDType::Storage,
                                                         1, ATV::ColGroup::Core,
                                                         0, ATV::kColNone,
                                                         ATV::kDynamicFieldId});
    auto rho = cf.span(rho_key);
    rho[0] = 1.5;
    rho[1] = 2.5;
    rho[2] = 3.5;
    constexpr ATV::MetaKey<std::string> annealing_stage{"annealing_stage",
                                                        true};
    cf.meta.insert(annealing_stage, std::string("quench"));

    // A contract field still carries across — the extras do not disturb it.
    cf.span(ATV::kAtom_x)[1] = 6.75;
    write_back(cf, *fr);

    auto rx = molrs::frame_column_f64(*fr, "atoms", "x");
    ASSERT_EQ(static_cast<int>(rx.size()), 3);
    EXPECT_DOUBLE_EQ(rx[1], 6.75);

    // The extension column is NOT exported: molrs has no "rho" column.
    auto exported_rho = molrs::frame_column_f64(*fr, "atoms", "rho");
    EXPECT_EQ(exported_rho.size(), 0u)
        << "an extension column leaked across the FFI boundary — extras are "
           "engine-local scratch; the exchange schema is the schema";

    // …and it is still intact on the engine side (write_back is a read-only
    // export; it must not disturb the frame).
    auto rho_after = cf.span(rho_key);
    EXPECT_EQ(rho_after[1], ATV::StorageReal(2.5));
    EXPECT_EQ(cf.meta.get(annealing_stage), "quench");

    bool found_stage = false;
    for (const auto& entry : molrs::frame_meta_entries(*fr)) {
        if (std::string(entry.key.data(), entry.key.size())
            == "annealing_stage") {
            found_stage = true;
            EXPECT_EQ(entry.dtype, molrs::MetaType::String);
            EXPECT_EQ(std::string(entry.string_value.data(),
                                  entry.string_value.size()),
                      "quench");
        }
    }
    EXPECT_TRUE(found_stage);
}

TEST(Materialize, MaterializedFrameCarriesNoExtras) {
    // The reverse direction: materialize builds a frame with the schema
    // columns only — no extension columns are invented from stray molrs
    // columns, and the mapping facade reports exactly that.
    auto fr = make_contract_frame();
    Frame cf = materialize(*fr);

    EXPECT_EQ(static_cast<int>(cf.block("atoms").columns().size()),
              ATV::kAtomFieldCount);
    EXPECT_TRUE(cf.has("atoms", "x"));
    EXPECT_FALSE(cf.has("atoms", "rho"));

    // Bookkeeping meta views are live off the materialized frame.
    EXPECT_EQ(cf.meta.get(ATV::kNLocal), 3);
}

TEST(Materialize, RejectsSyntheticResultBlocks) {
    for (const char* block : {"scalars", "stress"}) {
        auto fr = make_minimal_frame();
        const std::vector<double> values = {0.0};
        molrs::frame_set_column_f64(
            *fr, block, "energy",
            ::rust::Slice<const double>(values.data(), values.size()));
        EXPECT_THROW(
            {
                try {
                    (void)materialize(*fr);
                } catch (const std::runtime_error& error) {
                    EXPECT_NE(std::string(error.what()).find(block),
                              std::string::npos);
                    throw;
                }
            },
            std::runtime_error);
    }
}

TEST(Materialize, RejectsUnknownAtomColumns) {
    auto fr = make_minimal_frame();
    const std::vector<double> rho = {1.0, 2.0};
    molrs::frame_set_column_f64(
        *fr, "atoms", "rho",
        ::rust::Slice<const double>(rho.data(), rho.size()));
    EXPECT_THROW((void)materialize(*fr), std::runtime_error);
}

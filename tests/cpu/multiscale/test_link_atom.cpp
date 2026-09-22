/// @file tests/cpu/multiscale/test_link_atom.cpp
/// @brief Unit tests for H link-atom placement and topology-derived cuts.
///
/// Full QMMM electrostatic launch/SCF: regressions/multiscale/qmmm_link_ch3oh
/// (e2e-in-tests wave-2).
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/column.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/multiscale/link_atom.h"
#include "atomiverse/cpu/multiscale/qm_mask.h"
#include "atomiverse/cpu/multiscale/qmmm_electrostatic.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

namespace {

void insert_qm_z(cpu::Frame& f) {
    f.blocks.at("atoms").insert(
        "qm_z", cpu::ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone,
                                kDynamicFieldId});
}

} // namespace

TEST(LinkAtom, PlacementFormula) {
    double hx, hy, hz;
    cpu::place_link_hydrogen(0, 0, 0, 2, 0, 0, 0.5, hx, hy, hz);
    EXPECT_NEAR(hx, 1.0, 1e-14);
    EXPECT_NEAR(hy, 0.0, 1e-14);
    EXPECT_NEAR(hz, 0.0, 1e-14);
    cpu::place_link_hydrogen(1, 2, 3, 4, 2, 3, 0.709, hx, hy, hz);
    EXPECT_NEAR(hx, 1.0 + 0.709 * 3.0, 1e-12);
    EXPECT_NEAR(hy, 2.0, 1e-12);
    EXPECT_NEAR(hz, 3.0, 1e-12);
}

TEST(LinkAtom, ForceProjectionPartition) {
    double fqx = 0, fqy = 0, fqz = 0;
    double fmx = 0, fmy = 0, fmz = 0;
    const double g = 0.7;
    const double fhx = 10.0, fhy = -4.0, fhz = 2.0;
    cpu::project_link_force(g, fhx, fhy, fhz, fqx, fqy, fqz, fmx, fmy, fmz);
    EXPECT_NEAR(fqx, (1.0 - g) * fhx, 1e-14);
    EXPECT_NEAR(fqy, (1.0 - g) * fhy, 1e-14);
    EXPECT_NEAR(fqz, (1.0 - g) * fhz, 1e-14);
    EXPECT_NEAR(fmx, g * fhx, 1e-14);
    EXPECT_NEAR(fmy, g * fhy, 1e-14);
    EXPECT_NEAR(fmz, g * fhz, 1e-14);
    EXPECT_NEAR(fqx + fmx, fhx, 1e-14);
    EXPECT_NEAR(fqy + fmy, fhy, 1e-14);
    EXPECT_NEAR(fqz + fmz, fhz, 1e-14);
}

#ifdef ATV_HAS_HF
TEST(LinkAtom, NoBondsMeansZeroLinks) {
    // LJ-style: mask only, no topology → 0 link cuts (still valid QM/MM).
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_x)[1] = 0.74;
    f.span(kAtom_x)[2] = 5.0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 18;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.span(kAtom_tag)[2] = 3;
    for (int i = 0; i < 3; ++i)
        f.span(kAtom_charge)[i] = 0;
    f.box = cpu::Box::cubic(20);
    f.meta.get(kEnergy) = 0;
    insert_qm_z(f);
    f.span(cpu::kAtomQmZ)[0] = 1;
    f.span(cpu::kAtomQmZ)[1] = 1;
    f.span(cpu::kAtomQmZ)[2] = 0;

    cpu::Context ctx;
    cpu::QmmmElectrostatic q{
        {{1, 1, 0.01, 3.4}, {18, 18, 0.01, 3.4}}, 10.0, 12.0};
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    q.set_engine(std::move(rhf));
    q.build({ctx, f});
    EXPECT_EQ(q.last_n_links(), 0);
}

TEST(LinkAtom, QmmmPlacesLinkHydrogenFromBond) {
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_x)[1] = 1.5;
    f.span(kAtom_x)[2] = 4.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 6;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.span(kAtom_tag)[2] = 3;
    f.span(kAtom_charge)[0] = f.span(kAtom_charge)[1] =
        f.span(kAtom_charge)[2] = 0;
    f.box = cpu::Box::cubic(20);
    f.meta.get(kEnergy) = 0;
    insert_qm_z(f);
    auto qmcol = f.span(cpu::kAtomQmZ);
    qmcol[0] = 1; // H QM
    qmcol[1] = 0; // C MM
    qmcol[2] = 0;

    f.rebuild_tag_index(0, 3);
    f.add_bond(/*ti=*/1, /*tj=*/2, /*type=*/1); // QM H — MM C cross cut

    cpu::Context ctx;
    cpu::QmmmElectrostatic qm{{{1, 1, 0.01, 3.4}}, 10.0, 12.0};
    qm.link_scale() = 0.709;
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    qm.set_engine(std::move(rhf));
    qm.build({ctx, f});
    EXPECT_EQ(qm.last_n_links(), 1);
}

TEST(LinkAtom, BothQmBondIsNotALink) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_x)[1] = 0.74;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.span(kAtom_charge)[0] = f.span(kAtom_charge)[1] = 0;
    f.box = cpu::Box::cubic(20);
    f.meta.get(kEnergy) = 0;
    insert_qm_z(f);
    f.span(cpu::kAtomQmZ)[0] = 1;
    f.span(cpu::kAtomQmZ)[1] = 1;
    f.rebuild_tag_index(0, 2);
    f.add_bond(1, 2, 1);

    cpu::Context ctx;
    cpu::QmmmElectrostatic q{{{1, 1, 0.01, 3.4}}, 10.0, 12.0};
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    q.set_engine(std::move(rhf));
    q.build({ctx, f});
    EXPECT_EQ(q.last_n_links(), 0);
}
#endif

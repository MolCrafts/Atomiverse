/// @file tests/cpu/potentials/test_bond_fene.cpp
#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/bond_fene.h"

using namespace ATV;

namespace {

cpu::Frame dimer_with_bond(double r) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 0;
    f.span(kAtom_tag)[0] = 0;
    f.span(kAtom_tag)[1] = 1;
    f.box = cpu::Box::cubic(50.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    f.add_bond(0, 1, 0);
    return f;
}

} // namespace

TEST(BondFene, AnalyticEnergy) {
    const double K = 30.0, R0 = 1.5, r = 1.0;
    const double U = -0.5 * K * R0 * R0 * std::log(1.0 - (r / R0) * (r / R0));

    auto f = dimer_with_bond(r);
    f.meta.get(kEnergy) = 0;
    f.meta.get(kEnergyBond) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;

    cpu::Context ctx;
    cpu::BondFene bond({K}, {R0});
    bond.build({ctx, f});
    bond.launch({ctx, f, 0, 0});
    EXPECT_NEAR(f.meta.get(kEnergy), U, 1e-12);
    // Attractive: atom 0 force +x
    EXPECT_GT(f.span(kAtom_fx)[0], 0.0);
    EXPECT_LT(f.span(kAtom_fx)[1], 0.0);
    bond.destroy({ctx, f});
}

TEST(BondFene, ThrowsAtOrBeyondR0) {
    auto f = dimer_with_bond(1.5);
    f.meta.get(kEnergy) = 0;
    cpu::Context ctx;
    cpu::BondFene bond({30.0}, {1.5});
    bond.build({ctx, f});
    EXPECT_THROW(bond.launch({ctx, f, 0, 0}), std::runtime_error);
    bond.destroy({ctx, f});
}

TEST(BondFene, MissingPartnerFailsBeforePartialEvaluation) {
    auto f = dimer_with_bond(1.0);
    f.add_bond(0, 999, 0);
    cpu::Context ctx;
    cpu::BondFene bond({30.0}, {1.5});
    bond.build({ctx, f});
    f.meta.get(kEnergy) = 123;
    f.span(kAtom_fx)[0] = 456;
    EXPECT_THROW(bond.launch({ctx, f, 0, 0}), std::runtime_error);
    EXPECT_EQ(f.meta.get(kEnergy), 123);
    EXPECT_EQ(f.span(kAtom_fx)[0], 456);
    EXPECT_THROW(bond.launch({ctx, f, 0, 1}), std::runtime_error);
    bond.destroy({ctx, f});
}

TEST(BondFene, AddedTopologyInvalidatesCachedTuples) {
    auto f = dimer_with_bond(1.0);
    cpu::Context ctx;
    cpu::BondFene bond({30.0}, {1.5});
    bond.build({ctx, f});
    f.meta.get(kEnergy) = 0;
    bond.launch({ctx, f, 0, 0});
    const double one_bond = f.meta.get(kEnergy);
    f.add_bond(0, 1, 0);
    f.meta.get(kEnergy) = 0;
    bond.launch({ctx, f, 0, 1});
    EXPECT_NEAR(f.meta.get(kEnergy), 2 * one_bond, 1e-12);
    bond.destroy({ctx, f});
}

TEST(BondFene, RequestsAreIndependentAndFrameRebindingRefreshesTuples) {
    auto f = dimer_with_bond(1.0);
    cpu::Context ctx;
    cpu::BondFene bond({30.0}, {1.5});
    bond.build({ctx, f});
    f.meta.get(kEnergy) = 0;
    f.meta.get(kStress).fill(0);
    f.span(kAtom_fx)[0] = f.span(kAtom_fx)[1] = 0;
    bond.launch({ctx, f, 0, 0});
    const double energy = f.meta.get(kEnergy), force = f.span(kAtom_fx)[0];
    const auto stress = f.meta.get(kStress);
    auto reordered = f;
    reordered.copy_row_from(f, 0, 1);
    reordered.copy_row_from(f, 1, 0);
    reordered.clear_tag_index();
    reordered.rebuild_tag_index(0, 2);
    for (bool forces : {false, true})
        for (bool virial : {false, true}) {
            reordered.meta.get(kEnergy) = 0;
            reordered.meta.get(kStress).fill(7);
            reordered.span(kAtom_fx)[0] = reordered.span(kAtom_fx)[1] = 8;
            bond.launch({ctx, reordered, 0, 1, forces, virial});
            EXPECT_NEAR(reordered.meta.get(kEnergy), energy, 1e-12);
            EXPECT_NEAR(reordered.span(kAtom_fx)[1], 8 + (forces ? force : 0),
                        1e-12);
            EXPECT_NEAR(reordered.span(kAtom_fx)[0], 8 - (forces ? force : 0),
                        1e-12);
            for (int d = 0; d < 6; ++d)
                EXPECT_NEAR(reordered.meta.get(kStress)[d],
                            7 + (virial ? stress[d] : 0), 1e-12);
        }
    bond.destroy({ctx, reordered});
}

TEST(BondedForceScratch, ReductionUsesTheActualTeam) {
    cpu::Frame f(2);
    cpu::md::BondedForceScratch scratch;
    scratch.build(2, f.capacity());
    scratch.bind(f);
#ifdef _OPENMP
    const int old_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
#endif
#pragma omp parallel num_threads(2)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto force = scratch.begin_thread(tid, 2);
        force.fx[0] = 100 + tid;
    }
    // A later serialized/nested team must ignore the prior thread 1 buffer.
    auto force = scratch.begin_thread(0, 2);
    force.fx[0] = 3;
    force.fx[1] = 4;
    f.span(kAtom_fx)[0] = f.span(kAtom_fx)[1] = 0;
    scratch.reduce_into(2);
    EXPECT_EQ(f.span(kAtom_fx)[0], 3);
    EXPECT_EQ(f.span(kAtom_fx)[1], 4);
#ifdef _OPENMP
    omp_set_dynamic(old_dynamic);
#endif
}

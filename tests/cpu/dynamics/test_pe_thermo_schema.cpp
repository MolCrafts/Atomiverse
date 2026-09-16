/// @file tests/cpu/md/test_pe_thermo_schema.cpp
/// @brief MSU-PE thermo-schema invariants — `msu-pe-cuda-components-01`
///        ac-001 (per-term typed Frame metadata default-zero) and ac-006
///        (`energy ≈ Σ per-term energies` invariant), exercised by
///        directly calling the CPU evaluators on a 4-atom PE-like
///        fragment (one bond, one angle, one dihedral).
///
/// The benchmark thermo.csv 17-column extended schema (Phase G.2 + G.3)
/// is verified end-to-end by the operator-action smoke runs of
/// `tools/run_msu_atomiverse.sh`; this unit test pins the underlying
/// CPU invariants that the schema rides on.

#include <gtest/gtest.h>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

#include <cmath>

using namespace ATV;

// ── ac-001: per-term energy fields exist and default to zero ─────────

TEST(PeThermoSchema, TypedMetaDefaultsToZero) {
    cpu::Frame frame;
    EXPECT_EQ(frame.meta.get(kEnergy), 0.0);
    EXPECT_EQ(frame.meta.get(kEnergyPair), 0.0);
    EXPECT_EQ(frame.meta.get(kEnergyBond), 0.0);
    EXPECT_EQ(frame.meta.get(kEnergyAngle), 0.0);
    EXPECT_EQ(frame.meta.get(kEnergyDihedral), 0.0);
    EXPECT_EQ(frame.meta.get(kEnergyImproper), 0.0);
    EXPECT_EQ(frame.meta.get(kEnergyChargingWork), 0.0);
    EXPECT_EQ(frame.meta.get(kKinetic), 0.0);
    EXPECT_EQ(frame.meta.get(kConserved), 0.0);
    const auto& stress = frame.meta.get(kStress);
    const auto& stress_kinetic = frame.meta.get(kStressKinetic);
    for (int k = 0; k < 6; ++k) {
        EXPECT_EQ(stress[static_cast<std::size_t>(k)], 0.0);
        EXPECT_EQ(stress_kinetic[static_cast<std::size_t>(k)], 0.0);
    }
}

TEST(PeThermoSchema, ZeroEnergyClearsAllEnergyAccumulators) {
    cpu::Frame frame;
    frame.meta.set(kEnergy, 1.5);
    frame.meta.set(kEnergyPair, 0.5);
    frame.meta.set(kEnergyBond, 0.25);
    frame.meta.set(kEnergyAngle, 0.5);
    frame.meta.set(kEnergyDihedral, 0.25);
    frame.meta.set(kEnergyImproper, 0.1);
    frame.meta.set(kEnergyChargingWork, 0.4);
    frame.meta.set(kKinetic, 99.0); // should NOT be touched by ZeroEnergy
    auto stress = frame.meta.get(kStress);
    stress[0] = 1.0; // should NOT be touched by ZeroEnergy
    frame.meta.set(kStress, stress);

    cpu::Context runtime;
    cpu::ZeroEnergy zero;
    zero.launch(cpu::LaunchContext{runtime, frame, 0, 0});

    EXPECT_EQ(frame.meta.get(kEnergy), 0.0);
    EXPECT_EQ(frame.meta.get(kEnergyPair), 0.0);
    EXPECT_EQ(frame.meta.get(kEnergyBond), 0.0);
    EXPECT_EQ(frame.meta.get(kEnergyAngle), 0.0);
    EXPECT_EQ(frame.meta.get(kEnergyDihedral), 0.0);
    EXPECT_EQ(frame.meta.get(kEnergyImproper), 0.0);
    EXPECT_EQ(frame.meta.get(kEnergyChargingWork), 0.0);
    // ZeroEnergy must NOT touch kinetic or stress.
    EXPECT_EQ(frame.meta.get(kKinetic), 99.0);
    EXPECT_EQ(frame.meta.get(kStress)[0], 1.0);
}

// ── ac-006: per-term energies sum to total potential energy ──────────

namespace {

/// @brief Construct a 4-atom PE-like fragment (one bond, one angle, one
///        dihedral, plus a pair of well-separated extra atoms to exercise
///        a non-trivial LJ pair). Geometry is hand-tuned so every evaluator
///        contributes a non-zero energy — bond stretched off equilibrium,
///        angle off 109.5°, dihedral off cosφ=1.
cpu::Frame make_pe_fragment() {
    cpu::Frame f(6);
    // Linear chain 0-1-2-3 stretched along x with mild zig.
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_x)[1] = 1.6;
    f.span(ATV::kAtom_y)[1] = 0.0;
    f.span(ATV::kAtom_z)[1] = 0.0;
    f.span(ATV::kAtom_x)[2] = 2.8;
    f.span(ATV::kAtom_y)[2] = 1.2;
    f.span(ATV::kAtom_z)[2] = 0.0;
    f.span(ATV::kAtom_x)[3] = 4.2;
    f.span(ATV::kAtom_y)[3] = 1.5;
    f.span(ATV::kAtom_z)[3] = 0.6;
    // Two extra atoms placed inside LJ cutoff but outside the bonded set.
    f.span(ATV::kAtom_x)[4] = 6.0;
    f.span(ATV::kAtom_y)[4] = 0.0;
    f.span(ATV::kAtom_z)[4] = 0.0;
    f.span(ATV::kAtom_x)[5] = 7.0;
    f.span(ATV::kAtom_y)[5] = 0.0;
    f.span(ATV::kAtom_z)[5] = 0.0;
    for (int i = 0; i < 6; ++i) {
        f.span(ATV::kAtom_vx)[i] = f.span(ATV::kAtom_vy)[i] =
            f.span(ATV::kAtom_vz)[i] = 0.0;
        f.span(ATV::kAtom_fx)[i] = f.span(ATV::kAtom_fy)[i] =
            f.span(ATV::kAtom_fz)[i] = 0.0;
        f.span(ATV::kAtom_mass)[i] = 14.0; // UA CH₂
        f.span(ATV::kAtom_type)[i] = 0;
    }
    f.box = ATV::cpu::Box::cubic(40.0); // large box -> no PBC interference

    // One bond (0-1), one angle (0-1-2), one dihedral (0-1-2-3), in the
    // per-atom topology the force kernels actually read (num_bond/bond_atom,
    // num_angle/angle_atom, …). The legacy flat `f.bonds/angles/dihedrals`
    // blocks are dead storage the kernels ignore, so populating those left
    // every per-term energy at zero. Mirrors the MSU-PE pipeline idiom.
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    f.add_bond(0, 1, 0);
    f.add_angle(0, 1, 2, 0);
    f.add_dihedral(0, 1, 2, 3, 0);
    f.build_special_list();
    return f;
}

} // namespace

TEST(PeThermoSchema, PerTermEnergiesSumToTotalPotentialEnergy) {
    auto frame = make_pe_fragment();

    // Force-field constants — values mirror the LAMMPS Dreiding-UA setup
    // used by the MSU-PE benchmark, chosen so each evaluator returns a
    // non-trivial positive (bond/angle) or signed (LJ/dihedral) energy
    // when the geometry is off-equilibrium.
    const double lj_eps = 0.112 / 23.060547830619026; // 0.112 kcal/mol → eV
    const double lj_sigma = 4.01;
    const double lj_cut = 10.0;
    const double bond_k_eV = 350.0 / 23.060547830619026 * 0.5; // K (eV/Å²)
    const double bond_r0 = 1.54;
    const double angle_k_eV = 60.0 / 23.060547830619026 * 0.5;
    const double angle_th0 = 109.5 * M_PI / 180.0;
    std::vector<double> dih_C = {0.5, 0.0, 0.0, 0.0, 0.0};

    cpu::md::NeighborList<> nl{lj_cut};
    cpu::PairLJCut lj =
        cpu::PairLJCut{lj_eps, lj_sigma, lj_cut}.set_special_bonds(0.0, 0.0,
                                                                   0.0);
    // Canonical names: cpu::BondHarmonic / cpu::AngleHarmonic /
    // cpu::DihedralMultiHarmonic.
    cpu::BondHarmonic hb{{bond_k_eV}, {bond_r0}};
    cpu::AngleHarmonic ha{{angle_k_eV}, {angle_th0}};
    cpu::DihedralMultiHarmonic hd{{dih_C}};
    cpu::ZeroEnergy ze;
    cpu::ZeroForces zf;
    cpu::ZeroStress zs;

    // One evaluator pass: zero accumulators (Σ per-term as well), then
    // run each evaluator. The Evaluate driver pipes them through the
    // single `evaluate::Phase::evaluate` slot in registration order;
    // for this invariant we want all four evaluators to write their
    // per-term and total energy on the same line — ZeroEnergy resets
    // both, ensuring no cross-step leakage.
    cpu::Evaluate eval;
    eval.on(ATV::evaluate::Phase::prepare)
        .add_ref(zf, -1000)
        .add_ref(ze, -999)
        .add_ref(zs, -998);
    eval.set_neighbor_ref(nl);
    eval.set_evaluator_ref(lj);
    eval.set_evaluator_ref(hb);
    eval.set_evaluator_ref(ha);
    eval.set_evaluator_ref(hd);
    eval.build();
    eval.run(frame);

    const double total = frame.meta.get(ATV::kEnergyPair)
                         + frame.meta.get(ATV::kEnergyBond)
                         + frame.meta.get(ATV::kEnergyAngle)
                         + frame.meta.get(ATV::kEnergyDihedral);

    // Each evaluator wrote `+= energy_total` and `+= energy_<term>` on
    // the SAME line — the invariant is a literal equality (no parallel
    // reduction order shuffle, no FMA reordering). Compare against
    // FP-strict tolerance.
    EXPECT_NEAR(frame.meta.get(ATV::kEnergy), total,
                std::fabs(frame.meta.get(ATV::kEnergy)) * 1e-12 + 1e-15);

    // All four evaluators should have actually contributed.
    EXPECT_NE(frame.meta.get(ATV::kEnergyBond), 0.0);
    EXPECT_NE(frame.meta.get(ATV::kEnergyAngle), 0.0);
    EXPECT_NE(frame.meta.get(ATV::kEnergyDihedral), 0.0);
}

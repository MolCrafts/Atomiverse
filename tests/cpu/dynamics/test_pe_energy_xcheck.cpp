/// @file tests/cpu/md/test_pe_energy_xcheck.cpp
/// @brief Spec c20h42-energy-xcheck — C20H42 single-chain bonded-energy
///        cross-check: Atomiverse vs LAMMPS, term by term, to 1e-6 eV.
///
/// Carves Phase A.2 of `msu-pe-validation.md` into an automated CTest
/// gate, and is the real verification that `pe-ff-migrate-dreiding-ua`
/// ac-006 relocated here.
///
/// A deterministically-perturbed all-trans C20H42 united-atom chain is
/// the single source of truth: the 20×3 coordinates embedded below as
/// `kCoords` are byte-for-byte the coordinates in
/// the reference C20H42 chain data (see
/// `benchmarks/md/pe/validation/reference/SOURCES.md`), so both
/// engines evaluate identical geometry. A perfect-equilibrium chain
/// would give E_bond = E_angle = 0; the perturbation makes every term
/// non-trivial so the kernels are actually exercised.
///
/// Functional-form audit (verified against LAMMPS docs + Atomiverse
/// source): `bond_style harmonic`, `angle_style harmonic` and
/// `dihedral_style multi/harmonic` are term-identical between the two
/// engines — `K` absorbs the textbook ½ on both sides, and the dihedral
/// series `Σ A_n cos^{n-1}φ` (LAMMPS, 1-based) equals Atomiverse's
/// `Σ C_n cos^n φ` (0-based) with `C_n = A_{n+1}`. The only adjustments
/// the cross-check applies are kcal/mol→eV and θ0 degrees→radians; the
/// dihedral φ-sign convention is irrelevant (pure polynomial in cos φ).
///
/// LAMMPS reference: `lmp_serial -in in.c20h42_energy` (LAMMPS
/// 22 Jul 2025, `run 0`). See `benchmarks/md/pe/validation/
/// reference/SOURCES.md` for full provenance + regeneration commands.

#include <gtest/gtest.h>

#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"

using namespace ATV;

namespace {

inline constexpr double kEnergyTolerance = 1.0e-6;

// ── Units ────────────────────────────────────────────────────────────
// LAMMPS `units real` energy is kcal/mol; Atomiverse energy is eV. One
// full-precision constant, shared by the FF constants and the embedded
// reference (matches tests/cpu/md/test_pe_ff_constants.cpp).
inline constexpr double kKcalToEv = 1.0 / 23.060547830619026;

// ── MSU Dreiding-UA force field (verbatim from lammps_msu_paper.in) ───
// LAMMPS bond/angle `harmonic` absorb the textbook ½: V = K(r-r0)²,
// V = K(θ-θ0)². Atomiverse `BondHarmonic` / `AngleHarmonic` use the
// identical convention, so K converts by units alone.
inline constexpr double kBondK_kcal = 350.0; // kcal/mol/Å²
inline constexpr double kBondR0 = 1.53;      // Å
inline constexpr double kAngleK_kcal = 60.0; // kcal/mol/rad²
inline constexpr double kAngleTheta0 = 109.5 * units::pi / 180.0; // rad

inline double bond_K_eV() { return kBondK_kcal * kKcalToEv; }
inline double angle_K_eV() { return kAngleK_kcal * kKcalToEv; }

// Dihedral multi/harmonic A1..A5 → Atomiverse C0..C4 (same load order).
inline std::vector<double> dihedral_Cn_eV() {
    return {1.73 * kKcalToEv, -4.49 * kKcalToEv, 0.776 * kKcalToEv,
            6.99 * kKcalToEv, 0.0};
}

// ── LAMMPS reference energies ────────────────────────────────────────
// `lmp_serial -in in.c20h42_energy`, run 0, Step/E_bond/E_angle/E_dihed
// thermo line — verbatim, kcal/mol. Regenerate per SOURCES.md.
inline constexpr double kRefBondKcal = 122.63225361801;
inline constexpr double kRefAngleKcal = 30.774982565814;
inline constexpr double kRefDihedKcal = 3.5122664779742;

// ── Perturbed C20H42 geometry — single source of truth ───────────────
// Byte-identical to the reference chain data (SOURCES.md provenance).
// All-trans zigzag (r0=1.53 Å, θ0=109.5°) + per-atom displacement
// 0.10 Å·(sin(2.4i+0.7), sin(3.1i+1.3), sin(1.7i+2.1)). See SOURCES.md.
constexpr int kN = 20;
constexpr double kCoords[kN][3] = {
    {50.06442176872377, 50.09635581854172, 50.08632093666489},
    {51.2536196456407, 50.78787193336802, 49.938814210905726},
    {52.4283691262377, 50.09379999767747, 49.92944596744296},
    {53.848278872376085, 50.79075459859568, 50.07936678638492},
    {54.92107773661312, 50.090595474230845, 50.05010208564579},
    {56.26063110112884, 50.79427543739882, 49.907722457838716},
    {57.55388916335021, 50.08676441006417, 49.97367682086342},
    {58.64866845523476, 50.79841010033945, 50.09906073556949},
    {60.08245704524312, 50.082333300073806, 50.00079631837859},
    {61.21475775369521, 50.80312999289101, 49.90073406195294},
    {62.45267970236636, 50.077332788956625, 50.02478342079829},
    {63.83632065506337, 50.80840247319247, 50.092879523407724},
    {64.89943581193414, 50.071797459277164, 49.95128254875395},
    {66.28953937980128, 50.8141910777932, 49.919674427330605},
    {67.51793034484756, 50.06576559230139, 50.069416466825224},
    {68.65782461298532, 50.82045577382702, 50.06243771354164},
    {70.08994530023696, 50.05927890325083, 49.914494021922295},
    {71.17959083445773, 50.827153235871805, 49.95959623546769},
    {72.4820880007126, 50.05238225280825, 50.09591758329531},
    {73.81314940838905, 50.83423714557968, 50.01568685950484},
};

/// @brief Build the perturbed C20H42 chain Frame: 20 atoms from the
/// embedded coordinates, 19 bonds / 18 angles / 17 dihedrals all type 0.
cpu::Frame make_c20h42_perturbed() {
    cpu::Frame frame(kN);
    for (int i = 0; i < kN; ++i) {
        frame.span(ATV::kAtom_x)[i] = kCoords[i][0];
        frame.span(ATV::kAtom_y)[i] = kCoords[i][1];
        frame.span(ATV::kAtom_z)[i] = kCoords[i][2];
        frame.span(ATV::kAtom_mass)[i] = 14.027; // CH2 united atom
        frame.span(ATV::kAtom_type)[i] = 0;
    }

    const int nb = kN - 1, na = kN - 2, nd = kN - 3;

    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    for (int k = 0; k < nb; ++k) {
        frame.add_bond(k, k + 1, 0);
    }
    for (int k = 0; k < na; ++k) {
        frame.add_angle(k, k + 1, k + 2, 0);
    }
    for (int k = 0; k < nd; ++k) {
        frame.add_dihedral(k, k + 1, k + 2, k + 3, 0);
    }
    frame.build_special_list();

    frame.box = ATV::cpu::Box::cubic(100.0);
    return frame;
}

/// @brief Run one bonded component through `build → launch → destroy`
/// and return the energy it accumulated into `frame.meta.get(ATV::kEnergy)` [eV].
template <typename Comp>
double eval_one(cpu::Frame& frame, Comp& comp) {
    cpu::Context ctx;
    cpu::BuildContext bctx{ctx, frame};
    cpu::LaunchContext lctx{ctx, frame, 0, 0};

    comp.build(bctx);
    const int n = frame.nlocal();
    for (int i = 0; i < n; ++i) {
        frame.span(ATV::kAtom_fx)[i] = 0;
        frame.span(ATV::kAtom_fy)[i] = 0;
        frame.span(ATV::kAtom_fz)[i] = 0;
    }
    frame.meta.get(ATV::kEnergy) = 0;
    comp.launch(lctx);
    comp.destroy({ctx, frame});
    return frame.meta.get(ATV::kEnergy);
}

} // namespace

// ── ac-006: single-source-of-truth geometry + topology pinned ─────────

TEST(PeEnergyXcheck, GeometryAndTopologyPinned) {
    static_assert(sizeof(kCoords) / sizeof(kCoords[0]) == 20,
                  "embedded C20H42 geometry must have exactly 20 atoms");
    auto frame = make_c20h42_perturbed();
    EXPECT_EQ(frame.nlocal(), 20);
    int n_bond = 0, n_angle = 0, n_dihedral = 0;
    for (int i = 0; i < frame.nlocal(); ++i) {
        n_bond += frame.span(ATV::kAtom_num_bond)[i];
        n_angle += frame.span(ATV::kAtom_num_angle)[i];
        n_dihedral += frame.span(ATV::kAtom_num_dihedral)[i];
    }
    EXPECT_EQ(n_bond, 19);
    EXPECT_EQ(n_angle, 18);
    EXPECT_EQ(n_dihedral, 17);
}

// ── ac-004: the perturbation makes the bond/angle kernels non-trivial ─

TEST(PeEnergyXcheck, PerturbedGeometryExercisesKernels) {
    auto frame = make_c20h42_perturbed();
    cpu::BondHarmonic bond({bond_K_eV()}, {kBondR0});
    cpu::AngleHarmonic angle({angle_K_eV()}, {kAngleTheta0});

    const double e_bond = eval_one(frame, bond);
    const double e_angle = eval_one(frame, angle);

    // A perfect-equilibrium chain gives E_bond = E_angle = 0 and would
    // pass any comparison trivially. The deterministic perturbation must
    // drive both strictly positive.
    EXPECT_GT(e_bond, 1e-3) << "perturbed C20H42 must exercise the bond kernel";
    EXPECT_GT(e_angle, 1e-3)
        << "perturbed C20H42 must exercise the angle kernel";
}

// ── ac-005: bonded energies match LAMMPS term-by-term within 1e-6 eV ──

TEST(PeEnergyXcheck, BondEnergyMatchesLammps) {
    auto frame = make_c20h42_perturbed();
    cpu::BondHarmonic bond({bond_K_eV()}, {kBondR0});
    const double e_bond = eval_one(frame, bond);
    const double ref = kRefBondKcal * kKcalToEv;
    EXPECT_NEAR(e_bond, ref, kEnergyTolerance)
        << "E_bond: Atomiverse " << e_bond << " eV vs LAMMPS " << ref << " eV";
}

TEST(PeEnergyXcheck, AngleEnergyMatchesLammps) {
    auto frame = make_c20h42_perturbed();
    cpu::AngleHarmonic angle({angle_K_eV()}, {kAngleTheta0});
    const double e_angle = eval_one(frame, angle);
    const double ref = kRefAngleKcal * kKcalToEv;
    EXPECT_NEAR(e_angle, ref, kEnergyTolerance)
        << "E_angle: Atomiverse " << e_angle << " eV vs LAMMPS " << ref
        << " eV";
}

TEST(PeEnergyXcheck, DihedralEnergyMatchesLammps) {
    auto frame = make_c20h42_perturbed();
    cpu::DihedralMultiHarmonic dih({dihedral_Cn_eV()});
    const double e_dih = eval_one(frame, dih);
    const double ref = kRefDihedKcal * kKcalToEv;
    // The multi/harmonic A1 constant term contributes 17·A1 as a fixed
    // offset to edihed — physically real and present on both engines.
    EXPECT_NEAR(e_dih, ref, kEnergyTolerance)
        << "E_dihedral: Atomiverse " << e_dih << " eV vs LAMMPS " << ref
        << " eV";
}

TEST(PeEnergyXcheck, TotalBondedEnergyMatchesLammps) {
    auto frame = make_c20h42_perturbed();
    cpu::BondHarmonic bond({bond_K_eV()}, {kBondR0});
    cpu::AngleHarmonic angle({angle_K_eV()}, {kAngleTheta0});
    cpu::DihedralMultiHarmonic dih({dihedral_Cn_eV()});

    const double total =
        eval_one(frame, bond) + eval_one(frame, angle) + eval_one(frame, dih);
    const double ref =
        (kRefBondKcal + kRefAngleKcal + kRefDihedKcal) * kKcalToEv;
    EXPECT_NEAR(total, ref, kEnergyTolerance)
        << "E_bonded total: Atomiverse " << total << " eV vs LAMMPS " << ref
        << " eV";
}

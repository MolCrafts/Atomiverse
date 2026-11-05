/// @file tests/cuda/potentials/test_pair_lj_typed_cutoff.cu
/// @brief Unit: typed LJ with LAMMPS init_one per-pair cutoffs and the r⁻⁴
///        attraction on `cuda::PairLJCut` and `cuda::PairLJCutCoulLong`.
///
/// Mirrors `tests/cpu/potentials/test_pair_lj_typed_cutoff.cpp` assertion for
/// assertion; the analytic references live in `core/typed_lj_cases.h`.
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/longrange/ewald.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut_coul_long.cuh"
#include "core/typed_lj_cases.h"

using namespace ATV;
using namespace typed_lj_cases;

namespace {

constexpr double kTol = 1e-11;
constexpr double kAlpha = 0.3;

struct Result {
    double energy, fx0, fx1, wxx;
    bool typed = false; ///< PairLJCutCoulLong tile kernel selection.
};

/// Two atoms of types (ta, tb) a distance r apart along x, optionally bonded.
cpu::Frame dimer(int ta, int tb, double r, bool bonded = false, double qa = 0.0,
                 double qb = 0.0) {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_x)[i] = 10.0 + (i ? r : 0.0);
        f.span(kAtom_y)[i] = 10.0;
        f.span(kAtom_z)[i] = 10.0;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_tag)[i] = i;
    }
    f.span(kAtom_type)[0] = ta;
    f.span(kAtom_type)[1] = tb;
    f.span(kAtom_charge)[0] = qa;
    f.span(kAtom_charge)[1] = qb;
    f.box = cpu::Box::cubic(40.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, 2);
    if (bonded)
        f.add_bond(0, 1, 0);
    f.build_special_list();
    return f;
}

template <class Pair>
Result evaluate(cpu::Frame f, Pair pair) {
    constexpr bool kCoulLong = std::is_same_v<Pair, cuda::PairLJCutCoulLong>;
    Result out{};
    {
        cuda::Evaluate eval;
        eval.set_neighbor(cuda::NeighborList{kNeighbor, 0.5});
        eval.set_evaluator_ref(pair);
        if constexpr (kCoulLong)
            eval.on(evaluate::Phase::evaluate)
                .add(cuda::KSpaceEwald{kAlpha, 1e-6}, 100);
        eval.build();
        eval.run(f);
        if constexpr (kCoulLong)
            out.typed = pair.typed_kernel();
    }
    out.energy = f.meta.get(kEnergy);
    out.fx0 = f.span(kAtom_fx)[0];
    out.fx1 = f.span(kAtom_fx)[1];
    out.wxx = f.meta.get(kStress)[0];
    return out;
}

Result lj_cut(cpu::Frame f, cuda::PairLJCut pair) {
    return evaluate(std::move(f), std::move(pair));
}
Result coul_long(cpu::Frame f, cuda::PairLJCutCoulLong pair) {
    return evaluate(std::move(f), std::move(pair));
}

/// Energy, Newton pair force along x and the pair virial r F.
void expect_pair(const Result& got, double energy, double force, double r) {
    EXPECT_NEAR(got.energy, energy, kTol);
    EXPECT_NEAR(got.fx1, force, kTol);
    EXPECT_NEAR(got.fx0, -force, kTol);
    EXPECT_NEAR(got.wxx, r * force, kTol);
}

} // namespace

// ── PairLJCut ──────────────────────────────────────────────────────────────

TEST(CudaPairLJCutTypedCutoff, ReportsTheLargestPairCutoff) {
    cuda::PairLJCut pair{diagonal_rows(), kCutoff};
    EXPECT_NEAR(pair.pair_cutoff(), kCut1, kTol);
    pair.set_r4_attraction({{0, 1, kS, kSigR4, 12.0}});
    EXPECT_NEAR(pair.pair_cutoff(), 12.0, kTol);
    EXPECT_THROW(cuda::PairLJCut(1.0, 1.0, 3.0).set_r4_attraction({}),
                 std::logic_error);
}

// A per-type table needs the force-field type column; a Frame that left it
// out (species only) is refused with the how-to on both styles, never read
// as zeros. The device frame then has no type array at all.
TEST(CudaPairLJCutTypedCutoff, TypedTableRefusesAFrameWithoutTypes) {
    cpu::Frame f = dimer(0, 1, 7.0);
    f.block("atoms").erase(kAtom_type.key);
    try {
        (void)lj_cut(f, {diagonal_rows(), kCutoff});
        FAIL() << "cuda::PairLJCut ran a per-type table without types";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("type_id"), std::string::npos)
            << e.what();
    }
    try {
        (void)coul_long(f, {diagonal_rows(), kAlpha, kCutoff});
        FAIL() << "cuda::PairLJCutCoulLong ran without types";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("type_id"), std::string::npos)
            << e.what();
    }
}

TEST(CudaPairLJCutTypedCutoff, MixedPairInsideItsMixedCutoff) {
    const double r = 7.0;
    expect_pair(lj_cut(dimer(0, 1, r), {diagonal_rows(), kCutoff}),
                lj_shifted(kEps01, kSig01, kCut01, r),
                lj_force(kEps01, kSig01, kCut01, r), r);
}

TEST(CudaPairLJCutTypedCutoff, MixedPairOutsideItsMixedCutoff) {
    expect_pair(lj_cut(dimer(0, 1, 8.5), {diagonal_rows(), kCutoff}), 0.0, 0.0,
                8.5);
}

TEST(CudaPairLJCutTypedCutoff, DiagonalCutoffBeyondTheGlobalOne) {
    const double r = 9.5;
    expect_pair(lj_cut(dimer(1, 1, r), {diagonal_rows(), kCutoff}),
                lj_shifted(kEps1, kSig1, kCut1, r),
                lj_force(kEps1, kSig1, kCut1, r), r);
}

TEST(CudaPairLJCutTypedCutoff, ExplicitCrossRowOverridesTheMix) {
    auto rows = diagonal_rows();
    rows.push_back({0, 1, 0.3, 3.2, 7.5});
    const double r = 7.2;
    expect_pair(lj_cut(dimer(0, 1, r), {rows, kCutoff}),
                lj_shifted(0.3, 3.2, 7.5, r), lj_force(0.3, 3.2, 7.5, r), r);
}

TEST(CudaPairLJCutTypedCutoff, R4AttractionAddsToLJ) {
    const double r = 5.0;
    expect_pair(
        lj_cut(dimer(0, 1, r),
               cuda::PairLJCut{diagonal_rows(), kCutoff}.set_r4_attraction(
                   {{0, 1, kS, kSigR4, kCutR4}})),
        lj_shifted(kEps01, kSig01, kCut01, r) + r4(kS, kSigR4, kCutR4, r),
        lj_force(kEps01, kSig01, kCut01, r) + r4_force(kS, kSigR4, kCutR4, r),
        r);
}

TEST(CudaPairLJCutTypedCutoff, R4AloneIsContinuousAtItsCutoff) {
    auto make = [] {
        return cuda::PairLJCut{{{0, 0, 0.0, 1.0}, {1, 1, 0.0, 1.0}}, kCutoff}
            .set_r4_attraction({{0, 1, kS, kSigR4, kCutR4}});
    };
    const double r = 4.0;
    expect_pair(lj_cut(dimer(0, 1, r), make()), r4(kS, kSigR4, kCutR4, r),
                r4_force(kS, kSigR4, kCutR4, r), r);
    const Result in = lj_cut(dimer(0, 1, kCutR4 - 1e-4), make());
    EXPECT_LT(std::fabs(in.energy), 1e-9);
    EXPECT_LT(std::fabs(in.fx1), 1e-6);
    expect_pair(lj_cut(dimer(0, 1, kCutR4 + 1e-3), make()), 0.0, 0.0,
                kCutR4 + 1e-3);
}

TEST(CudaPairLJCutTypedCutoff, SpecialWeightScalesLJAndR4) {
    const double r = 5.0;
    auto make = [] {
        return cuda::PairLJCut{diagonal_rows(), kCutoff}.set_r4_attraction(
            {{0, 1, kS, kSigR4, kCutR4}});
    };
    expect_pair(
        lj_cut(dimer(0, 1, r, true), make().set_special_bonds(1.0, 1.0, 1.0)),
        lj_shifted(kEps01, kSig01, kCut01, r) + r4(kS, kSigR4, kCutR4, r),
        lj_force(kEps01, kSig01, kCut01, r) + r4_force(kS, kSigR4, kCutR4, r),
        r);
    expect_pair(
        lj_cut(dimer(0, 1, r, true), make().set_special_bonds(0.0, 1.0, 1.0)),
        0.0, 0.0, r);
}

// ── PairLJCutCoulLong ──────────────────────────────────────────────────────

TEST(CudaPairLJCutCoulLongTypedCutoff, ReportsTheLargestPairCutoff) {
    cuda::PairLJCutCoulLong pair{
        {{0, 0, kEps0, kSig0, kCut0}}, kAlpha, kCutoff};
    EXPECT_NEAR(pair.pair_cutoff(), kCutoff, kTol);
    EXPECT_NEAR(static_cast<double>(pair.long_range_contract().real_cutoff),
                kCutoff, kTol);
    cuda::PairLJCutCoulLong wide{diagonal_rows(), kAlpha, kCutoff};
    EXPECT_NEAR(wide.pair_cutoff(), kCut1, kTol);
    EXPECT_NEAR(static_cast<double>(wide.long_range_contract().real_cutoff),
                kCutoff, kTol);
}

TEST(CudaPairLJCutCoulLongTypedCutoff, MixedPairInsideItsMixedCutoff) {
    const double r = 7.0;
    const Result got =
        coul_long(dimer(0, 1, r), {diagonal_rows(), kAlpha, kCutoff});
    EXPECT_TRUE(got.typed);
    expect_pair(got, lj_shifted(kEps01, kSig01, kCut01, r),
                lj_force(kEps01, kSig01, kCut01, r), r);
}

TEST(CudaPairLJCutCoulLongTypedCutoff, MixedPairOutsideItsMixedCutoff) {
    expect_pair(coul_long(dimer(0, 1, 8.5), {diagonal_rows(), kAlpha, kCutoff}),
                0.0, 0.0, 8.5);
}

TEST(CudaPairLJCutCoulLongTypedCutoff, DiagonalCutoffBeyondTheCoulombOne) {
    const double r = 9.5;
    expect_pair(coul_long(dimer(1, 1, r), {diagonal_rows(), kAlpha, kCutoff}),
                lj_shifted(kEps1, kSig1, kCut1, r),
                lj_force(kEps1, kSig1, kCut1, r), r);
}

TEST(CudaPairLJCutCoulLongTypedCutoff, ExplicitCrossRowOverridesTheMix) {
    auto rows = diagonal_rows();
    rows.push_back({0, 1, 0.3, 3.2, 7.5});
    const double r = 7.2;
    expect_pair(coul_long(dimer(0, 1, r), {rows, kAlpha, kCutoff}),
                lj_shifted(0.3, 3.2, 7.5, r), lj_force(0.3, 3.2, 7.5, r), r);
}

TEST(CudaPairLJCutCoulLongTypedCutoff, EnergyShiftOffIsPlainLJ) {
    const double r = 7.0;
    expect_pair(
        coul_long(dimer(0, 1, r),
                  cuda::PairLJCutCoulLong{diagonal_rows(), kAlpha, kCutoff}
                      .set_energy_shift(false)),
        lj(kEps01, kSig01, r), lj_force(kEps01, kSig01, kCut01, r), r);
}

TEST(CudaPairLJCutCoulLongTypedCutoff,
     LorentzBerthelotTableShiftsAtTheGlobalCutoff) {
    // Pure LB diagonals at the global cutoff select the per-atom fast tile.
    const double r = 5.0;
    const Result got = coul_long(
        dimer(0, 1, r),
        {{{0, 0, kEps0, kSig0}, {1, 1, kEps1, kSig1}}, kAlpha, kCutoff});
    EXPECT_FALSE(got.typed);
    expect_pair(got, lj_shifted(kEps01, kSig01, kCutoff, r),
                lj_force(kEps01, kSig01, kCutoff, r), r);
}

TEST(CudaPairLJCutCoulLongTypedCutoff, R4AttractionAddsToLJ) {
    const double r = 5.0;
    expect_pair(
        coul_long(dimer(0, 1, r),
                  cuda::PairLJCutCoulLong{diagonal_rows(), kAlpha, kCutoff}
                      .set_r4_attraction({{0, 1, kS, kSigR4, kCutR4}})),
        lj_shifted(kEps01, kSig01, kCut01, r) + r4(kS, kSigR4, kCutR4, r),
        lj_force(kEps01, kSig01, kCut01, r) + r4_force(kS, kSigR4, kCutR4, r),
        r);
}

TEST(CudaPairLJCutCoulLongTypedCutoff, R4AloneIsContinuousAtItsCutoff) {
    auto make = [] {
        return cuda::PairLJCutCoulLong{
            {{0, 0, 0.0, 1.0}, {1, 1, 0.0, 1.0}}, kAlpha, kCutoff}
            .set_r4_attraction({{0, 1, kS, kSigR4, kCutR4}});
    };
    const double r = 4.0;
    expect_pair(coul_long(dimer(0, 1, r), make()), r4(kS, kSigR4, kCutR4, r),
                r4_force(kS, kSigR4, kCutR4, r), r);
    const Result in = coul_long(dimer(0, 1, kCutR4 - 1e-4), make());
    EXPECT_LT(std::fabs(in.energy), 1e-9);
    EXPECT_LT(std::fabs(in.fx1), 1e-6);
    expect_pair(coul_long(dimer(0, 1, kCutR4 + 1e-3), make()), 0.0, 0.0,
                kCutR4 + 1e-3);
}

TEST(CudaPairLJCutCoulLongTypedCutoff, LJSpecialWeightScalesLJAndR4) {
    const double r = 5.0;
    auto make = [] {
        return cuda::PairLJCutCoulLong{diagonal_rows(), kAlpha, kCutoff}
            .set_r4_attraction({{0, 1, kS, kSigR4, kCutR4}});
    };
    expect_pair(
        coul_long(dimer(0, 1, r, true),
                  make().set_special_bonds(1, 1, 1, 1, 1, 1)),
        lj_shifted(kEps01, kSig01, kCut01, r) + r4(kS, kSigR4, kCutR4, r),
        lj_force(kEps01, kSig01, kCut01, r) + r4_force(kS, kSigR4, kCutR4, r),
        r);
    // 1-2 LJ weight 0: an exception pass pair with LJ and r^-4 scaled away.
    expect_pair(coul_long(dimer(0, 1, r, true),
                          make().set_special_bonds(0, 1, 1, 1, 1, 1)),
                0.0, 0.0, r);
    // Partial weight: exception pass applies factor_lj * (LJ - offset + r4).
    expect_pair(coul_long(dimer(0, 1, r, true),
                          make().set_special_bonds(0.5, 1, 1, 1, 1, 1)),
                0.5
                    * (lj_shifted(kEps01, kSig01, kCut01, r)
                       + r4(kS, kSigR4, kCutR4, r)),
                0.5
                    * (lj_force(kEps01, kSig01, kCut01, r)
                       + r4_force(kS, kSigR4, kCutR4, r)),
                r);
}

TEST(CudaPairLJCutCoulLongTypedCutoff, CoulombSpecialRemovesThePlainShare) {
    const double r = 3.0, q = 0.5;
    auto make = [](double c12) {
        return cuda::PairLJCutCoulLong{{{0, 0, 0.0, 1.0}}, kAlpha, kCutoff}
            .set_special_bonds(1, 1, 1, c12, 1, 1);
    };
    const Result on = coul_long(dimer(0, 0, r, true, q, -q), make(1.0));
    const Result off = coul_long(dimer(0, 0, r, true, q, -q), make(0.0));
    const double qq = units::md_qqr2e * q * -q;
    EXPECT_NEAR(on.energy - off.energy, qq / r, 1e-10);
    EXPECT_NEAR(on.fx1 - off.fx1, qq / (r * r), 1e-10);
}

// ── Exceptions beyond the cutoffs, fractional weights, tail, set_type_coeff ─

namespace {

struct Full {
    double energy;
    std::vector<double> fx;
    std::array<double, 6> w;
};

/// Atoms of `types` on a line at x = `x` (y = z = 10), optionally bonded
/// in sequence, in a cubic box of edge `edge`.
cpu::Frame line(const std::vector<int>& types, const std::vector<double>& x,
                bool bonded, double edge = 40.0) {
    const int n = static_cast<int>(types.size());
    cpu::Frame f(n);
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_x)[i] = x[i];
        f.span(kAtom_y)[i] = 10.0;
        f.span(kAtom_z)[i] = 10.0;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_tag)[i] = i;
        f.span(kAtom_type)[i] = types[i];
        f.span(kAtom_charge)[i] = 0.0;
    }
    f.box = cpu::Box::cubic(edge);
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    if (bonded)
        for (int i = 0; i + 1 < n; ++i)
            f.add_bond(i, i + 1, 0);
    f.build_special_list();
    return f;
}

template <class Pair>
Full evaluate_full(cpu::Frame f, Pair pair) {
    constexpr bool kCoulLong = std::is_same_v<Pair, cuda::PairLJCutCoulLong>;
    {
        cuda::Evaluate eval;
        eval.set_neighbor(cuda::NeighborList{kNeighbor, 0.5});
        eval.set_evaluator_ref(pair);
        if constexpr (kCoulLong)
            eval.on(evaluate::Phase::evaluate)
                .add(cuda::KSpaceEwald{kAlpha, 1e-6}, 100);
        eval.build();
        eval.run(f);
    }
    Full out{f.meta.get(kEnergy), {}, {}};
    for (int i = 0; i < f.nlocal(); ++i)
        out.fx.push_back(f.span(kAtom_fx)[i]);
    for (int d = 0; d < 6; ++d)
        out.w[d] = f.meta.get(kStress)[d];
    return out;
}

/// Types {0, 0, 1} 12 and 16 Å apart: beyond every pair cutoff, so only the
/// tail acts.
cpu::Frame far_triplet(double edge = 40.0) {
    return line({0, 0, 1}, {5.0, 17.0, 29.0}, false, edge);
}

/// Tail of `diagonal_rows()` over the counts {2, 1}, times V.
double triplet_tail_energy() {
    return lj_tail_energy(kEps0, kSig0, kCut0, 2, 2)
           + 2.0 * lj_tail_energy(kEps01, kSig01, kCut01, 2, 1)
           + lj_tail_energy(kEps1, kSig1, kCut1, 1, 1);
}
double triplet_tail_virial() {
    return lj_tail_virial(kEps0, kSig0, kCut0, 2, 2)
           + 2.0 * lj_tail_virial(kEps01, kSig01, kCut01, 2, 1)
           + lj_tail_virial(kEps1, kSig1, kCut1, 1, 1);
}

void expect_tail(const Full& got, double energy, double virial) {
    const auto near = [](double a, double b) {
        EXPECT_NEAR(a, b, 1e-12 * std::fabs(b) + 1e-16);
    };
    near(got.energy, energy);
    for (int d = 0; d < 3; ++d)
        near(got.w[d], virial);
    for (int d = 3; d < 6; ++d)
        EXPECT_EQ(got.w[d], 0.0);
}

} // namespace

TEST(CudaPairLJCutCoulLongTypedCutoff, BondedPairBeyondTheCutoffsAddsNothing) {
    // LAMMPS lj/cut/coul/long: an excluded (Amber 1-2) pair beyond the LJ
    // and Coulomb cutoffs gets no LJ and no -(1 - factor_coul) qq/r term, so
    // the bonded (exception-pass) and unbonded runs agree (k-space only).
    const double r = 9.5, q = 0.5;
    auto run = [&](bool bonded) {
        return coul_long(dimer(0, 0, r, bonded, q, -q),
                         {diagonal_rows(), kAlpha, kCutoff});
    };
    const Result bonded = run(true), free = run(false);
    ASSERT_NE(free.energy, 0.0);
    EXPECT_NEAR(bonded.energy, free.energy, kTol);
    EXPECT_NEAR(bonded.fx1, free.fx1, kTol);
    EXPECT_NEAR(bonded.fx0, free.fx0, kTol);
    EXPECT_NEAR(bonded.wxx, free.wxx, kTol);
}

TEST(CudaPairLJCutTypedCutoff, FractionalWeightScalesLJAndR4) {
    const double r = 5.0;
    expect_pair(lj_cut(dimer(0, 1, r, true),
                       cuda::PairLJCut{diagonal_rows(), kCutoff}
                           .set_r4_attraction({{0, 1, kS, kSigR4, kCutR4}})
                           .set_special_bonds(0.5, 1.0, 1.0)),
                0.5
                    * (lj_shifted(kEps01, kSig01, kCut01, r)
                       + r4(kS, kSigR4, kCutR4, r)),
                0.5
                    * (lj_force(kEps01, kSig01, kCut01, r)
                       + r4_force(kS, kSigR4, kCutR4, r)),
                r);
}

TEST(CudaPairLJCutTypedCutoff, FractionalWeightBeyondThePairCutoffIsZero) {
    expect_pair(lj_cut(dimer(0, 1, 8.5, true),
                       cuda::PairLJCut{diagonal_rows(), kCutoff}
                           .set_r4_attraction({{0, 1, kS, kSigR4, kCutR4}})
                           .set_special_bonds(0.5, 1.0, 1.0)),
                0.0, 0.0, 8.5);
}

TEST(CudaPairLJCutTypedCutoff, FractionalOneThreeWeightOnAChain) {
    // 0-1-2 bonded: 1-2 pairs dropped, the 1-3 pair (r = 9) at weight 0.5.
    const double r = 9.0;
    const Full got = evaluate_full(
        line({1, 1, 1}, {10.0, 14.5, 19.0}, true),
        cuda::PairLJCut{diagonal_rows(), kCutoff}.set_special_bonds(0.0, 0.5,
                                                                    1.0));
    const double f = 0.5 * lj_force(kEps1, kSig1, kCut1, r);
    EXPECT_NEAR(got.energy, 0.5 * lj_shifted(kEps1, kSig1, kCut1, r), kTol);
    EXPECT_NEAR(got.fx[0], -f, kTol);
    EXPECT_NEAR(got.fx[1], 0.0, kTol);
    EXPECT_NEAR(got.fx[2], f, kTol);
    EXPECT_NEAR(got.w[0], r * f, kTol);
}

TEST(CudaPairLJCutTypedCutoff, SingleTypeFractionalWeight) {
    const double r = 4.0;
    expect_pair(lj_cut(dimer(0, 0, r, true),
                       cuda::PairLJCut{kEps0, kSig0, kCut0}.set_special_bonds(
                           0.5, 1.0, 1.0)),
                0.5 * lj_shifted(kEps0, kSig0, kCut0, r),
                0.5 * lj_force(kEps0, kSig0, kCut0, r), r);
}

TEST(CudaPairLJCutTail, TypedTailIsTheIntegralBeyondEachPairCutoff) {
    const double v = 40.0 * 40.0 * 40.0;
    expect_tail(
        evaluate_full(far_triplet(), cuda::PairLJCut{diagonal_rows(), kCutoff}
                                         .set_tail_correction(true)),
        triplet_tail_energy() / v, triplet_tail_virial() / v);
    // Off by default: nothing acts.
    expect_tail(
        evaluate_full(far_triplet(), cuda::PairLJCut{diagonal_rows(), kCutoff}),
        0.0, 0.0);
}

TEST(CudaPairLJCutTail, SingleTypeTailCountsEveryAtom) {
    const double edge = 45.0, v = edge * edge * edge;
    expect_tail(
        evaluate_full(
            far_triplet(edge),
            cuda::PairLJCut{kEps0, kSig0, kCut0}.set_tail_correction(true)),
        lj_tail_energy(kEps0, kSig0, kCut0, 3, 3) / v,
        lj_tail_virial(kEps0, kSig0, kCut0, 3, 3) / v);
}

TEST(CudaPairLJCutCoulLongTail, TypedTailIsTheIntegralBeyondEachPairCutoff) {
    const double v = 40.0 * 40.0 * 40.0;
    expect_tail(
        evaluate_full(far_triplet(),
                      cuda::PairLJCutCoulLong{diagonal_rows(), kAlpha, kCutoff}
                          .set_tail_correction(true)),
        triplet_tail_energy() / v, triplet_tail_virial() / v);
}

TEST(CudaPairLJCutTypedCutoff, SetTypeCoeffOverridesAndExtends) {
    const double r = 7.2;
    cuda::PairLJCut pair{diagonal_rows(), kCutoff};
    pair.set_type_coeff({0, 1, 0.3, 3.2, 7.5});
    expect_pair(lj_cut(dimer(0, 1, r), std::move(pair)),
                lj_shifted(0.3, 3.2, 7.5, r), lj_force(0.3, 3.2, 7.5, r), r);
    cuda::PairLJCut wide{diagonal_rows(), kCutoff};
    wide.set_type_coeff({2, 2, 0.01, 3.0, 12.0});
    EXPECT_NEAR(wide.pair_cutoff(), 12.0, kTol);
    EXPECT_THROW(cuda::PairLJCut(1.0, 1.0, 3.0).set_type_coeff({0, 0, 1, 1}),
                 std::logic_error);
}

TEST(CudaPairLJCutCoulLongTypedCutoff, SetTypeCoeffOverridesAndExtends) {
    const double r = 7.2;
    cuda::PairLJCutCoulLong pair{diagonal_rows(), kAlpha, kCutoff};
    pair.set_type_coeff({0, 1, 0.3, 3.2, 7.5});
    expect_pair(coul_long(dimer(0, 1, r), std::move(pair)),
                lj_shifted(0.3, 3.2, 7.5, r), lj_force(0.3, 3.2, 7.5, r), r);
    cuda::PairLJCutCoulLong wide{diagonal_rows(), kAlpha, kCutoff};
    wide.set_type_coeff({2, 2, 0.01, 3.0, 12.0});
    EXPECT_NEAR(wide.pair_cutoff(), 12.0, kTol);
    EXPECT_NEAR(wide.long_range_contract().real_cutoff, kCutoff, kTol);
    EXPECT_THROW(wide.set_type_coeff({0, 0, -1.0, 3.0}), std::invalid_argument);
}

TEST(CudaPairLJCutCoulLongTypedCutoff,
     ZeroEpsilonRowCutoffDoesNotExtendCoulombOrLJ) {
    // An epsilon = 0 row with a cutoff beyond the Coulomb one raises the
    // reported (neighbor) cutoff but must not move the Coulomb cutoff or an
    // interacting LJ pair's cutoff: the table is still Lorentz-Berthelot at
    // the global cutoff (the CUDA per-atom tile path).
    const double r = 9.5, q = 0.5;
    const std::vector<LJTypeCoeff> wide = {{0, 0, 0.0, 1.0, 10.0},
                                           {1, 1, kEps0, kSig0}};
    const std::vector<LJTypeCoeff> plain = {{0, 0, 0.0, 1.0},
                                            {1, 1, kEps0, kSig0}};
    cuda::PairLJCutCoulLong probe{wide, kAlpha, kCutoff};
    EXPECT_NEAR(static_cast<double>(probe.pair_cutoff()), 10.0, kTol);
    // Charged type-0 pair beyond the Coulomb cutoff: k-space only, as with
    // the plain row.
    const Result a =
        coul_long(dimer(0, 0, r, false, q, -q), {wide, kAlpha, kCutoff});
    const Result b =
        coul_long(dimer(0, 0, r, false, q, -q), {plain, kAlpha, kCutoff});
    ASSERT_NE(b.energy, 0.0);
    EXPECT_FALSE(a.typed); // the per-atom Lorentz-Berthelot tile ran
    EXPECT_NEAR(a.energy, b.energy, kTol);
    EXPECT_NEAR(a.fx1, b.fx1, kTol);
    EXPECT_NEAR(a.wxx, b.wxx, kTol);
    // Interacting LJ pair (1, 1) beyond its cutoff (the global one): zero.
    expect_pair(coul_long(dimer(1, 1, r), {wide, kAlpha, kCutoff}), 0.0, 0.0,
                r);
}

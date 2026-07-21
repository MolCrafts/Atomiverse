/// @file tests/cpu/scf/stages/test_scf_stages.cpp
/// @brief RED-first unit tests for the five SCF stage components
///        (scf-driver-refactor-02-stages).
///
/// Pins the behavioural contract of the five `scf::Phase`-bound stage
/// components that sub-spec 02 wraps around the shipping internal SCF helpers,
/// WITHOUT wiring them into any driver (that is 03). Everything here is an
/// ISOLATED bit-identical parity check: each stage reproduces the helper it
/// wraps, byte-for-byte, on a fixed synthetic single-step input — never a full
/// SCF self-consistency loop (that is e2e, forbidden in tests/).
///
/// ## Expected RED
/// None of the five stage headers exist yet, so this translation unit FAILS TO
/// COMPILE on the missing includes:
///   - atomiverse/cpu/scf/stages/fock_stage.h        (FockStage)
///   - atomiverse/cpu/scf/stages/mix_stage.h         (MixStage)
///   - atomiverse/cpu/scf/stages/diagonalize_stage.h (DiagonalizeStage)
///   - atomiverse/cpu/scf/stages/density_stage.h     (DensityStage)
///   - atomiverse/cpu/scf/stages/scf_converged.h     (ScfConverged)
/// The assertions must NOT be weakened to make it compile.
///
/// ## Contract encoded here (the RED-defined constructor shapes)
/// Every stage inherits `cpu::Component`; state flows through NAMED ctor
/// references (const& for reads, & for writes) — no config bag, no Workspace
/// bag, no `ctx.scf` slice. `ScfConverged` derives `cpu::Criterion`. The
/// constructor signatures this file assumes (discrete params, then read refs,
/// then write refs) are:
///
///   FockStage(bool restricted, bool is_hf, XCType, GridLevel, int n_radial,
///             int n_angular, double hybrid_alpha_override,
///             const MolecularGrid* fixed_grid,
///             const IntegralPack& integrals,
///             const RealMatrix& P_alpha, const RealMatrix& P_beta,
///             RealMatrix& F_alpha, RealMatrix& F_beta, RealMatrix& J,
///             double& e_xc_grid, double& e_exact_exchange,
///             double& int_n_alpha, double& int_n_beta)
///
///   MixStage(bool restricted, bool use_diis, int diis_start_iter,
///            int diis_max_vectors, double level_shift,
///            MixerKind mixer_kind, double mixer_switch_tol,
///            bool soscf, double soscf_start,
///            const RealMatrix& S, const RealMatrix& X,
///            const RealMatrix& P_alpha, const RealMatrix& P_beta,
///            const int& iter, const double& energy,
///            RealMatrix& F_alpha, RealMatrix& F_beta,
///            RealMatrix& err_alpha, RealMatrix& err_beta,
///            bool& soscf_active)   // + reset()
///
///   DiagonalizeStage(bool restricted,
///                    const RealMatrix& F_alpha, const RealMatrix& F_beta,
///                    const RealMatrix& S, const RealMatrix& X,
///                    const int& n_dropped,
///                    RealMatrix& C_alpha, RealMatrix& C_beta,
///                    RealVector& eig_alpha, RealVector& eig_beta,
///                    internal::SOSCFStepper* soscf,
///                    const bool& soscf_active)
///
///   DensityStage(bool restricted, bool is_hf, double damping,
///                const RealMatrix& C_alpha, const RealMatrix& C_beta,
///                const RealMatrix& F_alpha, const RealMatrix& F_beta,
///                const RealMatrix& Hcore, const RealMatrix& J,
///                const int& n_alpha, const int& n_beta, const int& n_basis,
///                const bool& diis_active, const double& e_xc_grid,
///                const double& e_exact_exchange, const double& v_nuc,
///                RealMatrix& P_alpha, RealMatrix& P_beta,   // old in / new out
///                double& e_total, double& e_one, double& e_coulomb)
///
///   ScfConverged(double tol_energy, double tol_grad, bool restricted,
///                const double& e_total,
///                const RealMatrix& err_alpha, const RealMatrix& err_beta)
///                // + double delta_energy() const, double diis_error() const,
///                //   void reset()
///
/// Harness: a stage's `launch()` is driven directly with a minimal
/// `cpu::LaunchContext{runtime, frame, slot_of(phase), step}` (mirrors the
/// component tests under tests/cpu/); `ctx.at(scf::Phase::…)` is the phase
/// self-guard. `build()` runs once before `launch()`.

#ifdef ATV_HAS_HF

#include <cmath>
#include <cstring>
#include <iomanip>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/facade/loop.h"
#include "atomiverse/facade/stage.h"

#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/xc_functional.h"

#include "atomiverse/cpu/scf/internal/density_builder.h"
#include "atomiverse/cpu/scf/internal/diis_mixer.h"
#include "atomiverse/cpu/scf/internal/dual_convergence_check.h"
#include "atomiverse/cpu/scf/internal/eigen_solver.h"
#include "atomiverse/cpu/scf/internal/hamiltonian_builder.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/level_shift.h"
#include "atomiverse/cpu/scf/internal/scf_types.h"

// ── The five stage headers under construction (RED: not present yet) ──────────
#include "atomiverse/cpu/scf/stages/density_stage.h"
#include "atomiverse/cpu/scf/stages/diagonalize_stage.h"
#include "atomiverse/cpu/scf/stages/fock_stage.h"
#include "atomiverse/cpu/scf/stages/mix_stage.h"
#include "atomiverse/cpu/scf/stages/scf_converged.h"

#include "cpu/driver_test_helpers.h"

using namespace ATV;
using cpu::scf::RealMatrix;
using cpu::scf::RealVector;
using cpu::scf::internal::apply_level_shift;
using cpu::scf::internal::DensityBuilder;
using cpu::scf::internal::diis_error;
using cpu::scf::internal::DIISMixer;
using cpu::scf::internal::DualConvergenceCheck;
using cpu::scf::internal::EigenSolver;
using cpu::scf::internal::FockResult;
using cpu::scf::internal::HamiltonianBuilder;

namespace {

// ── bit-identical comparison helpers (逐位一致 == same code path, same bits) ──

::testing::AssertionResult ScalarBitEqual(double got, double want) {
    if (std::memcmp(&got, &want, sizeof(double)) == 0)
        return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << std::setprecision(17) << "got " << got << " != want " << want;
}

::testing::AssertionResult MatBitEqual(const RealMatrix& got,
                                       const RealMatrix& want) {
    if (got.shape()[0] != want.shape()[0] || got.shape()[1] != want.shape()[1])
        return ::testing::AssertionFailure()
               << "shape mismatch: (" << got.shape()[0] << "," << got.shape()[1]
               << ") vs (" << want.shape()[0] << "," << want.shape()[1] << ")";
    for (std::size_t i = 0; i < got.shape()[0]; ++i)
        for (std::size_t j = 0; j < got.shape()[1]; ++j) {
            double a = got(i, j), b = want(i, j);
            if (std::memcmp(&a, &b, sizeof(double)) != 0)
                return ::testing::AssertionFailure()
                       << std::setprecision(17) << "element (" << i << "," << j
                       << ") got " << a << " != want " << b;
        }
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult VecBitEqual(const RealVector& got,
                                       const RealVector& want) {
    if (got.shape()[0] != want.shape()[0])
        return ::testing::AssertionFailure()
               << "size mismatch: " << got.shape()[0] << " vs "
               << want.shape()[0];
    for (std::size_t i = 0; i < got.shape()[0]; ++i) {
        double a = got(i), b = want(i);
        if (std::memcmp(&a, &b, sizeof(double)) != 0)
            return ::testing::AssertionFailure()
                   << std::setprecision(17) << "element (" << i << ") got " << a
                   << " != want " << b;
    }
    return ::testing::AssertionSuccess();
}

/// @brief Element-wise sum of A ∘ B — the scf_solver.cpp energy contraction.
double frob_dot(const RealMatrix& a, const RealMatrix& b) {
    return xt::sum(a * b)();
}

/// @brief H2 at 0.74 Å, STO-3G-scale (2 basis fns, 2 electrons) — the smallest
///        Fock-build input; never solved to self-consistency here.
cpu::Frame make_h2_frame() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.74;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_type)[0] = 1; // Z = 1
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

/// @brief A minimal LaunchContext that reports "this is `phase`" to `ctx.at`.
template <typename PhaseEnum>
cpu::LaunchContext at_stage(cpu::Context& rt, cpu::Frame& fr, PhaseEnum phase,
                            int step = 0) {
    return cpu::LaunchContext{rt, fr, slot_of(phase), step};
}

// A symmetric positive-ish 2×2 overlap-like matrix and a canonical transform.
RealMatrix sample_S() {
    RealMatrix S = xt::zeros<double>({2, 2});
    S(0, 0) = 1.0;
    S(0, 1) = 0.2;
    S(1, 0) = 0.2;
    S(1, 1) = 1.0;
    return S;
}

RealMatrix sample_X() {
    // Any dimensionless transform; identical on both sides of a parity check.
    RealMatrix X = xt::zeros<double>({2, 2});
    X(0, 0) = 1.0;
    X(0, 1) = -0.1;
    X(1, 0) = 0.0;
    X(1, 1) = 1.05;
    return X;
}

RealMatrix sym2(double a, double b, double c) {
    RealMatrix M = xt::zeros<double>({2, 2});
    M(0, 0) = a;
    M(0, 1) = b;
    M(1, 0) = b;
    M(1, 1) = c;
    return M;
}

} // namespace

// ── Structural contract: base classes + closed kernel + bindings ─────────────

static_assert(std::is_base_of_v<cpu::Component, cpu::scf::FockStage>,
              "FockStage is a Component");
// SOSCF wiring stand-ins for the pre-existing stage tests: the latch stays
// false (soscf disabled), keeping every path bit-identical to before.
namespace { bool soscf_latch = false; const bool kNoSoscf = false; }

static_assert(std::is_base_of_v<cpu::Component, cpu::scf::MixStage>,
              "MixStage is a Component");
static_assert(std::is_base_of_v<cpu::Component, cpu::scf::DiagonalizeStage>,
              "DiagonalizeStage is a Component");
static_assert(std::is_base_of_v<cpu::Component, cpu::scf::DensityStage>,
              "DensityStage is a Component");
static_assert(std::is_base_of_v<cpu::Criterion, cpu::scf::ScfConverged>,
              "ScfConverged is a Criterion (measure + judge in one component)");
static_assert(std::is_base_of_v<cpu::Component, cpu::scf::ScfConverged>,
              "Criterion is itself a Component kind");
static_assert(!std::is_base_of_v<cpu::Component, EigenSolver>,
              "EigenSolver stays a closed numeric kernel, not a Component");

// ── Absorbed helper known-answer pins (from the retired
//    test_scf_solver_internals.cpp; scf-driver-refactor-03b) ─────────────────
//
// The retired solver-internals test carried six meaningful helper assertions.
// The three DualConvergence assertions (iter-0-is-never-converged with dE=∞,
// both-criteria-required, reset-clears-history) are ALREADY covered here on the
// ScfConverged Criterion — see ScfConverged.BindsCheckPhaseAndFirstIteration-
// Continues, ScfConverged.ConvergesOnlyWhenBothCriteriaMet, and
// ScfConverged.ResetClearsPrevEnergy below. The remaining three are absorbed as
// the explicit known-answer pins below: the stage-parity tests exercise
// DensityBuilder / EigenSolver only as bit-parity ORACLES (a shared kernel bug
// would move both sides together), so the standalone known-answer numbers were
// not otherwise pinned.

TEST(DensityBuilder, RestrictedFactor2KnownAnswer) {
    // One occupied orbital c0 = (1, 0)^T with the restricted factor 2 ⇒
    // P = 2 · c0 c0^T has a single nonzero entry P(0,0) = 2.
    RealMatrix C = xt::zeros<double>({2, 2});
    C(0, 0) = 1.0;
    DensityBuilder db;
    db.prepare(2);
    RealMatrix P = db.step(C, /*n_occ=*/1, /*n_basis=*/2, /*factor=*/2.0);
    EXPECT_NEAR(P(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(P(0, 1), 0.0, 1e-12);
    EXPECT_NEAR(P(1, 0), 0.0, 1e-12);
    EXPECT_NEAR(P(1, 1), 0.0, 1e-12);
}

TEST(DensityBuilder, EmptyOccupationYieldsZero) {
    // n_occ = 0 ⇒ no occupied columns ⇒ P is the zero matrix (e.g. the β
    // channel of a one-electron cation). Edge case: no factor is applied.
    RealMatrix C = xt::eye<double>(3);
    DensityBuilder db;
    RealMatrix P = db.step(C, /*n_occ=*/0, /*n_basis=*/3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(P(i, j), 0.0, 1e-12);
}

TEST(EigenSolver, GeneralizedEigIdentityOverlap) {
    // F = diag(2, 1), S = I ⇒ the generalized eigenproblem F C = S C ε reduces
    // to the standard one; eigenvalues are returned ascending {1, 2}.
    RealMatrix F = xt::eye<double>(2);
    F(0, 0) = 2.0;
    F(1, 1) = 1.0;
    RealMatrix S = xt::eye<double>(2);
    EigenSolver es;
    auto [eig, C] = es.step(F, S);
    EXPECT_NEAR(eig(0), 1.0, 1e-12);
    EXPECT_NEAR(eig(1), 2.0, 1e-12);
    EXPECT_EQ(C.shape()[0], 2u);
    EXPECT_EQ(C.shape()[1], 2u);
}

// ── FockStage: wraps HamiltonianBuilder bit-identically, config-free ─────────

TEST(FockStage, ReproducesHamiltonianBuilderRestrictedHF) {
    cpu::Frame h2 = make_h2_frame();
    cpu::scf::IntegralPack ints{h2, cpu::scf::BasisSet::STO3G};

    // A nonzero density (core guess) so J/K actually contribute.
    RealMatrix S = ints.S;
    RealMatrix Hcore = ints.Hcore;
    EigenSolver es;
    auto [eig0, C0] = es.step(Hcore, S);
    DensityBuilder db;
    RealMatrix P = db.step(C0, ints.n_occ(), ints.n_basis, 2.0);

    // Oracle: the surviving discrete prepare/step entries, called with the same
    // parameters the FockStage ctor forwards. The discrete and (retired) config
    // entries share one private core, so this is bit-identical by construction.
    HamiltonianBuilder hb;
    hb.prepare(ints, /*is_hf=*/true, cpu::scf::GridLevel::SG0,
               /*n_radial=*/50, /*n_angular=*/110, /*fixed_grid=*/nullptr);
    FockResult ref = hb.step(ints, /*restricted=*/true, /*is_hf=*/true,
                             cpu::scf::XCType::LSDA,
                             /*hybrid_alpha_override=*/-1.0, P, P);

    // Under test: the config-free discrete FockStage on the same P.
    RealMatrix F_a, F_b, J;
    double e_xc = -123.0, e_exact = -123.0, int_na = -123.0, int_nb = -123.0;
    cpu::scf::FockStage fock(
        /*restricted=*/true, /*is_hf=*/true, cpu::scf::XCType::LSDA,
        cpu::scf::GridLevel::SG0, /*n_radial=*/50, /*n_angular=*/110,
        /*hybrid_alpha_override=*/-1.0, /*fixed_grid=*/nullptr, ints, P, P, F_a,
        F_b, J, e_xc, e_exact, int_na, int_nb);

    cpu::Context rt;
    cpu::BuildContext bc{rt, h2};
    fock.build(bc);
    auto lc = at_stage(rt, h2, scf::Phase::fock);
    fock.launch(lc);

    EXPECT_TRUE(MatBitEqual(F_a, ref.F_alpha));
    EXPECT_TRUE(MatBitEqual(F_b, ref.F_beta));
    EXPECT_TRUE(MatBitEqual(J, ref.J));
    EXPECT_TRUE(ScalarBitEqual(e_xc, ref.E_xc_grid));
    EXPECT_TRUE(ScalarBitEqual(e_exact, ref.E_exact_exchange));
    EXPECT_TRUE(ScalarBitEqual(int_na, ref.int_n_alpha));
    EXPECT_TRUE(ScalarBitEqual(int_nb, ref.int_n_beta));
}

TEST(FockStage, ReproducesHamiltonianBuilderRestrictedKS) {
    cpu::Frame h2 = make_h2_frame();
    cpu::scf::IntegralPack ints{h2, cpu::scf::BasisSet::STO3G};

    RealMatrix S = ints.S, Hcore = ints.Hcore;
    EigenSolver es;
    auto [eig0, C0] = es.step(Hcore, S);
    DensityBuilder db;
    RealMatrix P = db.step(C0, ints.n_occ(), ints.n_basis, 2.0);

    // Oracle: the surviving discrete prepare/step entries with the SG0 grid, the
    // same parameters the FockStage ctor forwards. The discrete and (retired) KS
    // config entries share one private core, so this is bit-identical.
    HamiltonianBuilder hb;
    hb.prepare(ints, /*is_hf=*/false, cpu::scf::GridLevel::SG0,
               /*n_radial=*/50, /*n_angular=*/110, /*fixed_grid=*/nullptr);
    FockResult ref = hb.step(ints, /*restricted=*/true, /*is_hf=*/false,
                             cpu::scf::XCType::LSDA,
                             /*hybrid_alpha_override=*/-1.0, P, P);

    RealMatrix F_a, F_b, J;
    double e_xc = 0, e_exact = 0, int_na = 0, int_nb = 0;
    cpu::scf::FockStage fock(
        /*restricted=*/true, /*is_hf=*/false, cpu::scf::XCType::LSDA,
        cpu::scf::GridLevel::SG0, /*n_radial=*/50, /*n_angular=*/110,
        /*hybrid_alpha_override=*/-1.0, /*fixed_grid=*/nullptr, ints, P, P, F_a,
        F_b, J, e_xc, e_exact, int_na, int_nb);

    cpu::Context rt;
    cpu::BuildContext bc{rt, h2};
    fock.build(bc);
    auto lc = at_stage(rt, h2, scf::Phase::fock);
    fock.launch(lc);

    EXPECT_TRUE(MatBitEqual(F_a, ref.F_alpha));
    EXPECT_TRUE(MatBitEqual(J, ref.J));
    EXPECT_TRUE(ScalarBitEqual(e_xc, ref.E_xc_grid));
    EXPECT_NE(ref.E_xc_grid, 0.0) << "KS branch must exercise a nonzero E_xc";
    EXPECT_TRUE(ScalarBitEqual(e_exact, ref.E_exact_exchange));
}

TEST(FockStage, BindsFockPhaseAndGuardsWithCtxAt) {
    cpu::Frame h2 = make_h2_frame();
    cpu::scf::IntegralPack ints{h2, cpu::scf::BasisSet::STO3G};
    RealMatrix S = ints.S, Hcore = ints.Hcore;
    EigenSolver es;
    auto [eig0, C0] = es.step(Hcore, S);
    DensityBuilder db;
    RealMatrix P = db.step(C0, ints.n_occ(), ints.n_basis, 2.0);

    RealMatrix F_a = xt::zeros<double>({2, 2}); // sentinel
    RealMatrix F_b, J;
    double e_xc = 0, e_exact = 0, int_na = 0, int_nb = 0;
    cpu::scf::FockStage fock(
        true, true, cpu::scf::XCType::LSDA, cpu::scf::GridLevel::SG0, 50, 110,
        -1.0, nullptr, ints, P, P, F_a, F_b, J, e_xc, e_exact, int_na, int_nb);

    // default_bindings() declares exactly the fock phase, via make_binding.
    auto bindings = fock.default_bindings();
    ASSERT_EQ(bindings.size(), 1u);
    EXPECT_EQ(bindings[0].kind, PhaseKind::SCF);
    EXPECT_EQ(bindings[0].phase, slot_of(scf::Phase::fock));

    cpu::Context rt;
    cpu::BuildContext bc{rt, h2};
    fock.build(bc);

    // Wrong phase → the self-guard skips: the sentinel F stays untouched.
    auto wrong = at_stage(rt, h2, scf::Phase::diagonalize);
    fock.launch(wrong);
    EXPECT_TRUE(MatBitEqual(F_a, RealMatrix(xt::zeros<double>({2, 2}))))
        << "ctx.at(fock) must gate the work off any non-fock stage";

    // Right phase → F is filled.
    auto right = at_stage(rt, h2, scf::Phase::fock);
    fock.launch(right);
    EXPECT_GT(std::abs(F_a(0, 0)), 0.0);
}

// ── MixStage: DIIS error + extrapolation + level shift, history internal ─────

TEST(MixStage, ReproducesDiisErrorAndExtrapolationRestricted) {
    RealMatrix S = sample_S(), X = sample_X();
    RealMatrix P = sym2(1.0, 0.3, 0.8);

    // A sequence of distinct Fock matrices — one "built" F per iteration.
    RealMatrix F_seq[3] = {sym2(-1.0, 0.4, -0.6), sym2(-1.1, 0.35, -0.55),
                           sym2(-1.15, 0.33, -0.52)};

    int iter = 0;
    double e_tot = 0.0;
    RealMatrix F_a, F_b, err_a, err_b;
    cpu::scf::MixStage mix(/*restricted=*/true, /*use_diis=*/true,
                           /*diis_start_iter=*/0, /*diis_max_vectors=*/6,
                           /*level_shift=*/0.0, cpu::scf::MixerKind::CDIIS,
                           /*mixer_switch_tol=*/1e-2, /*soscf=*/false,
                           /*soscf_start=*/0.25, S, X, P, P, iter, e_tot,
                           F_a, F_b, err_a, err_b, soscf_latch);
    cpu::Context rt;
    cpu::Frame fr = atv_test::make_trivial_frame();
    cpu::BuildContext bc{rt, fr};
    mix.build(bc);

    DIISMixer oracle_diis(6);
    for (iter = 0; iter < 3; ++iter) {
        F_a = F_seq[iter]; // fresh Fock into the wired in/out reference
        F_b = F_seq[iter];
        auto lc = at_stage(rt, fr, scf::Phase::mix, iter);
        mix.launch(lc);

        RealMatrix err_ref = diis_error(F_seq[iter], P, S, X);
        RealMatrix F_ref = oracle_diis.step(F_seq[iter], err_ref);

        EXPECT_TRUE(MatBitEqual(err_a, err_ref)) << "iter " << iter;
        EXPECT_TRUE(MatBitEqual(F_a, F_ref)) << "iter " << iter;
        EXPECT_TRUE(MatBitEqual(F_b, F_ref))
            << "restricted β mirrors α, iter " << iter;
    }
}

TEST(MixStage, ReproducesDiisErrorAndExtrapolationUnrestricted) {
    RealMatrix S = sample_S(), X = sample_X();
    RealMatrix Pa = sym2(0.9, 0.2, 0.7), Pb = sym2(0.6, 0.1, 0.5);
    RealMatrix Fa_seq[2] = {sym2(-1.0, 0.4, -0.6), sym2(-1.05, 0.38, -0.58)};
    RealMatrix Fb_seq[2] = {sym2(-0.8, 0.2, -0.5), sym2(-0.82, 0.19, -0.49)};

    int iter = 0;
    double e_tot = 0.0;
    RealMatrix F_a, F_b, err_a, err_b;
    cpu::scf::MixStage mix(/*restricted=*/false, /*use_diis=*/true,
                           /*diis_start_iter=*/0, /*diis_max_vectors=*/6,
                           /*level_shift=*/0.0, cpu::scf::MixerKind::CDIIS,
                           /*mixer_switch_tol=*/1e-2, /*soscf=*/false,
                           /*soscf_start=*/0.25, S, X, Pa, Pb, iter,
                           e_tot, F_a, F_b, err_a, err_b, soscf_latch);
    cpu::Context rt;
    cpu::Frame fr = atv_test::make_trivial_frame();
    cpu::BuildContext bc{rt, fr};
    mix.build(bc);

    DIISMixer oracle_diis(6);
    for (iter = 0; iter < 2; ++iter) {
        F_a = Fa_seq[iter];
        F_b = Fb_seq[iter];
        auto lc = at_stage(rt, fr, scf::Phase::mix, iter);
        mix.launch(lc);

        RealMatrix ea_ref = diis_error(Fa_seq[iter], Pa, S, X);
        RealMatrix eb_ref = diis_error(Fb_seq[iter], Pb, S, X);
        auto [Fa_ref, Fb_ref] =
            oracle_diis.step(Fa_seq[iter], ea_ref, Fb_seq[iter], eb_ref);

        EXPECT_TRUE(MatBitEqual(err_a, ea_ref)) << "iter " << iter;
        EXPECT_TRUE(MatBitEqual(err_b, eb_ref)) << "iter " << iter;
        EXPECT_TRUE(MatBitEqual(F_a, Fa_ref)) << "iter " << iter;
        EXPECT_TRUE(MatBitEqual(F_b, Fb_ref)) << "iter " << iter;
    }
}

TEST(MixStage, AppliesLevelShiftRestrictedWhenAboveThreshold) {
    RealMatrix S = sample_S(), X = sample_X();
    RealMatrix P = sym2(1.0, 0.3, 0.8);
    RealMatrix F_in = sym2(-1.0, 0.4, -0.6);

    int iter = 0;
    double e_tot = 0.0;
    RealMatrix F_a = F_in, F_b = F_in, err_a, err_b;
    const double lambda = 0.5;
    cpu::scf::MixStage mix(/*restricted=*/true, /*use_diis=*/false,
                           /*diis_start_iter=*/0, /*diis_max_vectors=*/6,
                           /*level_shift=*/lambda, cpu::scf::MixerKind::CDIIS,
                           /*mixer_switch_tol=*/1e-2, /*soscf=*/false,
                           /*soscf_start=*/0.25, S, X, P, P, iter, e_tot,
                           F_a, F_b, err_a, err_b, soscf_latch);
    cpu::Context rt;
    cpu::Frame fr = atv_test::make_trivial_frame();
    cpu::BuildContext bc{rt, fr};
    mix.build(bc);
    auto lc = at_stage(rt, fr, scf::Phase::mix, iter);
    mix.launch(lc);

    // Oracle: no DIIS, then F' = F + λ(S − S·D_occ·S) with D_occ = 0.5·P.
    RealMatrix err_ref = diis_error(F_in, P, S, X);
    RealMatrix D_occ = 0.5 * P;
    RealMatrix F_ref = apply_level_shift(F_in, S, D_occ, lambda);

    EXPECT_TRUE(MatBitEqual(err_a, err_ref));
    EXPECT_TRUE(MatBitEqual(F_a, F_ref));
    EXPECT_TRUE(MatBitEqual(F_b, F_ref)) << "restricted β mirrors α";
}

TEST(MixStage, SkipsLevelShiftAtOrBelowThreshold) {
    RealMatrix S = sample_S(), X = sample_X();
    RealMatrix P = sym2(1.0, 0.3, 0.8);
    RealMatrix F_in = sym2(-1.0, 0.4, -0.6);

    int iter = 0;
    double e_tot = 0.0;
    RealMatrix F_a = F_in, F_b = F_in, err_a, err_b;
    cpu::scf::MixStage mix(true, /*use_diis=*/false, 0, 6,
                           /*level_shift=*/1e-5, cpu::scf::MixerKind::CDIIS,
                           /*mixer_switch_tol=*/1e-2, /*soscf=*/false,
                           /*soscf_start=*/0.25, S, X, P, P, iter, e_tot,
                           F_a, F_b, err_a, err_b, soscf_latch);
    cpu::Context rt;
    cpu::Frame fr = atv_test::make_trivial_frame();
    cpu::BuildContext bc{rt, fr};
    mix.build(bc);
    auto lc = at_stage(rt, fr, scf::Phase::mix, iter);
    mix.launch(lc);

    // |λ| ≤ 1e-4 is a strict no-op (PySCF skip rule): F unchanged element-wise.
    EXPECT_TRUE(MatBitEqual(F_a, F_in));
}

TEST(MixStage, ResetClearsDiisHistory) {
    RealMatrix S = sample_S(), X = sample_X();
    RealMatrix P = sym2(1.0, 0.3, 0.8);
    RealMatrix F0 = sym2(-1.0, 0.4, -0.6);
    RealMatrix F1 = sym2(-1.1, 0.35, -0.55);
    RealMatrix F2 = sym2(-1.2, 0.30, -0.50);

    cpu::Context rt;
    cpu::Frame fr = atv_test::make_trivial_frame();

    // Fresh mixer, single step on F0.
    int iterA = 0;
    double e_totA = 0.0;
    RealMatrix Fa_A = F0, Fb_A = F0, ea_A, eb_A;
    cpu::scf::MixStage fresh(true, true, 0, 6, 0.0,
                             cpu::scf::MixerKind::CDIIS, 1e-2, false, 0.25,
                             S, X, P, P,
                             iterA, e_totA, Fa_A, Fb_A, ea_A, eb_A,
                             soscf_latch);
    cpu::BuildContext bcA{rt, fr};
    fresh.build(bcA);
    Fa_A = F0;
    auto lcA = at_stage(rt, fr, scf::Phase::mix, iterA);
    fresh.launch(lcA);
    RealMatrix fresh_out = Fa_A;

    // Grown mixer: two steps (history size 2), reset(), then the same F0 step.
    int iterB = 0;
    double e_totB = 0.0;
    RealMatrix Fa_B, Fb_B, ea_B, eb_B;
    cpu::scf::MixStage grown(true, true, 0, 6, 0.0,
                             cpu::scf::MixerKind::CDIIS, 1e-2, false, 0.25,
                             S, X, P, P,
                             iterB, e_totB, Fa_B, Fb_B, ea_B, eb_B,
                             soscf_latch);
    cpu::BuildContext bcB{rt, fr};
    grown.build(bcB);
    for (iterB = 0; iterB < 2; ++iterB) {
        Fa_B = (iterB == 0) ? F1 : F2;
        auto lc = at_stage(rt, fr, scf::Phase::mix, iterB);
        grown.launch(lc);
    }
    grown.reset();
    iterB = 0;
    Fa_B = F0;
    auto lcB = at_stage(rt, fr, scf::Phase::mix, iterB);
    grown.launch(lcB);

    // After reset the grown mixer behaves as if it had no history at all.
    EXPECT_TRUE(MatBitEqual(Fa_B, fresh_out));
}

// ── DiagonalizeStage: closed EigenSolver, healthy + orthogonal paths ─────────

TEST(DiagonalizeStage, ReproducesEigenSolverStepRestricted) {
    RealMatrix F = sym2(2.0, 0.3, 1.0);
    RealMatrix S = sample_S();
    RealMatrix X = sample_X(); // unused when n_dropped==0
    int n_dropped = 0;

    RealMatrix C_a, C_b;
    RealVector eig_a, eig_b;
    cpu::scf::DiagonalizeStage diag(/*restricted=*/true, F, F, S, X, n_dropped,
                                    C_a, C_b, eig_a, eig_b,
                                    /*soscf=*/nullptr, kNoSoscf);
    cpu::Context rt;
    cpu::Frame fr = atv_test::make_trivial_frame();
    cpu::BuildContext bc{rt, fr};
    diag.build(bc);
    auto lc = at_stage(rt, fr, scf::Phase::diagonalize);
    diag.launch(lc);

    EigenSolver es;
    auto [eig_ref, C_ref] = es.step(F, S);
    EXPECT_TRUE(VecBitEqual(eig_a, eig_ref));
    EXPECT_TRUE(MatBitEqual(C_a, C_ref));
    // Restricted reuses the α result for β.
    EXPECT_TRUE(VecBitEqual(eig_b, eig_ref));
    EXPECT_TRUE(MatBitEqual(C_b, C_ref));
}

TEST(DiagonalizeStage, ReproducesEigenSolverStepUnrestricted) {
    RealMatrix Fa = sym2(2.0, 0.3, 1.0);
    RealMatrix Fb = sym2(1.5, 0.1, 0.9);
    RealMatrix S = sample_S();
    RealMatrix X = sample_X();
    int n_dropped = 0;

    RealMatrix C_a, C_b;
    RealVector eig_a, eig_b;
    cpu::scf::DiagonalizeStage diag(/*restricted=*/false, Fa, Fb, S, X,
                                    n_dropped, C_a, C_b, eig_a, eig_b,
                                    /*soscf=*/nullptr, kNoSoscf);
    cpu::Context rt;
    cpu::Frame fr = atv_test::make_trivial_frame();
    cpu::BuildContext bc{rt, fr};
    diag.build(bc);
    auto lc = at_stage(rt, fr, scf::Phase::diagonalize);
    diag.launch(lc);

    EigenSolver es;
    auto [ea_ref, Ca_ref] = es.step(Fa, S);
    auto [eb_ref, Cb_ref] = es.step(Fb, S);
    EXPECT_TRUE(VecBitEqual(eig_a, ea_ref));
    EXPECT_TRUE(MatBitEqual(C_a, Ca_ref));
    EXPECT_TRUE(VecBitEqual(eig_b, eb_ref));
    EXPECT_TRUE(MatBitEqual(C_b, Cb_ref));
}

TEST(DiagonalizeStage, UsesStepOrthogonalWhenColumnsDropped) {
    RealMatrix F = sym2(2.0, 0.3, 1.0);
    RealMatrix S = sample_S(); // unused on the orthogonal path
    RealMatrix X = sample_X(); // the (possibly rectangular) transform
    int n_dropped = 1;         // >0 ⇒ canonical-orthogonalization path

    RealMatrix C_a, C_b;
    RealVector eig_a, eig_b;
    cpu::scf::DiagonalizeStage diag(/*restricted=*/true, F, F, S, X, n_dropped,
                                    C_a, C_b, eig_a, eig_b,
                                    /*soscf=*/nullptr, kNoSoscf);
    cpu::Context rt;
    cpu::Frame fr = atv_test::make_trivial_frame();
    cpu::BuildContext bc{rt, fr};
    diag.build(bc);
    auto lc = at_stage(rt, fr, scf::Phase::diagonalize);
    diag.launch(lc);

    EigenSolver es;
    auto [eig_ref, C_ref] = es.step_orthogonal(F, X);
    EXPECT_TRUE(VecBitEqual(eig_a, eig_ref));
    EXPECT_TRUE(MatBitEqual(C_a, C_ref));
}

// ── DensityStage: density build + damping + four-branch energy ───────────────

namespace {

/// @brief Drive one DensityStage launch and return the branch's own energies.
struct DensityRun {
    RealMatrix Pa_out, Pb_out;
    double e_total, e_one, e_coulomb;
};

DensityRun run_density(bool restricted, bool is_hf, double damping,
                       bool diis_active, const RealMatrix& Ca,
                       const RealMatrix& Cb, const RealMatrix& Fa,
                       const RealMatrix& Fb, const RealMatrix& Hcore,
                       const RealMatrix& J, int n_alpha, int n_beta,
                       int n_basis, double e_xc_grid, double e_exact_exchange,
                       double v_nuc, const RealMatrix& Pa_old,
                       const RealMatrix& Pb_old) {
    DensityRun r;
    r.Pa_out = Pa_old; // in/out: old density in, new density out
    r.Pb_out = Pb_old;
    r.e_total = r.e_one = r.e_coulomb = 0.0;

    bool diis_flag = diis_active;
    int na = n_alpha, nb = n_beta, nbf = n_basis;
    double exc = e_xc_grid, exx = e_exact_exchange, vnuc = v_nuc;

    cpu::scf::DensityStage dens(restricted, is_hf, damping, Ca, Cb, Fa, Fb,
                                Hcore, J, na, nb, nbf, diis_flag, exc, exx,
                                vnuc, r.Pa_out, r.Pb_out, r.e_total, r.e_one,
                                r.e_coulomb);
    cpu::Context rt;
    cpu::Frame fr = atv_test::make_trivial_frame();
    cpu::BuildContext bc{rt, fr};
    dens.build(bc);
    auto lc = at_stage(rt, fr, scf::Phase::density);
    dens.launch(lc);
    return r;
}

} // namespace

TEST(DensityStage, RestrictedHFOldDensityEnergyWithDamping) {
    RealMatrix C = sym2(0.8, 0.6, -0.8); // occupied col = (0.8, 0.6)
    RealMatrix Hcore = sym2(-1.2, 0.1, -0.9);
    RealMatrix F = sym2(-1.0, 0.4, -0.6);
    RealMatrix J = xt::zeros<double>({2, 2});
    RealMatrix P_old = sym2(1.0, 0.25, 0.7);
    const double damping = 0.4, v_nuc = 0.7;

    DensityRun got =
        run_density(/*restricted=*/true, /*is_hf=*/true, damping,
                    /*diis_active=*/false, C, C, F, F, Hcore, J,
                    /*n_alpha=*/1, /*n_beta=*/1, /*n_basis=*/2,
                    /*e_xc=*/0.0, /*e_exact=*/0.0, v_nuc, P_old, P_old);

    DensityBuilder db;
    RealMatrix P_new = db.step(C, 1, 2, 2.0);
    RealMatrix P_next = (1.0 - damping) * P_new + damping * P_old; // DIIS off
    double e_total = 0.5 * frob_dot(Hcore + F, P_old) + v_nuc;     // OLD P
    double e_one = frob_dot(Hcore, P_old);

    EXPECT_TRUE(MatBitEqual(got.Pa_out, P_next));
    EXPECT_TRUE(MatBitEqual(got.Pb_out, P_next)) << "restricted β mirrors α";
    EXPECT_TRUE(ScalarBitEqual(got.e_total, e_total));
    EXPECT_TRUE(ScalarBitEqual(got.e_one, e_one));
    EXPECT_TRUE(ScalarBitEqual(got.e_coulomb, 0.0));
}

TEST(DensityStage, DiisActiveSkipsDamping) {
    RealMatrix C = sym2(0.8, 0.6, -0.8);
    RealMatrix Hcore = sym2(-1.2, 0.1, -0.9);
    RealMatrix F = sym2(-1.0, 0.4, -0.6);
    RealMatrix J = xt::zeros<double>({2, 2});
    RealMatrix P_old = sym2(1.0, 0.25, 0.7);

    DensityRun got = run_density(true, true, /*damping=*/0.4,
                                 /*diis_active=*/true, C, C, F, F, Hcore, J, 1,
                                 1, 2, 0.0, 0.0, 0.7, P_old, P_old);

    DensityBuilder db;
    RealMatrix P_new = db.step(C, 1, 2, 2.0); // damping suppressed when DIIS on
    EXPECT_TRUE(MatBitEqual(got.Pa_out, P_new));
}

TEST(DensityStage, RestrictedKSNewDensityEnergy) {
    RealMatrix C = sym2(0.8, 0.6, -0.8);
    RealMatrix Hcore = sym2(-1.2, 0.1, -0.9);
    RealMatrix F = sym2(-1.0, 0.4, -0.6); // unused by KS energy
    RealMatrix J = sym2(0.9, 0.2, 0.7);
    RealMatrix P_old = sym2(1.0, 0.25, 0.7);
    const double e_xc = -0.55, e_exact = 0.0, v_nuc = 0.7;

    DensityRun got =
        run_density(/*restricted=*/true, /*is_hf=*/false,
                    /*damping=*/0.0, /*diis_active=*/false, C, C, F, F, Hcore,
                    J, 1, 1, 2, e_xc, e_exact, v_nuc, P_old, P_old);

    DensityBuilder db;
    RealMatrix P_new = db.step(C, 1, 2, 2.0);
    double e_one = frob_dot(Hcore, P_new); // NEW P
    double e_coulomb = 0.5 * frob_dot(J, P_new);
    double e_total = e_one + e_coulomb + e_exact + e_xc + v_nuc;

    EXPECT_TRUE(MatBitEqual(got.Pa_out, P_new));
    EXPECT_TRUE(ScalarBitEqual(got.e_one, e_one));
    EXPECT_TRUE(ScalarBitEqual(got.e_coulomb, e_coulomb));
    EXPECT_TRUE(ScalarBitEqual(got.e_total, e_total));
}

TEST(DensityStage, UnrestrictedHFOldDensityEnergy) {
    RealMatrix Ca = sym2(0.8, 0.6, -0.8);
    RealMatrix Cb = sym2(0.6, 0.8, -0.6);
    RealMatrix Hcore = sym2(-1.2, 0.1, -0.9);
    RealMatrix Fa = sym2(-1.0, 0.4, -0.6);
    RealMatrix Fb = sym2(-0.9, 0.3, -0.5);
    RealMatrix J = xt::zeros<double>({2, 2});
    RealMatrix Pa_old = sym2(1.0, 0.25, 0.7);
    RealMatrix Pb_old = sym2(0.6, 0.15, 0.4);
    const double v_nuc = 0.7;

    DensityRun got =
        run_density(/*restricted=*/false, /*is_hf=*/true, /*damping=*/0.0,
                    /*diis_active=*/false, Ca, Cb, Fa, Fb, Hcore, J, 1, 1, 2,
                    0.0, 0.0, v_nuc, Pa_old, Pb_old);

    DensityBuilder db;
    RealMatrix Pa_new = db.step(Ca, 1, 2);
    RealMatrix Pb_new = db.step(Cb, 1, 2);
    double e_total = 0.5
                         * (frob_dot(Hcore + Fa, Pa_old) // OLD densities
                            + frob_dot(Hcore + Fb, Pb_old))
                     + v_nuc;
    double e_one = frob_dot(Hcore, Pa_old) + frob_dot(Hcore, Pb_old);

    EXPECT_TRUE(MatBitEqual(got.Pa_out, Pa_new));
    EXPECT_TRUE(MatBitEqual(got.Pb_out, Pb_new));
    EXPECT_TRUE(ScalarBitEqual(got.e_total, e_total));
    EXPECT_TRUE(ScalarBitEqual(got.e_one, e_one));
}

TEST(DensityStage, UnrestrictedKSNewDensityEnergy) {
    RealMatrix Ca = sym2(0.8, 0.6, -0.8);
    RealMatrix Cb = sym2(0.6, 0.8, -0.6);
    RealMatrix Hcore = sym2(-1.2, 0.1, -0.9);
    RealMatrix Fa = sym2(-1.0, 0.4, -0.6); // unused by KS energy
    RealMatrix Fb = sym2(-0.9, 0.3, -0.5);
    RealMatrix J = sym2(0.9, 0.2, 0.7);
    RealMatrix Pa_old = sym2(1.0, 0.25, 0.7);
    RealMatrix Pb_old = sym2(0.6, 0.15, 0.4);
    const double e_xc = -0.6, e_exact = 0.05, v_nuc = 0.7;

    DensityRun got =
        run_density(/*restricted=*/false, /*is_hf=*/false, /*damping=*/0.0,
                    /*diis_active=*/false, Ca, Cb, Fa, Fb, Hcore, J, 1, 1, 2,
                    e_xc, e_exact, v_nuc, Pa_old, Pb_old);

    DensityBuilder db;
    RealMatrix Pa_new = db.step(Ca, 1, 2);
    RealMatrix Pb_new = db.step(Cb, 1, 2);
    RealMatrix P_total = Pa_new + Pb_new;
    double e_one = frob_dot(Hcore, Pa_new) + frob_dot(Hcore, Pb_new);
    double e_coulomb = 0.5 * frob_dot(J, P_total);
    double e_total = e_one + e_coulomb + e_exact + e_xc + v_nuc;

    EXPECT_TRUE(MatBitEqual(got.Pa_out, Pa_new));
    EXPECT_TRUE(MatBitEqual(got.Pb_out, Pb_new));
    EXPECT_TRUE(ScalarBitEqual(got.e_one, e_one));
    EXPECT_TRUE(ScalarBitEqual(got.e_coulomb, e_coulomb));
    EXPECT_TRUE(ScalarBitEqual(got.e_total, e_total));
}

// ── ScfConverged: one Criterion — launch() measures, verdict() judges ────────

namespace {

/// @brief A LoopState carrying sentinel metrics — ScfConverged must ignore it
///        (it judges its own members, never ConvergenceMetrics).
cpu::LoopState garbage_state(cpu::Frame& fr, cpu::Context& rt,
                             ConvergenceMetrics& metrics) {
    metrics.energy = 1e300; // sentinel: a metrics-reader would trip on this
    metrics.max_force = 1e300;
    return cpu::LoopState{/*iteration=*/7, fr, rt, metrics};
}

} // namespace

TEST(ScfConverged, BindsCheckPhaseAndFirstIterationContinues) {
    double e_total = -1.0;
    RealMatrix err_a = xt::zeros<double>({2, 2}); // tiny error
    RealMatrix err_b = xt::zeros<double>({2, 2});

    cpu::scf::ScfConverged crit(/*tol_energy=*/1e-9, /*tol_grad=*/1e-4,
                                /*restricted=*/true, e_total, err_a, err_b);

    auto bindings = crit.default_bindings();
    ASSERT_EQ(bindings.size(), 1u);
    EXPECT_EQ(bindings[0].kind, PhaseKind::SCF);
    EXPECT_EQ(bindings[0].phase, slot_of(scf::Phase::check));

    cpu::Context rt;
    cpu::Frame fr = atv_test::make_trivial_frame();
    ConvergenceMetrics m;
    crit.reset();
    auto lc = at_stage(rt, fr, scf::Phase::check);
    crit.launch(lc); // measures: ΔE = +∞ on the first iteration

    EXPECT_TRUE(std::isinf(crit.delta_energy()));
    auto state = garbage_state(fr, rt, m);
    EXPECT_EQ(crit.verdict(state), Verdict::Continue)
        << "iter 0 has no prior energy ⇒ never converged, whatever the error";
}

TEST(ScfConverged, ConvergesOnlyWhenBothCriteriaMet) {
    double e_total = -1.0;
    RealMatrix err_a = xt::ones<double>({2, 2}); // ‖err‖ = 2 (large)
    RealMatrix err_b = xt::zeros<double>({2, 2});

    cpu::scf::ScfConverged crit(1e-9, 1e-4, /*restricted=*/true, e_total, err_a,
                                err_b);
    cpu::Context rt;
    cpu::Frame fr = atv_test::make_trivial_frame();
    ConvergenceMetrics m;
    crit.reset();

    auto lc = at_stage(rt, fr, scf::Phase::check);

    // Iter 0: prime prev_energy.
    crit.launch(lc);
    auto s0 = garbage_state(fr, rt, m);
    EXPECT_EQ(crit.verdict(s0), Verdict::Continue);

    // Iter 1: ΔE ≈ 0 < tol, but the error is large ⇒ Continue.
    e_total = -1.0;
    crit.launch(lc);
    auto s1 = garbage_state(fr, rt, m);
    EXPECT_EQ(crit.verdict(s1), Verdict::Continue);

    // Iter 2: ΔE ≈ 0 and the error collapses to 0 ⇒ Converged.
    err_a = xt::zeros<double>({2, 2});
    crit.launch(lc);
    auto s2 = garbage_state(fr, rt, m);
    EXPECT_EQ(crit.verdict(s2), Verdict::Converged);
}

TEST(ScfConverged, ResetClearsPrevEnergy) {
    double e_total = -1.0;
    RealMatrix err_a = xt::zeros<double>({2, 2});
    RealMatrix err_b = xt::zeros<double>({2, 2});
    cpu::scf::ScfConverged crit(1e-9, 1e-4, true, e_total, err_a, err_b);

    cpu::Context rt;
    cpu::Frame fr = atv_test::make_trivial_frame();
    auto lc = at_stage(rt, fr, scf::Phase::check);

    crit.reset();
    crit.launch(lc); // primes prev_energy
    crit.reset();
    crit.launch(lc); // fresh again ⇒ ΔE = +∞
    EXPECT_TRUE(std::isinf(crit.delta_energy()));
}

TEST(ScfConverged, MatchesDualConvergenceCheckRestricted) {
    double e_total = 0.0;
    RealMatrix err_a = xt::zeros<double>({2, 2});
    RealMatrix err_b = xt::zeros<double>({2, 2});

    cpu::scf::ScfConverged crit(1e-9, 1e-4, /*restricted=*/true, e_total, err_a,
                                err_b);
    DualConvergenceCheck oracle(1e-9, 1e-4);
    crit.reset();
    oracle.reset();

    cpu::Context rt;
    cpu::Frame fr = atv_test::make_trivial_frame();
    ConvergenceMetrics m;
    auto lc = at_stage(rt, fr, scf::Phase::check);

    const double energies[4] = {-1.0, -1.0000001, -1.0000001, -1.0000001};
    const double err_scales[4] = {1.0, 0.5, 0.0, 0.0};
    for (int i = 0; i < 4; ++i) {
        e_total = energies[i];
        err_a = err_scales[i] * xt::ones<double>({2, 2});

        crit.launch(lc);
        auto v_ref = oracle.step(energies[i], err_a); // restricted ⇒ no β

        EXPECT_TRUE(ScalarBitEqual(crit.diis_error(), v_ref.err_norm))
            << "i=" << i;
        auto state = garbage_state(fr, rt, m);
        Verdict expected =
            v_ref.converged ? Verdict::Converged : Verdict::Continue;
        EXPECT_EQ(crit.verdict(state), expected) << "i=" << i;
    }
}

TEST(ScfConverged, CombinesAlphaBetaFrobeniusUnrestricted) {
    double e_total = 0.0;
    RealMatrix err_a = xt::zeros<double>({2, 2});
    RealMatrix err_b = xt::zeros<double>({2, 2});

    cpu::scf::ScfConverged crit(1e-9, 1e-4, /*restricted=*/false, e_total,
                                err_a, err_b);
    DualConvergenceCheck oracle(1e-9, 1e-4);
    crit.reset();
    oracle.reset();

    cpu::Context rt;
    cpu::Frame fr = atv_test::make_trivial_frame();
    auto lc = at_stage(rt, fr, scf::Phase::check);

    const double ea_scales[3] = {0.3, 0.1, 0.0};
    const double eb_scales[3] = {0.4, 0.2, 0.0};
    for (int i = 0; i < 3; ++i) {
        e_total = -1.0;
        err_a = ea_scales[i] * xt::ones<double>({2, 2});
        err_b = eb_scales[i] * xt::ones<double>({2, 2});

        crit.launch(lc);
        auto v_ref = oracle.step(-1.0, err_a, &err_b); // combined α/β Frobenius
        EXPECT_TRUE(ScalarBitEqual(crit.diis_error(), v_ref.err_norm))
            << "i=" << i;
    }
}

#endif // ATV_HAS_HF

/// @file tests/cpu/scf/test_method_driver_parity.cpp
/// @brief scf-driver-refactor-03 — RHF/UHF/RKS/UKS are
///        `cpu::CPUDriver<ATV::scf::Phase>` loop nodes; this file pins their
///        converged energies/eigenvalues and their internal-consistency wiring.
///
/// ## History
///
/// Spec 03a converted the four concrete methods to the driver loop and proved
/// each bit-identical (1e-12 Ha) to the in-tree solver reference while that
/// reference still shipped. Spec 03b RETIRES that reference. With one
/// implementation left, "parity vs a second implementation" is meaningless, so
/// this file becomes a REGRESSION guard: the per-method expectations are PINNED
/// literals captured from the driver-ized methods at 03a bit-parity (see the
/// provenance on each constant), and the remaining checks are engine-internal
/// invariants that need no external oracle.
///
/// ## What the suite pins
///
///   - **Derivation (compile-time).** Each method IS a
///     `cpu::CPUDriver<ATV::scf::Phase>` (four static_asserts).
///   - **Energy / eigenvalue regression.** RHF/UHF/RKS/UKS on the small
///     molecules the other `tests/cpu/scf/*` use (H / H₂ / He / H₂O, STO-3G)
///     reproduce the pinned converged energies (and, for the water cases and the
///     level shift, the pinned orbital energies).
///   - **One funnel, two entries.** The same molecule via `run(frame)` and via
///     binding the method as an `HF` component into `cpu::Evaluate` (the `launch`
///     path) yields the identical SCF energy — both route through the private
///     `drive_scf_`.
///   - **Wiring.** After a converging run the loop node reports `converged()` and
///     `stop_reason() == "scf_converged"` — the ctor registered `ScfConverged`
///     and the driver polled it.
///   - **Edges.** `level_shift ≠ 0` converges and reports de-shifted orbital
///     energies matching the pinned unshifted eigenvalues; `mp2()` / `fci()`
///     still route through the driven method (both corrections vanish for He).
///
/// The path under test (`method.run`, `Evaluate` + the method component) names no
/// retired machinery. Threads are pinned to 1 in CMake so the pinned literals are
/// reproducible (see `test_cpu_scf_stages` for the same reason).

#ifdef ATV_HAS_HF

#include <cmath>
#include <cstddef>
#include <string>

#include <gtest/gtest.h>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/driver/cpu_driver.h"
#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h" // RHF/UHF/RKS/UKS umbrella (path under test)
#include "atomiverse/cpu/scf/slater_determinant.h" // wavefunction read surface
#include "atomiverse/facade/stage.h" // ATV::scf::Phase, ATV::evaluate::Phase

using namespace ATV;
using Spin = cpu::scf::SlaterDeterminant::Spin;

// ── Compile-time derivation: each concrete method IS a CPUDriver<scf::Phase> ───
//
// After 03a `MeanFieldMethod<D>` derives BOTH `CPUDriver<ATV::scf::Phase>` and
// `HF`, so all four hold.
static_assert(std::is_base_of_v<cpu::CPUDriver<ATV::scf::Phase>, cpu::scf::RHF>,
              "03a: RHF must derive cpu::CPUDriver<ATV::scf::Phase>");
static_assert(std::is_base_of_v<cpu::CPUDriver<ATV::scf::Phase>, cpu::scf::UHF>,
              "03a: UHF must derive cpu::CPUDriver<ATV::scf::Phase>");
static_assert(std::is_base_of_v<cpu::CPUDriver<ATV::scf::Phase>, cpu::scf::RKS>,
              "03a: RKS must derive cpu::CPUDriver<ATV::scf::Phase>");
static_assert(std::is_base_of_v<cpu::CPUDriver<ATV::scf::Phase>, cpu::scf::UKS>,
              "03a: UKS must derive cpu::CPUDriver<ATV::scf::Phase>");

namespace {

// ── Geometries reused verbatim from the existing SCF solver tests ─────────────

/// @brief H₂O / STO-3G (10 electrons) — test_scf_solver.cpp:17-36 and
///        test_uks_solver.cpp:27-46 (identical).
cpu::Frame make_h2o() {
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.7572;
    f.span(kAtom_x)[2] = -0.7572;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_y)[2] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_z)[1] = -0.5860;
    f.span(kAtom_z)[2] = -0.5860;
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_mass)[0] = 15.999;
    f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_mass)[2] = 1.008;
    f.simbox = cpu::Box::cubic(50.0);
    return f;
}

/// @brief Single H atom (1 electron) — test_uhf_solver.cpp:16-25.
cpu::Frame make_h() {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

/// @brief He atom (2 electrons) — test_uks_solver.cpp:16-25 / test_mp2.cpp:16-25.
cpu::Frame make_he() {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[0] = 2;
    f.span(kAtom_mass)[0] = 4.0026;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

/// @brief H₂ at r = 0.74 Å (2 electrons) — test_mp2.cpp:27-41 /
///        test_level_shift.cpp:6-20.
cpu::Frame make_h2() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.74;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.span(kAtom_mass)[1] = 1.008;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

// ── FP tolerances (project scientific-test table) ─────────────────────────────
constexpr double kEnergyPin = 1e-9; ///< Ha — pinned converged-energy tol.
constexpr double kEigPin = 1e-8;    ///< Ha — pinned orbital-energy tol.
constexpr double kPhysTol = 1e-8;   ///< reference-free physical invariants.
constexpr double kFunnelTol =
    1e-10; ///< Ha — same-code run() vs launch() agree.

// ── Pinned reference values ───────────────────────────────────────────────────
//
// Captured 2026-07-20 from the driver-ized RHF/UHF/RKS/UKS on the geometries
// above, which spec 03a proved bit-identical (1e-12 Ha) to the retired solver
// reference — these literals ARE that reference, frozen (spec 03b). Regenerate
// only with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 (as CMake pins this test).
// All energies [Hartree].
constexpr double kRhfWaterEnergy = -74.9629648020;   ///< H₂O/STO-3G RHF [Ha].
constexpr double kUhfHydrogenEnergy = -0.4665818496; ///< H/STO-3G UHF [Ha].
constexpr double kRksWaterEnergy = -75.2242376913; ///< H₂O/STO-3G RKS-PBE [Ha].
constexpr double kUksHeliumEnergy = -2.7718860444; ///< He/STO-3G UKS-LSDA [Ha].
constexpr double kLevelShiftEnergy = -1.1167593074; ///< H₂/STO-3G RHF [Ha].

void expect_vector_near(const xt::xtensor<double, 1>& got,
                        const xt::xtensor<double, 1>& ref, double tol) {
    ASSERT_EQ(got.shape(0), ref.shape(0));
    for (std::size_t i = 0; i < got.shape(0); ++i)
        EXPECT_NEAR(got(i), ref(i), tol) << "index " << i;
}

void expect_matrix_near(const xt::xtensor<double, 2>& got,
                        const xt::xtensor<double, 2>& ref, double tol) {
    ASSERT_EQ(got.shape(0), ref.shape(0));
    ASSERT_EQ(got.shape(1), ref.shape(1));
    for (std::size_t i = 0; i < got.shape(0); ++i)
        for (std::size_t j = 0; j < got.shape(1); ++j)
            EXPECT_NEAR(got(i, j), ref(i, j), tol)
                << "(" << i << "," << j << ")";
}

} // namespace

// ── Per-method energy / eigenvalue regression (pinned) ────────────────────────

TEST(MethodDriverParity, RhfWaterMatchesPinnedEnergy) {
    auto frame = make_h2o();
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.run(frame);

    ASSERT_TRUE(rhf.converged());
    EXPECT_NEAR(rhf.energy(), kRhfWaterEnergy, kEnergyPin);

    const cpu::scf::SlaterDeterminant& det = rhf.determinant();
    // Pinned occupied+virtual orbital energies [Ha], ascending (STO-3G → 7).
    xt::xtensor<double, 1> ref_eigs = {
        -20.2417608103, -1.2682616792, -0.6178033132, -0.4529445213,
        -0.3912210574,  0.6053917496,  0.7420994336};
    expect_vector_near(det.orbital_energies(Spin::alpha), ref_eigs, kEigPin);

    // Structural, reference-free: the density matrix is symmetric.
    // (Restricted run: density_total() carries the old factor-2 P_alpha.)
    const cpu::scf::RealMatrix P = det.density_total();
    expect_matrix_near(P, xt::transpose(P), kPhysTol);
}

TEST(MethodDriverParity, UhfHydrogenMatchesPinnedEnergy) {
    auto frame = make_h();
    cpu::scf::UHF uhf(/*n_alpha=*/1, /*n_beta=*/0);
    uhf.basis = cpu::scf::BasisSet::STO3G;
    uhf.run(frame);

    ASSERT_TRUE(uhf.converged());
    const cpu::scf::SlaterDeterminant& det = uhf.determinant();
    EXPECT_EQ(det.n_electrons(Spin::alpha), 1);
    EXPECT_EQ(det.n_electrons(Spin::beta), 0);
    EXPECT_NEAR(uhf.energy(), kUhfHydrogenEnergy, kEnergyPin);
    // A lone electron is a pure doublet ⇒ ⟨S²⟩ = 0.75, contamination = 0.
    EXPECT_NEAR(det.spin_contamination(), 0.0, kPhysTol);
}

TEST(MethodDriverParity, RksWaterMatchesPinnedEnergy) {
    auto frame = make_h2o();
    cpu::scf::RKS rks(cpu::scf::XCType::PBE);
    rks.basis = cpu::scf::BasisSet::STO3G;
    // Light grid for unit-test speed (test_uks_solver.cpp:69-80); the pinned
    // eigenvalues below were captured at exactly this grid.
    rks.grid_level = cpu::scf::GridLevel::SG0;
    rks.n_radial = 30;
    rks.n_angular = 50;
    rks.run(frame);

    ASSERT_TRUE(rks.converged());
    EXPECT_NEAR(rks.energy(), kRksWaterEnergy, kEnergyPin);

    xt::xtensor<double, 1> ref_eigs = {
        -18.4438887162, -0.8425040694, -0.3887271763, -0.1610996780,
        -0.0653909417,  0.3103799258,  0.4203563646};
    expect_vector_near(rks.determinant().orbital_energies(Spin::alpha),
                       ref_eigs, kEigPin);
}

TEST(MethodDriverParity, UksHeliumMatchesPinnedEnergy) {
    auto frame = make_he();
    cpu::scf::UKS uks(/*n_alpha=*/1, /*n_beta=*/1, cpu::scf::XCType::LSDA);
    uks.basis = cpu::scf::BasisSet::STO3G;
    uks.run(frame);

    ASSERT_TRUE(uks.converged());
    EXPECT_NEAR(uks.energy(), kUksHeliumEnergy, kEnergyPin);
    // Closed-shell singlet: the α and β densities coincide, ⟨S²⟩ = 0.
    // (Unrestricted run: per-spin density(σ) matches the old P_alpha/P_beta.)
    const cpu::scf::SlaterDeterminant& det = uks.determinant();
    expect_matrix_near(det.density(Spin::alpha), det.density(Spin::beta),
                       kPhysTol);
    EXPECT_NEAR(det.spin_contamination(), 0.0, kPhysTol);
}

// ── Wiring: the loop node knows convergence via the ScfConverged criterion ─────
// Probed behaviorally through the public CPUDriver API, not by introspecting the
// Plan.

TEST(MethodDriverParity, ConvergenceWiringFlipsDriverStatus) {
    auto frame = make_h2o();
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.run(frame);

    // The old duplicate SCFResult.converged flag is gone (03-unbag): the loop
    // node's converged() is the single source of truth.
    EXPECT_TRUE(rhf.converged()); // the LOOP knows (CPUDriver::converged).
    EXPECT_EQ(rhf.stop_reason(), "scf_converged"); // ScfConverged ended it.
}

// ── One funnel, two entries: run(frame) vs Evaluate + method component ─────────

TEST(MethodDriverParity, RunAndComponentLaunchAgree) {
    auto frame_run = make_h2o();
    cpu::scf::RHF direct;
    direct.basis = cpu::scf::BasisSet::STO3G;
    direct.run(frame_run);
    const double e_run = direct.energy(); // [Hartree]

    // Same molecule, but driven through the Component `launch` path by binding
    // the method into Evaluate. compute_force=false so launch writes the pure
    // SCF energy (no gradient recompute) and energy() carries it.
    auto frame_eval = make_h2o();
    cpu::scf::RHF component;
    component.basis = cpu::scf::BasisSet::STO3G;
    component.with_gradient(false);

    cpu::Evaluate eval;
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(component, /*order=*/0);
    eval.build();
    eval.run(frame_eval);

    // Both entries route through the same private drive_scf_ → identical energy.
    EXPECT_NEAR(component.energy(), e_run, kFunnelTol);
    // Frame is the currency: the component wrote a bound electronic energy [eV].
    const double e_meta = static_cast<double>(frame_eval.meta.get(kEnergy));
    EXPECT_TRUE(std::isfinite(e_meta));
    EXPECT_LT(e_meta, 0.0);
    EXPECT_NEAR(e_meta, e_run * units::kHartreeInEv, 1e-6);
}

// ── Edge: level shift converges and reports de-shifted orbital energies ────────

TEST(MethodDriverParity, LevelShiftMatchesPinnedDeShifted) {
    auto frame = make_h2(); // test_level_shift.cpp geometry (H₂, r = 0.74).
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.level_shift = 0.5;
    rhf.run(frame);

    ASSERT_TRUE(rhf.converged());
    EXPECT_NEAR(rhf.energy(), kLevelShiftEnergy, kEnergyPin);
    // De-shift post-processing must reproduce the pinned UNSHIFTED eigenvalues
    // (STO-3G H₂ → 2 orbitals), not the level-shifted ones.
    xt::xtensor<double, 1> ref_eigs = {-0.5785538598, 0.6711434919};
    expect_vector_near(rhf.determinant().orbital_energies(Spin::alpha),
                       ref_eigs, kEigPin);
}

// ── Edge: mp2()/fci() still route through the driven method ────────────────────

TEST(MethodDriverParity, Mp2AndFciStillRouteThroughDrivenMethod) {
    // He / STO-3G has no virtuals → both correlation corrections vanish, and
    // both must sit on top of the driven-method SCF energy (test_mp2.cpp,
    // test_fci.cpp inputs).
    auto frame_mp2 = make_he();
    cpu::scf::RHF rhf_mp2;
    rhf_mp2.basis = cpu::scf::BasisSet::STO3G;
    rhf_mp2.tol = 1e-12;
    const cpu::scf::MP2Result mp2 = rhf_mp2.mp2(frame_mp2);
    EXPECT_TRUE(rhf_mp2.converged());
    EXPECT_NEAR(mp2.e_corr, 0.0, 1e-14);
    EXPECT_NEAR(mp2.e_tot, rhf_mp2.energy(), 1e-12);

    auto frame_fci = make_he();
    cpu::scf::RHF rhf_fci;
    rhf_fci.basis = cpu::scf::BasisSet::STO3G;
    rhf_fci.tol = 1e-12;
    const cpu::scf::FCIResult fci = rhf_fci.fci(frame_fci);
    EXPECT_TRUE(rhf_fci.converged());
    EXPECT_NEAR(fci.e_corr, 0.0, 1e-12);
    EXPECT_NEAR(fci.e_fci, rhf_fci.energy(), 1e-12);
}

#endif // ATV_HAS_HF

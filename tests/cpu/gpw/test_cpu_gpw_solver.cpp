/// @file tests/cpu/gpw/test_cpu_gpw_solver.cpp
/// @brief End-to-end GPW SCF: the closed-shell loop converges for a minimal H2.
///
/// This exercises the whole GPW chain in one self-consistent loop — basis S/T,
/// GTH local PP, collocation, FFT-Poisson electrostatics (Gaussian core charge),
/// PADE XC, generalized eigensolve, density mixing. The assertion is that it
/// CONVERGES to a finite fixed point; exact CP2K bit-parity is the M1 milestone.
///
/// The `GpwSolverClass` suite (arch-refactor-04-engines) pins the GPWSolver
/// class contract: build/run lifecycle, fail-fast validation, and deterministic
/// energies across independent solver instances. Bit-parity tests pin
/// OMP to 1 thread: `integrate_potential` merges per-thread buffers under
/// `omp critical` in arrival order, so multi-thread runs are not bit-stable
/// across invocations (verified empirically; same pattern as
/// tests/cpu/scf/test_spin_resolution.cpp).

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"

using ATV::cpu::gpw::GpwAtom;
using ATV::cpu::gpw::GpwConfig;
using ATV::cpu::gpw::GpwResult;
using ATV::cpu::gpw::GPWSolver;
using ATV::cpu::gpw::GridBasisFunction;
using ATV::cpu::gpw::GridPrimitive;

// Public face parity with PWSolver: move-only value type (basics).
static_assert(!std::is_copy_constructible<GPWSolver>::value,
              "GPWSolver must not be copyable (owns workspace/result)");
static_assert(!std::is_copy_assignable<GPWSolver>::value,
              "GPWSolver must not be copy-assignable");
static_assert(std::is_move_constructible<GPWSolver>::value,
              "GPWSolver must be move-constructible (same as PWSolver)");
static_assert(std::is_move_assignable<GPWSolver>::value,
              "GPWSolver must be move-assignable (same as PWSolver)");

namespace {

// One normalized s-type primitive at center c.
GridBasisFunction s_basis(double alpha, const double c[3]) {
    GridPrimitive p;
    p.alpha = alpha;
    p.center[0] = c[0];
    p.center[1] = c[1];
    p.center[2] = c[2];
    p.l[0] = p.l[1] = p.l[2] = 0;
    p.coeff = std::pow(2.0 * alpha / M_PI, 0.75); // L2-normalized s-Gaussian
    return {p};
}

GpwAtom hydrogen(const double pos[3]) {
    GpwAtom a;
    a.z_val = 1.0;
    a.pos[0] = pos[0];
    a.pos[1] = pos[1];
    a.pos[2] = pos[2];
    a.basis = {s_basis(0.5, pos)};
    // GTH-PADE-q1-like local pseudopotential for H (rloc + short-range C coeffs;
    // q1 H has no nonlocal projectors). Values near the GTH H parameterization.
    a.rloc = 0.2;
    a.cloc = {-4.17890044, 0.72446331};
    return a;
}

// Small known-converging inline H2 (8 Bohr box, 120 Ry): converges in a couple
// of DIIS iterations, ~ms per SCF — keeps the class-contract suite in the
// fast gate. Verified against the direct solver lifecycle.
std::vector<GpwAtom> small_h2(double box = 8.0) {
    const double rA[3] = {box / 2 - 0.7, box / 2, box / 2};
    const double rB[3] = {box / 2 + 0.7, box / 2, box / 2}; // 1.4 Bohr apart
    return {hydrogen(rA), hydrogen(rB)};
}

GpwConfig small_cfg(double box = 8.0) {
    GpwConfig cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = box;
    cfg.cutoff_ry = 120.0;
    cfg.n_occ = 1; // 2 electrons, closed shell
    cfg.max_iter = 100;
    cfg.tol = 1e-7;
    cfg.mix = 0.5;
    return cfg;
}

void pin_single_thread() {
#ifdef _OPENMP
    omp_set_num_threads(1); // bit-stable FP sums (see file header)
#endif
}

} // namespace

TEST(GpwSolver, MinimalH2Converges) {
    const double box = 10.0;
    const double rA[3] = {box / 2 - 0.7, box / 2, box / 2};
    const double rB[3] = {box / 2 + 0.7, box / 2, box / 2}; // 1.4 Bohr apart

    std::vector<GpwAtom> atoms = {hydrogen(rA), hydrogen(rB)};

    GpwConfig cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = box;
    cfg.cutoff_ry = 250.0;
    cfg.n_occ = 1; // 2 electrons, closed shell
    cfg.max_iter = 100;
    cfg.tol = 1e-7;
    cfg.mix = 0.5;

    GPWSolver solver{cfg};
    solver.build(atoms);
    const GpwResult r = solver.run(atoms);

    EXPECT_TRUE(r.converged)
        << "SCF did not converge in " << r.iters << " iters";
    EXPECT_LT(r.iters, cfg.max_iter);
    EXPECT_TRUE(std::isfinite(r.e_total)) << "E_total = " << r.e_total;
    EXPECT_TRUE(std::isfinite(r.e_kinetic) && std::isfinite(r.e_es)
                && std::isfinite(r.e_xc));
    // Kinetic energy of a real density is positive; XC is negative.
    EXPECT_GT(r.e_kinetic, 0.0);
    EXPECT_LT(r.e_xc, 0.0);
}

// ── GPWSolver class contract (arch-refactor-04-engines Design §2) ──────────

// Independent cold solvers must be bit-identical in every energy channel
// [Hartree] and in their iteration count.
TEST(GpwSolverClass, IndependentColdRunsAreBitIdentical) {
    pin_single_thread();
    const auto atoms = small_h2();
    const GpwConfig cfg = small_cfg();

    GPWSolver reference_solver(cfg);
    reference_solver.build(atoms);
    const GpwResult ref = reference_solver.run(atoms);
    ASSERT_TRUE(ref.converged) << "reference solver did not converge";

    GPWSolver solver(cfg);
    solver.build(atoms);
    const GpwResult& r = solver.run(atoms);

    EXPECT_TRUE(r.converged);
    EXPECT_EQ(r.iters, ref.iters);
    EXPECT_DOUBLE_EQ(r.e_total, ref.e_total);
    EXPECT_DOUBLE_EQ(r.e_kinetic, ref.e_kinetic);
    EXPECT_DOUBLE_EQ(r.e_loc_sr, ref.e_loc_sr);
    EXPECT_DOUBLE_EQ(r.e_nonlocal, ref.e_nonlocal);
    EXPECT_DOUBLE_EQ(r.e_es, ref.e_es);
    EXPECT_DOUBLE_EQ(r.e_xc, ref.e_xc);
    ASSERT_EQ(r.eps.size(), ref.eps.size());
    for (std::size_t i = 0; i < ref.eps.size(); ++i)
        EXPECT_DOUBLE_EQ(r.eps[i], ref.eps[i]) << "eps[" << i << "]";
}

// Integration: the warm-start guess_C parameter must reach the same fixed point
// through an independent solver (AIMD/geometry sweeps rely on it).
TEST(GpwSolverClass, WarmStartReachesColdFixedPoint) {
    pin_single_thread();
    const auto atoms = small_h2();
    const GpwConfig cfg = small_cfg();

    GPWSolver cold_solver(cfg);
    cold_solver.build(atoms);
    const GpwResult cold = cold_solver.run(atoms);
    ASSERT_TRUE(cold.converged);

    GPWSolver solver(cfg);
    solver.build(atoms);
    const GpwResult& r = solver.run(atoms, &cold.C);

    EXPECT_TRUE(r.converged);
    EXPECT_DOUBLE_EQ(r.e_total, cold.e_total);
}

// Lifecycle: build() is one-time, run() is re-entrant — two cold runs on the
// same built solver give the identical result (geometry-dependent structures
// are rebuilt per run, same as PWSolver).
TEST(GpwSolverClass, RunIsReentrantAfterSingleBuild) {
    pin_single_thread();
    const auto atoms = small_h2();

    GPWSolver solver(small_cfg());
    solver.build(atoms);

    const double e1 = solver.run(atoms).e_total;
    const int it1 = solver.last_result().iters;
    const GpwResult& r2 = solver.run(atoms);

    EXPECT_TRUE(r2.converged);
    EXPECT_EQ(r2.iters, it1);
    EXPECT_DOUBLE_EQ(r2.e_total, e1);
}

// Lifecycle: run() before build() is a programming error — fail fast with
// std::logic_error, never a silent auto-build.
TEST(GpwSolverClass, RunBeforeBuildThrowsLogicError) {
    const auto atoms = small_h2();
    GPWSolver solver(small_cfg());
    EXPECT_THROW(solver.run(atoms), std::logic_error);
}

// Edge case: build() rejects n_occ < 1 (no-fallback rule).
TEST(GpwSolverClass, BuildRejectsNonPositiveNOcc) {
    const auto atoms = small_h2();
    GpwConfig cfg = small_cfg();
    cfg.n_occ = 0;
    GPWSolver solver(cfg);
    EXPECT_THROW(solver.build(atoms), std::invalid_argument);
}

// Edge case: build() rejects a periodic cell with a zero-length edge.
TEST(GpwSolverClass, BuildRejectsZeroCellEdgeWhenPeriodic) {
    const auto atoms = small_h2();
    GpwConfig cfg = small_cfg();
    cfg.cell[1] = 0.0;
    cfg.nonperiodic = false;
    GPWSolver solver(cfg);
    EXPECT_THROW(solver.build(atoms), std::invalid_argument);
}

// D3 always draws from the in-tree tables (basis_data/dftd3_tables.h); an
// element outside their coverage THROWS at solve time rather than silently
// contributing zero dispersion (no-fallback rule).
TEST(GpwSolverClass, BuildAcceptsD3) {
    const auto atoms = small_h2();
    GpwConfig cfg = small_cfg();
    cfg.d3 = true;
    GPWSolver solver(cfg);
    EXPECT_NO_THROW(solver.build(atoms));
}

// Edge case: build() rejects an empty atom list (Design §2: atoms/basis
// non-empty is part of the one-time fail-fast validation).
TEST(GpwSolverClass, BuildRejectsEmptyAtoms) {
    GPWSolver solver(small_cfg());
    EXPECT_THROW(solver.build(std::vector<GpwAtom>{}), std::invalid_argument);
}

// Basics: config() echoes the constructor argument; last_result() aliases the
// owned result buffer returned by run() and starts out unconverged.
TEST(GpwSolverClass, AccessorsExposeConfigAndLastResult) {
    pin_single_thread();
    const auto atoms = small_h2();
    const GpwConfig cfg = small_cfg();

    GPWSolver solver(cfg);
    EXPECT_DOUBLE_EQ(solver.config().cutoff_ry, cfg.cutoff_ry);
    EXPECT_EQ(solver.config().n_occ, cfg.n_occ);
    EXPECT_DOUBLE_EQ(solver.config().cell[0], cfg.cell[0]);
    EXPECT_DOUBLE_EQ(solver.config().cell[1], cfg.cell[1]);
    EXPECT_DOUBLE_EQ(solver.config().cell[2], cfg.cell[2]);
    EXPECT_DOUBLE_EQ(solver.config().tol, cfg.tol);
    EXPECT_FALSE(solver.last_result().converged); // owned default before run()

    solver.build(atoms);
    const GpwResult& r = solver.run(atoms);
    EXPECT_EQ(&solver.last_result(), &r); // same owned buffer
    EXPECT_TRUE(solver.last_result().converged);
}

// Lifecycle: move construction transfers the built state — the moved-to
// solver runs without a second build() (move-only face, same as PWSolver).
TEST(GpwSolverClass, MoveConstructionPreservesBuiltState) {
    pin_single_thread();
    const auto atoms = small_h2();

    GPWSolver a(small_cfg());
    a.build(atoms);
    GPWSolver b(std::move(a));

    const GpwResult& r = b.run(atoms);
    EXPECT_TRUE(r.converged);
    EXPECT_TRUE(std::isfinite(r.e_total));
}

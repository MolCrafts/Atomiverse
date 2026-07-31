/// @file tests/cpu/gpw/test_cpu_gpw_solver.cpp
/// @brief End-to-end GPW SCF: the closed-shell loop converges for a minimal H2.
///
/// This exercises the whole GPW chain in one self-consistent loop — basis S/T,
/// GTH local PP, collocation, FFT-Poisson electrostatics (Gaussian core charge),
/// PADE XC, generalized eigensolve, density mixing. The assertion is that it
/// CONVERGES to a finite fixed point; exact CP2K bit-parity is the M1 milestone.
///
/// The `GpwSolverClass` suite (arch-refactor-04-engines) pins the GpwDft
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
using ATV::cpu::gpw::GpwDft;
using ATV::cpu::gpw::GpwResult;
using ATV::cpu::gpw::GridBasisFunction;
using ATV::cpu::gpw::GridPrimitive;

// Public face parity with PW: move-only value type (basics).
static_assert(!std::is_copy_constructible<GpwDft>::value,
              "GpwDft must not be copyable (owns workspace/result)");
static_assert(!std::is_copy_assignable<GpwDft>::value,
              "GpwDft must not be copy-assignable");
static_assert(std::is_move_constructible<GpwDft>::value,
              "GpwDft must be move-constructible (same as PW)");
static_assert(std::is_move_assignable<GpwDft>::value,
              "GpwDft must be move-assignable (same as PW)");

namespace {

GpwDft copy_gpw_test(const GpwDft& m) {
    GpwDft p;
    for (int d = 0; d < 3; ++d)
        p.cell[d] = m.cell[d];
    p.cutoff_ry = m.cutoff_ry;
    p.ngrids = m.ngrids;
    p.rel_cutoff = m.rel_cutoff;
    p.n_occ = m.n_occ;
    p.max_iter = m.max_iter;
    p.tol = m.tol;
    p.eps_default = m.eps_default;
    p.mix = m.mix;
    p.diis_depth = m.diis_depth;
    p.lindep_threshold = m.lindep_threshold;
    p.level_shift = m.level_shift;
    p.xc = m.xc;
    p.nonperiodic = m.nonperiodic;
    p.local_pp_on_grid = m.local_pp_on_grid;
    p.d3 = m.d3;
    p.d3bj = m.d3bj;
    p.use_ot = m.use_ot;
    return p;
}

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

GpwDft small_cfg(double box = 8.0) {
    GpwDft cfg;
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

    GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = box;
    cfg.cutoff_ry = 250.0;
    cfg.n_occ = 1; // 2 electrons, closed shell
    cfg.max_iter = 100;
    cfg.tol = 1e-7;
    cfg.mix = 0.5;

    GpwDft solver = small_cfg();
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

// ── GpwDft class contract (arch-refactor-04-engines Design §2) ──────────

// Independent cold solvers must be bit-identical in every energy channel
// [Hartree] and in their iteration count.
TEST(GpwSolverClass, IndependentColdRunsAreBitIdentical) {
    pin_single_thread();
    const auto atoms = small_h2();
    GpwDft cfg = small_cfg();

    GpwDft reference_solver = small_cfg();
    reference_solver.build(atoms);
    const GpwResult ref = reference_solver.run(atoms);
    ASSERT_TRUE(ref.converged) << "reference solver did not converge";

    GpwDft solver = small_cfg();
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
    GpwDft cfg = small_cfg();

    GpwDft cold_solver = small_cfg();
    cold_solver.build(atoms);
    const GpwResult cold = cold_solver.run(atoms);
    ASSERT_TRUE(cold.converged);

    GpwDft solver = small_cfg();
    solver.build(atoms);
    const GpwResult& r = solver.run(atoms, &cold.C);

    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.e_total, cold.e_total, 1e-12);
}

// Lifecycle: build() is one-time, run() is re-entrant — two cold runs on the
// same built solver give the identical result (geometry-dependent structures
// are rebuilt per run, same as PW).
TEST(GpwSolverClass, RunIsReentrantAfterSingleBuild) {
    pin_single_thread();
    const auto atoms = small_h2();

    GpwDft solver(small_cfg());
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
    GpwDft solver(small_cfg());
    EXPECT_THROW(solver.run(atoms), std::logic_error);
}

// Edge case: build() rejects n_occ < 1 (no-fallback rule).
TEST(GpwSolverClass, BuildRejectsNonPositiveNOcc) {
    const auto atoms = small_h2();
    GpwDft solver = small_cfg();
    solver.n_occ = 0;
    EXPECT_THROW(solver.build(atoms), std::invalid_argument);
}

// Edge case: build() rejects a periodic cell with a zero-length edge.
TEST(GpwSolverClass, BuildRejectsZeroCellEdgeWhenPeriodic) {
    const auto atoms = small_h2();
    GpwDft solver = small_cfg();
    solver.cell[1] = 0.0;
    solver.nonperiodic = false;
    EXPECT_THROW(solver.build(atoms), std::invalid_argument);
}

// D3 always draws from the in-tree tables (basis_data/dftd3_tables.h); an
// element outside their coverage THROWS at solve time rather than silently
// contributing zero dispersion (no-fallback rule).
TEST(GpwSolverClass, BuildAcceptsD3) {
    const auto atoms = small_h2();
    GpwDft solver = small_cfg();
    solver.d3 = true;
    EXPECT_NO_THROW(solver.build(atoms));
}

// k-gate: closed-shell multi-k accepts REL_CUTOFF multigrid (ngrids>1 with a
// positive rel_cutoff) — the gpw-multigrid-kpoints capability.
TEST(GpwSolverClass, BuildAcceptsKpointMultigridRks) {
    const auto atoms = small_h2();
    GpwDft solver = small_cfg();
    solver.k_mesh[0] = solver.k_mesh[1] = solver.k_mesh[2] = 2;
    solver.ngrids = 4;
    solver.rel_cutoff = 60.0;
    EXPECT_NO_THROW(solver.build(atoms));
}

// k-gate: multi-k multigrid without a positive rel_cutoff is rejected
// fail-fast (no silent single-grid fallback on the k path).
TEST(GpwSolverClass, BuildRejectsKpointMultigridWithoutRelCutoff) {
    const auto atoms = small_h2();
    GpwDft solver = small_cfg();
    solver.k_mesh[0] = 2;
    solver.ngrids = 4;
    solver.rel_cutoff = 0.0;
    EXPECT_THROW(solver.build(atoms), std::invalid_argument);
}

// k-gate: multi-k collinear UKS accepts REL_CUTOFF multigrid (ngrids>1 with a
// positive rel_cutoff) — the gpw-uks-multigrid-kpoints capability.
TEST(GpwSolverClass, BuildAcceptsKpointMultigridUks) {
    const auto atoms = small_h2();
    GpwDft solver = small_cfg();
    solver.k_mesh[0] = 2;
    solver.n_occ = 0;
    solver.n_alpha = 1;
    solver.n_beta = 1;
    solver.ngrids = 4;
    solver.rel_cutoff = 60.0;
    EXPECT_NO_THROW(solver.build(atoms));
}

// k-gate: nonlocal vdW is wired on the closed-shell multi-k RKS path (SCF
// V_nl in build_k_energy + post-SCF E_c^nl in fold_dispersion_k) — the
// gpw-vdw-kpoints capability.
TEST(GpwSolverClass, BuildAcceptsKpointNonlocalVdw) {
    const auto atoms = small_h2();
    GpwDft solver = small_cfg();
    solver.k_mesh[0] = 2;
    solver.vdw_df = true;
    EXPECT_NO_THROW(solver.build(atoms));
}

// k-gate: nonlocal vdW composes with REL_CUTOFF multigrid on the k path (V_nl
// on the finest grid before the per-level restriction, same as Γ).
TEST(GpwSolverClass, BuildAcceptsKpointMultigridNonlocalVdw) {
    const auto atoms = small_h2();
    GpwDft solver = small_cfg();
    solver.k_mesh[0] = 2;
    solver.rvv10 = true;
    solver.ngrids = 4;
    solver.rel_cutoff = 40.0;
    EXPECT_NO_THROW(solver.build(atoms));
}

// k-gate: multi-k UKS + nonlocal vdW stays rejected fail-fast (E_c^nl fold is
// closed-shell RKS only; UKS vdW is a separate queue pick).
TEST(GpwSolverClass, BuildRejectsKpointUksNonlocalVdw) {
    const auto atoms = small_h2();
    GpwDft solver = small_cfg();
    solver.k_mesh[0] = 2;
    solver.n_occ = 0;
    solver.n_alpha = 1;
    solver.n_beta = 1;
    solver.vdw_df = true;
    EXPECT_THROW(solver.build(atoms), std::invalid_argument);
}

// Γ gate: nonlocal vdW rides the REL_CUTOFF multigrid (V_nl on the finest
// grid before per-level restriction) — the gpw-vdw-multigrid capability.
TEST(GpwSolverClass, BuildAcceptsGammaMultigridNonlocalVdw) {
    const auto atoms = small_h2();
    GpwDft solver = small_cfg();
    solver.vdw_df = true;
    solver.ngrids = 4;
    solver.rel_cutoff = 40.0;
    EXPECT_NO_THROW(solver.build(atoms));
}

// Γ gate: UKS has no nonlocal-vdW fold (SCF V_nl and post-SCF E_c^nl live in
// the closed-shell RKS runner) — rejected instead of silently dropping E_c^nl.
TEST(GpwSolverClass, BuildRejectsUksNonlocalVdw) {
    const auto atoms = small_h2();
    GpwDft solver = small_cfg();
    solver.n_occ = 0;
    solver.n_alpha = 1;
    solver.n_beta = 1;
    solver.rvv10 = true;
    EXPECT_THROW(solver.build(atoms), std::invalid_argument);
}

// Edge case: build() rejects an empty atom list (Design §2: atoms/basis
// non-empty is part of the one-time fail-fast validation).
TEST(GpwSolverClass, BuildRejectsEmptyAtoms) {
    GpwDft solver(small_cfg());
    EXPECT_THROW(solver.build(std::vector<GpwAtom>{}), std::invalid_argument);
}

// Basics: config() echoes the constructor argument; last_result() aliases the
// owned result buffer returned by run() and starts out unconverged.
TEST(GpwSolverClass, AccessorsExposeConfigAndLastResult) {
    pin_single_thread();
    const auto atoms = small_h2();
    GpwDft cfg = small_cfg();

    GpwDft solver = small_cfg();
    EXPECT_DOUBLE_EQ(solver.cutoff_ry, cfg.cutoff_ry);
    EXPECT_EQ(solver.n_occ, cfg.n_occ);
    EXPECT_DOUBLE_EQ(solver.cell[0], cfg.cell[0]);
    EXPECT_DOUBLE_EQ(solver.cell[1], cfg.cell[1]);
    EXPECT_DOUBLE_EQ(solver.cell[2], cfg.cell[2]);
    EXPECT_DOUBLE_EQ(solver.tol, cfg.tol);
    EXPECT_FALSE(solver.last_result().converged); // owned default before run()

    solver.build(atoms);
    const GpwResult& r = solver.run(atoms);
    EXPECT_EQ(&solver.last_result(), &r); // same owned buffer
    EXPECT_TRUE(solver.last_result().converged);
}

// Lifecycle: move construction transfers the built state — the moved-to
// solver runs without a second build() (move-only face, same as PW).
TEST(GpwSolverClass, MoveConstructionPreservesBuiltState) {
    pin_single_thread();
    const auto atoms = small_h2();

    GpwDft a(small_cfg());
    a.build(atoms);
    GpwDft b(std::move(a));

    const GpwResult& r = b.run(atoms);
    EXPECT_TRUE(r.converged);
    EXPECT_TRUE(std::isfinite(r.e_total));
}

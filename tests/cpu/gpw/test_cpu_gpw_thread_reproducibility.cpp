/// @file tests/cpu/gpw/test_cpu_gpw_thread_reproducibility.cpp
/// @brief The GPW grid kernels must return BITWISE-identical results at every
///        OpenMP thread count.
///
/// This is a correctness gate, not a numerical-quality one. Floating-point
/// addition is not associative, so a reduction whose split across threads is
/// decided by the OpenMP runtime returns a different sum on a different thread
/// count — and, with a dynamic schedule, on a different *run at the same* thread
/// count. Those kernels feed the Kohn-Sham matrix, so the SCF sees a slightly
/// different Hamiltonian each time. That is not harmless last-digit noise: the
/// OT minimizer's Armijo line search compares two total energies whose true
/// difference falls below the noise near convergence, so the comparison stops
/// being a decision about the physics. Measured on the ot-minimizer-water
/// regression before this gate existed, the same binary on the same input took
/// 30, 31, 32, 34 or 45 SCF iterations at 8 threads, and one run in eight
/// exhausted the 200-iteration budget without converging at all.
///
/// Each test therefore asserts `==` on doubles deliberately: a tolerance here
/// would pass exactly the code this gate exists to reject.
///
/// The 1-thread result is the reference; 2/4/8/16 threads must reproduce it
/// exactly. Repeating at the widest count additionally catches a split that
/// depends on the *schedule* rather than the count.

#include <cmath>
#include <complex>
#include <vector>

#include <gtest/gtest.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/gpw/internal/multigrid.h"
#include "atomiverse/cpu/gpw/internal/xc_grid.h"

using ATV::cpu::gpw::build_compact_basis_grid;
using ATV::cpu::gpw::build_products;
using ATV::cpu::gpw::CompactBasisGrid;
using ATV::cpu::gpw::GridBasisFunction;
using ATV::cpu::gpw::GridPrimitive;
using ATV::cpu::gpw::make_multigrid;
using ATV::cpu::gpw::MultiGrid;
using ATV::cpu::gpw::ProductSet;
using ATV::cpu::gpw::UniformGrid;

namespace {

/// Thread counts every kernel below must agree on, bitwise.
const std::vector<int> kWidths = {1, 2, 4, 8, 16};

/// Set the OpenMP width for the next kernel call. Returns false when the build
/// has no OpenMP, in which case there is nothing to compare.
bool set_width(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n);
    return true;
#else
    (void)n;
    return false;
#endif
}

/// A 4x4x4 lattice of narrow s-Gaussians in a 10 Bohr cube: 64 basis functions,
/// enough that a `schedule(dynamic, 8)` loop splits across threads, with tails
/// short enough that only a handful overlap at any grid point.
std::vector<GridBasisFunction> lattice_basis() {
    std::vector<GridBasisFunction> basis;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k) {
                GridPrimitive p;
                p.alpha = 4.0;
                p.coeff = 1.0;
                p.l[0] = p.l[1] = p.l[2] = 0;
                p.center[0] = 1.25 + 2.5 * i;
                p.center[1] = 1.25 + 2.5 * j;
                p.center[2] = 1.25 + 2.5 * k;
                basis.push_back(GridBasisFunction{p});
            }
    return basis;
}

UniformGrid cube_grid(long n) {
    UniformGrid g;
    g.n[0] = g.n[1] = g.n[2] = n;
    g.L[0] = g.L[1] = g.L[2] = 10.0;
    g.origin[0] = g.origin[1] = g.origin[2] = 0.0;
    return g;
}

/// A reproducible, structure-free density matrix — no symmetry an accidental
/// cancellation could hide behind.
std::vector<double> pseudo_density(int nb) {
    std::vector<double> P(static_cast<std::size_t>(nb) * nb);
    for (int u = 0; u < nb; ++u)
        for (int v = 0; v < nb; ++v)
            P[static_cast<std::size_t>(u) * nb + v] =
                std::sin(0.7 * u + 1.3 * v) / (1.0 + 0.1 * (u + v));
    return P;
}

/// A reproducible grid potential/density field.
std::vector<double> pseudo_field(const UniformGrid& g) {
    const long ng = g.npts();
    std::vector<double> v(static_cast<std::size_t>(ng));
    for (long i = 0; i < ng; ++i)
        v[static_cast<std::size_t>(i)] =
            0.05 * std::sin(0.013 * static_cast<double>(i))
            + 0.02 * std::cos(0.0031 * static_cast<double>(i));
    return v;
}

} // namespace

// ── The compact basis grid: its CSR LAYOUT must not depend on the thread count.
// The layout fixes the summation order of every density and KS contraction
// built on it, so a thread-dependent layout is a thread-dependent Hamiltonian
// even when every kernel reading it is itself deterministic.
TEST(GpwThreadReproducibility, CompactBasisGridLayoutIsThreadIndependent) {
    if (!set_width(1))
        GTEST_SKIP() << "no OpenMP in this build";
    const UniformGrid grid = cube_grid(32);
    const std::vector<GridBasisFunction> basis = lattice_basis();

    const CompactBasisGrid ref =
        build_compact_basis_grid(grid, basis, /*periodic*/ true);
    ASSERT_FALSE(ref.fu.empty());

    for (int w : kWidths) {
        set_width(w);
        const CompactBasisGrid got =
            build_compact_basis_grid(grid, basis, /*periodic*/ true);
        ASSERT_EQ(got.gidx, ref.gidx) << "thread count " << w;
        ASSERT_EQ(got.off, ref.off) << "thread count " << w;
        ASSERT_EQ(got.fu, ref.fu) << "thread count " << w;
        ASSERT_EQ(got.fv, ref.fv) << "thread count " << w;
    }
    set_width(1);
}

// ── ρ(r) from the compact grid.
TEST(GpwThreadReproducibility, CompactCollocateDensityIsThreadIndependent) {
    if (!set_width(1))
        GTEST_SKIP() << "no OpenMP in this build";
    const UniformGrid grid = cube_grid(32);
    const std::vector<GridBasisFunction> basis = lattice_basis();
    const int nb = static_cast<int>(basis.size());
    const std::vector<double> P = pseudo_density(nb);

    const CompactBasisGrid cbg =
        build_compact_basis_grid(grid, basis, /*periodic*/ true);
    const std::vector<double> ref = collocate_density(cbg, nb, P);

    for (int w : kWidths) {
        set_width(w);
        EXPECT_EQ(collocate_density(cbg, nb, P), ref) << "thread count " << w;
    }
    set_width(1);
}

// ── The KS grid block: the reduction that carried the OT defect.
TEST(GpwThreadReproducibility, CompactIntegratePotentialIsThreadIndependent) {
    if (!set_width(1))
        GTEST_SKIP() << "no OpenMP in this build";
    const UniformGrid grid = cube_grid(32);
    const std::vector<GridBasisFunction> basis = lattice_basis();
    const int nb = static_cast<int>(basis.size());
    const std::vector<double> V_r = pseudo_field(grid);

    const CompactBasisGrid cbg =
        build_compact_basis_grid(grid, basis, /*periodic*/ true);
    const std::vector<double> ref = integrate_potential(cbg, nb, V_r);

    for (int w : kWidths) {
        set_width(w);
        EXPECT_EQ(integrate_potential(cbg, nb, V_r), ref)
            << "thread count " << w;
    }
    // Same width twice: catches a split that follows the schedule, not the count.
    set_width(16);
    for (int rep = 0; rep < 4; ++rep)
        EXPECT_EQ(integrate_potential(cbg, nb, V_r), ref) << "repeat " << rep;
    set_width(1);
}

// ── E_xc: an OpenMP `reduction(+ : e)` merges its partials in thread-completion
// order, so this differed run to run even at one fixed width.
TEST(GpwThreadReproducibility, XcGridPadeEnergyIsThreadIndependent) {
    if (!set_width(1))
        GTEST_SKIP() << "no OpenMP in this build";
    const UniformGrid grid = cube_grid(48);
    const long ng = grid.npts();
    std::vector<double> rho(static_cast<std::size_t>(ng));
    for (long i = 0; i < ng; ++i)
        rho[static_cast<std::size_t>(i)] =
            0.3 + 0.29 * std::sin(0.0007 * static_cast<double>(i));

    const double ref = ATV::cpu::gpw::xc_grid_pade(grid, rho).e_xc;
    ASSERT_NE(ref, 0.0);

    for (int w : kWidths) {
        set_width(w);
        EXPECT_EQ(ATV::cpu::gpw::xc_grid_pade(grid, rho).e_xc, ref)
            << "thread count " << w;
    }
    set_width(16);
    for (int rep = 0; rep < 4; ++rep)
        EXPECT_EQ(ATV::cpu::gpw::xc_grid_pade(grid, rho).e_xc, ref)
            << "repeat " << rep;
    set_width(1);
}

// ── Spin-resolved E_xc (UKS / PADE LSD): same ChunkedRange reduction as the
// closed-shell path; this gate closes the dual-density gap that reg_cpmd-uks
// hit under threading (~3e-5) before 0f78ff6d. XcOnGridSpin exposes only e_xc
// (no separate e_x / e_c), so the energy compare is on that single double.
TEST(GpwThreadReproducibility, XcGridPadeLsdEnergyIsThreadIndependent) {
    if (!set_width(1))
        GTEST_SKIP() << "no OpenMP in this build";
    const UniformGrid grid = cube_grid(48);
    const long ng = grid.npts();
    // Unequal spin channels so the path is not a collapse of xc_grid_pade.
    std::vector<double> rho_a(static_cast<std::size_t>(ng));
    std::vector<double> rho_b(static_cast<std::size_t>(ng));
    for (long i = 0; i < ng; ++i) {
        const double t = static_cast<double>(i);
        rho_a[static_cast<std::size_t>(i)] = 0.22 + 0.18 * std::sin(0.0007 * t);
        rho_b[static_cast<std::size_t>(i)] = 0.11 + 0.09 * std::cos(0.0011 * t);
    }

    const double ref = ATV::cpu::gpw::xc_grid_pade_lsd(grid, rho_a, rho_b).e_xc;
    ASSERT_NE(ref, 0.0);

    for (int w : kWidths) {
        set_width(w);
        EXPECT_EQ(ATV::cpu::gpw::xc_grid_pade_lsd(grid, rho_a, rho_b).e_xc, ref)
            << "thread count " << w;
    }
    set_width(16);
    for (int rep = 0; rep < 4; ++rep)
        EXPECT_EQ(ATV::cpu::gpw::xc_grid_pade_lsd(grid, rho_a, rho_b).e_xc, ref)
            << "repeat " << rep;
    set_width(1);
}

// ── The multigrid product path: the product LIST order is itself a reduction
// order, and both the density scatter and the KS integration reduce over it.
TEST(GpwThreadReproducibility, ProductCollocationIsThreadIndependent) {
    if (!set_width(1))
        GTEST_SKIP() << "no OpenMP in this build";
    const double L[3] = {10.0, 10.0, 10.0};
    const MultiGrid mg = make_multigrid(L, /*cutoff_ry*/ 60.0, /*ngrids*/ 2,
                                        /*rel_cutoff*/ 20.0);
    const std::vector<GridBasisFunction> basis = lattice_basis();
    const int nb = static_cast<int>(basis.size());
    const std::vector<double> P = pseudo_density(nb);
    const std::vector<double> V_l = pseudo_field(mg.levels[0]);

    const ProductSet ref_ps =
        build_products(mg, basis, L, /*rel_cutoff*/ 20.0, /*eps*/ 1e-10,
                       /*periodic*/ true);
    ASSERT_FALSE(ref_ps.by_level[0].empty());
    const std::vector<double> ref_rho = collocate_products(
        mg.levels[0], ref_ps.by_level[0], ref_ps.tasks_by_level[0],
        ref_ps.prims, ref_ps.contribs, nb, P, 1e-10);
    const std::vector<double> ref_ks = integrate_products(
        mg.levels[0], ref_ps.by_level[0], ref_ps.tasks_by_level[0],
        ref_ps.prims, ref_ps.contribs, nb, V_l, 1e-10);

    for (int w : kWidths) {
        set_width(w);
        const ProductSet ps =
            build_products(mg, basis, L, 20.0, 1e-10, /*periodic*/ true);
        ASSERT_EQ(ps.by_level[0].size(), ref_ps.by_level[0].size())
            << "thread count " << w;
        EXPECT_EQ(collocate_products(mg.levels[0], ps.by_level[0],
                                     ps.tasks_by_level[0], ps.prims,
                                     ps.contribs, nb, P, 1e-10),
                  ref_rho)
            << "thread count " << w;
        EXPECT_EQ(integrate_products(mg.levels[0], ps.by_level[0],
                                     ps.tasks_by_level[0], ps.prims,
                                     ps.contribs, nb, V_l, 1e-10),
                  ref_ks)
            << "thread count " << w;
    }
    set_width(1);
}

// ── The multi-k KS block (the GW / k-point path).
TEST(GpwThreadReproducibility, IntegratePotentialKIsThreadIndependent) {
    if (!set_width(1))
        GTEST_SKIP() << "no OpenMP in this build";
    const UniformGrid grid = cube_grid(24);
    const std::vector<GridBasisFunction> basis = lattice_basis();
    std::vector<ATV::cpu::gpw::internal::KPoint> kpts(1);
    kpts[0].frac[0] = 0.25;
    kpts[0].frac[1] = 0.0;
    kpts[0].frac[2] = 0.0;
    kpts[0].weight = 1.0;
    const std::vector<double> V_r = pseudo_field(grid);

    const ATV::cpu::gpw::internal::KpointBasisGrid kbg =
        ATV::cpu::gpw::internal::build_kpoint_basis_grid(grid, basis, kpts,
                                                         1e-10);
    const ATV::cpu::gpw::internal::CplxMatrix ref =
        ATV::cpu::gpw::internal::integrate_potential_k(kbg, 0, V_r);

    for (int w : kWidths) {
        set_width(w);
        const ATV::cpu::gpw::internal::CplxMatrix got =
            ATV::cpu::gpw::internal::integrate_potential_k(kbg, 0, V_r);
        ASSERT_EQ(got.shape(0), ref.shape(0)) << "thread count " << w;
        for (std::size_t u = 0; u < ref.shape(0); ++u)
            for (std::size_t t = 0; t < ref.shape(1); ++t)
                ASSERT_EQ(got(u, t), ref(u, t))
                    << "thread count " << w << " at (" << u << "," << t << ")";
    }
    set_width(1);
}

/// @file tests/cpu/gpw/test_cpu_gpw_periodic_overlap.cpp
/// @brief The Γ-point overlap matrix must be POSITIVE DEFINITE at every cell size.
///
/// S^Γ_μν = Σ_T ⟨χ_μ(0) | χ_ν(T)⟩ is the overlap of Bloch functions at k=0. It is a
/// Gram matrix by construction, so every eigenvalue is > 0 — no matter how small the
/// cell gets relative to the basis reach. A negative eigenvalue means the lattice sum
/// is wrong, not that the basis is "ill-conditioned".
///
/// This is the minimal model system for that invariant: ONE H atom with
/// DZVP-MOLOPT-SR-GTH (5 AOs) in a cubic cell, shrunk until the cell edge is far
/// below the basis reach (~13.6 Å), which is the regime the diffuse MOLOPT functions
/// live in.
///
/// Reference (PySCF 2.13.1, pbc.gto.Cell, basis gth-dzvp-molopt-sr, pseudo gth-pbe,
/// precision 1e-8, `cell.pbc_intor('int1e_ovlp')`; rcut = 25.69 Bohr at every L, i.e.
/// the reference sums many images and stays PSD throughout). Compared on the
/// UNIT-DIAGONAL overlap S̃_ij = S_ij/sqrt(S_ii·S_jj), because the two codes use
/// different basis normalizations — the GpwAtom constructor folds N(α,L) into the coefficients
/// and does not normalize the contraction, so our raw S has S_ii ≠ 1. That is a
/// congruence S = D·S̃·D: it changes the eigenvalues but not the span and not the
/// positive-definiteness, so S̃ is the normalization-invariant thing to pin.
///
///   L [Å]   s_min(S̃)     s_max(S̃)
///   12.0    8.983588e-01 1.1016
///    8.0    8.984236e-01 1.1016
///    5.0    9.251716e-01 1.0748
///    4.0    9.770366e-01 1.0230
///    3.0    6.719470e-01 1.3281
///
/// Unit-scale: builds S only, no SCF.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <tuple>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/one_electron.h"
#include "atomiverse/cpu/gpw/internal/periodic.h"
#include "atomiverse/cpu/linalg/matrix.h"

using namespace ATV::cpu::gpw;
using ATV::cpu::linalg::RealMatrix;

namespace {

constexpr double kAng = 1.8897259885789; // Å -> Bohr
constexpr double kEps = 1e-8;            // GpwDft eps default

/// Build the Γ-point overlap for H atoms at the given fractional positions in a
/// cubic cell of edge L [Å], exactly as the solver does.
RealMatrix gamma_overlap_h(double L_ang,
                           const std::vector<std::array<double, 3>>& frac) {
    const double cellL = L_ang * kAng;
    std::vector<ATV::cpu::gpw::internal::GpwAtom> atoms;
    for (const auto& f : frac) {
        const double p[3] = {f[0] * cellL, f[1] * cellL, f[2] * cellL};
        atoms.push_back(ATV::cpu::gpw::internal::GpwAtom(
            p, GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
            GthPotential(molrs::Element::H, "GTH-PBE-q1"), 1));
    }
    std::vector<GridBasisFunction> basis;
    for (const auto& a : atoms)
        for (const auto& bf : a.basis)
            basis.push_back(bf);
    const int nb = static_cast<int>(basis.size());

    const double cell[3] = {cellL, cellL, cellL};
    std::vector<double> radii(static_cast<std::size_t>(nb));
    double rmax = 0.0;
    for (int u = 0; u < nb; ++u) {
        radii[static_cast<std::size_t>(u)] =
            basis_function_radius(basis[u], kEps);
        rmax = std::max(rmax, radii[static_cast<std::size_t>(u)]);
    }
    int mimg[3];
    pair_image_counts(cell, /*periodic=*/true, 2.0 * rmax, mimg);

    RealMatrix S = xt::zeros<double>({nb, nb});
    for (int i = 0; i < nb; ++i)
        for (int j = 0; j < nb; ++j) {
            const double reach = radii[static_cast<std::size_t>(i)]
                                 + radii[static_cast<std::size_t>(j)];
            S(i, j) = overlap(basis[i], basis[j], cell, mimg, reach);
        }
    return S;
}

/// One atom at the origin — the degenerate case where R_A = R_B.
RealMatrix gamma_overlap_single_h(double L_ang) {
    return gamma_overlap_h(L_ang, {{{0.0, 0.0, 0.0}}});
}

} // namespace

// The invariant. A Gram matrix cannot have a negative eigenvalue at ANY cell size.
TEST(GpwPeriodicOverlap, SingleHydrogenStaysPositiveDefinite) {
    for (double L : {12.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0}) {
        const RealMatrix S = gamma_overlap_single_h(L);
        const auto w = std::get<0>(xt::linalg::eigh(S));
        const double s_min = w(0);
        const double s_max = w(w.size() - 1);
        std::printf("[S^gamma] L=%5.1f A  s_min=%14.6e  s_max=%9.4f\n", L,
                    s_min, s_max);
        EXPECT_GT(s_min, 0.0)
            << "the Gamma-point overlap is NOT positive definite at L=" << L
            << " A — the lattice sum is wrong (an overlap matrix cannot have a "
               "negative eigenvalue)";
    }
}

// Pinned against PySCF on the unit-diagonal overlap (see the header for provenance).
// This is the check that the LATTICE SUM itself is right, independent of how either
// code normalizes its basis functions.
TEST(GpwPeriodicOverlap, SingleHydrogenMatchesReferenceSpectrum) {
    struct Ref {
        double L, s_min, s_max;
    };
    const Ref refs[] = {{12.0, 8.983588e-01, 1.1016},
                        {8.0, 8.984236e-01, 1.1016},
                        {5.0, 9.251716e-01, 1.0748},
                        {4.0, 9.770366e-01, 1.0230},
                        {3.0, 6.719470e-01, 1.3281}};
    for (const Ref& r : refs) {
        RealMatrix S = gamma_overlap_single_h(r.L);
        const int n = static_cast<int>(S.shape(0));
        std::vector<double> d(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            d[static_cast<std::size_t>(i)] = 1.0 / std::sqrt(S(i, i));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                S(i, j) *= d[static_cast<std::size_t>(i)]
                           * d[static_cast<std::size_t>(j)];

        const auto w = std::get<0>(xt::linalg::eigh(S));
        EXPECT_NEAR(w(0), r.s_min, 1e-5) << "s_min(S~) at L=" << r.L;
        EXPECT_NEAR(w(w.size() - 1), r.s_max, 1e-4) << "s_max(S~) at L=" << r.L;
    }
}

// The regression that the single-atom case CANNOT see.
//
// `image_counts(L, rcut)` covers |T| <= rcut. The pair sum screens on
// |R_A - R_B - T| <= reach, so it needs |T| <= |R_A - R_B| + reach — and two atoms
// can sit a full cell edge apart. With one atom at the origin R_A = R_B, the extra
// shell is empty and the missing images are invisible; put the second atom in the
// opposite corner and the truncated sum stops being a Gram matrix.
//
// This is the shape of the bug that made a large elongated-slab overlap (13726 AOs) come out at
// s_min = -1.92 (unit-diagonal) where the reference is +1.1e-13.
// Multi-atom overlap vs PySCF — the check the single-atom case CANNOT make. On one
// atom the p functions are orthogonal to s and to each other by symmetry, so the
// intra-atomic spectrum only exercises the s contraction; every INTER-atomic block
// (s-p, p-p) is untested there. A 3x3 planar H grid at spacing d exercises all of it,
// and its overlap is near-singular at small d, which is where errors show up first.
//
// Reference (PySCF 2.13.1, same cell/basis/pseudo, unit-diagonal overlap, s_min):
//   d [Å]   L=14 Å        L=24 Å
//   0.4     2.57149e-04   2.57149e-04
//   0.6     1.16314e-03   1.16314e-03
//   0.8     1.07507e-02   1.07507e-02
//   1.2     6.21384e-02   6.21384e-02
TEST(GpwPeriodicOverlap, DenseHydrogenGridMatchesReference) {
    struct Ref {
        double L, d, s_min;
    };
    const Ref refs[] = {{14.0, 0.4, 2.57149e-04}, {14.0, 0.6, 1.16314e-03},
                        {14.0, 0.8, 1.07507e-02}, {14.0, 1.2, 6.21384e-02},
                        {24.0, 0.4, 2.57149e-04}, {24.0, 1.2, 6.21384e-02}};
    for (const Ref& r : refs) {
        std::vector<std::array<double, 3>> frac;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                frac.push_back(
                    {{0.5 + i * r.d / r.L, 0.5 + j * r.d / r.L, 0.5}});

        RealMatrix S = gamma_overlap_h(r.L, frac);
        const int n = static_cast<int>(S.shape(0));
        ASSERT_EQ(n, 45); // 9 atoms x 5 AOs — same as PySCF's nao
        std::vector<double> dg(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            dg[static_cast<std::size_t>(i)] = 1.0 / std::sqrt(S(i, i));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                S(i, j) *= dg[static_cast<std::size_t>(i)]
                           * dg[static_cast<std::size_t>(j)];

        const double s_min = std::get<0>(xt::linalg::eigh(S))(0);
        EXPECT_GT(s_min, 0.0) << "L=" << r.L << " d=" << r.d;
        EXPECT_NEAR(s_min, r.s_min, 1e-8 + 1e-4 * r.s_min)
            << "inter-atomic overlap disagrees with the reference at L=" << r.L
            << " d=" << r.d;
    }
}

// The defect itself, as pure arithmetic: a PAIR sum screens on |R_A - R_B - T| <= reach,
// so it needs |T| <= |R_A - R_B| + reach — and two centers can be a full cell edge
// apart. `image_counts` only covers |T| <= rcut and is therefore one shell short for
// every axis. This is what made the large-slab overlap non-Gram.
TEST(GpwPeriodicOverlap, PairImageCountCoversFullCenterSeparation) {
    const double L[3] = {24.808, 19.997, 88.919}; // elongated slab cell [Bohr]
    const double rcut = 33.3;                     // ~2*rmax for that basis

    int m_old[3], m_pair[3];
    image_counts(L, /*periodic=*/true, rcut, m_old);
    pair_image_counts(L, /*periodic=*/true, rcut, m_pair);

    for (int d = 0; d < 3; ++d) {
        // A pair separated by a full cell edge needs images out to L[d] + rcut.
        const int needed =
            static_cast<int>(std::ceil((L[d] + rcut) / L[d] - 1e-12));
        EXPECT_GE(m_pair[d], needed) << "axis " << d;
        EXPECT_LT(m_old[d], needed)
            << "axis " << d
            << ": image_counts is supposed to be the one that falls short here";
    }
}

TEST(GpwPeriodicOverlap, AtomsAtOppositeCornersStayPositiveDefinite) {
    // Cell sizes matter, and NOT in the direction you would guess: the bug needs the
    // image count to be SMALL (m = 1-2), i.e. L comparable to 2·rmax (~26 Bohr ≈ 14 Å
    // for H). In a *tight* cell m is already large and the missing shell is empty —
    // which is why the small-cell sweep above stays green either way.
    const std::vector<std::array<double, 3>> corners = {{{0.0, 0.0, 0.0}},
                                                        {{0.5, 0.5, 0.5}},
                                                        {{0.0, 0.5, 0.5}},
                                                        {{0.5, 0.0, 0.5}}};
    for (double L : {14.0, 16.0, 20.0, 24.0}) {
        const RealMatrix S = gamma_overlap_h(L, corners);
        const auto w = std::get<0>(xt::linalg::eigh(S));
        std::printf("[S^gamma corners] L=%5.1f A  s_min=%14.6e  s_max=%9.4f\n",
                    L, w(0), w(w.size() - 1));
        EXPECT_GT(w(0), 0.0)
            << "the Gamma-point overlap is NOT positive definite at L=" << L
            << " A with atoms spread across the cell — the pair lattice sum is "
               "missing "
               "its outermost image shell (see pair_image_counts)";
    }
}

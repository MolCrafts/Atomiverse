/// @file cpu/scf/rys_roots_iterative.h
/// @brief Iterative Rys-quadrature root finder for nroots 6 and 7
///        (f-shell electron-repulsion integrals).
///
/// The closed-form rys_root1..5 fits (rys_roots.cpp) cover nroots <= 5,
/// sufficient for d-shells (Ltot <= 8). f-shells push Ltot to 12, and a
/// derivative of an (f,f|f,f) quartet to Ltot = 13, both of which need a
/// 7-point Rys quadrature. libcint does NOT extend its closed-form
/// Chebyshev fits to nroots 6-7 — high-order fits degrade badly in the
/// intermediate-T regime — and instead switches to an iterative scheme.
/// This unit ports that scheme.
///
/// Algorithm (libcint `CINTrys_schmidt`, src/rys_roots.c, BSD-2):
///   1. Compute the Boys-like "gamma incomplete" moments
///        fmt_ints[m] = F_m(x),  m = 0 .. 2*nroots
///      via `gamma_inc_like` — a power-series expansion for small T and
///      the erf-form plus upward recurrence for large T.
///   2. Build, by modified Gram-Schmidt orthogonalization over those
///      moments, the coefficient matrix `cs` of the discrete orthogonal
///      (Rys) polynomials up to degree nroots (`R_dsmit`).
///   3. Extract the nroots roots of the degree-nroots Rys polynomial as
///      the eigenvalues of its companion matrix via a Hessenberg-QR
///      iteration (`_CINT_polynomial_roots` / `_hessenberg_qr`), with an
///      `R_dnode` Newton/bisection polish fallback.
///   4. Convert each polynomial root `r` to the libcint Rys convention
///        u = r / (1 - r),
///      and recover the weight from the orthonormal-polynomial sum
///        w = 1 / (1/F_0 + sum_{j>=1} P_j(r)^2).
///
/// This is the modified-Gram-Schmidt branch of libcint's nroots-6/7
/// `segment_solve(CINTrys_jacobi, CINTrys_schmidt)`. It is numerically
/// robust across the full small/intermediate/large-x range for the
/// 6- and 7-root orders required by f-shell ERIs (validated to <= 1e-10
/// against libcint and against the Rys quadrature moment identity in
/// tests/cpu/scf/test_rys_roots.cpp).
///
/// Upstream attribution:
///   libcint: https://github.com/sunqm/libcint (BSD 2-clause)
///   Copyright (c) 2012 Qiming Sun.
///   gamma_inc_like ported from libcint src/fmt.c; the Schmidt
///   orthogonalization and companion-matrix root extraction from
///   src/rys_roots.c and src/find_roots.c.
///
/// @see rys_roots.cpp for the closed-form nroots-1..5 dispatcher.
#pragma once

namespace ATV {
namespace cpu {
namespace scf {
namespace native {

/// @brief Largest Rys order handled by the iterative finder.
/// @details f-shell ERIs need 7 roots ((f,f|f,f): Ltot = 12; its first
///   derivative: Ltot = 13 — both map to N = floor(Ltot/2)+1 = 7).
constexpr int kIterativeRysMaxRoots = 7;

/// @brief Compute Rys quadrature roots and weights for nroots 6 or 7
///        via the iterative modified-Gram-Schmidt scheme.
///
/// @details Drop-in replacement for the closed-form `rys_root1..5`
///   branches when the Rys order exceeds 5. The output convention is
///   identical to `native::rys_roots`: `u[i]` stores the transformed
///   root t_i^2 / (1 - t_i^2) (libcint convention), `w[i]` the
///   associated quadrature weight. Callers must size `u` and `w` for at
///   least `nroots` entries.
///
/// @param nroots Number of Rys roots; must be 6 or 7. Other values are
///        a programming error (the closed-form path covers 1..5).
/// @param x      Boys parameter T = rho * |P-Q|^2, dimensionless,
///        non-negative. Any T >= 0 is accepted; the SMALLX / LARGEX
///        analytic shortcuts in `rys_roots()` are applied before this
///        function is reached, so it sees only the intermediate range.
/// @param u      Output array [nroots] — transformed roots
///        t_i^2 / (1 - t_i^2).
/// @param w      Output array [nroots] — quadrature weights.
/// @return 0 on success; non-zero if the Schmidt orthogonalization or
///         the companion-matrix eigensolve failed to converge (the
///         caller treats a non-zero return as fatal, mirroring libcint).
int rys_roots_iterative(int nroots, double x, double* u, double* w);

} // namespace native
} // namespace scf
} // namespace cpu
} // namespace ATV

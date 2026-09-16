/// @file cpu/scf/rys_eri.h
/// @brief Native Rys-quadrature electron repulsion integrals.
///
/// This engine replicates the algorithm used by libcint / PySCF for
/// evaluating two-electron integrals (ab|cd) over contracted Cartesian
/// Gaussians. It is algorithmically equivalent to libcint (same Rys
/// roots, same 2D-VRR, same HRR) and is the production ERI path in
/// native_integrals.cpp for all s/p/d/f shells (l <= 3).
///
/// Data flow per shell quartet (A B | C D):
///   1. For each primitive-pair quartet (p,q,r,s):
///        a. Compute bra/ket Gaussian-product centres P, Q and the Boys
///           parameter x = ρ·|P-Q|².
///        b. Call native::rys_roots(nroots, x, u, w) — closed-form
///           piecewise polynomial approximation ported from libcint.
///        c. Per root, form the 9 Rys coefficients b00, b10, b01,
///           c00{xyz}, c0p{xyz}.
///        d. Build the 2D Rys intermediate gx[i][k], gy[i][k], gz[i][k]
///           via the standard Obara–Saika-style recurrence along the
///           bra (i) and ket (k) axes — see `CINTg0_2e_2d` in libcint.
///   2. Apply the 2D→4D HRR to shift angular momentum from
///      (li+lj, 0 | lk+ll, 0) → (li, lj | lk, ll). HRR is a pure
///      geometric transform and does not depend on the primitive
///      quartet.
///   3. For each Cartesian component tuple (a, b, c, d), assemble the
///      contracted ERI as
///          ERI(a,b,c,d) = Σ_root gx(...) · gy(...) · gz(...)
///      and scatter into `out` with the primitive coefficient product.
///
/// Public API: one function, `rys_eri_block`. The output buffer layout
/// is identical to `native::eri_block`:
///
///     out[ia*nb*nc*nd + ib*nc*nd + ic*nd + id] += (ia ib | ic id)
///
/// @see rys_roots.h for the ported root/weight generator.
/// @see native_integrals.h for the ShellBlock data structure.
#pragma once

#include "atomiverse/cpu/scf/cart.h"
#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace native {

/// @brief Maximum supported angular momentum per shell (inclusive).
/// l = 0 (s), 1 (p), 2 (d), 3 (f), 4 (g). g shells serve the 2-center /
/// 3-center density-fitting paths only (def2 JKFIT auxiliary basis);
/// the orbital basis stays capped at f (`kMaxLOrbital`), so f-shell
/// quartets remain the heaviest 4-center case: Ltot = 12 ((ff|ff)) and
/// Ltot = 13 for the first derivative of an (ff|ff) quartet, both
/// served by the 7-point iterative root finder (rys_roots_iterative).
constexpr int kRysMaxL = 4;

/// @brief Maximum Rys roots supported. nroots = floor(Ltot/2) + 1; the
/// largest case is the +1-derivative of an (ff|ff) quartet — Ltot = 13,
/// nroots = 7. Quartets demanding more roots (only reachable by
/// misusing g shells in a 4-center quartet) throw std::runtime_error.
constexpr int kRysMaxRoots = 7;

// Root-budget proof for the density-fitting g extension: the heaviest
// 3-center case is (ff|g·1) — Ltot = 3+3+4 = 10 → 6 roots; the heaviest
// 2-center case is (g·1|g·1) — Ltot = 8 → 5 roots; the heaviest 4-center
// orbital case stays (ff|ff)+deriv — Ltot = 13 → 7 roots. All within
// kRysMaxRoots; no new root order is required.
static_assert((2 * 3 + kMaxL) / 2 + 1 <= kRysMaxRoots,
              "3-center (ff|g) exceeds the Rys root budget");
static_assert((2 * kMaxL) / 2 + 1 <= kRysMaxRoots,
              "2-center (g|g) exceeds the Rys root budget");
static_assert((4 * 3 + 1) / 2 + 1 <= kRysMaxRoots,
              "4-center (ff|ff) derivative exceeds the Rys root budget");

/// @brief Compute the full Cartesian ERI block (AB|CD) using Rys
///        quadrature.
///
/// The output buffer is accumulated, not overwritten — callers can
/// pre-zero once and build many block contributions into the same
/// buffer. Input shells may have any angular momentum up to @ref
/// kRysMaxL and any contraction length supported by `ShellBlock`
/// (alpha/coeff inline arrays).
///
/// The function is algorithmically equivalent to libcint's
/// `cint2e_cart` for point-nucleus, non-range-separated Coulomb.
///
/// @param A,B Shell blocks for the bra (electron 1), in that order.
/// @param C,D Shell blocks for the ket (electron 2), in that order.
/// @param out Pre-zeroed flat buffer of size
///            A.n_cart·B.n_cart·C.n_cart·D.n_cart.
void rys_eri_block(const ShellBlock& A, const ShellBlock& B,
                   const ShellBlock& C, const ShellBlock& D, double* out);

/// @brief Block-level erf-attenuated Coulomb ERI (AB|CD)_ω.
///
/// Evaluates
/// \f[
///   (\mu\nu|\lambda\sigma)_\omega
///   = \iint \chi_\mu(\mathbf{r}_1)\chi_\nu(\mathbf{r}_1)\,
///     \frac{\mathrm{erf}(\omega r_{12})}{r_{12}}\,
///     \chi_\lambda(\mathbf{r}_2)\chi_\sigma(\mathbf{r}_2)\,
///     d\mathbf{r}_1\,d\mathbf{r}_2
/// \f]
/// with \(\omega\) in Bohr\(^{-1}\). \(\omega = 0\) is defined as the bare
/// \(1/r\) operator and is bit-identical to `rys_eri_block` (libcint
/// convention: a zero range-separation parameter means "no attenuation",
/// not the mathematical \(\mathrm{erf}(0\cdot r)/r = 0\) limit).
///
/// Output layout and accumulation semantics match `rys_eri_block`.
///
/// @param A,B Shell blocks for the bra (electron 1).
/// @param C,D Shell blocks for the ket (electron 2).
/// @param omega Range-separation parameter [1/Bohr]; must be ≥ 0.
/// @param out Pre-zeroed flat buffer of size
///            A.n_cart·B.n_cart·C.n_cart·D.n_cart.
/// @throws std::invalid_argument when omega < 0.
void rys_eri_erf_block(const ShellBlock& A, const ShellBlock& B,
                       const ShellBlock& C, const ShellBlock& D, double omega,
                       double* out);

/// @brief Block-level derivative of the ERI w.r.t. one nuclear centre.
///
/// Computes ∂(AB|CD)/∂R_X — all three Cartesian components — into an
/// axis-split output buffer. Uses a single extended 2D Rys grid per
/// primitive quartet (bra or ket direction extended by +1 angular
/// momentum on the deriv side) and a nabla recurrence
///   f[i] = i·g[i-1] - 2α·g[i+1]
/// to read out the derivative from the already-built grid. All three
/// Cartesian axes share the same grid; this is the same algorithmic
/// skeleton used by libcint's `int2e_ip1`.
///
/// @param A,B,C,D Shell blocks in the conventional ERI ordering (AB|CD).
/// @param deriv_center 0 = ∂/∂R_A, 1 = ∂/∂R_B, 2 = ∂/∂R_C, 3 = ∂/∂R_D.
/// @param out_axes Pre-zeroed flat buffer of size
///        3 · A.n_cart · B.n_cart · C.n_cart · D.n_cart, interpreted as
///        `out_axes[axis * (na*nb*nc*nd) + ia*nbcd + ib*ncd + ic*nd + id]`.
///        Output is accumulated, not overwritten (mirrors
///        `rys_eri_block`).
///
/// @note Cartesian Gaussians up to l = 3 (s/p/d/f) are supported. The
///       derivative raises one shell's angular momentum by +1, so the
///       worst case — the +1-derivative of an (ff|ff) quartet — reaches
///       Ltot = 13 and stays within the `kRysMaxRoots = 7` budget served
///       by the iterative root finder.
void rys_eri_block_deriv(const ShellBlock& A, const ShellBlock& B,
                         const ShellBlock& C, const ShellBlock& D,
                         int deriv_center, double* out_axes);

} // namespace native
} // namespace scf
} // namespace cpu
} // namespace ATV

/// @file cpu/scf/native_integrals.h
/// @brief Native C++ molecular integral functions for s+p type Gaussians.
///
/// Primary integral engine. Validated element-by-element against libcint
/// as ground truth (enable -DATV_BUILD_LIBCINT=ON for validation tests).
///
/// All functions operate in atomic units (Bohr, Hartree).
/// Uses physical normalization: (2*alpha/pi)^{3/4} for s-type.
/// Shell.l selects angular momentum; Shell.cart selects Cartesian component.
///
/// For direct-SCF Fock building, prefer the shell-block interface:
///   ShellBlock + eri_block() amortize Boys/R-table computation across all
///   angular combinations in a shell-block quartet (libcint/PySCF approach).
///
/// @see cpu::scf::IntegralPack for the output data structure
/// @see cpu::scf::IntegralPack for the entry point
#pragma once

namespace ATV {
namespace cpu {
namespace scf {
namespace native {

/// @brief Contracted Gaussian shell (s and p type), single-AO view.
/// Used for one-electron integrals and legacy direct-SCF.
struct Shell {
    int n_prim;          ///< Number of primitive Gaussians.
    const double* alpha; ///< Exponents [n_prim].
    const double* coeff; ///< Normalized contraction coefficients [n_prim].
    double center[3];    ///< Shell center in Bohr [x, y, z].
    int l;               ///< Angular momentum (0=s, 1=p).
    int cart;            ///< Cartesian component: s->0, p->0=x,1=y,2=z.
};

/// @brief Shell block for shell-level ERI computation (libcint/PySCF approach).
///
/// Represents all Cartesian AOs of a single angular-momentum shell:
///   l=0 (s): n_cart=1, function 0 = (0,0,0)
///   l=1 (p): n_cart=3, functions 0=px, 1=py, 2=pz
///
/// eri_block() amortizes Boys function + R-table evaluation across all
/// (na × nb × nc × nd) angular combinations per primitive quartet,
/// giving an ~81× speedup for (pp|pp) shell-block quartets in STO-3G.
///
/// @see eri_block
/// @see cpu::scf::ShellBlockPairList
struct ShellBlock {
    int n_prim; ///< Number of primitive Gaussians (same for all AOs in this shell).
    const double* alpha; ///< Exponents [n_prim].
    const double* coeff; ///< Normalized contraction coefficients [n_prim].
    double center[3];    ///< Shell center in Bohr [x, y, z].
    int l;               ///< Angular momentum: 0=s, 1=p.
    int n_cart;          ///< Number of Cartesian AOs: l=0→1, l=1→3.
    int ao_start;        ///< First global AO index for this shell block.
};

/// @brief Atom descriptor for nuclear attraction integrals.
struct Atom {
    int Z;         ///< Atomic number (nuclear charge).
    double pos[3]; ///< Position in Bohr [x, y, z].
};

// ── Utility ───────────────────────────────────────────────────────────

/// @brief GTO normalization (libcint convention). Delegates to ATV::gto_norm.
/// @param l     Angular momentum quantum number.
/// @param alpha Gaussian exponent.
/// @return Normalization factor.
double gto_norm(int l, double alpha);

/// @brief Physical GTO normalization for s-type. Delegates to ATV::phys_norm_s.
/// @param alpha Gaussian exponent.
/// @return Normalization factor (2*alpha/pi)^{3/4}.
double phys_norm(double alpha);

/// @brief Physical normalization for d-type (l=2) Gaussians (libcint convention).
/// @param alpha Exponent.
/// @return Shared normalization factor for all six Cartesian d AOs.
double phys_norm_d(double alpha);

/// @brief Physical normalization for p-type (l=1) Gaussians.
/// @param alpha Gaussian exponent.
/// @return Normalization factor.
double phys_norm_p(double alpha);

/// @brief Physical normalization for f-type (l=3) Gaussians (libcint
///        convention). Delegates to ATV::phys_norm_f.
/// @param alpha Gaussian exponent.
/// @return Shared normalization factor for all ten Cartesian f AOs.
double phys_norm_f(double alpha);

/// @brief Physical normalization for g-type (l=4) Gaussians (libcint
///        convention). Delegates to ATV::phys_norm_g. Auxiliary-basis
///        (density-fitting) shells only.
/// @param alpha Gaussian exponent.
/// @return Shared normalization factor for all fifteen Cartesian g AOs.
double phys_norm_g(double alpha);

/// @brief Boys function F_0(x) = integral_0^1 exp(-x*t^2) dt.
/// @details F_0(0) = 1. For x > 0: F_0(x) = 0.5 * sqrt(pi/x) * erf(sqrt(x)).
/// @param x Non-negative argument.
/// @return F_0(x).
double boys_f0(double x);

/// @brief General Boys function F_n(x) = integral_0^1 t^{2n} exp(-x*t^2) dt.
/// @details Uses upward recurrence from F_0. F_n(0) = 1/(2n+1).
/// @param n Order (n >= 0).
/// @param x Non-negative argument.
/// @return F_n(x).
double boys_fn(int n, double x);

// ── One-electron integrals ────────────────────────────────────────────

/// @brief Overlap integral between two contracted s-type shells.
/// @details S = sum_pq c_p c_q (pi/gamma)^{3/2} exp(-mu * R^2)
///   where gamma = alpha_p + alpha_q, mu = alpha_p*alpha_q/gamma.
/// @param a First shell.
/// @param b Second shell.
/// @return Overlap integral <a|b>.
double overlap_ss(const Shell& a, const Shell& b);

/// @brief Kinetic energy integral between two contracted s-type shells.
/// @details T_pq = mu * (3 - 2*mu*R^2) * S_pq for each primitive pair.
/// @param a First shell.
/// @param b Second shell.
/// @return Kinetic integral <a|-nabla^2/2|b>.
double kinetic_ss(const Shell& a, const Shell& b);

/// @brief Nuclear attraction integral between two contracted s-type shells.
/// @details V = sum_C (-Z_C) sum_pq c_p c_q (2pi/gamma) exp(-mu*R^2) F_0(gamma*|P-C|^2)
///   summed over all nuclei C.
/// @param a First shell.
/// @param b Second shell.
/// @param atoms Array of atoms (nuclei).
/// @param natm Number of atoms.
/// @return Nuclear attraction integral <a|sum(-Z/r)|b>.
double nuclear_ss(const Shell& a, const Shell& b, const Atom* atoms, int natm);

// ── Two-electron integrals ────────────────────────────────────────────

/// @brief Two-electron repulsion integral (ab|cd) for s-type shells.
/// @details (ab|cd) = sum c_p c_q c_r c_s * 2*pi^{5/2} / (g1*g2*sqrt(g1+g2))
///            * exp(-mu1*R_AB^2 - mu2*R_CD^2) * F_0(rho*|PQ|^2)
/// @param a Shell for electron 1, bra.
/// @param b Shell for electron 1, ket.
/// @param c Shell for electron 2, bra.
/// @param d Shell for electron 2, ket.
/// @return ERI in chemist's notation (ab|cd).
double eri_ssss(const Shell& a, const Shell& b, const Shell& c, const Shell& d);

// ── General integrals (s+p type) ─────────────────────────────────────

/// @brief General overlap integral for shells with l in {0,1}.
/// @details Dispatches to Obara-Saika recurrence based on a.l, b.l, a.cart, b.cart.
/// @param a First shell.
/// @param b Second shell.
/// @return Overlap integral <a|b>.
double overlap(const Shell& a, const Shell& b);

/// @brief General kinetic energy integral for shells with l in {0,1}.
/// @param a First shell.
/// @param b Second shell.
/// @return Kinetic integral <a|-nabla^2/2|b>.
double kinetic(const Shell& a, const Shell& b);

/// @brief General nuclear attraction integral for shells with l in {0,1}.
/// @param a First shell.
/// @param b Second shell.
/// @param atoms Array of atoms (nuclei).
/// @param natm Number of atoms.
/// @return Nuclear attraction integral <a|sum(-Z/r)|b>.
double nuclear(const Shell& a, const Shell& b, const Atom* atoms, int natm);

/// @brief General two-electron repulsion integral for shells with l in {0,1}.
/// @param a Shell for electron 1, bra.
/// @param b Shell for electron 1, ket.
/// @param c Shell for electron 2, bra.
/// @param d Shell for electron 2, ket.
/// @return ERI in chemist's notation (ab|cd).
double eri(const Shell& a, const Shell& b, const Shell& c, const Shell& d);

// ── Shell-block ERI (libcint/PySCF approach) ─────────────────────────

/// @brief Compute all ERIs for a shell-block quartet into a flat output buffer.
///
/// @details Amortizes Boys function and R-table evaluation across all
///   (A.n_cart × B.n_cart × C.n_cart × D.n_cart) angular combinations
///   per primitive quartet.  For a (pp|pp) block with 3 primitives/shell
///   (STO-3G), this reduces Boys evaluations from 81×81 to 81 (one per
///   primitive quartet instead of one per scalar ERI call).
///
///   The E-coefficient tables E_bra[3][2][2][3] and E_ket[3][2][2][3] are
///   precomputed once per primitive quartet (indexed by angular quantum number
///   in each direction: 0 or 1), and shared across the inner angular loop.
///   The R-table (shared for all angular combinations) is built to size
///   [5][5][5][5] covering Ltot ≤ 4 = A.l+B.l+C.l+D.l for l_max=1.
///
///   Supports l ∈ {0, 1} only (s and p shells, STO-3G basis).
///
/// @param A,B Shell blocks for the bra (electron 1).
/// @param C,D Shell blocks for the ket (electron 2).
/// @param out Flat output buffer of size A.n_cart×B.n_cart×C.n_cart×D.n_cart.
///   Indexing: out[ia*B.n_cart*C.n_cart*D.n_cart + ib*C.n_cart*D.n_cart
///             + ic*D.n_cart + id] = (ia_in_A  ib_in_B | ic_in_C  id_in_D).
///   The buffer is zeroed on entry and accumulated into (supports warm accumulation).
/// @note out must be pre-zeroed; the function accumulates into it.
void eri_block(const ShellBlock& A, const ShellBlock& B, const ShellBlock& C,
               const ShellBlock& D, double* out);

// ── Shell-block one-electron integrals ────────────────────────────────

/// @brief Compute all overlap integrals for a shell-block pair.
///
/// @details For each primitive pair (p, q) the 1D Obara–Saika tables
///   are built once and reused across all (A.n_cart × B.n_cart) angular
///   combinations — the same amortization strategy as eri_block for
///   the 2e path. On d-shells this is ~36× the per-element work on
///   the shared primitive setup.
///
/// @param A,B  Shell blocks (any l ≤ 2).
/// @param out  Pre-zeroed buffer of size A.n_cart * B.n_cart, flat
///             layout `out[ia * B.n_cart + ib] = <A_ia|B_ib>`.
///             Accumulated into (ss! caller must pre-zero).
void overlap_block(const ShellBlock& A, const ShellBlock& B, double* out);

/// @brief Compute all kinetic-energy integrals for a shell-block pair.
///
/// Same batching strategy as overlap_block. Requires 1D overlap tables
/// up to (imax, jmax+2) per axis; builds them once per primitive pair.
void kinetic_block(const ShellBlock& A, const ShellBlock& B, double* out);

/// @brief Compute all nuclear-attraction integrals for a shell-block pair.
///
/// Per primitive pair, builds the full Hermite E table and the
/// McMurchie–Davidson R^0_{tuv} table once, then contracts across all
/// angular combinations. For d-shells this amortizes the R-table build
/// over 36 (ia, ib) pairs.
///
/// @param A,B    Shell blocks.
/// @param atoms  Nuclei (charges + positions in Bohr).
/// @param natm   Number of nuclei.
/// @param out    Pre-zeroed buffer of size A.n_cart * B.n_cart.
void nuclear_block(const ShellBlock& A, const ShellBlock& B, const Atom* atoms,
                   int natm, double* out);

// ── Primitive-level functions (exposed for derivative code) ──────────

/// @brief Primitive overlap integral for arbitrary Cartesian angular momenta.
/// @details Computes the 3D overlap integral for a single primitive pair using
///   the Obara-Saika recurrence. Supports angular momenta up to 2 per component.
/// @param a_ang Cartesian angular momenta on center A [3] (each 0..2).
/// @param b_ang Cartesian angular momenta on center B [3] (each 0..2).
/// @param alpha_a Exponent on center A.
/// @param A Center A position [3].
/// @param alpha_b Exponent on center B.
/// @param B Center B position [3].
/// @return Primitive overlap integral value.
double overlap_prim_3d(const int a_ang[3], const int b_ang[3], double alpha_a,
                       const double A[3], double alpha_b, const double B[3]);

/// @brief Primitive kinetic energy integral for arbitrary Cartesian angular momenta.
/// @details Computes the 3D kinetic integral for a single primitive pair.
///   Supports angular momenta up to 2 per component.
/// @param a_ang Cartesian angular momenta on center A [3] (each 0..2).
/// @param b_ang Cartesian angular momenta on center B [3] (each 0..2).
/// @param alpha_a Exponent on center A.
/// @param A Center A position [3].
/// @param alpha_b Exponent on center B.
/// @param B Center B position [3].
/// @return Primitive kinetic integral value.
double kinetic_prim_3d(const int a_ang[3], const int b_ang[3], double alpha_a,
                       const double A[3], double alpha_b, const double B[3]);

/// @brief Primitive nuclear attraction integral for one nucleus.
/// @details McMurchie-Davidson scheme with Hermite Coulomb integrals.
///   Supports angular momenta up to 2 per component (Ltot <= 6).
///   Returns the integral without the -Z_C prefactor.
/// @param a_ang Angular momenta on center A [3] (each 0..2).
/// @param b_ang Angular momenta on center B [3] (each 0..2).
/// @param gamma Sum of exponents (alpha_a + alpha_b).
/// @param PA P - A vector [3].
/// @param PB P - B vector [3].
/// @param PC P - C vector [3].
/// @param K_AB exp(-mu * |A-B|^2).
/// @return Primitive nuclear attraction integral value (without -Z).
double nuclear_prim(const int a_ang[3], const int b_ang[3], double gamma,
                    const double PA[3], const double PB[3], const double PC[3],
                    double K_AB);

/// @brief Primitive two-electron repulsion integral (McMurchie-Davidson).
/// @details Supports angular momenta up to 2 per component (Ltot <= 8).
/// @param a_ang Angular momenta on center A [3] (each 0..2).
/// @param b_ang Angular momenta on center B [3] (each 0..2).
/// @param c_ang Angular momenta on center C [3] (each 0..2).
/// @param d_ang Angular momenta on center D [3] (each 0..2).
/// @param alpha_a Exponent on A.
/// @param A Center A position [3].
/// @param alpha_b Exponent on B.
/// @param B Center B position [3].
/// @param alpha_c Exponent on C.
/// @param C Center C position [3].
/// @param alpha_d Exponent on D.
/// @param D Center D position [3].
/// @return Primitive ERI value.
double eri_prim(const int a_ang[3], const int b_ang[3], const int c_ang[3],
                const int d_ang[3], double alpha_a, const double A[3],
                double alpha_b, const double B[3], double alpha_c,
                const double C[3], double alpha_d, const double D[3]);

} // namespace native
} // namespace scf
} // namespace cpu
} // namespace ATV

/// @file cpu/scf/integral_derivatives.cpp
/// @brief Analytic derivatives of molecular integrals w.r.t. nuclear positions.
///
/// Uses the angular momentum raising formula for derivatives:
///   d/dA_d <a|O|b> = 2*alpha_a * <a+1_d|O|b> - a_d * <a-1_d|O|b>
///
/// where <a+1_d| bumps angular momentum on center A in direction d by 1,
/// and <a-1_d| lowers it by 1 (vanishes if a_d == 0).
///
/// For two-center integrals, translational invariance gives d/dB = -d/dA.
/// Nuclear attraction has an additional term from the nuclear position.
///
/// Supports any l <= kMaxL (s, p, d shells) through the generic
/// primitive helpers in `native::`.
///
/// References:
///   Obara & Saika, JCP 84, 3963 (1986) -- recurrence relations
///   Szabo & Ostlund, Appendix C -- RHF gradient formulas
///   Helgaker, Jorgensen, Olsen, Ch. 9 -- molecular gradients

#include "atomiverse/cpu/scf/internal/integral_derivatives.h"
#include "atomiverse/core/gto.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/cart.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/libcint_bridge.h"
#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/rys_eri.h"
#include "atomiverse/cpu/scf/xc_functional.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef ATV_HAS_LIBCINT
extern "C" {
#include <cint.h>
#include <cint_funcs.h>
}
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Get Cartesian angular momentum exponents for a shell.
/// @details Decodes `sh.cart` (the per-shell-block index 0..n_cart_of(l)-1)
///   into `(l_x, l_y, l_z)` using the canonical libcint/PySCF enumeration
///   in `fill_cart_comps`. Supports any l ≤ kMaxL.
static inline void cart_exponents(const native::Shell& sh, int a_out[3]) {
    int comps[kMaxNCart][3];
    fill_cart_comps(sh.l, comps);
    a_out[0] = comps[sh.cart][0];
    a_out[1] = comps[sh.cart][1];
    a_out[2] = comps[sh.cart][2];
}

// ── Primitive-level derivative helpers ───────────────────────────────

/// @brief Derivative of primitive overlap w.r.t. center A in all 3 directions.
///
/// d/dA_d S(a,b) = 2*alpha_a * S(a+1_d, b) - a_d * S(a-1_d, b)
///
/// @param a_ang Cartesian angular momenta on A [3].
/// @param b_ang Cartesian angular momenta on B [3].
/// @param alpha_a Exponent on center A.
/// @param A Center A position [3].
/// @param alpha_b Exponent on center B.
/// @param B Center B position [3].
/// @param dS_dA Output: dS/dA in each direction [3].
static void overlap_deriv_prim_general(const int a_ang[3], const int b_ang[3],
                                       double alpha_a, const double A[3],
                                       double alpha_b, const double B[3],
                                       double dS_dA[3]) {
    for (int d = 0; d < 3; ++d) {
        // Raising term: 2*alpha_a * S(a+1_d, b)
        int a_plus[3] = {a_ang[0], a_ang[1], a_ang[2]};
        a_plus[d] += 1;
        double S_plus =
            native::overlap_prim_3d(a_plus, b_ang, alpha_a, A, alpha_b, B);

        double result = 2.0 * alpha_a * S_plus;

        // Lowering term: -a_d * S(a-1_d, b) -- only if a_d > 0
        if (a_ang[d] > 0) {
            int a_minus[3] = {a_ang[0], a_ang[1], a_ang[2]};
            a_minus[d] -= 1;
            double S_minus =
                native::overlap_prim_3d(a_minus, b_ang, alpha_a, A, alpha_b, B);
            result -= a_ang[d] * S_minus;
        }

        dS_dA[d] = result;
    }
}

/// @brief Derivative of primitive kinetic integral w.r.t. center A in all 3 directions.
///
/// d/dA_d T(a,b) = 2*alpha_a * T(a+1_d, b) - a_d * T(a-1_d, b)
///
/// @param a_ang Cartesian angular momenta on A [3].
/// @param b_ang Cartesian angular momenta on B [3].
/// @param alpha_a Exponent on center A.
/// @param A Center A position [3].
/// @param alpha_b Exponent on center B.
/// @param B Center B position [3].
/// @param dT_dA Output: dT/dA in each direction [3].
static void kinetic_deriv_prim_general(const int a_ang[3], const int b_ang[3],
                                       double alpha_a, const double A[3],
                                       double alpha_b, const double B[3],
                                       double dT_dA[3]) {
    for (int d = 0; d < 3; ++d) {
        int a_plus[3] = {a_ang[0], a_ang[1], a_ang[2]};
        a_plus[d] += 1;
        double T_plus =
            native::kinetic_prim_3d(a_plus, b_ang, alpha_a, A, alpha_b, B);

        double result = 2.0 * alpha_a * T_plus;

        if (a_ang[d] > 0) {
            int a_minus[3] = {a_ang[0], a_ang[1], a_ang[2]};
            a_minus[d] -= 1;
            double T_minus =
                native::kinetic_prim_3d(a_minus, b_ang, alpha_a, A, alpha_b, B);
            result -= a_ang[d] * T_minus;
        }

        dT_dA[d] = result;
    }
}

/// @brief Derivative of primitive nuclear attraction w.r.t. center A (Gaussian center).
///
/// d/dA_d V(a,b) = 2*alpha_a * V(a+1_d, b) - a_d * V(a-1_d, b)
///
/// V is computed via native::nuclear_prim for a single nucleus.
/// The -Z_C prefactor is NOT included; caller must multiply.
///
/// @param a_ang Cartesian angular momenta on A [3].
/// @param b_ang Cartesian angular momenta on B [3].
/// @param gamma Sum of exponents.
/// @param alpha_a Exponent on center A.
/// @param A Center A position [3].
/// @param alpha_b Exponent on center B.
/// @param B Center B position [3].
/// @param C Nuclear position [3].
/// @param dV_dA Output: dV/dA in each direction [3].
static void nuclear_deriv_prim_center_general(
    const int a_ang[3], const int b_ang[3], double alpha_a, const double A[3],
    double alpha_b, const double B[3], const double C[3], double dV_dA[3]) {
    const double gamma = alpha_a + alpha_b;
    const double mu = alpha_a * alpha_b / gamma;
    const double R2_AB = (A[0] - B[0]) * (A[0] - B[0])
                         + (A[1] - B[1]) * (A[1] - B[1])
                         + (A[2] - B[2]) * (A[2] - B[2]);
    const double K_AB = std::exp(-mu * R2_AB);

    double P[3], PA[3], PB[3], PC[3];
    for (int dd = 0; dd < 3; ++dd) {
        P[dd] = (alpha_a * A[dd] + alpha_b * B[dd]) / gamma;
        PA[dd] = P[dd] - A[dd];
        PB[dd] = P[dd] - B[dd];
        PC[dd] = P[dd] - C[dd];
    }

    for (int d = 0; d < 3; ++d) {
        int a_plus[3] = {a_ang[0], a_ang[1], a_ang[2]};
        a_plus[d] += 1;
        double V_plus =
            native::nuclear_prim(a_plus, b_ang, gamma, PA, PB, PC, K_AB);

        double result = 2.0 * alpha_a * V_plus;

        if (a_ang[d] > 0) {
            int a_minus[3] = {a_ang[0], a_ang[1], a_ang[2]};
            a_minus[d] -= 1;
            double V_minus =
                native::nuclear_prim(a_minus, b_ang, gamma, PA, PB, PC, K_AB);
            result -= a_ang[d] * V_minus;
        }

        dV_dA[d] = result;
    }
}

/// @brief Derivative of nuclear attraction w.r.t. nuclear position C.
///
/// The nucleus C appears in the Coulomb potential |r-C|^{-1}.
/// d/dC_d V(a,b;C) is computed by finite difference of the McMurchie-Davidson
/// R integrals w.r.t. C. Equivalently:
///   d/dC_d V = -d/dP_d V (for the P-C dependence only, keeping K_AB constant)
///
/// For the McMurchie-Davidson scheme:
///   V = (2pi/gamma) K_AB sum_{tuv} E[t]E[u]E[v] R^0_{tuv}(P-C)
///
/// d/dC_d only affects R^0_{tuv} via its PC dependence:
///   dR^0_{tuv}/dC_d = -dR^0_{tuv}/dPC_d = -(R^0_{t+delta_dx, u+delta_dy, v+delta_dz})
///     at next order = -R^0 with shifted index
///
/// Actually, the simplest correct approach: dV/dC_d = -dV/dP_d (for R dependence).
/// But dV/dP_d acts on both E and R. For the nuclear derivative w.r.t. C:
///   dV/dC_d = -(2pi/gamma) K_AB sum_{tuv} E[t]E[u]E[v] * dR^0_{tuv}/dPC_d
/// where dR^0_{tuv}/dPC_d = R^0_{t+delta_dx, u+delta_dy, v+delta_dz} (at the bumped index).
///
/// More precisely: R^0_{tuv} depends on PC, and the derivative is:
///   dR^0_{0,0,0}/dPC_d = R^1_{1_d, 0, 0} (via the downward recurrence)
///   But since we build R^0 from R^m, a cleaner way is:
///     dR^0_{tuv}/dC_d = -R^0_{t+delta_d(x), u+delta_d(y), v+delta_d(z)}  -- NO, this is wrong.
///
/// The correct way: from dV_pq^C/dC_d, the result is:
///   dV/dC_d = (2pi/gamma) K_AB sum_{tuv} E[t]E[u]E[v] * R^0_{t+delta_dx, u+delta_dy, v+delta_dz}
/// where delta_dx = (d==0), etc. This comes from differentiating the Hermite Coulomb integral
/// R^0_{tuv}(T) w.r.t. C_d.
///
/// Actually, the simplest and most correct formula uses translational invariance:
/// For a two-center integral V^C_ab summed over C:
///   d(total V)/dC_d = (contribution from C as a Gaussian center if C hosts a shell)
///                   + (contribution from C as a nuclear charge center)
///
/// The nuclear contribution dV_pq^C/dC_d has already been derived. We compute it
/// via a direct formula using Boys functions.
static void nuclear_deriv_prim_nuc_general(const int a_ang[3],
                                           const int b_ang[3], double alpha_a,
                                           const double A[3], double alpha_b,
                                           const double B[3], const double C[3],
                                           double dV_dC[3]) {
    const double gamma = alpha_a + alpha_b;
    const double mu = alpha_a * alpha_b / gamma;
    const double R2_AB = (A[0] - B[0]) * (A[0] - B[0])
                         + (A[1] - B[1]) * (A[1] - B[1])
                         + (A[2] - B[2]) * (A[2] - B[2]);
    const double K_AB = std::exp(-mu * R2_AB);

    double P[3], PA[3], PB[3], PC[3];
    for (int dd = 0; dd < 3; ++dd) {
        P[dd] = (alpha_a * A[dd] + alpha_b * B[dd]) / gamma;
        PA[dd] = P[dd] - A[dd];
        PB[dd] = P[dd] - B[dd];
        PC[dd] = P[dd] - C[dd];
    }

    // The derivative dV/dC_d of the McMurchie-Davidson nuclear integral
    // w.r.t. nuclear position C is obtained by differentiating R^0_{tuv}:
    //   dR^0_{tuv}/dC_d = -dR^0_{tuv}/dPC_d
    // where dR^0_{tuv}/dPC_d involves the next-order R integrals.
    //
    // For the complete integral:
    //   V = (2pi/gamma) K_AB sum_{tuv} E[t]E[u]E[v] R^0_{tuv}
    //
    //   dV/dC_d = -(2pi/gamma) K_AB sum_{tuv} E[t]E[u]E[v] R^0_{t+d_x, u+d_y, v+d_z}
    //   where d_x = (d==0), d_y = (d==1), d_z = (d==2).
    //
    // This requires R^0 built to one higher order in direction d.
    // We implement this by evaluating V with a bumped angular momentum
    // on a fictitious direction, using the identity:
    //   dV/dC_d = -(2pi/gamma) K_AB sum_{tuv} E[t]E[u]E[v] R^0_{t+d_x, u+d_y, v+d_z}
    //
    // This equals: -V_prim(a_ang, b_ang + delta_d) evaluated with
    //   modified E coefficients for the b side... Actually this is not quite right.
    //
    // Simpler: just use the translational invariance of the total integral.
    // For a given nucleus C:
    //   d/dA_d V_C + d/dB_d V_C + d/dC_d V_C = 0  (translational invariance)
    // so:
    //   d/dC_d V_C = -(d/dA_d V_C + d/dB_d V_C)
    //
    // d/dA_d V_C uses angular momentum raising on center A.
    // d/dB_d V_C uses angular momentum raising on center B.
    // This is the cleanest approach.

    // Compute d/dA_d V_C (raising on center A)
    double dV_dA[3];
    for (int d = 0; d < 3; ++d) {
        int a_plus[3] = {a_ang[0], a_ang[1], a_ang[2]};
        a_plus[d] += 1;
        double V_plus =
            native::nuclear_prim(a_plus, b_ang, gamma, PA, PB, PC, K_AB);
        double result = 2.0 * alpha_a * V_plus;
        if (a_ang[d] > 0) {
            int a_minus[3] = {a_ang[0], a_ang[1], a_ang[2]};
            a_minus[d] -= 1;
            double V_minus =
                native::nuclear_prim(a_minus, b_ang, gamma, PA, PB, PC, K_AB);
            result -= a_ang[d] * V_minus;
        }
        dV_dA[d] = result;
    }

    // Compute d/dB_d V_C (raising on center B)
    double dV_dB[3];
    for (int d = 0; d < 3; ++d) {
        int b_plus[3] = {b_ang[0], b_ang[1], b_ang[2]};
        b_plus[d] += 1;
        double V_plus =
            native::nuclear_prim(a_ang, b_plus, gamma, PA, PB, PC, K_AB);
        double result = 2.0 * alpha_b * V_plus;
        if (b_ang[d] > 0) {
            int b_minus[3] = {b_ang[0], b_ang[1], b_ang[2]};
            b_minus[d] -= 1;
            double V_minus =
                native::nuclear_prim(a_ang, b_minus, gamma, PA, PB, PC, K_AB);
            result -= b_ang[d] * V_minus;
        }
        dV_dB[d] = result;
    }

    // Translational invariance: dV/dC = -(dV/dA + dV/dB)
    for (int d = 0; d < 3; ++d) {
        dV_dC[d] = -(dV_dA[d] + dV_dB[d]);
    }
}

/// @brief Derivative of primitive ERI w.r.t. center A in all 3 directions.
///
/// d/dA_d (ab|cd) = 2*alpha_a * (a+1_d b|cd) - a_d * (a-1_d b|cd)
///
/// @param a_ang, b_ang, c_ang, d_ang  Cartesian angular momenta [3] each.
/// @param alpha_a..alpha_d  Primitive exponents.
/// @param A..D  Center positions [3] each.
/// @param dERI_dA  Output: d(ERI)/dA in each direction [3].
static void eri_deriv_prim_general(const int a_ang[3], const int b_ang[3],
                                   const int c_ang[3], const int d_ang[3],
                                   double alpha_a, const double A[3],
                                   double alpha_b, const double B[3],
                                   double alpha_c, const double C[3],
                                   double alpha_d, const double D[3],
                                   double dERI_dA[3]) {
    for (int d = 0; d < 3; ++d) {
        int a_plus[3] = {a_ang[0], a_ang[1], a_ang[2]};
        a_plus[d] += 1;
        double ERI_plus =
            native::eri_prim(a_plus, b_ang, c_ang, d_ang, alpha_a, A, alpha_b,
                             B, alpha_c, C, alpha_d, D);

        double result = 2.0 * alpha_a * ERI_plus;

        if (a_ang[d] > 0) {
            int a_minus[3] = {a_ang[0], a_ang[1], a_ang[2]};
            a_minus[d] -= 1;
            double ERI_minus =
                native::eri_prim(a_minus, b_ang, c_ang, d_ang, alpha_a, A,
                                 alpha_b, B, alpha_c, C, alpha_d, D);
            result -= a_ang[d] * ERI_minus;
        }

        dERI_dA[d] = result;
    }
}

struct QuartetDerivTotals {
    double dA[3] = {0.0, 0.0, 0.0};
    double dB[3] = {0.0, 0.0, 0.0};
    double dC[3] = {0.0, 0.0, 0.0};
    double dD[3] = {0.0, 0.0, 0.0};
};

static inline int pair_multiplicity(int a, int b) { return (a == b) ? 1 : 2; }

static std::vector<int>
build_shell_to_atom_from_integrals(const IntegralPack& integrals) {
    const int n = integrals.n_basis;
    const int natm = static_cast<int>(integrals.atoms.size());
    std::vector<int> shell_to_atom(n, -1);

    for (int mu = 0; mu < n; ++mu) {
        const auto& sh = integrals.shells[mu];
        for (int a = 0; a < natm; ++a) {
            const double dx = sh.center[0] - integrals.atoms[a].pos[0];
            const double dy = sh.center[1] - integrals.atoms[a].pos[1];
            const double dz = sh.center[2] - integrals.atoms[a].pos[2];
            if (dx * dx + dy * dy + dz * dz < 1e-20) {
                shell_to_atom[mu] = a;
                break;
            }
        }
    }

    return shell_to_atom;
}

#ifdef ATV_HAS_LIBCINT

static inline int libcint_shell_dim(const LibcintContext& ctx, int sh) {
    const int begin = ctx.shell_offsets[sh];
    const int end =
        (sh + 1 < ctx.nshells) ? ctx.shell_offsets[sh + 1] : ctx.nao;
    return end - begin;
}

static inline std::size_t libcint_ip1_index(int comp, int i, int j, int k,
                                            int l, int di, int dj, int dk,
                                            int dl) {
    return static_cast<std::size_t>(
        i + di * (j + dj * (k + dk * (l + dl * comp))));
}

#endif

static QuartetDerivTotals
compute_quartet_derivatives(const native::Shell& shell_a, const int a_ang[3],
                            const native::Shell& shell_b, const int b_ang[3],
                            const native::Shell& shell_c, const int c_ang[3],
                            const native::Shell& shell_d, const int d_ang[3]) {
    QuartetDerivTotals totals;

    for (int p = 0; p < shell_a.n_prim; ++p) {
        for (int q = 0; q < shell_b.n_prim; ++q) {
            for (int r = 0; r < shell_c.n_prim; ++r) {
                for (int s = 0; s < shell_d.n_prim; ++s) {
                    const double coeff = shell_a.coeff[p] * shell_b.coeff[q]
                                         * shell_c.coeff[r] * shell_d.coeff[s];

                    double dA[3];
                    eri_deriv_prim_general(
                        a_ang, b_ang, c_ang, d_ang, shell_a.alpha[p],
                        shell_a.center, shell_b.alpha[q], shell_b.center,
                        shell_c.alpha[r], shell_c.center, shell_d.alpha[s],
                        shell_d.center, dA);

                    double dB[3];
                    eri_deriv_prim_general(
                        b_ang, a_ang, c_ang, d_ang, shell_b.alpha[q],
                        shell_b.center, shell_a.alpha[p], shell_a.center,
                        shell_c.alpha[r], shell_c.center, shell_d.alpha[s],
                        shell_d.center, dB);

                    double dC[3];
                    eri_deriv_prim_general(
                        c_ang, d_ang, a_ang, b_ang, shell_c.alpha[r],
                        shell_c.center, shell_d.alpha[s], shell_d.center,
                        shell_a.alpha[p], shell_a.center, shell_b.alpha[q],
                        shell_b.center, dC);

                    double dD[3];
                    eri_deriv_prim_general(
                        d_ang, c_ang, a_ang, b_ang, shell_d.alpha[s],
                        shell_d.center, shell_c.alpha[r], shell_c.center,
                        shell_a.alpha[p], shell_a.center, shell_b.alpha[q],
                        shell_b.center, dD);

                    for (int axis = 0; axis < 3; ++axis) {
                        totals.dA[axis] += coeff * dA[axis];
                        totals.dB[axis] += coeff * dB[axis];
                        totals.dC[axis] += coeff * dC[axis];
                        totals.dD[axis] += coeff * dD[axis];
                    }
                }
            }
        }
    }

    return totals;
}

static void accumulate_quartet_into_gradient(xt::xtensor<double, 2>& grad,
                                             int atom_a, int atom_b, int atom_c,
                                             int atom_d,
                                             const QuartetDerivTotals& totals,
                                             double factor) {
    if (std::abs(factor) < 1e-18)
        return;

    for (int axis = 0; axis < 3; ++axis) {
        grad(atom_a, axis) += factor * totals.dA[axis];
        grad(atom_b, axis) += factor * totals.dB[axis];
        grad(atom_c, axis) += factor * totals.dC[axis];
        grad(atom_d, axis) += factor * totals.dD[axis];
    }
}

// ── Main entry point ─────────────────────────────────────────────────

IntegralDerivPack::IntegralDerivPack(const IntegralPack& ipack) {
    const int natm = static_cast<int>(ipack.atoms.size());
    const int n_basis = ipack.n_basis;

    this->n_atoms = natm;
    this->n_basis = n_basis;
    dS = xt::zeros<double>({(int)natm, (int)3, n_basis, n_basis});
    dT = xt::zeros<double>({(int)natm, (int)3, n_basis, n_basis});
    dV = xt::zeros<double>({(int)natm, (int)3, n_basis, n_basis});
    dV_nuc = xt::zeros<double>({(int)natm, (int)3});

    // Reuse the shell list already resolved in the IntegralPack — same
    // normalization, same ordering as the SCF path, works for any l.
    const auto& shells = ipack.shells;
    const auto& atoms = ipack.atoms;

    // Shell → atom mapping via shell-center vs atom-position match.
    // Both are in Bohr and constructed from the same Frame positions,
    // so bit-exact equality holds.
    std::vector<int> shell_to_atom(n_basis, -1);
    for (int i = 0; i < n_basis; ++i) {
        const double* c = shells[i].center;
        for (int a = 0; a < natm; ++a) {
            const double* p = atoms[a].pos;
            if (c[0] == p[0] && c[1] == p[1] && c[2] == p[2]) {
                shell_to_atom[i] = a;
                break;
            }
        }
        if (shell_to_atom[i] < 0) {
            throw std::runtime_error("cpu::scf::IntegralDerivPack: shell "
                                     + std::to_string(i)
                                     + " center has no matching atom");
        }
    }

    // ── 1e integral derivatives ──────────────────────────────────────

    for (int i = 0; i < n_basis; ++i) {
        int atom_i = shell_to_atom[i];
        int a_ang[3];
        cart_exponents(shells[i], a_ang);

        for (int j = 0; j <= i; ++j) {
            int atom_j = shell_to_atom[j];
            int b_ang[3];
            cart_exponents(shells[j], b_ang);

            // Overlap and kinetic derivatives
            double dS_total[3] = {0, 0, 0};
            double dT_total[3] = {0, 0, 0};

            for (int p = 0; p < shells[i].n_prim; ++p) {
                for (int q = 0; q < shells[j].n_prim; ++q) {
                    double ci = shells[i].coeff[p];
                    double cj = shells[j].coeff[q];

                    double dS_dA[3], dT_dA[3];
                    overlap_deriv_prim_general(
                        a_ang, b_ang, shells[i].alpha[p], shells[i].center,
                        shells[j].alpha[q], shells[j].center, dS_dA);
                    kinetic_deriv_prim_general(
                        a_ang, b_ang, shells[i].alpha[p], shells[i].center,
                        shells[j].alpha[q], shells[j].center, dT_dA);

                    for (int d = 0; d < 3; ++d) {
                        dS_total[d] += ci * cj * dS_dA[d];
                        dT_total[d] += ci * cj * dT_dA[d];
                    }
                }
            }

            // Assign to atom_i (derivative w.r.t. center of shell i)
            for (int d = 0; d < 3; ++d) {
                dS(atom_i, d, i, j) += static_cast<double>(dS_total[d]);
                dT(atom_i, d, i, j) += static_cast<double>(dT_total[d]);
                // Translational invariance: d/dB = -d/dA for S and T
                dS(atom_j, d, i, j) += static_cast<double>(-dS_total[d]);
                dT(atom_j, d, i, j) += static_cast<double>(-dT_total[d]);
                // Symmetry: S(i,j) = S(j,i)
                if (i != j) {
                    dS(atom_i, d, j, i) += static_cast<double>(dS_total[d]);
                    dT(atom_i, d, j, i) += static_cast<double>(dT_total[d]);
                    dS(atom_j, d, j, i) += static_cast<double>(-dS_total[d]);
                    dT(atom_j, d, j, i) += static_cast<double>(-dT_total[d]);
                }
            }

            // Nuclear attraction derivatives: contributions from each nucleus
            for (int c = 0; c < natm; ++c) {
                double dV_center_i[3] = {0, 0, 0};
                double dV_nuc_c[3] = {0, 0, 0};

                for (int p = 0; p < shells[i].n_prim; ++p) {
                    for (int q = 0; q < shells[j].n_prim; ++q) {
                        double ci = shells[i].coeff[p];
                        double cj = shells[j].coeff[q];

                        // d/dA (center of shell i) -- angular momentum raising on A
                        double dV_dA[3];
                        nuclear_deriv_prim_center_general(
                            a_ang, b_ang, shells[i].alpha[p], shells[i].center,
                            shells[j].alpha[q], shells[j].center, atoms[c].pos,
                            dV_dA);

                        // d/dC (nuclear position) -- via translational invariance
                        double dV_dC[3];
                        nuclear_deriv_prim_nuc_general(
                            a_ang, b_ang, shells[i].alpha[p], shells[i].center,
                            shells[j].alpha[q], shells[j].center, atoms[c].pos,
                            dV_dC);

                        for (int d = 0; d < 3; ++d) {
                            dV_center_i[d] +=
                                ci * cj * (-atoms[c].Z) * dV_dA[d];
                            dV_nuc_c[d] += ci * cj * (-atoms[c].Z) * dV_dC[d];
                        }
                    }
                }

                // For V, the Gaussian center derivative d/dA_d applies to atom_i.
                // The derivative d/dB_d (center of shell j) is obtained via
                // translational invariance: dV/dA + dV/dB + dV/dC = 0
                // so dV/dB = -(dV/dA + dV/dC).
                // But we already computed dV/dC via translational invariance too.
                // Actually: dV_dC above already uses d/dC = -(d/dA + d/dB).
                // So we need dV/dB separately.
                //
                // Compute d/dB (center of shell j) by angular momentum raising on B.
                double dV_center_j[3] = {0, 0, 0};
                for (int p = 0; p < shells[i].n_prim; ++p) {
                    for (int q = 0; q < shells[j].n_prim; ++q) {
                        double ci = shells[i].coeff[p];
                        double cj = shells[j].coeff[q];

                        // d/dB uses raising on center B -- swap roles of A and B
                        double dV_dB[3];
                        nuclear_deriv_prim_center_general(
                            b_ang, a_ang, shells[j].alpha[q], shells[j].center,
                            shells[i].alpha[p], shells[i].center, atoms[c].pos,
                            dV_dB);

                        for (int d = 0; d < 3; ++d) {
                            dV_center_j[d] +=
                                ci * cj * (-atoms[c].Z) * dV_dB[d];
                        }
                    }
                }

                for (int d = 0; d < 3; ++d) {
                    dV(atom_i, d, i, j) += static_cast<double>(dV_center_i[d]);
                    dV(atom_j, d, i, j) += static_cast<double>(dV_center_j[d]);
                    dV(c, d, i, j) += static_cast<double>(dV_nuc_c[d]);
                    if (i != j) {
                        dV(atom_i, d, j, i) +=
                            static_cast<double>(dV_center_i[d]);
                        dV(atom_j, d, j, i) +=
                            static_cast<double>(dV_center_j[d]);
                        dV(c, d, j, i) += static_cast<double>(dV_nuc_c[d]);
                    }
                }
            }
        }
    }

    // ── Nuclear repulsion derivative ─────────────────────────────────
    // V_nuc = sum_{A<B} Z_A * Z_B / R_AB
    // dV_nuc/dA_d = sum_{B!=A} Z_A * Z_B * (B_d - A_d) / R_AB^3
    for (int a = 0; a < natm; ++a) {
        for (int b = 0; b < natm; ++b) {
            if (a == b)
                continue;
            double dx = atoms[b].pos[0] - atoms[a].pos[0];
            double dy = atoms[b].pos[1] - atoms[a].pos[1];
            double dz = atoms[b].pos[2] - atoms[a].pos[2];
            double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            double r3_inv = 1.0 / (r * r * r);
            double ZZ = atoms[a].Z * atoms[b].Z;
            dV_nuc(a, 0) += static_cast<double>(ZZ * dx * r3_inv);
            dV_nuc(a, 1) += static_cast<double>(ZZ * dy * r3_inv);
            dV_nuc(a, 2) += static_cast<double>(ZZ * dz * r3_inv);
        }
    }
}

void accumulate_two_electron_gradient(
    xt::xtensor<double, 2>& grad, const Frame& frame,
    const IntegralPack& integrals, const xt::xtensor<double, 2>& P_total,
    const SlaterDeterminant& det, MeanFieldType method, SpinType spin,
    XCType xc_type, bool force_native_two_electron_deriv) {
    const auto t_total = LogClock::now();
    const int natm = static_cast<int>(integrals.atoms.size());
    const int n = integrals.n_basis;
    const bool is_ks = (method == MeanFieldType::KS);
    if (spin == SpinType::RestrictedOpen)
        throw std::runtime_error(
            "accumulate_two_electron_gradient: SpinType::RestrictedOpen "
            "(ROHF) has no analytic two-electron gradient — refusing to "
            "take the unrestricted branch silently.");
    if (spin == SpinType::Generalized)
        throw std::runtime_error(
            "accumulate_two_electron_gradient: SpinType::Generalized (GHF) "
            "has no analytic two-electron gradient — refusing to take the "
            "unrestricted branch silently.");
    const bool restricted = (spin == SpinType::Restricted);

    // Per-spin factor-1 densities for the unrestricted exchange contraction
    // and Schwarz bound. Hoisted ONCE (density() returns by value); the
    // restricted branches read only P_total, so skip the build there.
    xt::xtensor<double, 2> Pa, Pb;
    if (!restricted) {
        Pa = det.density(SlaterDeterminant::Spin::alpha);
        Pb = det.density(SlaterDeterminant::Spin::beta);
    }

    // Exact-exchange (K-channel) fraction.
    //
    //   dE_2e/dR = dJ/dR - alpha_hf * dK/dR  (PySCF grad/rks.py: vk *= hyb).
    //
    // alpha_hf is 1.0 for pure HF, 0.0 for a pure (semi-local) functional —
    // in which case the exchange-matrix derivative is skipped entirely — and
    // the functional's global-hybrid admixture (0.20 for B3LYP, 0.25 for
    // PBE0) for a KS hybrid. It enters as a pure scalar multiplier on top of
    // the existing closed-shell exchange prefactors and the
    // 0.5*[P(i,k)P(j,l)+P(j,k)P(i,l)] symmetrization, which are unchanged.
    const double alpha_hf = is_ks ? hybrid_exchange_fraction(xc_type) : 1.0;
    const bool do_exchange = (alpha_hf > 0.0);

#ifdef ATV_HAS_LIBCINT
    LibcintContext cint_ctx{frame};
    if (cint_ctx.nao == n && !force_native_two_electron_deriv) {
        const int nshells = cint_ctx.nshells;
        ATV_INFO("[gradient] two_e engine=libcint natm={} basis={} shells={}",
                 natm, n, nshells);
        std::vector<int> shell_to_atom(nshells, -1);
        for (int sh = 0; sh < nshells; ++sh)
            shell_to_atom[sh] = cint_ctx.bas[sh * BAS_SLOTS + ATOM_OF];

        std::vector<std::array<int, 2>> shell_pairs;
        shell_pairs.reserve(
            static_cast<std::size_t>(nshells * (nshells + 1) / 2));
        for (int ish = 0; ish < nshells; ++ish)
            for (int jsh = 0; jsh <= ish; ++jsh)
                shell_pairs.push_back({ish, jsh});

        const int n_shell_pairs = static_cast<int>(shell_pairs.size());
        const long long n_unique_shell_quartets =
            1LL * n_shell_pairs * (n_shell_pairs + 1) / 2;
        const int progress_stride =
            (n_shell_pairs >= 10) ? (n_shell_pairs / 10) : 1;

        CINTOpt* ip1_opt = nullptr;
        int2e_ip1_optimizer(&ip1_opt, cint_ctx.atm.data(), cint_ctx.natm,
                            cint_ctx.bas.data(), cint_ctx.nshells,
                            const_cast<double*>(cint_ctx.env.data()));

        constexpr std::size_t kMaxIP1Block = 3u * 3u * 3u * 3u * 3u;
        std::array<double, kMaxIP1Block> dA_buf{};
        std::array<double, kMaxIP1Block> dB_buf{};
        std::array<double, kMaxIP1Block> dC_buf{};
        std::array<double, kMaxIP1Block> dD_buf{};

        const auto eval_ip1 = [&](std::array<double, kMaxIP1Block>& buf,
                                  const int shls[4]) {
            std::fill(buf.begin(), buf.end(), 0.0);
            int2e_ip1_cart(buf.data(), nullptr, const_cast<int*>(shls),
                           cint_ctx.atm.data(), cint_ctx.natm,
                           cint_ctx.bas.data(), cint_ctx.nshells,
                           const_cast<double*>(cint_ctx.env.data()), ip1_opt,
                           nullptr);
        };

        const auto t_coulomb = LogClock::now();
        ATV_DEBUG("[gradient] two_e coulomb start engine=libcint natm={} "
                  "basis={} shells={} unique_shell_quartets={}",
                  natm, n, nshells, n_unique_shell_quartets);

        for (int left = 0; left < n_shell_pairs; ++left) {
            const int ish = shell_pairs[left][0];
            const int jsh = shell_pairs[left][1];
            const int i0 = cint_ctx.shell_offsets[ish];
            const int j0 = cint_ctx.shell_offsets[jsh];
            const int di = libcint_shell_dim(cint_ctx, ish);
            const int dj = libcint_shell_dim(cint_ctx, jsh);
            const int ai = shell_to_atom[ish];
            const int aj = shell_to_atom[jsh];
            const int mult_left = pair_multiplicity(ish, jsh);

            for (int right = 0; right <= left; ++right) {
                const int ksh = shell_pairs[right][0];
                const int lsh = shell_pairs[right][1];
                const int k0 = cint_ctx.shell_offsets[ksh];
                const int l0 = cint_ctx.shell_offsets[lsh];
                const int dk = libcint_shell_dim(cint_ctx, ksh);
                const int dl = libcint_shell_dim(cint_ctx, lsh);
                const int ak = shell_to_atom[ksh];
                const int al = shell_to_atom[lsh];
                const int mult_right = pair_multiplicity(ksh, lsh);
                const int bra_ket_mult = (left == right) ? 1 : 2;

                const int shls_a[4] = {ish, jsh, ksh, lsh};
                const int shls_b[4] = {jsh, ish, ksh, lsh};
                const int shls_c[4] = {ksh, lsh, ish, jsh};
                const int shls_d[4] = {lsh, ksh, ish, jsh};
                eval_ip1(dA_buf, shls_a);
                eval_ip1(dB_buf, shls_b);
                eval_ip1(dC_buf, shls_c);
                eval_ip1(dD_buf, shls_d);

                const double quartet_prefactor =
                    0.5 * mult_left * mult_right * bra_ket_mult;
                for (int ii = 0; ii < di; ++ii) {
                    const int i = i0 + ii;
                    for (int jj = 0; jj < dj; ++jj) {
                        const int j = j0 + jj;
                        const double Pij = P_total(i, j);
                        if (std::abs(Pij) < 1e-18)
                            continue;
                        for (int kk = 0; kk < dk; ++kk) {
                            const int k = k0 + kk;
                            for (int ll = 0; ll < dl; ++ll) {
                                const int l = l0 + ll;
                                const double factor =
                                    quartet_prefactor * Pij * P_total(k, l);
                                if (std::abs(factor) < 1e-18)
                                    continue;

                                for (int axis = 0; axis < 3; ++axis) {
                                    const double dA = -dA_buf[libcint_ip1_index(
                                        axis, ii, jj, kk, ll, di, dj, dk, dl)];
                                    const double dB = -dB_buf[libcint_ip1_index(
                                        axis, jj, ii, kk, ll, dj, di, dk, dl)];
                                    const double dC = -dC_buf[libcint_ip1_index(
                                        axis, kk, ll, ii, jj, dk, dl, di, dj)];
                                    const double dD = -dD_buf[libcint_ip1_index(
                                        axis, ll, kk, ii, jj, dl, dk, di, dj)];
                                    grad(ai, axis) += factor * dA;
                                    grad(aj, axis) += factor * dB;
                                    grad(ak, axis) += factor * dC;
                                    grad(al, axis) += factor * dD;
                                }
                            }
                        }
                    }
                }
            }

            const bool should_log = (((left + 1) % progress_stride) == 0)
                                    || ((left + 1) == n_shell_pairs);
            if (should_log) {
                const long long unique_done = 1LL * (left + 1) * (left + 2) / 2;
                ATV_DEBUG("[gradient] two_e coulomb progress shell_pairs={}/{} "
                          "unique_done={}/{} ms={:.1f}",
                          left + 1, n_shell_pairs, unique_done,
                          n_unique_shell_quartets, elapsed_ms(t_coulomb));
            }
        }

        ATV_DEBUG("[gradient] two_e coulomb done ms={:.1f}",
                  elapsed_ms(t_coulomb));

        if (do_exchange) {
            const auto t_exchange = LogClock::now();
            ATV_DEBUG(
                "[gradient] two_e exchange start engine=libcint natm={} "
                "basis={} shells={} unique_shell_quartets={} alpha_hf={:.3f}",
                natm, n, nshells, n_unique_shell_quartets, alpha_hf);

            // Exchange iterates the SAME canonical (ij)×(kl) pairs as
            // Coulomb — the ERI (ij|kl) has 8-fold symmetry under
            // (i↔j), (k↔l), (bra↔ket), none of which match (i↔k) or
            // (j↔l). A previous version canonicalized (ik)+(jl), which
            // mis-counted multiplicities (e.g. (11|00) got mult=4 vs
            // the correct 2; (10|10) got mult=2 vs the correct 4) and
            // made exchange coincidentally track Coulomb so closely
            // that the total force over-cancelled by ~225×. The density
            // contraction factor for exchange remains P(i,k)·P(j,l)
            // (Szabo 3.184 with the (μλ|νσ) → (ij|kl) relabelling:
            // i=μ, j=λ, k=ν, l=σ).
            for (int left = 0; left < n_shell_pairs; ++left) {
                const int ish = shell_pairs[left][0];
                const int jsh = shell_pairs[left][1];
                const int i0 = cint_ctx.shell_offsets[ish];
                const int j0 = cint_ctx.shell_offsets[jsh];
                const int di = libcint_shell_dim(cint_ctx, ish);
                const int dj = libcint_shell_dim(cint_ctx, jsh);
                const int ai = shell_to_atom[ish];
                const int aj = shell_to_atom[jsh];
                const int mult_left = pair_multiplicity(ish, jsh);

                for (int right = 0; right <= left; ++right) {
                    const int ksh = shell_pairs[right][0];
                    const int lsh = shell_pairs[right][1];
                    const int k0 = cint_ctx.shell_offsets[ksh];
                    const int l0 = cint_ctx.shell_offsets[lsh];
                    const int dk = libcint_shell_dim(cint_ctx, ksh);
                    const int dl = libcint_shell_dim(cint_ctx, lsh);
                    const int ak = shell_to_atom[ksh];
                    const int al = shell_to_atom[lsh];
                    const int mult_right = pair_multiplicity(ksh, lsh);
                    const int bra_ket_mult = (left == right) ? 1 : 2;

                    const int shls_a[4] = {ish, jsh, ksh, lsh};
                    const int shls_b[4] = {jsh, ish, ksh, lsh};
                    const int shls_c[4] = {ksh, lsh, ish, jsh};
                    const int shls_d[4] = {lsh, ksh, ish, jsh};
                    eval_ip1(dA_buf, shls_a);
                    eval_ip1(dB_buf, shls_b);
                    eval_ip1(dC_buf, shls_c);
                    eval_ip1(dD_buf, shls_d);

                    const double quartet_prefactor =
                        mult_left * mult_right * bra_ket_mult;
                    for (int ii = 0; ii < di; ++ii) {
                        const int i = i0 + ii;
                        for (int jj = 0; jj < dj; ++jj) {
                            const int j = j0 + jj;
                            for (int kk = 0; kk < dk; ++kk) {
                                const int k = k0 + kk;
                                for (int ll = 0; ll < dl; ++ll) {
                                    const int l = l0 + ll;

                                    // Exchange density factor. The canonical enumeration
                                    // of ERI (ij|kl) uses 8-fold symmetry, but the
                                    // exchange *density* product P(i,k)·P(j,l) is NOT
                                    // symmetric under i↔j or k↔l — it maps to
                                    // P(j,k)·P(i,l) and P(i,l)·P(j,k). Symmetrize so
                                    // canonical-enumeration * mult gives the full-sum
                                    // exchange energy.
                                    //
                                    // Averaged over the 8 ERI permutations:
                                    //   sum = 4·[P(i,k)·P(j,l) + P(j,k)·P(i,l)]
                                    //   sym_factor = (1/8) · sum = ½·[P(i,k)·P(j,l) + P(j,k)·P(i,l)]
                                    // leading to the factor 0.5 inside the bracket.
                                    //
                                    // The global-hybrid admixture alpha_hf
                                    // (1.0 for HF) is a pure scalar
                                    // multiplier on top of the prefactors.
                                    double density_factor = 0.0;
                                    if (restricted) {
                                        density_factor =
                                            -0.25 * alpha_hf * 0.5
                                            * (P_total(i, k) * P_total(j, l)
                                               + P_total(j, k) * P_total(i, l));
                                    } else {
                                        density_factor =
                                            -0.5 * alpha_hf * 0.5
                                            * (Pa(i, k) * Pa(j, l)
                                               + Pa(j, k) * Pa(i, l)
                                               + Pb(i, k) * Pb(j, l)
                                               + Pb(j, k) * Pb(i, l));
                                    }

                                    const double factor =
                                        quartet_prefactor * density_factor;
                                    if (std::abs(factor) < 1e-18)
                                        continue;

                                    for (int axis = 0; axis < 3; ++axis) {
                                        const double dA =
                                            -dA_buf[libcint_ip1_index(
                                                axis, ii, jj, kk, ll, di, dj,
                                                dk, dl)];
                                        const double dB =
                                            -dB_buf[libcint_ip1_index(
                                                axis, jj, ii, kk, ll, dj, di,
                                                dk, dl)];
                                        const double dC =
                                            -dC_buf[libcint_ip1_index(
                                                axis, kk, ll, ii, jj, dk, dl,
                                                di, dj)];
                                        const double dD =
                                            -dD_buf[libcint_ip1_index(
                                                axis, ll, kk, ii, jj, dl, dk,
                                                di, dj)];
                                        grad(ai, axis) += factor * dA;
                                        grad(aj, axis) += factor * dB;
                                        grad(ak, axis) += factor * dC;
                                        grad(al, axis) += factor * dD;
                                    }
                                }
                            }
                        }
                    }
                }

                const bool should_log = (((left + 1) % progress_stride) == 0)
                                        || ((left + 1) == n_shell_pairs);
                if (should_log) {
                    const long long unique_done =
                        1LL * (left + 1) * (left + 2) / 2;
                    ATV_DEBUG("[gradient] two_e exchange progress "
                              "shell_pairs={}/{} unique_done={}/{} ms={:.1f}",
                              left + 1, n_shell_pairs, unique_done,
                              n_unique_shell_quartets, elapsed_ms(t_exchange));
                }
            }

            ATV_DEBUG("[gradient] two_e exchange done ms={:.1f}",
                      elapsed_ms(t_exchange));
        }

        CINTdel_2e_optimizer(&ip1_opt);
        ATV_DEBUG("[gradient] two_e contraction total_ms={:.1f}",
                  elapsed_ms(t_total));
        return;
    }

    if (force_native_two_electron_deriv) {
        ATV_INFO("[gradient] two_e engine=native "
                 "(force_native_two_electron_deriv=true)");
    } else {
        ATV_WARN("[gradient] libcint two_e derivative nao mismatch "
                 "libcint_nao={} basis={}, falling back to native engine",
                 cint_ctx.nao, n);
    }
#endif

    // ── Native 2e gradient: block-level + Schwarz + OpenMP + Rys ───────
    //
    // Mirrors the SCF Fock build's direct-SCF pattern in operators.cpp:
    //   1. Iterate canonical shell-BLOCK pairs (bra × ket, canonical).
    //   2. Cauchy-Schwarz gate each block quartet against Q_ij·Q_kl·|P|max.
    //   3. For each surviving quartet, compute ∂/∂R_A, ∂/∂R_B, ∂/∂R_C via
    //      `rys_eri_block_deriv` (3 block-tensor calls) and use
    //      translational invariance ∂/∂R_D = -(∂A + ∂B + ∂C).
    //   4. Contract with the density (J + K combined in one pass) and
    //      accumulate into per-thread [natm, 3] force arrays.
    //   5. Reduce thread-local arrays into `grad` at the end.
    //
    // AO canonical ordering inside the block quartet: global indices
    // (i, j, k, l) must satisfy i≥j, k≥l, (i,j)≥(k,l). When the outer
    // block pair has si>sj (distinct blocks), i>j is automatic; the
    // same holds across bra/ket when the block pairs differ. Only
    // same-block cases (si==sj, or bp_l==bp_r) need explicit AO-level
    // canonicalization.

    const ShellBlockPairList bpl{integrals, /*thr=*/1e-10};
    const int n_bpairs = static_cast<int>(bpl.pairs.size());

    // Shell-block → atom mapping (each block is single-centered).
    std::vector<int> block_to_atom(bpl.blocks.size(), -1);
    for (std::size_t b = 0; b < bpl.blocks.size(); ++b) {
        const auto& Bl = bpl.blocks[b];
        for (int a = 0; a < natm; ++a) {
            const double dx = Bl.center[0] - integrals.atoms[a].pos[0];
            const double dy = Bl.center[1] - integrals.atoms[a].pos[1];
            const double dz = Bl.center[2] - integrals.atoms[a].pos[2];
            if (dx * dx + dy * dy + dz * dz < 1e-20) {
                block_to_atom[b] = a;
                break;
            }
        }
    }

    // |P|_max for Schwarz contraction-bound.
    double D_max = 0.0;
    {
        auto max_abs = [](const xt::xtensor<double, 2>& M) {
            double m = 0.0;
            for (std::size_t i = 0; i < M.size(); ++i)
                m = std::max(m, std::abs(M.data()[i]));
            return m;
        };
        if (restricted)
            D_max = max_abs(P_total);
        else
            D_max = std::max(max_abs(Pa), max_abs(Pb));
        if (D_max < 1.0)
            D_max = 1.0; // keep Schwarz conservative at startup
    }
    const double schwarz_tol = 1e-13;

    // Per-thread gradient accumulators (avoid atomics on `grad`).
    const int nT =
#ifdef _OPENMP
        omp_get_max_threads();
#else
        1;
#endif
    std::vector<xt::xtensor<double, 2>> grad_local(
        nT, xt::zeros<double>(
                {static_cast<std::size_t>(natm), static_cast<std::size_t>(3)}));

    const auto t_two_e = LogClock::now();
    ATV_DEBUG("[gradient] two_e engine=native block-deriv natm={} basis={} "
              "blocks={} block_pairs={}",
              natm, n, bpl.blocks.size(), n_bpairs);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int left = 0; left < n_bpairs; ++left) {
        const int tid =
#ifdef _OPENMP
            omp_get_thread_num();
#else
            0;
#endif
        auto& Gl = grad_local[tid];
        const auto& bp_l = bpl.pairs[left];
        const auto& Bsi = bpl.blocks[bp_l.si];
        const auto& Bsj = bpl.blocks[bp_l.sj];
        const int na = Bsi.n_cart, nb = Bsj.n_cart;
        const int i0 = Bsi.ao_start, j0 = Bsj.ao_start;
        const int atom_i = block_to_atom[bp_l.si];
        const int atom_j = block_to_atom[bp_l.sj];
        const bool diag_bra = (bp_l.si == bp_l.sj);

        for (int right = 0; right <= left; ++right) {
            const auto& bp_r = bpl.pairs[right];
            if (bp_l.schwarz * bp_r.schwarz * D_max < schwarz_tol)
                continue;

            const auto& Bsk = bpl.blocks[bp_r.si];
            const auto& Bsl = bpl.blocks[bp_r.sj];
            const int nc = Bsk.n_cart, nd = Bsl.n_cart;
            const int k0 = Bsk.ao_start, l0 = Bsl.ao_start;
            const int atom_k = block_to_atom[bp_r.si];
            const int atom_l = block_to_atom[bp_r.sj];
            const bool diag_ket = (bp_r.si == bp_r.sj);
            const bool same_pair = (left == right);
            const int bra_ket_m = same_pair ? 1 : 2;
            const int block_n = na * nb * nc * nd;

            std::vector<double> deriv_A(3 * block_n, 0.0);
            std::vector<double> deriv_B(3 * block_n, 0.0);
            std::vector<double> deriv_C(3 * block_n, 0.0);
            native::rys_eri_block_deriv(Bsi, Bsj, Bsk, Bsl, 0, deriv_A.data());
            native::rys_eri_block_deriv(Bsi, Bsj, Bsk, Bsl, 1, deriv_B.data());
            native::rys_eri_block_deriv(Bsi, Bsj, Bsk, Bsl, 2, deriv_C.data());

            // Multiplicities are BLOCK-level (matching the libcint
            // branch above and operators.cpp's Fock build): within a
            // canonical shell-block quartet we iterate every AO tuple
            // and rely on `mult_left * mult_right * bra_ket_m` — all
            // three at block-level — to account for the 8-fold ERI
            // permutation symmetry. An additional AO-level canonical
            // check would over-restrict and under-count off-diagonal
            // contributions.
            const int mult_left = diag_bra ? 1 : 2;
            const int mult_right = diag_ket ? 1 : 2;
            const int mult = mult_left * mult_right * bra_ket_m;

            for (int ia = 0; ia < na; ++ia) {
                const int i = i0 + ia;
                for (int ib = 0; ib < nb; ++ib) {
                    const int j = j0 + ib;

                    for (int ic = 0; ic < nc; ++ic) {
                        const int k = k0 + ic;
                        for (int id = 0; id < nd; ++id) {
                            const int l = l0 + id;

                            // Coulomb factor (J): +0.5·mult·P_ij·P_kl
                            const double factor_J =
                                0.5 * mult * P_total(i, j) * P_total(k, l);

                            // Exchange factor (K) — scaled by the exact-
                            // exchange fraction alpha_hf: 1.0 for HF, 0.0 for
                            // a pure functional (K skipped), and the
                            // global-hybrid admixture for B3LYP / PBE0. The
                            // grid XC handles the semi-local exchange
                            // separately.
                            //
                            // Canonical enumeration of (ij|kl) uses 8-fold
                            // ERI symmetry, but the exchange density product
                            // P(i,k)·P(j,l) is NOT symmetric under i↔j or
                            // k↔l. Symmetrize (average over the 8 perms):
                            //   sym_factor = ½·[P(i,k)·P(j,l) + P(j,k)·P(i,l)]
                            // so canonical + mult recovers the full sum.
                            // alpha_hf multiplies that whole expression.
                            double factor_K = 0.0;
                            if (do_exchange) {
                                if (restricted) {
                                    factor_K =
                                        -0.25 * mult * alpha_hf * 0.5
                                        * (P_total(i, k) * P_total(j, l)
                                           + P_total(j, k) * P_total(i, l));
                                } else {
                                    factor_K = -0.5 * mult * alpha_hf * 0.5
                                               * (Pa(i, k) * Pa(j, l)
                                                  + Pa(j, k) * Pa(i, l)
                                                  + Pb(i, k) * Pb(j, l)
                                                  + Pb(j, k) * Pb(i, l));
                                }
                            }
                            const double factor = factor_J + factor_K;
                            if (std::abs(factor) < 1e-18)
                                continue;

                            const int ao_idx =
                                ((ia * nb + ib) * nc + ic) * nd + id;
                            for (int axis = 0; axis < 3; ++axis) {
                                const double dA =
                                    deriv_A[axis * block_n + ao_idx];
                                const double dB =
                                    deriv_B[axis * block_n + ao_idx];
                                const double dC =
                                    deriv_C[axis * block_n + ao_idx];
                                const double dD = -(dA + dB + dC);
                                Gl(atom_i, axis) += factor * dA;
                                Gl(atom_j, axis) += factor * dB;
                                Gl(atom_k, axis) += factor * dC;
                                Gl(atom_l, axis) += factor * dD;
                            }
                        }
                    }
                }
            }
        }
    }

    // Reduce thread-local force accumulators.
    for (const auto& Gl : grad_local)
        grad += Gl;

    ATV_DEBUG("[gradient] two_e block-deriv done ms={:.1f}",
              elapsed_ms(t_two_e));
    ATV_DEBUG("[gradient] two_e contraction total_ms={:.1f}",
              elapsed_ms(t_total));
}

// ═════════════════════════════════════════════════════════════════════
// Second derivatives (hf-capability-gaps-13-cphf)
//
// Double application of the angular-momentum raising formula on the
// shared primitive kernels. Raise budget: at most +2 in total across the
// centers of one integral (either +2 on one center or +1 on two), which
// the primitive buffers in native_integrals.cpp are sized for.
// ═════════════════════════════════════════════════════════════════════

namespace {

/// @brief Second derivative d²/dX_i dX_j of a primitive integral w.r.t. ONE
///        Gaussian center X, by double application of the raising formula
///        D_i f(a) = 2α f(a+e_i) − a_i f(a−e_i) on the Cartesian powers.
/// @param eval  Callable double(const int a[3]) evaluating the primitive
///              with bumped powers on center X (all else fixed).
/// @param a_ang Unbumped Cartesian powers on X [3].
/// @param i     First derivative axis (0..2).
/// @param j     Second derivative axis (0..2).
/// @param alpha Gaussian exponent on X.
/// @return d²f/dX_i dX_j (units of `eval`, per Bohr²).
template <typename Eval>
static double d2_same_center(const Eval& eval, const int a_ang[3], int i,
                             int j, double alpha) {
    double result = 0.0;
    // + branch of D_i: 2α f(a+e_i), then D_j of it.
    int ap[3] = {a_ang[0], a_ang[1], a_ang[2]};
    ap[i] += 1;
    {
        int app[3] = {ap[0], ap[1], ap[2]};
        app[j] += 1;
        result += 4.0 * alpha * alpha * eval(app);
        if (ap[j] > 0) {
            int apm[3] = {ap[0], ap[1], ap[2]};
            apm[j] -= 1;
            result -= 2.0 * alpha * ap[j] * eval(apm);
        }
    }
    // − branch of D_i: −a_i f(a−e_i), then D_j of it.
    if (a_ang[i] > 0) {
        int am[3] = {a_ang[0], a_ang[1], a_ang[2]};
        am[i] -= 1;
        int amp[3] = {am[0], am[1], am[2]};
        amp[j] += 1;
        result -= 2.0 * alpha * a_ang[i] * eval(amp);
        if (am[j] > 0) {
            int amm[3] = {am[0], am[1], am[2]};
            amm[j] -= 1;
            result += a_ang[i] * am[j] * eval(amm);
        }
    }
    return result;
}

/// @brief Mixed second derivative d²/dA_i dB_j of a primitive integral
///        w.r.t. TWO different Gaussian centers (independent raisings).
/// @param eval  Callable double(const int a[3], const int b[3]) with bumped
///              powers on both centers.
/// @param a_ang Unbumped powers on A [3]; @param b_ang on B [3].
/// @param i     Axis of the A derivative; @param j axis of the B derivative.
/// @param alpha_a Exponent on A; @param alpha_b exponent on B.
/// @return d²f/dA_i dB_j (units of `eval`, per Bohr²).
template <typename Eval2>
static double d2_cross_center(const Eval2& eval, const int a_ang[3],
                              const int b_ang[3], int i, int j, double alpha_a,
                              double alpha_b) {
    int ap[3] = {a_ang[0], a_ang[1], a_ang[2]};
    ap[i] += 1;
    int bp[3] = {b_ang[0], b_ang[1], b_ang[2]};
    bp[j] += 1;

    double result = 4.0 * alpha_a * alpha_b * eval(ap, bp);
    if (b_ang[j] > 0) {
        int bm[3] = {b_ang[0], b_ang[1], b_ang[2]};
        bm[j] -= 1;
        result -= 2.0 * alpha_a * b_ang[j] * eval(ap, bm);
    }
    if (a_ang[i] > 0) {
        int am[3] = {a_ang[0], a_ang[1], a_ang[2]};
        am[i] -= 1;
        result -= 2.0 * alpha_b * a_ang[i] * eval(am, bp);
        if (b_ang[j] > 0) {
            int bm[3] = {b_ang[0], b_ang[1], b_ang[2]};
            bm[j] -= 1;
            result += a_ang[i] * b_ang[j] * eval(am, bm);
        }
    }
    return result;
}

} // namespace

IntegralSecondDerivPack::IntegralSecondDerivPack(const IntegralPack& ipack) {
    if (ipack.spherical())
        throw std::runtime_error(
            "cpu::scf::IntegralSecondDerivPack: spherical-convention packs "
            "are not supported — the derivative c2s sandwich is a follow-up "
            "chain link (no fallback)");

    const int natm = static_cast<int>(ipack.atoms.size());
    const int n = ipack.n_basis;
    n_atoms = natm;
    n_basis = n;
    d2S = xt::zeros<double>({natm, 3, natm, 3, n, n});
    d2T = xt::zeros<double>({natm, 3, natm, 3, n, n});
    d2V = xt::zeros<double>({natm, 3, natm, 3, n, n});
    d2V_nuc = xt::zeros<double>({natm, 3, natm, 3});

    const auto& shells = ipack.shells;
    const auto& atoms = ipack.atoms;
    const std::vector<int> shell_to_atom =
        build_shell_to_atom_from_integrals(ipack);
    for (int mu = 0; mu < n; ++mu)
        if (shell_to_atom[mu] < 0)
            throw std::runtime_error(
                "cpu::scf::IntegralSecondDerivPack: shell "
                + std::to_string(mu) + " center has no matching atom");

    // Scatter helper: one (μν) value into the rank-6 tensor at
    // [(X,i),(Y,j)] and, when μ ≠ ν, into the mirrored (νμ) element.
    const auto scatter = [n](xt::xtensor<double, 6>& t, int X, int di, int Y,
                             int dj, int mu, int nu, double v) {
        t(X, di, Y, dj, mu, nu) += v;
        if (mu != nu)
            t(X, di, Y, dj, nu, mu) += v;
    };

    for (int mu = 0; mu < n; ++mu) {
        const int atom_a = shell_to_atom[mu];
        int a_ang[3];
        cart_exponents(shells[mu], a_ang);

        for (int nu = 0; nu <= mu; ++nu) {
            const int atom_b = shell_to_atom[nu];
            int b_ang[3];
            cart_exponents(shells[nu], b_ang);

            // ── d²S and d²T: two-center — every second derivative follows
            //    from AA_ij (translational invariance: ∂/∂B = −∂/∂A).
            for (int di = 0; di < 3; ++di) {
                for (int dj = di; dj < 3; ++dj) {
                    double s_aa = 0.0, t_aa = 0.0;
                    for (int p = 0; p < shells[mu].n_prim; ++p) {
                        for (int q = 0; q < shells[nu].n_prim; ++q) {
                            const double cc =
                                shells[mu].coeff[p] * shells[nu].coeff[q];
                            const double aa = shells[mu].alpha[p];
                            const double* A = shells[mu].center;
                            const double ab = shells[nu].alpha[q];
                            const double* B = shells[nu].center;
                            s_aa += cc
                                    * d2_same_center(
                                        [&](const int a[3]) {
                                            return native::overlap_prim_3d(
                                                a, b_ang, aa, A, ab, B);
                                        },
                                        a_ang, di, dj, aa);
                            t_aa += cc
                                    * d2_same_center(
                                        [&](const int a[3]) {
                                            return native::kinetic_prim_3d(
                                                a, b_ang, aa, A, ab, B);
                                        },
                                        a_ang, di, dj, aa);
                        }
                    }
                    // Supermatrix scatter (AA symmetric in (di,dj)):
                    //   AA = +v, BB = +v, AB = BA = −v.
                    const auto scatter_2c = [&](xt::xtensor<double, 6>& t,
                                                double v) {
                        const int dirs[2][2] = {{di, dj}, {dj, di}};
                        const int n_dir = (di == dj) ? 1 : 2;
                        for (int k = 0; k < n_dir; ++k) {
                            const int i = dirs[k][0], j = dirs[k][1];
                            scatter(t, atom_a, i, atom_a, j, mu, nu, v);
                            scatter(t, atom_b, i, atom_b, j, mu, nu, v);
                            scatter(t, atom_a, i, atom_b, j, mu, nu, -v);
                            scatter(t, atom_b, i, atom_a, j, mu, nu, -v);
                        }
                    };
                    scatter_2c(d2S, s_aa);
                    scatter_2c(d2T, t_aa);
                }
            }

            // ── d²V: three-center (A, B, nucleus C). Compute AA/BB/AB by
            //    raising; complete C-involving pairs by translational
            //    invariance of each derivative slot.
            for (int c = 0; c < natm; ++c) {
                double AA[3][3] = {}, BB[3][3] = {}, AB[3][3] = {};
                for (int p = 0; p < shells[mu].n_prim; ++p) {
                    for (int q = 0; q < shells[nu].n_prim; ++q) {
                        const double cc =
                            shells[mu].coeff[p] * shells[nu].coeff[q];
                        const double aa = shells[mu].alpha[p];
                        const double ab = shells[nu].alpha[q];
                        const double* A = shells[mu].center;
                        const double* B = shells[nu].center;
                        const double gamma = aa + ab;
                        const double m = aa * ab / gamma;
                        const double r2 =
                            (A[0] - B[0]) * (A[0] - B[0])
                            + (A[1] - B[1]) * (A[1] - B[1])
                            + (A[2] - B[2]) * (A[2] - B[2]);
                        const double K_AB = std::exp(-m * r2);
                        double PA[3], PB[3], PC[3];
                        for (int d = 0; d < 3; ++d) {
                            const double Pd = (aa * A[d] + ab * B[d]) / gamma;
                            PA[d] = Pd - A[d];
                            PB[d] = Pd - B[d];
                            PC[d] = Pd - atoms[c].pos[d];
                        }
                        const double w = cc * (-static_cast<double>(atoms[c].Z));
                        const auto eval_ab = [&](const int a[3],
                                                 const int b[3]) {
                            return native::nuclear_prim(a, b, gamma, PA, PB,
                                                        PC, K_AB);
                        };
                        for (int di = 0; di < 3; ++di) {
                            for (int dj = 0; dj < 3; ++dj) {
                                if (dj >= di) {
                                    AA[di][dj] += w
                                                  * d2_same_center(
                                                      [&](const int a[3]) {
                                                          return eval_ab(
                                                              a, b_ang);
                                                      },
                                                      a_ang, di, dj, aa);
                                    BB[di][dj] += w
                                                  * d2_same_center(
                                                      [&](const int b[3]) {
                                                          return eval_ab(
                                                              a_ang, b);
                                                      },
                                                      b_ang, di, dj, ab);
                                }
                                AB[di][dj] += w
                                              * d2_cross_center(eval_ab, a_ang,
                                                                b_ang, di, dj,
                                                                aa, ab);
                            }
                        }
                    }
                }
                // Symmetric completion of the same-center blocks.
                for (int di = 0; di < 3; ++di)
                    for (int dj = 0; dj < di; ++dj) {
                        AA[di][dj] = AA[dj][di];
                        BB[di][dj] = BB[dj][di];
                    }

                for (int di = 0; di < 3; ++di) {
                    for (int dj = 0; dj < 3; ++dj) {
                        const double aa_v = AA[di][dj];
                        const double bb_v = BB[di][dj];
                        const double ab_v = AB[di][dj];
                        const double ba_v = AB[dj][di]; // ∂²/∂B_i∂A_j
                        const double ac_v = -(aa_v + ab_v);
                        const double ca_v = -(aa_v + ba_v);
                        const double bc_v = -(ba_v + bb_v);
                        const double cb_v = -(ab_v + bb_v);
                        const double cc_v = aa_v + ab_v + ba_v + bb_v;
                        scatter(d2V, atom_a, di, atom_a, dj, mu, nu, aa_v);
                        scatter(d2V, atom_b, di, atom_b, dj, mu, nu, bb_v);
                        scatter(d2V, atom_a, di, atom_b, dj, mu, nu, ab_v);
                        scatter(d2V, atom_b, di, atom_a, dj, mu, nu, ba_v);
                        scatter(d2V, atom_a, di, c, dj, mu, nu, ac_v);
                        scatter(d2V, c, di, atom_a, dj, mu, nu, ca_v);
                        scatter(d2V, atom_b, di, c, dj, mu, nu, bc_v);
                        scatter(d2V, c, di, atom_b, dj, mu, nu, cb_v);
                        scatter(d2V, c, di, c, dj, mu, nu, cc_v);
                    }
                }
            }
        }
    }

    // ── Nuclear repulsion second derivative ─────────────────────────
    // V = Σ_{a<b} Z_a Z_b / |R_a − R_b|:
    //   ∂²V/∂a_i∂a_j = Z_a Z_b (3 d_i d_j / r⁵ − δ_ij / r³), d = R_a − R_b;
    //   ∂²V/∂a_i∂b_j = −∂²V/∂a_i∂a_j;  b↔b equals a↔a.
    for (int a = 0; a < natm; ++a) {
        for (int b = 0; b < a; ++b) {
            const double dx = atoms[a].pos[0] - atoms[b].pos[0];
            const double dy = atoms[a].pos[1] - atoms[b].pos[1];
            const double dz = atoms[a].pos[2] - atoms[b].pos[2];
            const double d[3] = {dx, dy, dz};
            const double r2 = dx * dx + dy * dy + dz * dz;
            const double r = std::sqrt(r2);
            const double r3_inv = 1.0 / (r * r2);
            const double r5_inv = r3_inv / r2;
            const double ZZ = static_cast<double>(atoms[a].Z)
                              * static_cast<double>(atoms[b].Z);
            for (int di = 0; di < 3; ++di) {
                for (int dj = 0; dj < 3; ++dj) {
                    const double v = ZZ
                                     * (3.0 * d[di] * d[dj] * r5_inv
                                        - (di == dj ? r3_inv : 0.0));
                    d2V_nuc(a, di, a, dj) += v;
                    d2V_nuc(b, di, b, dj) += v;
                    d2V_nuc(a, di, b, dj) -= v;
                    d2V_nuc(b, di, a, dj) -= v;
                }
            }
        }
    }
}

void accumulate_two_electron_fock_derivative(
    xt::xtensor<double, 4>& dF, const IntegralPack& integrals,
    const xt::xtensor<double, 2>& P_total) {
    const auto t_total = LogClock::now();
    const int natm = static_cast<int>(integrals.atoms.size());
    const int n = integrals.n_basis;

    const ShellBlockPairList bpl{integrals, /*thr=*/1e-12};
    const int n_blocks = static_cast<int>(bpl.blocks.size());

    std::vector<int> block_to_atom(bpl.blocks.size(), -1);
    for (std::size_t b = 0; b < bpl.blocks.size(); ++b) {
        const auto& Bl = bpl.blocks[b];
        for (int a = 0; a < natm; ++a) {
            const double dx = Bl.center[0] - integrals.atoms[a].pos[0];
            const double dy = Bl.center[1] - integrals.atoms[a].pos[1];
            const double dz = Bl.center[2] - integrals.atoms[a].pos[2];
            if (dx * dx + dy * dy + dz * dz < 1e-20) {
                block_to_atom[b] = a;
                break;
            }
        }
    }

    const int nT =
#ifdef _OPENMP
        omp_get_max_threads();
#else
        1;
#endif
    std::vector<xt::xtensor<double, 4>> dF_local(
        nT, xt::zeros<double>({natm, 3, n, n}));

    ATV_DEBUG("[hessian] two_e fock-derivative start blocks={} basis={}",
              n_blocks, n);

    // Full ORDERED block-quartet enumeration: every AO quartet (ij|kl)
    // appears exactly once, so no permutational multiplicities are needed.
    // Coulomb feeds dJ(i,j) += P(k,l)·dg; exchange dK(i,k) −= ½·P(j,l)·dg
    // (closed-shell G = J − ½K contraction with the total density).
    // O(n_blocks⁴) without Schwarz screening — Hessian-assembly scale,
    // executed once per RHF::hessian() call.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int bi = 0; bi < n_blocks; ++bi) {
        const int tid =
#ifdef _OPENMP
            omp_get_thread_num();
#else
            0;
#endif
        auto& dFl = dF_local[tid];
        const auto& BA = bpl.blocks[bi];
        std::vector<double> deriv(0);
        for (int bj = 0; bj < n_blocks; ++bj) {
            const auto& BB = bpl.blocks[bj];
            for (int bk = 0; bk < n_blocks; ++bk) {
                const auto& BC = bpl.blocks[bk];
                for (int bl = 0; bl < n_blocks; ++bl) {
                    const auto& BD = bpl.blocks[bl];
                    const int na = BA.n_cart, nb = BB.n_cart, nc = BC.n_cart,
                              nd = BD.n_cart;
                    const int block_n = na * nb * nc * nd;
                    const int atom_of[4] = {block_to_atom[bi],
                                            block_to_atom[bj],
                                            block_to_atom[bk],
                                            block_to_atom[bl]};

                    deriv.assign(static_cast<std::size_t>(4) * 3 * block_n,
                                 0.0);
                    for (int ctr = 0; ctr < 3; ++ctr)
                        native::rys_eri_block_deriv(
                            BA, BB, BC, BD, ctr,
                            deriv.data()
                                + static_cast<std::size_t>(ctr) * 3 * block_n);
                    // ∂/∂R_D = −(∂A + ∂B + ∂C).
                    for (int k = 0; k < 3 * block_n; ++k)
                        deriv[static_cast<std::size_t>(3) * 3 * block_n + k] =
                            -(deriv[k]
                              + deriv[static_cast<std::size_t>(1) * 3 * block_n
                                      + k]
                              + deriv[static_cast<std::size_t>(2) * 3 * block_n
                                      + k]);

                    for (int ia = 0; ia < na; ++ia) {
                        const int i = BA.ao_start + ia;
                        for (int ib = 0; ib < nb; ++ib) {
                            const int j = BB.ao_start + ib;
                            for (int ic = 0; ic < nc; ++ic) {
                                const int k = BC.ao_start + ic;
                                for (int id = 0; id < nd; ++id) {
                                    const int l = BD.ao_start + id;
                                    const int ao =
                                        ((ia * nb + ib) * nc + ic) * nd + id;
                                    const double pj = P_total(k, l);
                                    const double pk = P_total(j, l);
                                    for (int ctr = 0; ctr < 4; ++ctr) {
                                        const int X = atom_of[ctr];
                                        for (int ax = 0; ax < 3; ++ax) {
                                            const double dg = deriv
                                                [(static_cast<std::size_t>(ctr)
                                                      * 3
                                                  + ax)
                                                     * block_n
                                                 + ao];
                                            if (dg == 0.0)
                                                continue;
                                            dFl(X, ax, i, j) += pj * dg;
                                            dFl(X, ax, i, k) -= 0.5 * pk * dg;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (const auto& dFl : dF_local)
        dF += dFl;

    ATV_DEBUG("[hessian] two_e fock-derivative done ms={:.1f}",
              elapsed_ms(t_total));
}

// ── Two-electron second-derivative direct contraction ────────────────

namespace {

/// t-index dimension of the local Hermite E tables: t ≤ ai + bj with
/// ai ≤ kMaxLOrbital + 2 (second-derivative raise) and bj ≤ kMaxLOrbital.
constexpr int kD2EDim = 2 * kMaxLOrbital + 3 + 2; // 11 for f-shells

/// Supermatrix dimension of one quartet's second-derivative block:
/// 4 centers × 3 axes.
constexpr int kSuper = 12;

/// @brief Local McMurchie-Davidson Hermite coefficients E^{ai,bj}_t
///        supporting ai, bj ≤ kMaxLOrbital + 2 (the +2 derivative raise).
///        Zero-fills all kD2EDim entries; populated for t = 0..ai+bj.
/// @param ai,bj  Cartesian powers on the two centers (one axis).
/// @param PA,PB  Gaussian-product offsets P−A, P−B on this axis [Bohr].
/// @param inv2g  1/(2γ) [1/Bohr²-exponent units].
/// @param E      Output, length kD2EDim.
void md_e_table(int ai, int bj, double PA, double PB, double inv2g,
                double* E) {
    for (int t = 0; t < kD2EDim; ++t)
        E[t] = 0.0;

    double E_buf[kMaxLOrbital + 3][kD2EDim] = {};
    E_buf[0][0] = 1.0;
    for (int i = 0; i < ai; ++i) {
        const int tmax = i + 1;
        for (int t = 0; t <= tmax; ++t) {
            const double etm1 = (t > 0) ? E_buf[i][t - 1] : 0.0;
            const double et = E_buf[i][t];
            const double etp1 = (t + 1 < kD2EDim) ? E_buf[i][t + 1] : 0.0;
            E_buf[i + 1][t] = inv2g * etm1 + PA * et + (t + 1) * etp1;
        }
    }

    double W[2][kD2EDim] = {};
    int cur = 0;
    for (int t = 0; t < kD2EDim; ++t)
        W[cur][t] = E_buf[ai][t];
    for (int jj = 0; jj < bj; ++jj) {
        const int nxt = 1 - cur;
        for (int t = 0; t < kD2EDim; ++t)
            W[nxt][t] = 0.0;
        const int tmax = ai + jj + 1;
        for (int t = 0; t <= tmax && t < kD2EDim; ++t) {
            const double etm1 = (t > 0) ? W[cur][t - 1] : 0.0;
            const double et = W[cur][t];
            const double etp1 = (t + 1 < kD2EDim) ? W[cur][t + 1] : 0.0;
            W[nxt][t] = inv2g * etm1 + PB * et + (t + 1) * etp1;
        }
        cur = nxt;
    }
    for (int t = 0; t <= ai + bj && t < kD2EDim; ++t)
        E[t] = W[cur][t];
}

/// Max per-axis contraction index of one side (bra or ket) after a +2
/// raise: l + l + 2 ≤ 2·kMaxLOrbital + 2, so dimension 2·kMaxLOrbital + 3.
constexpr int kSideDim = 2 * kMaxLOrbital + 3; // 9 for f-shells

/// @brief All second derivatives of one contracted shell-block quartet.
///
/// Fills `out[combo * 144 + su * 12 + sv]` with ∂²(AB|CD)/∂(su)∂(sv)
/// [Hartree/Bohr²], where su = center·3 + axis over the supermatrix order
/// (A, B, C, D) × (x, y, z) and `combo` is the AO tuple
/// ((ia·nb + ib)·nc + ic)·nd + id. Center-D rows/columns follow from
/// translational invariance. One shared R-table (built to Ltot+2) and the
/// bumped Hermite E tables serve every element; nothing four-index is
/// materialized beyond this quartet block.
void quartet_second_derivatives(const native::ShellBlock& A,
                                const native::ShellBlock& B,
                                const native::ShellBlock& C,
                                const native::ShellBlock& D,
                                std::vector<double>& out,
                                std::vector<double>& R_scratch) {
    const int na = A.n_cart, nb = B.n_cart, nc = C.n_cart, nd = D.n_cart;
    const int n_combo = na * nb * nc * nd;
    out.assign(static_cast<std::size_t>(n_combo) * kSuper * kSuper, 0.0);

    int compsA[kMaxNCart][3], compsB[kMaxNCart][3], compsC[kMaxNCart][3],
        compsD[kMaxNCart][3];
    fill_cart_comps(A.l, compsA);
    fill_cart_comps(B.l, compsB);
    fill_cart_comps(C.l, compsC);
    fill_cart_comps(D.l, compsD);

    const int Ltot = A.l + B.l + C.l + D.l;
    const int Lmax2 = Ltot + 2;   // deepest Boys order / R index sum
    const int Ldim = Ltot + 3;    // per-axis + m dimension of the R table
    const std::size_t Rsz = static_cast<std::size_t>(Ldim) * Ldim * Ldim
                            * Ldim;
    R_scratch.resize(Rsz);
    const auto Rat = [&](int m, int t, int u, int v) -> double& {
        return R_scratch[((static_cast<std::size_t>(m) * Ldim + t) * Ldim + u)
                             * Ldim
                         + v];
    };

    // Hermite E tables, bra and ket, with both indices extended by +2.
    const int adim = A.l + 3, bdim = B.l + 3, cdim = C.l + 3, ddim = D.l + 3;
    std::vector<double> Ebra(static_cast<std::size_t>(3) * adim * bdim
                             * kD2EDim);
    std::vector<double> Eket(static_cast<std::size_t>(3) * cdim * ddim
                             * kD2EDim);
    const auto EB = [&](int d, int ia, int jb) -> const double* {
        return &Ebra[((static_cast<std::size_t>(d) * adim + ia) * bdim + jb)
                     * kD2EDim];
    };
    const auto EK = [&](int d, int ic, int jd) -> const double* {
        return &Eket[((static_cast<std::size_t>(d) * cdim + ic) * ddim + jd)
                     * kD2EDim];
    };

    for (int p = 0; p < A.n_prim; ++p) {
        for (int q = 0; q < B.n_prim; ++q) {
            for (int r = 0; r < C.n_prim; ++r) {
                for (int s = 0; s < D.n_prim; ++s) {
                    const double aa = A.alpha[p], ab = B.alpha[q],
                                 ac = C.alpha[r], ad = D.alpha[s];
                    const double coeff = A.coeff[p] * B.coeff[q] * C.coeff[r]
                                         * D.coeff[s];
                    const double g1 = aa + ab, g2 = ac + ad;
                    const double delta = g1 + g2;
                    const double rho = g1 * g2 / delta;
                    const double inv2g1 = 0.5 / g1, inv2g2 = 0.5 / g2;

                    double P[3], Q[3], PA[3], PB[3], QC[3], QD[3], PQ[3];
                    double R2_AB = 0.0, R2_CD = 0.0, PQ2 = 0.0;
                    for (int d = 0; d < 3; ++d) {
                        P[d] = (aa * A.center[d] + ab * B.center[d]) / g1;
                        Q[d] = (ac * C.center[d] + ad * D.center[d]) / g2;
                        PA[d] = P[d] - A.center[d];
                        PB[d] = P[d] - B.center[d];
                        QC[d] = Q[d] - C.center[d];
                        QD[d] = Q[d] - D.center[d];
                        PQ[d] = P[d] - Q[d];
                        const double dAB = A.center[d] - B.center[d];
                        const double dCD = C.center[d] - D.center[d];
                        R2_AB += dAB * dAB;
                        R2_CD += dCD * dCD;
                        PQ2 += PQ[d] * PQ[d];
                    }
                    const double K_AB = std::exp(-aa * ab / g1 * R2_AB);
                    const double K_CD = std::exp(-ac * ad / g2 * R2_CD);
                    const double T = rho * PQ2;
                    const double pref = 2.0 * std::pow(M_PI, 2.5)
                                        / (g1 * g2 * std::sqrt(delta)) * K_AB
                                        * K_CD * coeff;

                    // Shared R-table to order Ltot + 2.
                    std::fill(R_scratch.begin(), R_scratch.end(), 0.0);
                    {
                        double neg2rho = -2.0 * rho;
                        double pw = 1.0;
                        for (int m = 0; m <= Lmax2; ++m) {
                            Rat(m, 0, 0, 0) = pw * native::boys_fn(m, T);
                            pw *= neg2rho;
                        }
                        for (int m = Lmax2 - 1; m >= 0; --m) {
                            const int max_tuv = Lmax2 - m;
                            for (int t = 1; t <= max_tuv; ++t) {
                                double v0 = PQ[0] * Rat(m + 1, t - 1, 0, 0);
                                if (t >= 2)
                                    v0 += (t - 1) * Rat(m + 1, t - 2, 0, 0);
                                Rat(m, t, 0, 0) = v0;
                            }
                            for (int t = 0; t <= max_tuv; ++t) {
                                for (int u = 1; t + u <= max_tuv; ++u) {
                                    double v0 =
                                        PQ[1] * Rat(m + 1, t, u - 1, 0);
                                    if (u >= 2)
                                        v0 += (u - 1)
                                              * Rat(m + 1, t, u - 2, 0);
                                    Rat(m, t, u, 0) = v0;
                                }
                            }
                            for (int t = 0; t <= max_tuv; ++t) {
                                for (int u = 0; t + u <= max_tuv; ++u) {
                                    for (int v = 1; t + u + v <= max_tuv;
                                         ++v) {
                                        double v0 = PQ[2]
                                                    * Rat(m + 1, t, u, v - 1);
                                        if (v >= 2)
                                            v0 += (v - 1)
                                                  * Rat(m + 1, t, u, v - 2);
                                        Rat(m, t, u, v) = v0;
                                    }
                                }
                            }
                        }
                    }

                    // Bumped Hermite E tables for this primitive quartet.
                    for (int d = 0; d < 3; ++d) {
                        for (int ia = 0; ia < adim; ++ia)
                            for (int jb = 0; jb < bdim; ++jb)
                                md_e_table(ia, jb, PA[d], PB[d], inv2g1,
                                           const_cast<double*>(EB(d, ia, jb)));
                        for (int ic = 0; ic < cdim; ++ic)
                            for (int jd = 0; jd < ddim; ++jd)
                                md_e_table(ic, jd, QC[d], QD[d], inv2g2,
                                           const_cast<double*>(EK(d, ic, jd)));
                    }

                    // Per-AO-combo assembly.
                    for (int ia = 0; ia < na; ++ia) {
                        const int* a = compsA[ia];
                        for (int ib = 0; ib < nb; ++ib) {
                            const int* b = compsB[ib];
                            for (int ic = 0; ic < nc; ++ic) {
                                const int* c = compsC[ic];
                                for (int id = 0; id < nd; ++id) {
                                    const int* dd = compsD[id];
                                    const int combo =
                                        ((ia * nb + ib) * nc + ic) * nd + id;
                                    double* M =
                                        &out[static_cast<std::size_t>(combo)
                                             * kSuper * kSuper];

                                    // Per-axis vectors: base + derivatives.
                                    double vA[3][kD2EDim], vB[3][kD2EDim],
                                        vAA[3][kD2EDim], vBB[3][kD2EDim],
                                        vAB[3][kD2EDim], vC[3][kD2EDim],
                                        vCC[3][kD2EDim];
                                    const double* Eb0[3];
                                    const double* Ek0[3];
                                    for (int d = 0; d < 3; ++d) {
                                        const int ax = a[d], bx = b[d];
                                        const int cx = c[d], dx = dd[d];
                                        Eb0[d] = EB(d, ax, bx);
                                        Ek0[d] = EK(d, cx, dx);
                                        const double* Ap1 = EB(d, ax + 1, bx);
                                        const double* Am1 =
                                            (ax > 0) ? EB(d, ax - 1, bx)
                                                     : nullptr;
                                        const double* Bp1 = EB(d, ax, bx + 1);
                                        const double* Bm1 =
                                            (bx > 0) ? EB(d, ax, bx - 1)
                                                     : nullptr;
                                        const double* Ap2 = EB(d, ax + 2, bx);
                                        const double* Am2 =
                                            (ax > 1) ? EB(d, ax - 2, bx)
                                                     : nullptr;
                                        const double* Bp2 = EB(d, ax, bx + 2);
                                        const double* Bm2 =
                                            (bx > 1) ? EB(d, ax, bx - 2)
                                                     : nullptr;
                                        const double* ApBp =
                                            EB(d, ax + 1, bx + 1);
                                        const double* ApBm =
                                            (bx > 0) ? EB(d, ax + 1, bx - 1)
                                                     : nullptr;
                                        const double* AmBp =
                                            (ax > 0) ? EB(d, ax - 1, bx + 1)
                                                     : nullptr;
                                        const double* AmBm =
                                            (ax > 0 && bx > 0)
                                                ? EB(d, ax - 1, bx - 1)
                                                : nullptr;
                                        const double* Cp1 = EK(d, cx + 1, dx);
                                        const double* Cm1 =
                                            (cx > 0) ? EK(d, cx - 1, dx)
                                                     : nullptr;
                                        const double* Cp2 = EK(d, cx + 2, dx);
                                        const double* Cm2 =
                                            (cx > 1) ? EK(d, cx - 2, dx)
                                                     : nullptr;
                                        for (int t = 0; t < kD2EDim; ++t) {
                                            vA[d][t] =
                                                2.0 * aa * Ap1[t]
                                                - (Am1 ? ax * Am1[t] : 0.0);
                                            vB[d][t] =
                                                2.0 * ab * Bp1[t]
                                                - (Bm1 ? bx * Bm1[t] : 0.0);
                                            vAA[d][t] =
                                                4.0 * aa * aa * Ap2[t]
                                                - 2.0 * aa * (2 * ax + 1)
                                                      * Eb0[d][t]
                                                + (Am2 ? ax * (ax - 1) * Am2[t]
                                                       : 0.0);
                                            vBB[d][t] =
                                                4.0 * ab * ab * Bp2[t]
                                                - 2.0 * ab * (2 * bx + 1)
                                                      * Eb0[d][t]
                                                + (Bm2 ? bx * (bx - 1) * Bm2[t]
                                                       : 0.0);
                                            vAB[d][t] =
                                                4.0 * aa * ab * ApBp[t]
                                                - (ApBm ? 2.0 * aa * bx
                                                              * ApBm[t]
                                                        : 0.0)
                                                - (AmBp ? 2.0 * ab * ax
                                                              * AmBp[t]
                                                        : 0.0)
                                                + (AmBm ? static_cast<double>(
                                                              ax * bx)
                                                              * AmBm[t]
                                                        : 0.0);
                                            vC[d][t] =
                                                2.0 * ac * Cp1[t]
                                                - (Cm1 ? cx * Cm1[t] : 0.0);
                                            vCC[d][t] =
                                                4.0 * ac * ac * Cp2[t]
                                                - 2.0 * ac * (2 * cx + 1)
                                                      * Ek0[d][t]
                                                + (Cm2 ? cx * (cx - 1) * Cm2[t]
                                                       : 0.0);
                                        }
                                    }

                                    const int braP[3] = {a[0] + b[0],
                                                         a[1] + b[1],
                                                         a[2] + b[2]};
                                    const int ketP[3] = {c[0] + dd[0],
                                                         c[1] + dd[1],
                                                         c[2] + dd[2]};

                                    // Ket-contracted partial (plain ket,
                                    // sign folded in): Gket[t][u][v].
                                    double Gket[kSideDim][kSideDim][kSideDim];
                                    for (int t = 0; t <= braP[0] + 2; ++t)
                                        for (int u = 0; u <= braP[1] + 2; ++u)
                                            for (int v = 0; v <= braP[2] + 2;
                                                 ++v) {
                                                double acc = 0.0;
                                                for (int tau = 0;
                                                     tau <= ketP[0]; ++tau)
                                                    for (int nu = 0;
                                                         nu <= ketP[1]; ++nu)
                                                        for (int ph = 0;
                                                             ph <= ketP[2];
                                                             ++ph) {
                                                            const double e2 =
                                                                Ek0[0][tau]
                                                                * Ek0[1][nu]
                                                                * Ek0[2][ph];
                                                            if (e2 == 0.0)
                                                                continue;
                                                            const int sg =
                                                                ((tau + nu
                                                                  + ph)
                                                                 & 1)
                                                                    ? -1
                                                                    : 1;
                                                            acc +=
                                                                sg * e2
                                                                * Rat(0,
                                                                      t + tau,
                                                                      u + nu,
                                                                      v + ph);
                                                        }
                                                Gket[t][u][v] = acc;
                                            }

                                    // Bra-contracted partial (plain bra, no
                                    // sign): Hbra[tau][nu][phi].
                                    double Hbra[kSideDim][kSideDim][kSideDim];
                                    for (int tau = 0; tau <= ketP[0] + 2;
                                         ++tau)
                                        for (int nu = 0; nu <= ketP[1] + 2;
                                             ++nu)
                                            for (int ph = 0; ph <= ketP[2] + 2;
                                                 ++ph) {
                                                double acc = 0.0;
                                                for (int t = 0; t <= braP[0];
                                                     ++t)
                                                    for (int u = 0;
                                                         u <= braP[1]; ++u)
                                                        for (int v = 0;
                                                             v <= braP[2];
                                                             ++v) {
                                                            const double e1 =
                                                                Eb0[0][t]
                                                                * Eb0[1][u]
                                                                * Eb0[2][v];
                                                            if (e1 == 0.0)
                                                                continue;
                                                            acc +=
                                                                e1
                                                                * Rat(0,
                                                                      t + tau,
                                                                      u + nu,
                                                                      v + ph);
                                                        }
                                                Hbra[tau][nu][ph] = acc;
                                            }

                                    // Bra-derivative partials K[X][i]
                                    // (X = A(0) or B(1), first-derivative
                                    // vector on axis i; ket indices free
                                    // to +1).
                                    static thread_local std::vector<double>
                                        Kbuf;
                                    Kbuf.assign(static_cast<std::size_t>(2) * 3
                                                    * kSideDim * kSideDim
                                                    * kSideDim,
                                                0.0);
                                    const auto Kat = [&](int X, int i, int tau,
                                                         int nu,
                                                         int ph) -> double& {
                                        return Kbuf
                                            [((((static_cast<std::size_t>(X)
                                                     * 3
                                                 + i)
                                                    * kSideDim
                                                + tau)
                                                   * kSideDim
                                               + nu)
                                                  * kSideDim
                                              + ph)];
                                    };
                                    for (int X = 0; X < 2; ++X) {
                                        for (int i = 0; i < 3; ++i) {
                                            const double* wx =
                                                (i == 0)
                                                    ? (X == 0 ? vA[0] : vB[0])
                                                    : Eb0[0];
                                            const double* wy =
                                                (i == 1)
                                                    ? (X == 0 ? vA[1] : vB[1])
                                                    : Eb0[1];
                                            const double* wz =
                                                (i == 2)
                                                    ? (X == 0 ? vA[2] : vB[2])
                                                    : Eb0[2];
                                            const int bx =
                                                braP[0] + (i == 0 ? 1 : 0);
                                            const int by =
                                                braP[1] + (i == 1 ? 1 : 0);
                                            const int bz =
                                                braP[2] + (i == 2 ? 1 : 0);
                                            for (int tau = 0;
                                                 tau <= ketP[0] + 1; ++tau)
                                                for (int nu = 0;
                                                     nu <= ketP[1] + 1; ++nu)
                                                    for (int ph = 0;
                                                         ph <= ketP[2] + 1;
                                                         ++ph) {
                                                        double acc = 0.0;
                                                        for (int t = 0;
                                                             t <= bx; ++t)
                                                            for (int u = 0;
                                                                 u <= by; ++u)
                                                                for (int v = 0;
                                                                     v <= bz;
                                                                     ++v) {
                                                                    const double
                                                                        e1 =
                                                                            wx[t]
                                                                            * wy[u]
                                                                            * wz[v];
                                                                    if (e1
                                                                        == 0.0)
                                                                        continue;
                                                                    acc += e1
                                                                           * Rat(
                                                                               0,
                                                                               t + tau,
                                                                               u + nu,
                                                                               v + ph);
                                                                }
                                                        Kat(X, i, tau, nu,
                                                            ph) = acc;
                                                    }
                                        }
                                    }

                                    // ── Elements among {A, B} (bra side) ──
                                    const auto bra_elem =
                                        [&](const double* wx, int bx,
                                            const double* wy, int by,
                                            const double* wz, int bz) {
                                            double acc = 0.0;
                                            for (int t = 0; t <= bx; ++t)
                                                for (int u = 0; u <= by; ++u)
                                                    for (int v = 0; v <= bz;
                                                         ++v) {
                                                        const double e1 =
                                                            wx[t] * wy[u]
                                                            * wz[v];
                                                        if (e1 != 0.0)
                                                            acc += e1
                                                                   * Gket[t][u]
                                                                         [v];
                                                    }
                                            return acc;
                                        };
                                    // ── Elements among {C} (ket side) ──
                                    const auto ket_elem =
                                        [&](const double* wx, int kx,
                                            const double* wy, int ky,
                                            const double* wz, int kz) {
                                            double acc = 0.0;
                                            for (int tau = 0; tau <= kx;
                                                 ++tau)
                                                for (int nu = 0; nu <= ky;
                                                     ++nu)
                                                    for (int ph = 0; ph <= kz;
                                                         ++ph) {
                                                        const double e2 =
                                                            wx[tau] * wy[nu]
                                                            * wz[ph];
                                                        if (e2 == 0.0)
                                                            continue;
                                                        const int sg =
                                                            ((tau + nu + ph)
                                                             & 1)
                                                                ? -1
                                                                : 1;
                                                        acc += sg * e2
                                                               * Hbra[tau][nu]
                                                                     [ph];
                                                    }
                                            return acc;
                                        };
                                    // ── Mixed bra-ket elements ──
                                    const auto mixed_elem = [&](int X, int i,
                                                                int j) {
                                        const double* wx =
                                            (j == 0) ? vC[0] : Ek0[0];
                                        const double* wy =
                                            (j == 1) ? vC[1] : Ek0[1];
                                        const double* wz =
                                            (j == 2) ? vC[2] : Ek0[2];
                                        const int kx =
                                            ketP[0] + (j == 0 ? 1 : 0);
                                        const int ky =
                                            ketP[1] + (j == 1 ? 1 : 0);
                                        const int kz =
                                            ketP[2] + (j == 2 ? 1 : 0);
                                        double acc = 0.0;
                                        for (int tau = 0; tau <= kx; ++tau)
                                            for (int nu = 0; nu <= ky; ++nu)
                                                for (int ph = 0; ph <= kz;
                                                     ++ph) {
                                                    const double e2 =
                                                        wx[tau] * wy[nu]
                                                        * wz[ph];
                                                    if (e2 == 0.0)
                                                        continue;
                                                    const int sg =
                                                        ((tau + nu + ph) & 1)
                                                            ? -1
                                                            : 1;
                                                    acc += sg * e2
                                                           * Kat(X, i, tau, nu,
                                                                 ph);
                                                }
                                        return acc;
                                    };

                                    for (int su = 0; su < 9; ++su) {
                                        const int cX = su / 3, i = su % 3;
                                        for (int sv = su; sv < 9; ++sv) {
                                            const int cY = sv / 3, j = sv % 3;
                                            double val = 0.0;
                                            if (cY <= 1) {
                                                // (A,A), (A,B), (B,B)
                                                const double* wx = Eb0[0];
                                                const double* wy = Eb0[1];
                                                const double* wz = Eb0[2];
                                                const double* w[3] = {wx, wy,
                                                                      wz};
                                                int bnd[3] = {braP[0], braP[1],
                                                              braP[2]};
                                                if (i == j && cX == cY) {
                                                    w[i] = (cX == 0) ? vAA[i]
                                                                     : vBB[i];
                                                    bnd[i] += 2;
                                                } else if (i == j) {
                                                    // cX = A, cY = B
                                                    w[i] = vAB[i];
                                                    bnd[i] += 2;
                                                } else {
                                                    w[i] = (cX == 0) ? vA[i]
                                                                     : vB[i];
                                                    w[j] = (cY == 0) ? vA[j]
                                                                     : vB[j];
                                                    bnd[i] += 1;
                                                    bnd[j] += 1;
                                                }
                                                val = bra_elem(w[0], bnd[0],
                                                               w[1], bnd[1],
                                                               w[2], bnd[2]);
                                            } else if (cX == 2) {
                                                // (C,C)
                                                const double* w[3] = {
                                                    Ek0[0], Ek0[1], Ek0[2]};
                                                int bnd[3] = {ketP[0], ketP[1],
                                                              ketP[2]};
                                                if (i == j) {
                                                    w[i] = vCC[i];
                                                    bnd[i] += 2;
                                                } else {
                                                    w[i] = vC[i];
                                                    w[j] = vC[j];
                                                    bnd[i] += 1;
                                                    bnd[j] += 1;
                                                }
                                                val = ket_elem(w[0], bnd[0],
                                                               w[1], bnd[1],
                                                               w[2], bnd[2]);
                                            } else {
                                                // (A,C) or (B,C)
                                                val = mixed_elem(cX, i, j);
                                            }
                                            val *= pref;
                                            M[su * kSuper + sv] += val;
                                            if (su != sv)
                                                M[sv * kSuper + su] += val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Translational-invariance completion: center-D rows and columns.
    for (int combo = 0; combo < n_combo; ++combo) {
        double* M = &out[static_cast<std::size_t>(combo) * kSuper * kSuper];
        for (int i = 0; i < 3; ++i) {
            for (int sv = 0; sv < 9; ++sv) {
                double acc = 0.0;
                for (int cX = 0; cX < 3; ++cX)
                    acc += M[(cX * 3 + i) * kSuper + sv];
                M[(9 + i) * kSuper + sv] = -acc;
                M[sv * kSuper + (9 + i)] = -acc;
            }
        }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double acc = 0.0;
                for (int cX = 0; cX < 3; ++cX)
                    acc += M[(cX * 3 + i) * kSuper + (9 + j)];
                M[(9 + i) * kSuper + (9 + j)] = -acc;
            }
    }
}

} // namespace

void accumulate_two_electron_hessian(xt::xtensor<double, 2>& hess,
                                     const IntegralPack& integrals,
                                     const xt::xtensor<double, 2>& P_total) {
    const auto t_total = LogClock::now();
    const int natm = static_cast<int>(integrals.atoms.size());
    const int n = integrals.n_basis;
    (void)n;

    const ShellBlockPairList bpl{integrals, /*thr=*/1e-12};
    const int n_bpairs = static_cast<int>(bpl.pairs.size());

    std::vector<int> block_to_atom(bpl.blocks.size(), -1);
    for (std::size_t b = 0; b < bpl.blocks.size(); ++b) {
        const auto& Bl = bpl.blocks[b];
        for (int a = 0; a < natm; ++a) {
            const double dx = Bl.center[0] - integrals.atoms[a].pos[0];
            const double dy = Bl.center[1] - integrals.atoms[a].pos[1];
            const double dz = Bl.center[2] - integrals.atoms[a].pos[2];
            if (dx * dx + dy * dy + dz * dz < 1e-20) {
                block_to_atom[b] = a;
                break;
            }
        }
    }

    double D_max = 0.0;
    for (std::size_t i = 0; i < P_total.size(); ++i)
        D_max = std::max(D_max, std::abs(P_total.data()[i]));
    if (D_max < 1.0)
        D_max = 1.0;
    const double schwarz_tol = 1e-14;

    const int nT =
#ifdef _OPENMP
        omp_get_max_threads();
#else
        1;
#endif
    std::vector<xt::xtensor<double, 2>> hess_local(
        nT, xt::zeros<double>({3 * natm, 3 * natm}));

    ATV_DEBUG("[hessian] two_e second-derivative start blocks={} "
              "block_pairs={}",
              bpl.blocks.size(), n_bpairs);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int left = 0; left < n_bpairs; ++left) {
        const int tid =
#ifdef _OPENMP
            omp_get_thread_num();
#else
            0;
#endif
        auto& Hl = hess_local[tid];
        std::vector<double> d2buf;
        std::vector<double> R_scratch;
        const auto& bp_l = bpl.pairs[left];
        const auto& BA = bpl.blocks[bp_l.si];
        const auto& BB = bpl.blocks[bp_l.sj];
        const int na = BA.n_cart, nb = BB.n_cart;
        const int i0 = BA.ao_start, j0 = BB.ao_start;
        const bool diag_bra = (bp_l.si == bp_l.sj);

        for (int right = 0; right <= left; ++right) {
            const auto& bp_r = bpl.pairs[right];
            if (bp_l.schwarz * bp_r.schwarz * D_max < schwarz_tol)
                continue;

            const auto& BC = bpl.blocks[bp_r.si];
            const auto& BD = bpl.blocks[bp_r.sj];
            const int nc = BC.n_cart, nd = BD.n_cart;
            const int k0 = BC.ao_start, l0 = BD.ao_start;
            const bool diag_ket = (bp_r.si == bp_r.sj);
            const bool same_pair = (left == right);
            const int mult = (diag_bra ? 1 : 2) * (diag_ket ? 1 : 2)
                             * (same_pair ? 1 : 2);

            const int atom_of[4] = {
                block_to_atom[bp_l.si], block_to_atom[bp_l.sj],
                block_to_atom[bp_r.si], block_to_atom[bp_r.sj]};

            quartet_second_derivatives(BA, BB, BC, BD, d2buf, R_scratch);

            for (int ia = 0; ia < na; ++ia) {
                const int i = i0 + ia;
                for (int ib = 0; ib < nb; ++ib) {
                    const int j = j0 + ib;
                    for (int ic = 0; ic < nc; ++ic) {
                        const int k = k0 + ic;
                        for (int id = 0; id < nd; ++id) {
                            const int l = l0 + id;
                            const double factor_J =
                                0.5 * mult * P_total(i, j) * P_total(k, l);
                            const double factor_K =
                                -0.25 * mult * 0.5
                                * (P_total(i, k) * P_total(j, l)
                                   + P_total(j, k) * P_total(i, l));
                            const double factor = factor_J + factor_K;
                            if (std::abs(factor) < 1e-18)
                                continue;
                            const int combo =
                                ((ia * nb + ib) * nc + ic) * nd + id;
                            const double* M =
                                &d2buf[static_cast<std::size_t>(combo)
                                       * kSuper * kSuper];
                            for (int su = 0; su < kSuper; ++su) {
                                const int row =
                                    3 * atom_of[su / 3] + su % 3;
                                for (int sv = 0; sv < kSuper; ++sv) {
                                    const int col =
                                        3 * atom_of[sv / 3] + sv % 3;
                                    Hl(row, col) +=
                                        factor * M[su * kSuper + sv];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (const auto& Hl : hess_local)
        hess += Hl;

    ATV_DEBUG("[hessian] two_e second-derivative done ms={:.1f}",
              elapsed_ms(t_total));
}

} // namespace scf
} // namespace cpu
} // namespace ATV

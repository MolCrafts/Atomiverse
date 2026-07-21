/// @file tests/libcint_reference.cuh
/// @brief libcint ground-truth reference for integral validation tests.
///
/// Isolates ALL libcint C-API interaction into a single test-only header.
/// Gated by ATV_HAS_LIBCINT — only available when -DATV_BUILD_LIBCINT=ON.
///
/// Supports s+p type shells for elements Z=1..8 (STO-3G basis) via
/// `build_integrals_libcint`, and any resolved IntegralPack including
/// d- and f-polarized sets (6-31G*, def2-SVP, def2-TZVP) via
/// `build_eri_libcint_from_pack`.
#pragma once

#ifdef ATV_HAS_LIBCINT

extern "C" {
#include <cint.h>
#include <cint_funcs.h>
}
#undef atm
#undef bas

#include "atomiverse/core/gto.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/basis_data/sto3g.h"
#include "atomiverse/cpu/scf/internal/integrals.h"

#include <cmath>
#include <numeric>
#include <vector>

namespace ATV {
namespace test {

/// @brief Wrap CINTgto_norm for test comparison.
inline double libcint_gto_norm(int l, double alpha) {
    return CINTgto_norm(l, alpha);
}

/// @brief Normalize contracted AO coefficients (PySCF convention).
/// After primitive normalization (coeff *= gto_norm), divide by the
/// contracted orbital norm so that <φ|φ> = 1. This matches the
/// normalization applied by libcint_bridge.cpp and PySCF's make_env.
inline void normalize_contracted_coeffs(int l, const double* exps,
                                        double* coeffs, int nprim) {
    double norm2 = 0.0;
    for (int p = 0; p < nprim; ++p)
        for (int q = 0; q < nprim; ++q) {
            double n1 = 0.5 * (l * 2 + 2 + 1);
            norm2 += coeffs[p] * coeffs[q] * 0.5 * std::tgamma(n1)
                     / std::pow(exps[p] + exps[q], n1);
        }
    double scale = 1.0 / std::sqrt(norm2);
    for (int p = 0; p < nprim; ++p)
        coeffs[p] *= scale;
}

/// @brief Compute STO-3G integrals via libcint (ground truth).
///
/// Mirrors the logic in build_integrals but uses libcint as the
/// integral engine. Supports elements Z=1..8 with s and p type shells.
/// Used exclusively for validation — never in production code.
inline cpu::scf::IntegralPack build_integrals_libcint(const cpu::Frame& frame) {
    int natm = static_cast<int>(frame.nlocal());
    int n_electrons = 0;
    for (int i = 0; i < natm; ++i)
        n_electrons += frame.span(ATV::kAtom_type)[i];

    // Count basis functions and shells from the checked-in basis_data table.
    // Generic table already separates s and p shells (no sp fusion) so shell
    // count is simply the number of shells per atom; n_basis sums
    // cart-dims over shells.
    const auto& sto3g_table = cpu::scf::basis_data::sto3g();
    int n_basis = 0;
    int n_shells = 0;
    for (int i = 0; i < natm; ++i) {
        int Z = static_cast<int>(frame.span(ATV::kAtom_type)[i]);
        for (const auto& sh : sto3g_table[Z].shells) {
            n_shells += 1;
            n_basis += (sh.l + 1) * (sh.l + 2) / 2; // Cartesian dim
        }
    }

    cpu::scf::IntegralPack pack;
    pack.n_basis = n_basis;
    pack.n_electrons = n_electrons;
    pack.S = xt::zeros<double>(
        {static_cast<int>(n_basis), static_cast<int>(n_basis)});
    pack.T = xt::zeros<double>(
        {static_cast<int>(n_basis), static_cast<int>(n_basis)});
    pack.V = xt::zeros<double>(
        {static_cast<int>(n_basis), static_cast<int>(n_basis)});
    pack.ERI = xt::zeros<double>(
        {static_cast<int>(n_basis), static_cast<int>(n_basis),
         static_cast<int>(n_basis), static_cast<int>(n_basis)});

    // Build libcint atm/bas/env arrays
    int off = PTR_ENV_START;
    std::vector<FINT> atm_arr(natm * ATM_SLOTS, 0);
    // Generous env allocation
    std::vector<double> env(off + natm * 3 + n_shells * 10, 0.0);

    for (int i = 0; i < natm; ++i) {
        atm_arr[CHARGE_OF + ATM_SLOTS * i] =
            static_cast<FINT>(frame.span(ATV::kAtom_type)[i]);
        atm_arr[PTR_COORD + ATM_SLOTS * i] = off;
        env[off + 0] = static_cast<double>(frame.span(ATV::kAtom_x)[i])
                       * units::kAngstromInBohr;
        env[off + 1] = static_cast<double>(frame.span(ATV::kAtom_y)[i])
                       * units::kAngstromInBohr;
        env[off + 2] = static_cast<double>(frame.span(ATV::kAtom_z)[i])
                       * units::kAngstromInBohr;
        off += 3;
    }

    // Build basis arrays: expand sp shells into separate s and p libcint shells
    std::vector<FINT> bas_arr(n_shells * BAS_SLOTS, 0);
    // Track which basis functions each libcint shell maps to
    std::vector<int> shell_bf_offset(n_shells); // first basis function index
    std::vector<int> shell_bf_count(n_shells);  // number of basis functions

    int shell_idx = 0;
    int bf_idx = 0;
    for (int a = 0; a < natm; ++a) {
        int Z = static_cast<int>(frame.span(ATV::kAtom_type)[a]);
        for (const auto& sh : sto3g_table[Z].shells) {
            const int l = sh.l;
            const int nprim = static_cast<int>(sh.alpha.size());
            int ptr_exp = off;
            for (int p = 0; p < nprim; ++p)
                env[off++] = sh.alpha[p];
            int ptr_coeff = off;
            for (int p = 0; p < nprim; ++p)
                env[off++] = sh.coeff[p] * CINTgto_norm(l, sh.alpha[p]);
            normalize_contracted_coeffs(l, &env[ptr_exp], &env[ptr_coeff],
                                        nprim);

            bas_arr[ATOM_OF + BAS_SLOTS * shell_idx] = a;
            bas_arr[ANG_OF + BAS_SLOTS * shell_idx] = l;
            bas_arr[NPRIM_OF + BAS_SLOTS * shell_idx] = nprim;
            bas_arr[NCTR_OF + BAS_SLOTS * shell_idx] = 1;
            bas_arr[PTR_EXP + BAS_SLOTS * shell_idx] = ptr_exp;
            bas_arr[PTR_COEFF + BAS_SLOTS * shell_idx] = ptr_coeff;
            const int n_cart = (l + 1) * (l + 2) / 2;
            shell_bf_offset[shell_idx] = bf_idx;
            shell_bf_count[shell_idx] = n_cart;
            ++shell_idx;
            bf_idx += n_cart;
        }
    }
    env.resize(off);

    FINT* atm_p = atm_arr.data();
    FINT* bas_p = bas_arr.data();
    double* env_p = env.data();

    // Use Cartesian integrals — for l<=1, Cartesian components are (x,y,z)
    // matching our convention (cart=0,1,2) directly. No reordering needed.

    // 1e integrals
    for (int i = 0; i < n_shells; ++i) {
        int ni = shell_bf_count[i];
        int oi = shell_bf_offset[i];
        for (int j = 0; j <= i; ++j) {
            int nj = shell_bf_count[j];
            int oj = shell_bf_offset[j];
            int buf_size = ni * nj;
            std::vector<double> buf(buf_size, 0.0);
            FINT shls[2] = {static_cast<FINT>(i), static_cast<FINT>(j)};

            // libcint Cartesian output: column-major buf[i + di * j]
            int1e_ovlp_cart(buf.data(), nullptr, shls, atm_p, natm, bas_p,
                            n_shells, env_p, nullptr, nullptr);
            for (int ii = 0; ii < ni; ++ii)
                for (int jj = 0; jj < nj; ++jj) {
                    pack.S(oi + ii, oj + jj) =
                        static_cast<double>(buf[ii + ni * jj]);
                    pack.S(oj + jj, oi + ii) = pack.S(oi + ii, oj + jj);
                }

            std::fill(buf.begin(), buf.end(), 0.0);
            int1e_kin_cart(buf.data(), nullptr, shls, atm_p, natm, bas_p,
                           n_shells, env_p, nullptr, nullptr);
            for (int ii = 0; ii < ni; ++ii)
                for (int jj = 0; jj < nj; ++jj) {
                    pack.T(oi + ii, oj + jj) =
                        static_cast<double>(buf[ii + ni * jj]);
                    pack.T(oj + jj, oi + ii) = pack.T(oi + ii, oj + jj);
                }

            std::fill(buf.begin(), buf.end(), 0.0);
            int1e_nuc_cart(buf.data(), nullptr, shls, atm_p, natm, bas_p,
                           n_shells, env_p, nullptr, nullptr);
            for (int ii = 0; ii < ni; ++ii)
                for (int jj = 0; jj < nj; ++jj) {
                    pack.V(oi + ii, oj + jj) =
                        static_cast<double>(buf[ii + ni * jj]);
                    pack.V(oj + jj, oi + ii) = pack.V(oi + ii, oj + jj);
                }
        }
    }

    // 2e integrals
    for (int i = 0; i < n_shells; ++i) {
        int ni = shell_bf_count[i];
        int oi = shell_bf_offset[i];
        for (int j = 0; j < n_shells; ++j) {
            int nj = shell_bf_count[j];
            int oj = shell_bf_offset[j];
            for (int k = 0; k < n_shells; ++k) {
                int nk = shell_bf_count[k];
                int ok = shell_bf_offset[k];
                for (int l = 0; l < n_shells; ++l) {
                    int nl = shell_bf_count[l];
                    int ol = shell_bf_offset[l];
                    int buf_size = ni * nj * nk * nl;
                    std::vector<double> buf(buf_size, 0.0);
                    FINT shls4[4] = {static_cast<FINT>(i), static_cast<FINT>(j),
                                     static_cast<FINT>(k),
                                     static_cast<FINT>(l)};
                    cint2e_cart(buf.data(), shls4, atm_p, natm, bas_p, n_shells,
                                env_p, nullptr);
                    // Column-major: i fastest
                    for (int ii = 0; ii < ni; ++ii)
                        for (int jj = 0; jj < nj; ++jj)
                            for (int kk = 0; kk < nk; ++kk)
                                for (int ll = 0; ll < nl; ++ll)
                                    pack.ERI(oi + ii, oj + jj, ok + kk,
                                             ol + ll) =
                                        static_cast<double>(
                                            buf[ii
                                                + ni
                                                      * (jj
                                                         + nj
                                                               * (kk
                                                                  + nk * ll))]);
                }
            }
        }
    }

    pack.Hcore = pack.T + pack.V;

    // Nuclear repulsion
    pack.V_nuc = 0;
    for (int a = 0; a < natm; ++a)
        for (int b = a + 1; b < natm; ++b) {
            double dx = (static_cast<double>(frame.span(ATV::kAtom_x)[a])
                         - static_cast<double>(frame.span(ATV::kAtom_x)[b]))
                        * units::kAngstromInBohr;
            double dy = (static_cast<double>(frame.span(ATV::kAtom_y)[a])
                         - static_cast<double>(frame.span(ATV::kAtom_y)[b]))
                        * units::kAngstromInBohr;
            double dz = (static_cast<double>(frame.span(ATV::kAtom_z)[a])
                         - static_cast<double>(frame.span(ATV::kAtom_z)[b]))
                        * units::kAngstromInBohr;
            double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            pack.V_nuc +=
                static_cast<double>(frame.span(ATV::kAtom_type)[a]
                                    * frame.span(ATV::kAtom_type)[b] / r);
        }

    return pack;
}

// ── Generic d-shell-capable ERI reference ────────────────────────────
//
// `build_integrals_libcint` is hardwired to the STO-3G (s/p only)
// table. The Phase-4 cc-pVDZ correctness gate needs a libcint ERI
// reference that handles d shells.
//
// `build_eri_libcint_from_pack` builds the libcint atm/bas/env arrays
// from an already-resolved `IntegralPack` (output of `build_integrals`
// for any basis, including the d-polarized 6-31G* / def2-SVP sets) and
// returns the dense Cartesian (ij|kl) tensor.
//
// CRITICAL — normalization (the d-shell subtlety):
//   The native engine normalizes every primitive coefficient with
//   `phys_norm_{s,p,d}(alpha)` and does NOT renormalize the contracted
//   shell.  `build_integrals_libcint` instead applies `CINTgto_norm`
//   plus a contracted renorm; for s/p those two schemes are equivalent,
//   but for l = 2 they differ by a constant per-d-AO factor (~1.092),
//   i.e. ~1.42 on a (dd|dd) integral.
//
//   `cint2e_cart` carries an internal per-shell Cartesian-vs-spherical
//   factor: feeding it the native `phys_norm`-coefficients verbatim
//   yields  libcint = g_la^-1·g_lb^-1·g_lc^-1·g_ld^-1 · native, with a
//   per-l constant g_l (verified empirically and α-independent because
//   CINTgto_norm and phys_norm share the same α-power (2l+3)/4):
//       g_0 = CINTgto_norm(0,α) / phys_norm_s(α)   ( = 2 sqrt pi )
//       g_1 = CINTgto_norm(1,α) / phys_norm_p(α)
//       g_2 = 1                                    (phys_norm_d already
//             matches libcint's Cartesian d convention, <d_xx|d_xx>=3)
//       g_3 = 1                                    (phys_norm_f already
//             matches libcint's Cartesian f convention — same
//             off-diagonal-component norm, α-power (2l+3)/4)
//   To make the reference reproduce the native engine bit-for-bit we
//   multiply every primitive coefficient by g_l; libcint's internal
//   factor then cancels and only the quadrature algebra is exercised.
//   No CINTgto_norm path, no contracted renorm.
//
// AO-level `IntegralPack.shells` carries one `native::Shell` per
// Cartesian component (all components of a shell share alpha/coeff and
// center); consecutive entries with the same center and l are regrouped
// here into one libcint basis function.
//
// Cartesian component order from `cint2e_cart` for l = 2 is
//   xx, xy, xz, yy, yz, zz
// identical to `fill_cart_comps` (cart.h) — no reordering needed.

/// @brief Compute the full Cartesian (ij|kl) ERI tensor via libcint,
///        reusing the exact normalized shell data of an IntegralPack.
///
/// @param frame Molecular geometry (positions in Angstrom, type = Z).
/// @param pack  IntegralPack from `cpu::scf::IntegralPack{frame, basis, ...}`.
/// @return Dense 4-index ERI tensor [n_basis^4], Cartesian, chemist's notation.
inline xt::xtensor<double, 4>
build_eri_libcint_from_pack(const cpu::Frame& frame,
                            const cpu::scf::IntegralPack& pack) {
    const int natm = static_cast<int>(frame.nlocal());

    // Regroup AO-level shells (one per Cartesian component) into shells:
    // a new shell starts whenever (center, l) changes.
    struct ShellGroup {
        int l, ao_start, first_ao_shell;
    };
    std::vector<ShellGroup> groups;
    {
        int ao = 0;
        const int n = static_cast<int>(pack.shells.size());
        while (ao < n) {
            const auto& s = pack.shells[ao];
            const int n_cart = (s.l + 1) * (s.l + 2) / 2;
            groups.push_back({s.l, ao, ao});
            ao += n_cart;
        }
    }
    const int n_shells = static_cast<int>(groups.size());

    int off = PTR_ENV_START;
    std::vector<FINT> atm_arr(natm * ATM_SLOTS, 0);
    std::vector<double> env(off + natm * 3 + n_shells * 64, 0.0);

    for (int i = 0; i < natm; ++i) {
        atm_arr[CHARGE_OF + ATM_SLOTS * i] =
            static_cast<FINT>(frame.span(ATV::kAtom_type)[i]);
        atm_arr[PTR_COORD + ATM_SLOTS * i] = off;
        env[off + 0] = static_cast<double>(frame.span(ATV::kAtom_x)[i])
                       * units::kAngstromInBohr;
        env[off + 1] = static_cast<double>(frame.span(ATV::kAtom_y)[i])
                       * units::kAngstromInBohr;
        env[off + 2] = static_cast<double>(frame.span(ATV::kAtom_z)[i])
                       * units::kAngstromInBohr;
        off += 3;
    }

    std::vector<FINT> bas_arr(n_shells * BAS_SLOTS, 0);
    std::vector<int> shell_bf_offset(n_shells);
    std::vector<int> shell_bf_count(n_shells);

    for (int s = 0; s < n_shells; ++s) {
        const auto& g = groups[s];
        const auto& sh = pack.shells[g.first_ao_shell];
        const int l = g.l;
        const int nprim = sh.n_prim;

        // Map this shell's center (Bohr) onto its owning atom.
        int atom_index = 0;
        double best = 1e30;
        for (int a = 0; a < natm; ++a) {
            const double ax = static_cast<double>(frame.span(ATV::kAtom_x)[a])
                              * units::kAngstromInBohr;
            const double ay = static_cast<double>(frame.span(ATV::kAtom_y)[a])
                              * units::kAngstromInBohr;
            const double az = static_cast<double>(frame.span(ATV::kAtom_z)[a])
                              * units::kAngstromInBohr;
            const double d2 = (ax - sh.center[0]) * (ax - sh.center[0])
                              + (ay - sh.center[1]) * (ay - sh.center[1])
                              + (az - sh.center[2]) * (az - sh.center[2]);
            if (d2 < best) {
                best = d2;
                atom_index = a;
            }
        }

        const int ptr_exp = off;
        for (int p = 0; p < nprim; ++p)
            env[off++] = sh.alpha[p];
        const int ptr_coeff = off;
        // Feed native phys_norm coefficients scaled by the per-l factor
        // g_l so libcint's internal Cartesian factor cancels exactly.
        // g_l is α-independent — evaluate at the shell's first exponent.
        const double a0 = sh.alpha[0];
        // g_l == 1 for l >= 2: phys_norm_{d,f} already match libcint's
        // Cartesian d/f convention (the off-diagonal-component norm with
        // alpha-power (2l+3)/4 shared with CINTgto_norm).
        double g_l = 1.0;
        if (l == 0)
            g_l = CINTgto_norm(0, a0) / ATV::phys_norm_s(a0);
        else if (l == 1)
            g_l = CINTgto_norm(1, a0) / ATV::phys_norm_p(a0);
        for (int p = 0; p < nprim; ++p)
            env[off++] = sh.coeff[p] * g_l;

        bas_arr[ATOM_OF + BAS_SLOTS * s] = atom_index;
        bas_arr[ANG_OF + BAS_SLOTS * s] = l;
        bas_arr[NPRIM_OF + BAS_SLOTS * s] = nprim;
        bas_arr[NCTR_OF + BAS_SLOTS * s] = 1;
        bas_arr[PTR_EXP + BAS_SLOTS * s] = ptr_exp;
        bas_arr[PTR_COEFF + BAS_SLOTS * s] = ptr_coeff;
        shell_bf_offset[s] = g.ao_start;
        shell_bf_count[s] = (l + 1) * (l + 2) / 2;
    }
    env.resize(off);

    FINT* atm_p = atm_arr.data();
    FINT* bas_p = bas_arr.data();
    double* env_p = env.data();

    const int n_basis = pack.n_basis;
    xt::xtensor<double, 4> ERI = xt::zeros<double>(
        {static_cast<std::size_t>(n_basis), static_cast<std::size_t>(n_basis),
         static_cast<std::size_t>(n_basis), static_cast<std::size_t>(n_basis)});

    for (int i = 0; i < n_shells; ++i) {
        const int ni = shell_bf_count[i], oi = shell_bf_offset[i];
        for (int j = 0; j < n_shells; ++j) {
            const int nj = shell_bf_count[j], oj = shell_bf_offset[j];
            for (int k = 0; k < n_shells; ++k) {
                const int nk = shell_bf_count[k], ok = shell_bf_offset[k];
                for (int l = 0; l < n_shells; ++l) {
                    const int nl = shell_bf_count[l], ol = shell_bf_offset[l];
                    std::vector<double> buf(
                        static_cast<std::size_t>(ni) * nj * nk * nl, 0.0);
                    FINT shls4[4] = {static_cast<FINT>(i), static_cast<FINT>(j),
                                     static_cast<FINT>(k),
                                     static_cast<FINT>(l)};
                    cint2e_cart(buf.data(), shls4, atm_p, natm, bas_p, n_shells,
                                env_p, nullptr);
                    // Column-major: i fastest.
                    for (int ii = 0; ii < ni; ++ii)
                        for (int jj = 0; jj < nj; ++jj)
                            for (int kk = 0; kk < nk; ++kk)
                                for (int ll = 0; ll < nl; ++ll)
                                    ERI(oi + ii, oj + jj, ok + kk, ol + ll) =
                                        buf[ii
                                            + ni * (jj + nj * (kk + nk * ll))];
                }
            }
        }
    }
    return ERI;
}

} // namespace test
} // namespace ATV

#endif // ATV_HAS_LIBCINT

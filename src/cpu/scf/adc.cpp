/// @file cpu/scf/adc.cpp
/// @brief ADC(2) / ADC(2)-x excitation energies on closed-shell RHF (PySCF twin).
///
/// Formulas follow PySCF `pyscf.adc.radc_ee` (get_imds / matvec for
/// `method='adc(2)'` and `method='adc(2)-x'`) and `radc_amplitudes`
/// first-order doubles t2: t_ij^ab = (ia|bj) / (ε_i + ε_j − ε_a − ε_b).
/// Non-symmetric Davidson on the singles+doubles ISR matrix.
///
/// Strict ADC(2): zeroth-order doubles diagonal only.
/// ADC(2)-x: adds first-order doubles–doubles (vvvv + ovvo/oovv + oooo).
///
/// References: Schirmer, Phys. Rev. A 26, 2395 (1982); Trofimov & Schirmer,
/// J. Phys. B 28, 2299 (1995); PySCF ADC module (doi:10.1063/5.0135290).

#include "atomiverse/cpu/scf/adc.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/scf/internal/ao2mo.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace {

inline int idx_ia(int i, int a, int nvir) { return i * nvir + a; }

inline int idx_ijab(int i, int j, int a, int b, int nocc, int nvir) {
    return ((i * nocc + j) * nvir + a) * nvir + b;
}

/// First-order MP2 doubles t_ij^ab = (ia|bj) / (ε_i+ε_j−ε_a−ε_b).
/// Chemist MO ERI (pq|rs); returns [nocc,nocc,nvir,nvir].
xt::xtensor<double, 4> mp2_t2_(const xt::xtensor<double, 4>& eri_mo,
                               const RealVector& eps, int nocc, int nvir) {
    xt::xtensor<double, 4> t2 = xt::zeros<double>(
        {static_cast<std::size_t>(nocc), static_cast<std::size_t>(nocc),
         static_cast<std::size_t>(nvir), static_cast<std::size_t>(nvir)});
    for (int i = 0; i < nocc; ++i) {
        for (int j = 0; j < nocc; ++j) {
            for (int a = 0; a < nvir; ++a) {
                for (int b = 0; b < nvir; ++b) {
                    const double denom =
                        eps(static_cast<std::size_t>(i))
                        + eps(static_cast<std::size_t>(j))
                        - eps(static_cast<std::size_t>(nocc + a))
                        - eps(static_cast<std::size_t>(nocc + b));
                    if (std::abs(denom) < 1e-12)
                        throw std::runtime_error(
                            "ADC::adc2: near-zero MP2 doubles denominator "
                            "(degenerate or non-aufbau reference)");
                    // (ia|bj)
                    const double ia_bj =
                        eri_mo(static_cast<std::size_t>(i),
                               static_cast<std::size_t>(nocc + a),
                               static_cast<std::size_t>(nocc + b),
                               static_cast<std::size_t>(j));
                    t2(i, j, a, b) = ia_bj / denom;
                }
            }
        }
    }
    return t2;
}

/// Extract chemist MO blocks with virtual indices 0..nvir-1.
struct MoBlocks {
    xt::xtensor<double, 4> oovv; // (ij|ab)
    xt::xtensor<double, 4> ovvo; // (ia|bj)
    xt::xtensor<double, 4> ovoo; // (ia|jk)
    xt::xtensor<double, 4> ovvv; // (ia|bc)
    xt::xtensor<double, 4> oooo; // (ij|kl) — ADC(2)-x doubles
    xt::xtensor<double, 4> vvvv; // (ab|cd) chemist — ADC(2)-x doubles
};

MoBlocks mo_blocks_(const xt::xtensor<double, 4>& eri_mo, int nocc, int nvir) {
    MoBlocks blk;
    blk.oovv = xt::zeros<double>(
        {static_cast<std::size_t>(nocc), static_cast<std::size_t>(nocc),
         static_cast<std::size_t>(nvir), static_cast<std::size_t>(nvir)});
    blk.ovvo = xt::zeros<double>(
        {static_cast<std::size_t>(nocc), static_cast<std::size_t>(nvir),
         static_cast<std::size_t>(nvir), static_cast<std::size_t>(nocc)});
    blk.ovoo = xt::zeros<double>(
        {static_cast<std::size_t>(nocc), static_cast<std::size_t>(nvir),
         static_cast<std::size_t>(nocc), static_cast<std::size_t>(nocc)});
    blk.ovvv = xt::zeros<double>(
        {static_cast<std::size_t>(nocc), static_cast<std::size_t>(nvir),
         static_cast<std::size_t>(nvir), static_cast<std::size_t>(nvir)});
    blk.oooo = xt::zeros<double>(
        {static_cast<std::size_t>(nocc), static_cast<std::size_t>(nocc),
         static_cast<std::size_t>(nocc), static_cast<std::size_t>(nocc)});
    blk.vvvv = xt::zeros<double>(
        {static_cast<std::size_t>(nvir), static_cast<std::size_t>(nvir),
         static_cast<std::size_t>(nvir), static_cast<std::size_t>(nvir)});
    for (int i = 0; i < nocc; ++i) {
        for (int j = 0; j < nocc; ++j) {
            for (int k = 0; k < nocc; ++k)
                for (int l = 0; l < nocc; ++l)
                    blk.oooo(i, j, k, l) = eri_mo(static_cast<std::size_t>(i),
                                                  static_cast<std::size_t>(j),
                                                  static_cast<std::size_t>(k),
                                                  static_cast<std::size_t>(l));
            for (int a = 0; a < nvir; ++a) {
                for (int bb = 0; bb < nvir; ++bb) {
                    blk.oovv(i, j, a, bb) =
                        eri_mo(static_cast<std::size_t>(i),
                               static_cast<std::size_t>(j),
                               static_cast<std::size_t>(nocc + a),
                               static_cast<std::size_t>(nocc + bb));
                    blk.ovvo(i, a, bb, j) =
                        eri_mo(static_cast<std::size_t>(i),
                               static_cast<std::size_t>(nocc + a),
                               static_cast<std::size_t>(nocc + bb),
                               static_cast<std::size_t>(j));
                }
            }
            for (int a = 0; a < nvir; ++a) {
                for (int k = 0; k < nocc; ++k) {
                    blk.ovoo(i, a, j, k) =
                        eri_mo(static_cast<std::size_t>(i),
                               static_cast<std::size_t>(nocc + a),
                               static_cast<std::size_t>(j),
                               static_cast<std::size_t>(k));
                }
            }
        }
        for (int a = 0; a < nvir; ++a) {
            for (int bb = 0; bb < nvir; ++bb) {
                for (int c = 0; c < nvir; ++c) {
                    blk.ovvv(i, a, bb, c) =
                        eri_mo(static_cast<std::size_t>(i),
                               static_cast<std::size_t>(nocc + a),
                               static_cast<std::size_t>(nocc + bb),
                               static_cast<std::size_t>(nocc + c));
                }
            }
        }
    }
    for (int a = 0; a < nvir; ++a)
        for (int b = 0; b < nvir; ++b)
            for (int c = 0; c < nvir; ++c)
                for (int d = 0; d < nvir; ++d)
                    blk.vvvv(a, b, c, d) =
                        eri_mo(static_cast<std::size_t>(nocc + a),
                               static_cast<std::size_t>(nocc + b),
                               static_cast<std::size_t>(nocc + c),
                               static_cast<std::size_t>(nocc + d));
    return blk;
}

/// Second-order singles–singles intermediate M_ss [n_s × n_s], PySCF radc_ee
/// get_imds for method == "adc(2)" (before the adc(3) branch). Layout
/// index = i*nvir + a.
xt::xtensor<double, 2> compute_mss_(const MoBlocks& bl,
                                    const xt::xtensor<double, 4>& t2,
                                    const RealVector& eps, int nocc, int nvir) {
    const int ns = nocc * nvir;
    // Work in M[I,D,L,A] then flatten.
    xt::xtensor<double, 4> M = xt::zeros<double>(
        {static_cast<std::size_t>(nocc), static_cast<std::size_t>(nvir),
         static_cast<std::size_t>(nocc), static_cast<std::size_t>(nvir)});
    const auto& t = t2;
    const auto& oovv = bl.oovv; // v_ccee
    const auto& ovvo = bl.ovvo; // v_cece / v_ceec

    // 000: orbital-energy diagonal
    for (int i = 0; i < nocc; ++i)
        for (int a = 0; a < nvir; ++a)
            M(i, a, i, a) = eps(static_cast<std::size_t>(nocc + a))
                            - eps(static_cast<std::size_t>(i));

    // 010: zeroth-order two-electron
    // M -= einsum('ILAD->IDLA', oovv)  => M[I,D,L,A] -= (IL|AD)
    // M += 2 * einsum('LADI->IDLA', ovvo) => M[I,D,L,A] += 2*(LA|DI)
    for (int I = 0; I < nocc; ++I)
        for (int D = 0; D < nvir; ++D)
            for (int L = 0; L < nocc; ++L)
                for (int A = 0; A < nvir; ++A) {
                    M(I, D, L, A) -= oovv(I, L, A, D);       // ILAD → IDLA
                    M(I, D, L, A) += 2.0 * ovvo(L, A, D, I); // LADI
                }

    // 020: second-order t2 × v terms (PySCF radc_ee get_imds)
    // Helper: accumulate all t2 contractions into M.
    auto add = [&](int I, int D, int L, int A, double v) {
        M(I, D, L, A) += v;
    };

    // Compact loop form of the einsums in get_imds (adc(2) only).
    for (int I = 0; I < nocc; ++I) {
        for (int D = 0; D < nvir; ++D) {
            for (int L = 0; L < nocc; ++L) {
                for (int A = 0; A < nvir; ++A) {
                    double s = 0.0;
                    for (int i = 0; i < nocc; ++i) {
                        for (int a = 0; a < nvir; ++a) {
                            // t[I,i,D,a] * v terms and permutations
                            const double t_IiDa = t(I, i, D, a);
                            const double t_iIDa = t(i, I, D, a);
                            const double t_LiAa = t(L, i, A, a);
                            const double t_iLAa = t(i, L, A, a);
                            // ovvo[L,A,a,i] = (LA|ai) = LAai
                            const double v_LAai = ovvo(L, A, a, i);
                            const double v_iAaL = ovvo(i, A, a, L);
                            const double v_IDai = ovvo(I, D, a, i);
                            const double v_iDaI = ovvo(i, D, a, I);

                            // First block (v_cece = ovvo)
                            s += 2.0 * t_IiDa * v_LAai;
                            s -= t_IiDa * v_iAaL;
                            s += 2.0 * t_LiAa * v_IDai;
                            s -= t_LiAa * v_iDaI;
                            s -= t_iIDa * v_LAai;
                            s += t_iIDa * v_iAaL;
                            s -= t_iLAa * v_IDai;
                            s += t_iLAa * v_iDaI;

                            // Second block (v_ceec = ovvo, half factors)
                            s -= v_LAai * t_IiDa;
                            s += 0.5 * v_LAai * t_iIDa;
                            s += 0.5 * v_iAaL * t_IiDa;
                            s -= 0.5 * v_iAaL * t_iIDa;
                            s -= t_LiAa * v_IDai;
                            s += 0.5 * t_LiAa * v_iDaI;
                            s += 0.5 * t_iLAa * v_IDai;
                            s -= 0.5 * t_iLAa * v_iDaI;
                            s -= t_IiDa * v_LAai;
                            s += 0.5 * t_IiDa * v_iAaL;
                            s += 0.5 * t_iIDa * v_LAai;

                            // Repeated trailing block from PySCF
                            s += 2.0 * t_IiDa * v_LAai;
                            s -= t_IiDa * v_iAaL;
                            s += 2.0 * t_LiAa * v_IDai;
                            s -= t_LiAa * v_iDaI;
                            s -= t_iIDa * v_LAai;
                            s -= t_iLAa * v_IDai;
                            s -= t_LiAa * v_IDai;
                            s += 0.5 * t_LiAa * v_iDaI;
                            s += 0.5 * t_iLAa * v_IDai;
                        }
                    }
                    add(I, D, L, A, s);
                }
            }
        }
    }

    // Diagonal blocks: M[:,a,:,a] += tmp_IL ; M[i,:,i,:] += tmp_DA
    // tmp_IL from t2 × ovvo contractions over ab / i
    xt::xtensor<double, 2> tmp_IL = xt::zeros<double>(
        {static_cast<std::size_t>(nocc), static_cast<std::size_t>(nocc)});
    xt::xtensor<double, 2> tmp_DA = xt::zeros<double>(
        {static_cast<std::size_t>(nvir), static_cast<std::size_t>(nvir)});
    for (int I = 0; I < nocc; ++I) {
        for (int L = 0; L < nocc; ++L) {
            double s = 0.0;
            for (int i = 0; i < nocc; ++i)
                for (int a = 0; a < nvir; ++a)
                    for (int b = 0; b < nvir; ++b) {
                        // -2 * t[I,i,a,b] * ovvo[L,a,b,i]  (Iiab, Labi)
                        s -= 2.0 * t(I, i, a, b) * ovvo(L, a, b, i);
                        // +   t[I,i,a,b] * ovvo[L,b,a,i]  (Iiab, Lbai)
                        s += t(I, i, a, b) * ovvo(L, b, a, i);
                        // -2 * t[L,i,a,b] * ovvo[I,a,b,i]
                        s -= 2.0 * t(L, i, a, b) * ovvo(I, a, b, i);
                        // +   t[L,i,a,b] * ovvo[I,b,a,i]
                        s += t(L, i, a, b) * ovvo(I, b, a, i);
                        // + ovvo[I,a,b,i] * t[L,i,a,b]  (Iabi, Liab)
                        s += ovvo(I, a, b, i) * t(L, i, a, b);
                        // -0.5 * ovvo[I,a,b,i] * t[L,i,b,a]
                        s -= 0.5 * ovvo(I, a, b, i) * t(L, i, b, a);
                        // + t[I,i,a,b] * ovvo[L,a,b,i]
                        s += t(I, i, a, b) * ovvo(L, a, b, i);
                        // -0.5 * t[I,i,a,b] * ovvo[L,b,a,i]
                        s -= 0.5 * t(I, i, a, b) * ovvo(L, b, a, i);
                    }
            tmp_IL(I, L) = s;
        }
    }
    for (int D = 0; D < nvir; ++D) {
        for (int A = 0; A < nvir; ++A) {
            double s = 0.0;
            for (int i = 0; i < nocc; ++i)
                for (int j = 0; j < nocc; ++j)
                    for (int a = 0; a < nvir; ++a) {
                        // -2 * t[i,j,A,a] * ovvo[i,D,a,j]  (ijAa, iDaj)
                        s -= 2.0 * t(i, j, A, a) * ovvo(i, D, a, j);
                        // +   t[i,j,A,a] * ovvo[j,D,a,i]
                        s += t(i, j, A, a) * ovvo(j, D, a, i);
                        // -2 * t[i,j,D,a] * ovvo[i,A,a,j]
                        s -= 2.0 * t(i, j, D, a) * ovvo(i, A, a, j);
                        // +   t[i,j,D,a] * ovvo[j,A,a,i]
                        s += t(i, j, D, a) * ovvo(j, A, a, i);
                        // + ovvo[i,A,a,j] * t[i,j,D,a]
                        s += ovvo(i, A, a, j) * t(i, j, D, a);
                        // -0.5 * ovvo[i,A,a,j] * t[j,i,D,a]
                        s -= 0.5 * ovvo(i, A, a, j) * t(j, i, D, a);
                        // + t[i,j,A,a] * ovvo[i,D,a,j]
                        s += t(i, j, A, a) * ovvo(i, D, a, j);
                        // -0.5 * t[i,j,A,a] * ovvo[j,D,a,i]
                        s -= 0.5 * t(i, j, A, a) * ovvo(j, D, a, i);
                    }
            tmp_DA(D, A) = s;
        }
    }
    for (int a = 0; a < nvir; ++a)
        for (int I = 0; I < nocc; ++I)
            for (int L = 0; L < nocc; ++L)
                M(I, a, L, a) += tmp_IL(I, L);
    for (int i = 0; i < nocc; ++i)
        for (int D = 0; D < nvir; ++D)
            for (int A = 0; A < nvir; ++A)
                M(i, D, i, A) += tmp_DA(D, A);

    // Flatten M[I,D,L,A] → row (I*nvir+D), col (L*nvir+A)
    xt::xtensor<double, 2> Mss = xt::zeros<double>(
        {static_cast<std::size_t>(ns), static_cast<std::size_t>(ns)});
    for (int I = 0; I < nocc; ++I)
        for (int D = 0; D < nvir; ++D)
            for (int L = 0; L < nocc; ++L)
                for (int A = 0; A < nvir; ++A)
                    Mss(idx_ia(I, D, nvir), idx_ia(L, A, nvir)) = M(I, D, L, A);
    return Mss;
}

/// ADC matvec: s = M r on singles+doubles.
/// @param extended true → ADC(2)-x first-order doubles–doubles (PySCF
///        `method='adc(2)-x'`); false → strict ADC(2) doubles diagonal only.
void matvec_(const xt::xtensor<double, 2>& Mss, const MoBlocks& bl,
             const RealVector& eps, int nocc, int nvir, bool extended,
             const std::vector<double>& r, std::vector<double>& s) {
    const int ns = nocc * nvir;
    const int nd = nocc * nocc * nvir * nvir;
    const int dim = ns + nd;
    s.assign(static_cast<std::size_t>(dim), 0.0);

    // s1 = Mss @ r1
    for (int p = 0; p < ns; ++p) {
        double v = 0.0;
        for (int q = 0; q < ns; ++q)
            v += Mss(p, q) * r[static_cast<std::size_t>(q)];
        s[static_cast<std::size_t>(p)] = v;
    }

    // s2 = D_ijab * r2  and coupling
    // Y[I,a] = r1
    for (int i = 0; i < nocc; ++i) {
        for (int j = 0; j < nocc; ++j) {
            for (int a = 0; a < nvir; ++a) {
                for (int b = 0; b < nvir; ++b) {
                    const int p = idx_ijab(i, j, a, b, nocc, nvir);
                    const double D = eps(static_cast<std::size_t>(nocc + a))
                                     + eps(static_cast<std::size_t>(nocc + b))
                                     - eps(static_cast<std::size_t>(i))
                                     - eps(static_cast<std::size_t>(j));
                    s[static_cast<std::size_t>(ns + p)] =
                        D * r[static_cast<std::size_t>(ns + p)];
                }
            }
        }
    }

    const auto& ovvv = bl.ovvv;
    const auto& ovoo = bl.ovoo;

    // s2 += einsum('Ia,JDaC->IJCD', Y, ovvv) + einsum('Ja,ICaD->IJCD', Y, ovvv)
    // Y[I,a], ovvv[J,D,a,C]=(JD|aC) wait: ovvv is (ia|bc)=[i,a,b,c]
    // einsum Ia,JDaC->IJCD : Y[I,a] * v[J,D,a,C] → s2[I,J,C,D]?
    // PySCF: v_ceee = ovvv unpacked as (nocc,nvir,nvir,nvir) = (i,a,b,c)=(ia|bc)
    // M_11Y0 = einsum('Ia,JDaC->IJCD', Y, v_ceee)
    //   Y[I,a] * v[J,D,a,C] → out[I,J,C,D]
    // Indices: I J C D with v_ceee[J,D,a,C] = (J D | a C)
    for (int I = 0; I < nocc; ++I) {
        for (int J = 0; J < nocc; ++J) {
            for (int C = 0; C < nvir; ++C) {
                for (int D = 0; D < nvir; ++D) {
                    double acc = 0.0;
                    for (int a = 0; a < nvir; ++a) {
                        const double Y_Ia =
                            r[static_cast<std::size_t>(idx_ia(I, a, nvir))];
                        const double Y_Ja =
                            r[static_cast<std::size_t>(idx_ia(J, a, nvir))];
                        // v[J,D,a,C] = ovvv(J,D,a,C)
                        acc += Y_Ia * ovvv(J, D, a, C);
                        // einsum Ja,ICaD->IJCD : Y[J,a]*v[I,C,a,D]
                        acc += Y_Ja * ovvv(I, C, a, D);
                    }
                    s[static_cast<std::size_t>(
                        ns + idx_ijab(I, J, C, D, nocc, nvir))] += acc;
                }
            }
        }
    }

    // s2 -= einsum('iC,JDIi->IJCD', Y, ovoo)  : Y[i,C]*ovoo[J,D,I,i]
    // s2 -= einsum('iD,ICJi->IJCD', Y, ovoo)
    for (int I = 0; I < nocc; ++I) {
        for (int J = 0; J < nocc; ++J) {
            for (int C = 0; C < nvir; ++C) {
                for (int D = 0; D < nvir; ++D) {
                    double acc = 0.0;
                    for (int i = 0; i < nocc; ++i) {
                        acc += r[static_cast<std::size_t>(idx_ia(i, C, nvir))]
                               * ovoo(J, D, I, i);
                        acc += r[static_cast<std::size_t>(idx_ia(i, D, nvir))]
                               * ovoo(I, C, J, i);
                    }
                    s[static_cast<std::size_t>(
                        ns + idx_ijab(I, J, C, D, nocc, nvir))] -= acc;
                }
            }
        }
    }

    // s1 += -einsum('Iiab,iabD->ID', r2, ovvv) + 2*einsum('Iiab,ibDa->ID', r2, ovvv)
    for (int I = 0; I < nocc; ++I) {
        for (int D = 0; D < nvir; ++D) {
            double acc = 0.0;
            for (int i = 0; i < nocc; ++i)
                for (int a = 0; a < nvir; ++a)
                    for (int b = 0; b < nvir; ++b) {
                        const double r2 = r[static_cast<std::size_t>(
                            ns + idx_ijab(I, i, a, b, nocc, nvir))];
                        // - r2[I,i,a,b] * ovvv[i,a,b,D]
                        acc -= r2 * ovvv(i, a, b, D);
                        // +2 r2[I,i,a,b] * ovvv[i,b,D,a]
                        acc += 2.0 * r2 * ovvv(i, b, D, a);
                    }
            s[static_cast<std::size_t>(idx_ia(I, D, nvir))] += acc;
        }
    }

    // s1 -= 2*einsum('ijDa,jaiI->ID', r2, ovoo) + einsum('ijDa,iajI->ID', r2, ovoo)
    for (int I = 0; I < nocc; ++I) {
        for (int D = 0; D < nvir; ++D) {
            double acc = 0.0;
            for (int i = 0; i < nocc; ++i)
                for (int j = 0; j < nocc; ++j)
                    for (int a = 0; a < nvir; ++a) {
                        const double r2 = r[static_cast<std::size_t>(
                            ns + idx_ijab(i, j, D, a, nocc, nvir))];
                        // -2 * r2[i,j,D,a] * ovoo[j,a,i,I]
                        acc -= 2.0 * r2 * ovoo(j, a, i, I);
                        // + r2[i,j,D,a] * ovoo[i,a,j,I]
                        acc += r2 * ovoo(i, a, j, I);
                    }
            s[static_cast<std::size_t>(idx_ia(I, D, nvir))] += acc;
        }
    }

    if (!extended)
        return;

    // ── ADC(2)-x first-order doubles–doubles (PySCF radc_ee matvec) ──────
    // Chemist blocks: vvvv=(ab|cd), ovvo=v_ceec, oovv=v_ccee, oooo=v_cccc.
    // Y ≡ r2.  vvvv term matches PySCF's transposed packing:
    //   s[I,J,C,D] += Σ_{a,b} r2[I,J,a,b] · (a C | b D)
    const auto& ovvo = bl.ovvo;
    const auto& oovv = bl.oovv;
    const auto& oooo = bl.oooo;
    const auto& vvvv = bl.vvvv;

    for (int I = 0; I < nocc; ++I) {
        for (int J = 0; J < nocc; ++J) {
            for (int C = 0; C < nvir; ++C) {
                for (int D = 0; D < nvir; ++D) {
                    double acc = 0.0;
                    // vvvv ladder
                    for (int a = 0; a < nvir; ++a)
                        for (int b = 0; b < nvir; ++b) {
                            const double Y = r[static_cast<std::size_t>(
                                ns + idx_ijab(I, J, a, b, nocc, nvir))];
                            acc += Y * vvvv(a, C, b, D); // (a C | b D)
                        }
                    // 2*einsum('IiCa,JDai->IJCD', Y, ovvo)
                    // -einsum('IiCa,iJDa->IJCD', Y, oovv)
                    // -einsum('IiaC,JDai->IJCD', Y, ovvo)
                    // -einsum('IiaD,iJCa->IJCD', Y, oovv)
                    for (int i = 0; i < nocc; ++i)
                        for (int a = 0; a < nvir; ++a) {
                            const double YiCa = r[static_cast<std::size_t>(
                                ns + idx_ijab(I, i, C, a, nocc, nvir))];
                            const double YiaC = r[static_cast<std::size_t>(
                                ns + idx_ijab(I, i, a, C, nocc, nvir))];
                            const double YiaD = r[static_cast<std::size_t>(
                                ns + idx_ijab(I, i, a, D, nocc, nvir))];
                            acc += 2.0 * YiCa * ovvo(J, D, a, i);
                            acc -= YiCa * oovv(i, J, D, a);
                            acc -= YiaC * ovvo(J, D, a, i);
                            acc -= YiaD * oovv(i, J, C, a);
                        }
                    // 2*einsum('JiDa,ICai->IJCD', Y, ovvo)
                    // -einsum('JiDa,iICa->IJCD', Y, oovv)
                    // -einsum('JiaC,iIDa->IJCD', Y, oovv)
                    // -einsum('JiaD,ICai->IJCD', Y, ovvo)
                    for (int i = 0; i < nocc; ++i)
                        for (int a = 0; a < nvir; ++a) {
                            const double YjDa = r[static_cast<std::size_t>(
                                ns + idx_ijab(J, i, D, a, nocc, nvir))];
                            const double YjaC = r[static_cast<std::size_t>(
                                ns + idx_ijab(J, i, a, C, nocc, nvir))];
                            const double YjaD = r[static_cast<std::size_t>(
                                ns + idx_ijab(J, i, a, D, nocc, nvir))];
                            acc += 2.0 * YjDa * ovvo(I, C, a, i);
                            acc -= YjDa * oovv(i, I, C, a);
                            acc -= YjaC * oovv(i, I, D, a);
                            acc -= YjaD * ovvo(I, C, a, i);
                        }
                    // einsum('ijCD,IiJj->IJCD', Y, oooo)
                    for (int i = 0; i < nocc; ++i)
                        for (int j = 0; j < nocc; ++j) {
                            const double Yij = r[static_cast<std::size_t>(
                                ns + idx_ijab(i, j, C, D, nocc, nvir))];
                            acc += Yij * oooo(I, i, J, j);
                        }
                    s[static_cast<std::size_t>(
                        ns + idx_ijab(I, J, C, D, nocc, nvir))] += acc;
                }
            }
        }
    }
}

/// Dense non-symmetric eigenproblem for the Davidson-projected subspace.
/// NOTE: kept as a local extern "C" binding because the ADC Davidson path
/// (davidson_adc_ / ea_adc2 projected solves) still calls dgeev with right
/// eigenvectors inline — those uses are out of scope for scf-review-01, which
/// only hoists the eigenvalue-only `dgeev_real_sorted_` below onto the shared
/// `linalg::eig_nonsymmetric`.
extern "C" void dgeev_(const char* jobvl, const char* jobvr, const int* n,
                       double* a, const int* lda, double* wr, double* wi,
                       double* vl, const int* ldvl, double* vr, const int* ldvr,
                       double* work, const int* lwork, int* info);

/// Dense non-symmetric eigenproblem via the shared `linalg::eig_nonsymmetric`.
/// Returns real eigenvalues sorted ascending (complex discarded if |Im|>tol).
std::vector<double> dgeev_real_sorted_(const xt::xtensor<double, 2>& Hrow,
                                       double imag_tol = 1e-8) {
    const int n = static_cast<int>(Hrow.shape(0));
    // RealMatrix is column-major, so A(i,j) = Hrow(i,j) reproduces the old
    // hand-rolled column-major buffer bit-for-bit under dgeev.
    linalg::RealMatrix A = xt::zeros<double>(
        {static_cast<std::size_t>(n), static_cast<std::size_t>(n)});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                Hrow(i, j);
    linalg::RealVector wr, wi;
    linalg::eig_nonsymmetric(A, wr, wi, nullptr, "ADC::adc2");
    std::vector<double> real_eigs;
    real_eigs.reserve(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < wr.size(); ++i) {
        if (std::abs(wi(i)) <= imag_tol && wr(i) > -1e-6)
            real_eigs.push_back(wr(i));
    }
    std::sort(real_eigs.begin(), real_eigs.end());
    return real_eigs;
}

/// Non-symmetric Davidson for the lowest `nroots` real Ritz values.
/// Falls back to dense dgeev when dim is small (≤ kDenseCap).
std::vector<double> davidson_adc_(const xt::xtensor<double, 2>& Mss,
                                  const MoBlocks& bl, const RealVector& eps,
                                  int nocc, int nvir, int nroots,
                                  bool extended) {
    const int ns = nocc * nvir;
    const int nd = nocc * nocc * nvir * nvir;
    const int dim = ns + nd;
    constexpr int kDenseCap = 400; // full dgeev only for tiny spaces
    const char* api = extended ? "ADC::adc2x" : "ADC::adc2";

    if (dim <= kDenseCap) {
        xt::xtensor<double, 2> H = xt::zeros<double>(
            {static_cast<std::size_t>(dim), static_cast<std::size_t>(dim)});
        std::vector<double> e(static_cast<std::size_t>(dim), 0.0);
        std::vector<double> s;
        for (int k = 0; k < dim; ++k) {
            e[static_cast<std::size_t>(k)] = 1.0;
            matvec_(Mss, bl, eps, nocc, nvir, extended, e, s);
            for (int i = 0; i < dim; ++i)
                H(i, k) = s[static_cast<std::size_t>(i)];
            e[static_cast<std::size_t>(k)] = 0.0;
        }
        auto w = dgeev_real_sorted_(H);
        if (static_cast<int>(w.size()) < nroots)
            throw std::runtime_error(
                std::string(api) + ": fewer real eigenvalues than n_states");
        w.resize(static_cast<std::size_t>(nroots));
        return w;
    }

    // Precompute diagonal preconditioner
    std::vector<double> diag(static_cast<std::size_t>(dim));
    for (int p = 0; p < ns; ++p)
        diag[static_cast<std::size_t>(p)] = Mss(p, p);
    for (int i = 0; i < nocc; ++i)
        for (int j = 0; j < nocc; ++j)
            for (int a = 0; a < nvir; ++a)
                for (int b = 0; b < nvir; ++b) {
                    const int p = ns + idx_ijab(i, j, a, b, nocc, nvir);
                    diag[static_cast<std::size_t>(p)] =
                        eps(static_cast<std::size_t>(nocc + a))
                        + eps(static_cast<std::size_t>(nocc + b))
                        - eps(static_cast<std::size_t>(i))
                        - eps(static_cast<std::size_t>(j));
                }

    // Initial guesses: lowest nroots singles diagonal entries
    std::vector<int> order(static_cast<std::size_t>(ns));
    for (int i = 0; i < ns; ++i)
        order[static_cast<std::size_t>(i)] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return diag[static_cast<std::size_t>(a)]
               < diag[static_cast<std::size_t>(b)];
    });

    const int max_space = std::min(dim, std::max(4 * nroots, 20));
    const int max_cycle = 80;
    const double tol = 1e-8;

    std::vector<std::vector<double>> V; // basis
    V.reserve(static_cast<std::size_t>(max_space));
    for (int k = 0; k < nroots; ++k) {
        std::vector<double> v(static_cast<std::size_t>(dim), 0.0);
        v[static_cast<std::size_t>(order[static_cast<std::size_t>(k)])] = 1.0;
        V.push_back(std::move(v));
    }

    auto orthonormalize_last = [&]() {
        auto& v = V.back();
        for (std::size_t j = 0; j + 1 < V.size(); ++j) {
            double dot = 0.0;
            for (int i = 0; i < dim; ++i)
                dot += V[j][static_cast<std::size_t>(i)]
                       * v[static_cast<std::size_t>(i)];
            for (int i = 0; i < dim; ++i)
                v[static_cast<std::size_t>(i)] -=
                    dot * V[j][static_cast<std::size_t>(i)];
        }
        double nrm = 0.0;
        for (double x : v)
            nrm += x * x;
        nrm = std::sqrt(nrm);
        if (nrm < 1e-14)
            return false;
        for (double& x : v)
            x /= nrm;
        return true;
    };

    std::vector<std::vector<double>> W; // M V
    std::vector<double> roots(static_cast<std::size_t>(nroots), 0.0);

    for (int cycle = 0; cycle < max_cycle; ++cycle) {
        // Compute missing W = M V
        while (W.size() < V.size()) {
            std::vector<double> s;
            matvec_(Mss, bl, eps, nocc, nvir, extended, V[W.size()], s);
            W.push_back(std::move(s));
        }
        const int m = static_cast<int>(V.size());
        // Projected Hm[i,j] = V_i · W_j
        xt::xtensor<double, 2> Hm = xt::zeros<double>(
            {static_cast<std::size_t>(m), static_cast<std::size_t>(m)});
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j) {
                double d = 0.0;
                for (int p = 0; p < dim; ++p)
                    d += V[static_cast<std::size_t>(i)]
                          [static_cast<std::size_t>(p)]
                         * W[static_cast<std::size_t>(j)]
                            [static_cast<std::size_t>(p)];
                Hm(i, j) = d;
            }
        // Dense eig of small Hm
        std::vector<double> A(static_cast<std::size_t>(m * m));
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j)
                A[static_cast<std::size_t>(i + j * m)] = Hm(i, j);
        std::vector<double> wr(static_cast<std::size_t>(m));
        std::vector<double> wi(static_cast<std::size_t>(m));
        std::vector<double> vr(static_cast<std::size_t>(m * m));
        std::vector<double> vl(1);
        char jobvl = 'N', jobvr = 'V';
        int ldvl = 1, ldvr = m, info = 0, lwork = -1;
        double wkopt = 0.0;
        dgeev_(&jobvl, &jobvr, &m, A.data(), &m, wr.data(), wi.data(),
               vl.data(), &ldvl, vr.data(), &ldvr, &wkopt, &lwork, &info);
        lwork = static_cast<int>(wkopt);
        std::vector<double> work(static_cast<std::size_t>(lwork));
        // dgeev destroys A; rebuild
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j)
                A[static_cast<std::size_t>(i + j * m)] = Hm(i, j);
        dgeev_(&jobvl, &jobvr, &m, A.data(), &m, wr.data(), wi.data(),
               vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, &info);
        if (info != 0)
            throw std::runtime_error(std::string(api)
                                     + ": Davidson projected dgeev failed");

        // Sort real Ritz pairs
        std::vector<int> idx;
        for (int k = 0; k < m; ++k)
            if (std::abs(wi[static_cast<std::size_t>(k)]) < 1e-8)
                idx.push_back(k);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return wr[static_cast<std::size_t>(a)]
                   < wr[static_cast<std::size_t>(b)];
        });
        if (static_cast<int>(idx.size()) < nroots)
            throw std::runtime_error(
                std::string(api) + ": Davidson found too few real Ritz values");

        bool all_conv = true;
        std::vector<std::vector<double>> residuals;
        for (int r = 0; r < nroots; ++r) {
            const int k = idx[static_cast<std::size_t>(r)];
            roots[static_cast<std::size_t>(r)] =
                wr[static_cast<std::size_t>(k)];
            // Ritz vector x = V c
            std::vector<double> x(static_cast<std::size_t>(dim), 0.0);
            std::vector<double> Mx(static_cast<std::size_t>(dim), 0.0);
            for (int j = 0; j < m; ++j) {
                const double c =
                    vr[static_cast<std::size_t>(j + k * m)]; // col k
                for (int p = 0; p < dim; ++p) {
                    x[static_cast<std::size_t>(p)] +=
                        c
                        * V[static_cast<std::size_t>(j)]
                           [static_cast<std::size_t>(p)];
                    Mx[static_cast<std::size_t>(p)] +=
                        c
                        * W[static_cast<std::size_t>(j)]
                           [static_cast<std::size_t>(p)];
                }
            }
            // residual r = Mx - θ x
            std::vector<double> res(static_cast<std::size_t>(dim));
            double rn = 0.0;
            const double theta = roots[static_cast<std::size_t>(r)];
            for (int p = 0; p < dim; ++p) {
                res[static_cast<std::size_t>(p)] =
                    Mx[static_cast<std::size_t>(p)]
                    - theta * x[static_cast<std::size_t>(p)];
                rn += res[static_cast<std::size_t>(p)]
                      * res[static_cast<std::size_t>(p)];
            }
            rn = std::sqrt(rn);
            if (rn > tol)
                all_conv = false;
            // preconditioned residual
            for (int p = 0; p < dim; ++p) {
                double den = diag[static_cast<std::size_t>(p)] - theta;
                if (std::abs(den) < 1e-8)
                    den = (den >= 0.0 ? 1e-8 : -1e-8);
                res[static_cast<std::size_t>(p)] /= den;
            }
            residuals.push_back(std::move(res));
        }
        if (all_conv)
            return roots;

        // Collapse if space full
        if (static_cast<int>(V.size()) + nroots > max_space) {
            std::vector<std::vector<double>> Vnew;
            std::vector<std::vector<double>> Wnew;
            for (int r = 0; r < nroots; ++r) {
                const int k = idx[static_cast<std::size_t>(r)];
                std::vector<double> x(static_cast<std::size_t>(dim), 0.0);
                std::vector<double> Mx(static_cast<std::size_t>(dim), 0.0);
                for (int j = 0; j < m; ++j) {
                    const double c = vr[static_cast<std::size_t>(j + k * m)];
                    for (int p = 0; p < dim; ++p) {
                        x[static_cast<std::size_t>(p)] +=
                            c
                            * V[static_cast<std::size_t>(j)]
                               [static_cast<std::size_t>(p)];
                        Mx[static_cast<std::size_t>(p)] +=
                            c
                            * W[static_cast<std::size_t>(j)]
                               [static_cast<std::size_t>(p)];
                    }
                }
                // normalize x
                double nrm = 0.0;
                for (double v : x)
                    nrm += v * v;
                nrm = std::sqrt(nrm);
                for (double& v : x)
                    v /= nrm;
                for (double& v : Mx)
                    v /= nrm;
                Vnew.push_back(std::move(x));
                Wnew.push_back(std::move(Mx));
            }
            V.swap(Vnew);
            W.swap(Wnew);
        }

        // Append preconditioned residuals
        for (int r = 0; r < nroots; ++r) {
            V.push_back(residuals[static_cast<std::size_t>(r)]);
            if (!orthonormalize_last())
                V.pop_back();
        }
    }
    throw std::runtime_error(std::string(api)
                             + ": Davidson did not converge within max_cycle");
}

double mp2_corr_ha_(const xt::xtensor<double, 4>& t2,
                    const xt::xtensor<double, 4>& ovvo, int nocc, int nvir) {
    // e_mp = 2 Σ t_ijab (ia|bj) − Σ t_ijab (ib|aj)  [PySCF compute_energy]
    double e = 0.0;
    for (int i = 0; i < nocc; ++i)
        for (int j = 0; j < nocc; ++j)
            for (int a = 0; a < nvir; ++a)
                for (int b = 0; b < nvir; ++b) {
                    const double tijab = t2(i, j, a, b);
                    e += 2.0 * tijab * ovvo(i, a, b, j);
                    e -= tijab * ovvo(i, b, a, j);
                }
    return e;
}

inline int idx_iab(int i, int a, int b, int nvir) {
    return (i * nvir + a) * nvir + b;
}

/// EA-ADC(2) singles intermediate M_ab [nvir × nvir] (PySCF radc_ea get_imds,
/// method == "adc(2)" only — no ADC(3) terms).
xt::xtensor<double, 2> compute_mab_ea_(const xt::xtensor<double, 4>& t2,
                                       const xt::xtensor<double, 4>& ovvo,
                                       const RealVector& eps, int nocc,
                                       int nvir) {
    xt::xtensor<double, 2> Mab = xt::zeros<double>(
        {static_cast<std::size_t>(nvir), static_cast<std::size_t>(nvir)});
    // Zeroth order: δ_ab ε_a
    for (int a = 0; a < nvir; ++a)
        Mab(a, a) = eps(static_cast<std::size_t>(nocc + a));

    // Second-order t2 × ovvo contractions (strict ADC(2)).
    // Coefficients: 1.5*0.5 = 0.75, 0.5*0.5 = 0.25.
    for (int a = 0; a < nvir; ++a) {
        for (int b = 0; b < nvir; ++b) {
            double s = 0.0;
            for (int l = 0; l < nocc; ++l) {
                for (int m = 0; m < nocc; ++m) {
                    for (int d = 0; d < nvir; ++d) {
                        const double t_lmad = t2(l, m, a, d);
                        const double t_mlad = t2(m, l, a, d);
                        const double t_lmbd = t2(l, m, b, d);
                        const double t_mlbd = t2(m, l, b, d);
                        const double ov_lbdm = ovvo(l, b, d, m); // lbdm
                        const double ov_ldbm = ovvo(l, d, b, m); // ldbm
                        const double ov_ladm = ovvo(l, a, d, m); // ladm
                        const double ov_ldam = ovvo(l, d, a, m); // ldam

                        s -= 0.75 * t_lmad * ov_lbdm;
                        s += 0.25 * t_mlad * ov_lbdm;
                        s += 0.25 * t_lmad * ov_ldbm;
                        s -= 0.25 * t_mlad * ov_ldbm;

                        s -= 0.75 * t_lmbd * ov_ladm;
                        s += 0.25 * t_mlbd * ov_ladm;
                        s += 0.25 * t_lmbd * ov_ldam;
                        s -= 0.25 * t_mlbd * ov_ldam;
                    }
                }
            }
            Mab(a, b) += s;
        }
    }
    return Mab;
}

/// EA-ADC(2) matvec on 1p + 2p–1h (PySCF radc_ea matvec, adc(2) only).
void matvec_ea_(const xt::xtensor<double, 2>& Mab, const MoBlocks& bl,
                const RealVector& eps, int nocc, int nvir,
                const std::vector<double>& r, std::vector<double>& s) {
    const int ns = nvir;
    const int nd = nocc * nvir * nvir;
    const int dim = ns + nd;
    s.assign(static_cast<std::size_t>(dim), 0.0);

    // s1 = Mab @ r1
    for (int a = 0; a < nvir; ++a) {
        double v = 0.0;
        for (int b = 0; b < nvir; ++b)
            v += Mab(a, b) * r[static_cast<std::size_t>(b)];
        s[static_cast<std::size_t>(a)] = v;
    }

    // s2 = D_iab * r2  (D = −ε_i + ε_a + ε_b)
    for (int i = 0; i < nocc; ++i)
        for (int a = 0; a < nvir; ++a)
            for (int b = 0; b < nvir; ++b) {
                const int p = idx_iab(i, a, b, nvir);
                const double D = -eps(static_cast<std::size_t>(i))
                                 + eps(static_cast<std::size_t>(nocc + a))
                                 + eps(static_cast<std::size_t>(nocc + b));
                s[static_cast<std::size_t>(ns + p)] =
                    D * r[static_cast<std::size_t>(ns + p)];
            }

    const auto& ovvv = bl.ovvv; // (ia|bc) = [i,a,b,c]

    // s1 += 2 * einsum('icab,ibc->a', ovvv, r2)
    //     −     einsum('ibac,ibc->a', ovvv, r2)
    for (int a = 0; a < nvir; ++a) {
        double acc = 0.0;
        for (int i = 0; i < nocc; ++i)
            for (int bb = 0; bb < nvir; ++bb)
                for (int c = 0; c < nvir; ++c) {
                    const double r2 = r[static_cast<std::size_t>(
                        ns + idx_iab(i, bb, c, nvir))];
                    // ovvv[i,c,a,b] = (ic|ab); ovvv[i,b,a,c] = (ib|ac)
                    acc += 2.0 * ovvv(i, c, a, bb) * r2;
                    acc -= ovvv(i, bb, a, c) * r2;
                }
        s[static_cast<std::size_t>(a)] += acc;
    }

    // s2 += einsum('icab,a->ibc', ovvv, r1)
    //       → s2[i,b,c] += Σ_a ovvv[i,c,a,b] * r1[a]
    for (int i = 0; i < nocc; ++i)
        for (int bb = 0; bb < nvir; ++bb)
            for (int c = 0; c < nvir; ++c) {
                double acc = 0.0;
                for (int a = 0; a < nvir; ++a)
                    acc += ovvv(i, c, a, bb) * r[static_cast<std::size_t>(a)];
                s[static_cast<std::size_t>(ns + idx_iab(i, bb, c, nvir))] +=
                    acc;
            }
}

/// Dense / Davidson eigensolver for EA-ADC(2).
std::vector<double> davidson_ea_adc_(const xt::xtensor<double, 2>& Mab,
                                     const MoBlocks& bl, const RealVector& eps,
                                     int nocc, int nvir, int nroots) {
    const int ns = nvir;
    const int nd = nocc * nvir * nvir;
    const int dim = ns + nd;
    constexpr int kDenseCap = 400;

    if (dim <= kDenseCap) {
        xt::xtensor<double, 2> H = xt::zeros<double>(
            {static_cast<std::size_t>(dim), static_cast<std::size_t>(dim)});
        std::vector<double> e(static_cast<std::size_t>(dim), 0.0);
        std::vector<double> s;
        for (int k = 0; k < dim; ++k) {
            e[static_cast<std::size_t>(k)] = 1.0;
            matvec_ea_(Mab, bl, eps, nocc, nvir, e, s);
            for (int i = 0; i < dim; ++i)
                H(i, k) = s[static_cast<std::size_t>(i)];
            e[static_cast<std::size_t>(k)] = 0.0;
        }
        auto w = dgeev_real_sorted_(H);
        if (static_cast<int>(w.size()) < nroots)
            throw std::runtime_error(
                "ADC::ea_adc2: fewer real eigenvalues than n_states");
        w.resize(static_cast<std::size_t>(nroots));
        return w;
    }

    // Preconditioner diagonal
    std::vector<double> diag(static_cast<std::size_t>(dim));
    for (int a = 0; a < nvir; ++a)
        diag[static_cast<std::size_t>(a)] = Mab(a, a);
    for (int i = 0; i < nocc; ++i)
        for (int a = 0; a < nvir; ++a)
            for (int b = 0; b < nvir; ++b) {
                const int p = ns + idx_iab(i, a, b, nvir);
                diag[static_cast<std::size_t>(p)] =
                    -eps(static_cast<std::size_t>(i))
                    + eps(static_cast<std::size_t>(nocc + a))
                    + eps(static_cast<std::size_t>(nocc + b));
            }

    std::vector<int> order(static_cast<std::size_t>(ns));
    for (int i = 0; i < ns; ++i)
        order[static_cast<std::size_t>(i)] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return diag[static_cast<std::size_t>(a)]
               < diag[static_cast<std::size_t>(b)];
    });

    const int max_space = std::min(dim, std::max(4 * nroots, 20));
    const int max_cycle = 80;
    const double tol = 1e-8;

    std::vector<std::vector<double>> V;
    V.reserve(static_cast<std::size_t>(max_space));
    for (int k = 0; k < nroots; ++k) {
        std::vector<double> v(static_cast<std::size_t>(dim), 0.0);
        v[static_cast<std::size_t>(order[static_cast<std::size_t>(k)])] = 1.0;
        V.push_back(std::move(v));
    }

    auto orthonormalize_last = [&]() {
        auto& v = V.back();
        for (std::size_t j = 0; j + 1 < V.size(); ++j) {
            double dot = 0.0;
            for (int i = 0; i < dim; ++i)
                dot += V[j][static_cast<std::size_t>(i)]
                       * v[static_cast<std::size_t>(i)];
            for (int i = 0; i < dim; ++i)
                v[static_cast<std::size_t>(i)] -=
                    dot * V[j][static_cast<std::size_t>(i)];
        }
        double nrm = 0.0;
        for (double x : v)
            nrm += x * x;
        nrm = std::sqrt(nrm);
        if (nrm < 1e-14)
            return false;
        for (double& x : v)
            x /= nrm;
        return true;
    };

    std::vector<std::vector<double>> W;
    std::vector<double> roots(static_cast<std::size_t>(nroots), 0.0);

    for (int cycle = 0; cycle < max_cycle; ++cycle) {
        while (W.size() < V.size()) {
            std::vector<double> s;
            matvec_ea_(Mab, bl, eps, nocc, nvir, V[W.size()], s);
            W.push_back(std::move(s));
        }
        const int m = static_cast<int>(V.size());
        xt::xtensor<double, 2> Hm = xt::zeros<double>(
            {static_cast<std::size_t>(m), static_cast<std::size_t>(m)});
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j) {
                double dot = 0.0;
                for (int k = 0; k < dim; ++k)
                    dot += V[static_cast<std::size_t>(i)]
                            [static_cast<std::size_t>(k)]
                           * W[static_cast<std::size_t>(j)]
                              [static_cast<std::size_t>(k)];
                Hm(i, j) = dot;
            }

        // Dense dgeev on the projected matrix; need right eigenvectors for
        // residual construction (reuse column-major path with jobvr='V').
        std::vector<double> A(static_cast<std::size_t>(m * m));
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j)
                A[static_cast<std::size_t>(i + j * m)] = Hm(i, j);
        std::vector<double> wr(static_cast<std::size_t>(m));
        std::vector<double> wi(static_cast<std::size_t>(m));
        std::vector<double> vr(static_cast<std::size_t>(m * m));
        char jobvl = 'N', jobvr = 'V';
        int ldvl = 1, ldvr = m, info = 0;
        int lwork = -1;
        double wkopt = 0.0;
        std::vector<double> vl(1);
        dgeev_(&jobvl, &jobvr, &m, A.data(), &m, wr.data(), wi.data(),
               vl.data(), &ldvl, vr.data(), &ldvr, &wkopt, &lwork, &info);
        lwork = static_cast<int>(wkopt);
        std::vector<double> work(static_cast<std::size_t>(lwork));
        dgeev_(&jobvl, &jobvr, &m, A.data(), &m, wr.data(), wi.data(),
               vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, &info);
        if (info != 0)
            throw std::runtime_error(
                "ADC::ea_adc2: Davidson projected dgeev failed");

        // Collect real Ritz values with indices
        std::vector<std::pair<double, int>> real_ritz;
        for (int i = 0; i < m; ++i) {
            if (std::abs(wi[static_cast<std::size_t>(i)]) <= 1e-8
                && wr[static_cast<std::size_t>(i)] > -1e-6)
                real_ritz.emplace_back(wr[static_cast<std::size_t>(i)], i);
        }
        std::sort(
            real_ritz.begin(), real_ritz.end(),
            [](const auto& x, const auto& y) { return x.first < y.first; });
        if (static_cast<int>(real_ritz.size()) < nroots)
            throw std::runtime_error(
                "ADC::ea_adc2: Davidson found too few real Ritz values");

        bool all_conv = true;
        std::vector<std::vector<double>> residuals;
        for (int r = 0; r < nroots; ++r) {
            roots[static_cast<std::size_t>(r)] =
                real_ritz[static_cast<std::size_t>(r)].first;
            const int idx = real_ritz[static_cast<std::size_t>(r)].second;
            // Ritz vector in full space: x = V * y (y = right eigenvector)
            std::vector<double> x(static_cast<std::size_t>(dim), 0.0);
            for (int j = 0; j < m; ++j) {
                const double yj = vr[static_cast<std::size_t>(j + idx * m)];
                for (int i = 0; i < dim; ++i)
                    x[static_cast<std::size_t>(i)] +=
                        yj
                        * V[static_cast<std::size_t>(j)]
                           [static_cast<std::size_t>(i)];
            }
            std::vector<double> Ax;
            matvec_ea_(Mab, bl, eps, nocc, nvir, x, Ax);
            std::vector<double> res(static_cast<std::size_t>(dim));
            double rn = 0.0;
            for (int i = 0; i < dim; ++i) {
                res[static_cast<std::size_t>(i)] =
                    Ax[static_cast<std::size_t>(i)]
                    - roots[static_cast<std::size_t>(r)]
                          * x[static_cast<std::size_t>(i)];
                rn += res[static_cast<std::size_t>(i)]
                      * res[static_cast<std::size_t>(i)];
            }
            rn = std::sqrt(rn);
            if (rn > tol)
                all_conv = false;
            residuals.push_back(std::move(res));
        }
        if (all_conv)
            return roots;

        // Expand subspace with preconditioned residuals
        if (static_cast<int>(V.size()) + nroots > max_space) {
            // Collapse onto current nroots Ritz vectors
            std::vector<std::vector<double>> Vnew;
            std::vector<std::vector<double>> Wnew;
            for (int r = 0; r < nroots; ++r) {
                const int idx = real_ritz[static_cast<std::size_t>(r)].second;
                std::vector<double> x(static_cast<std::size_t>(dim), 0.0);
                for (int j = 0; j < m; ++j) {
                    const double yj = vr[static_cast<std::size_t>(j + idx * m)];
                    for (int i = 0; i < dim; ++i)
                        x[static_cast<std::size_t>(i)] +=
                            yj
                            * V[static_cast<std::size_t>(j)]
                               [static_cast<std::size_t>(i)];
                }
                Vnew.push_back(std::move(x));
            }
            V.swap(Vnew);
            W.clear();
            // Re-orthonormalize collapsed basis
            for (std::size_t k = 0; k < V.size(); ++k) {
                for (std::size_t j = 0; j < k; ++j) {
                    double dot = 0.0;
                    for (int i = 0; i < dim; ++i)
                        dot += V[j][static_cast<std::size_t>(i)]
                               * V[k][static_cast<std::size_t>(i)];
                    for (int i = 0; i < dim; ++i)
                        V[k][static_cast<std::size_t>(i)] -=
                            dot * V[j][static_cast<std::size_t>(i)];
                }
                double nrm = 0.0;
                for (double x : V[k])
                    nrm += x * x;
                nrm = std::sqrt(nrm);
                if (nrm < 1e-14)
                    continue;
                for (double& x : V[k])
                    x /= nrm;
            }
            continue;
        }

        for (int r = 0; r < nroots; ++r) {
            std::vector<double> t = residuals[static_cast<std::size_t>(r)];
            const double lam = roots[static_cast<std::size_t>(r)];
            for (int i = 0; i < dim; ++i) {
                const double den = diag[static_cast<std::size_t>(i)] - lam;
                if (std::abs(den) > 1e-12)
                    t[static_cast<std::size_t>(i)] /= den;
            }
            V.push_back(std::move(t));
            if (!orthonormalize_last())
                V.pop_back();
        }
    }
    throw std::runtime_error(
        "ADC::ea_adc2: Davidson did not converge within max_cycle");
}

} // namespace

namespace {

ADC2Result run_adc_ee_member_(const RHF& self, int n_states, bool extended,
                              const internal::IntegralView& pack,
                              const SlaterDeterminant& det) {
    const char* api = extended ? "ADC::adc2x" : "ADC::adc2";
    const xt::xtensor<double, 4> eri_ao =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();

    const RealMatrix& C = det.coefficients(SlaterDeterminant::Spin::alpha);
    const RealVector& f = det.occupations(SlaterDeterminant::Spin::alpha);
    const RealVector& eps =
        det.orbital_energies(SlaterDeterminant::Spin::alpha);
    const int n_mo = static_cast<int>(C.shape()[1]);
    int n_occ = 0;
    for (int p = 0; p < n_mo; ++p)
        if (f(static_cast<std::size_t>(p)) > 0.0)
            ++n_occ;
    const int n_vir = n_mo - n_occ;
    if (n_occ == 0 || n_vir == 0)
        throw std::invalid_argument(
            std::string(api)
            + ": no virtual orbitals (e.g. He/STO-3G — no fallback)");
    const int n_ov = n_occ * n_vir;
    if (n_states < 1 || n_states > n_ov)
        throw std::invalid_argument(std::string(api)
                                    + ": n_states must be in [1, n_occ*n_virt="
                                    + std::to_string(n_ov) + "]");
    (void)self;

    const xt::xtensor<double, 4> eri_mo = ao2mo_full(eri_ao, C, n_mo);
    const MoBlocks bl = mo_blocks_(eri_mo, n_occ, n_vir);
    const xt::xtensor<double, 4> t2 = mp2_t2_(eri_mo, eps, n_occ, n_vir);
    const xt::xtensor<double, 2> Mss = compute_mss_(bl, t2, eps, n_occ, n_vir);

    const std::vector<double> omega_ha =
        davidson_adc_(Mss, bl, eps, n_occ, n_vir, n_states, extended);

    ADC2Result out;
    out.e_corr_mp2_ha = mp2_corr_ha_(t2, bl.ovvo, n_occ, n_vir);
    out.energy_ev.resize(static_cast<std::size_t>(n_states));
    for (int s = 0; s < n_states; ++s)
        out.energy_ev[static_cast<std::size_t>(s)] =
            omega_ha[static_cast<std::size_t>(s)] * units::kHartreeInEv;
    return out;
}

} // namespace

template <typename Method>
ADC<Method>::ADC(const Method& mf) : mf_(&mf) {}

template <typename Method>
ADC2Result ADC<Method>::adc2(int n_states) const {
    const Method& mf = *mf_;
    if (!mf.has_result())
        throw std::runtime_error(
            "ADC::adc2: no SCF result — call run(frame) first");
    return run_adc_ee_member_(mf, n_states, /*extended=*/false,
                              mf.integral_view_(), mf.determinant());
}

template <typename Method>
ADC2Result ADC<Method>::adc2x(int n_states) const {
    const Method& mf = *mf_;
    if (!mf.has_result())
        throw std::runtime_error(
            "ADC::adc2x: no SCF result — call run(frame) first");
    return run_adc_ee_member_(mf, n_states, /*extended=*/true,
                              mf.integral_view_(), mf.determinant());
}

inline int idx_ajk_(int a, int j, int k, int nocc) {
    return (a * nocc + j) * nocc + k;
}

/// Second-order occupied–occupied intermediate M_ij [nocc × nocc].
/// Twin of PySCF `radc_ip.get_imds` for `method == "adc(2)"` (no ADC(3)).
xt::xtensor<double, 2> compute_mij_ip_(const xt::xtensor<double, 4>& t2,
                                       const xt::xtensor<double, 4>& ovvo,
                                       const RealVector& eps, int nocc,
                                       int nvir) {
    xt::xtensor<double, 2> M = xt::zeros<double>(
        {static_cast<std::size_t>(nocc), static_cast<std::size_t>(nocc)});
    // Zeroth order: M_ij = δ_ij ε_j
    for (int i = 0; i < nocc; ++i)
        M(i, i) = eps(static_cast<std::size_t>(i));

    // Second-order t2 × ovvo (PySCF radc_ip get_imds, adc(2) branch only).
    for (int i = 0; i < nocc; ++i) {
        for (int j = 0; j < nocc; ++j) {
            double s = 0.0;
            for (int l = 0; l < nocc; ++l) {
                for (int d = 0; d < nvir; ++d) {
                    for (int e = 0; e < nvir; ++e) {
                        const double t_ilde = t2(i, l, d, e);
                        const double t_lide = t2(l, i, d, e);
                        const double v_jdel = ovvo(j, d, e, l); // (jd|el)
                        const double v_jedl = ovvo(j, e, d, l); // (je|dl)
                        // 0.25*(ilde,jdel) − 0.25*(lide,jdel)
                        // −0.25*(ilde,jedl) + 0.25*(lide,jedl)
                        // +0.5*(ilde,jdel)
                        s += 0.25 * t_ilde * v_jdel;
                        s -= 0.25 * t_lide * v_jdel;
                        s -= 0.25 * t_ilde * v_jedl;
                        s += 0.25 * t_lide * v_jedl;
                        s += 0.5 * t_ilde * v_jdel;

                        const double t_jlde = t2(j, l, d, e);
                        const double t_ljde = t2(l, j, d, e);
                        const double v_idel = ovvo(i, d, e, l); // (id|el)
                        const double v_ldei = ovvo(l, d, e, i); // (ld|ei)
                        // 0.25*(jlde,idel) − 0.25*(ljde,idel)
                        // −0.25*(jlde,ldei) + 0.25*(ljde,ldei)
                        // +0.5*(jlde,idel)
                        s += 0.25 * t_jlde * v_idel;
                        s -= 0.25 * t_ljde * v_idel;
                        s -= 0.25 * t_jlde * v_ldei;
                        s += 0.25 * t_ljde * v_ldei;
                        s += 0.5 * t_jlde * v_idel;
                    }
                }
            }
            M(i, j) += s;
        }
    }
    return M;
}

/// IP-ADC(2) matvec in the PySCF pre-flip convention (caller applies s*=−1).
/// Layout: r[0:nocc] = 1h; r[nocc:] = 2h1p with index a*nocc*nocc + j*nocc + k.
void matvec_ip_(const xt::xtensor<double, 2>& Mij,
                const xt::xtensor<double, 4>& ovoo, const RealVector& eps,
                int nocc, int nvir, const std::vector<double>& r,
                std::vector<double>& s) {
    const int ns = nocc;
    const int nd = nvir * nocc * nocc;
    const int dim = ns + nd;
    s.assign(static_cast<std::size_t>(dim), 0.0);

    // s1 = M_ij @ r1
    for (int i = 0; i < nocc; ++i) {
        double v = 0.0;
        for (int j = 0; j < nocc; ++j)
            v += Mij(i, j) * r[static_cast<std::size_t>(j)];
        s[static_cast<std::size_t>(i)] = v;
    }

    // s1 += 2 * einsum('jaki,ajk->i', ovoo, r2)
    //     −     einsum('kaji,ajk->i', ovoo, r2)
    for (int i = 0; i < nocc; ++i) {
        double acc = 0.0;
        for (int a = 0; a < nvir; ++a) {
            for (int j = 0; j < nocc; ++j) {
                for (int k = 0; k < nocc; ++k) {
                    const double r2 = r[static_cast<std::size_t>(
                        ns + idx_ajk_(a, j, k, nocc))];
                    // ovoo[j,a,k,i] = (ja|ki)
                    acc += 2.0 * ovoo(j, a, k, i) * r2;
                    // ovoo[k,a,j,i] = (ka|ji)
                    acc -= ovoo(k, a, j, i) * r2;
                }
            }
        }
        s[static_cast<std::size_t>(i)] += acc;
    }

    // s2 = D_ajk * r2 + einsum('jaki,i->ajk', ovoo, r1)
    // D_ajk = ε_j + ε_k − ε_a
    for (int a = 0; a < nvir; ++a) {
        for (int j = 0; j < nocc; ++j) {
            for (int k = 0; k < nocc; ++k) {
                const int p = ns + idx_ajk_(a, j, k, nocc);
                const double D = eps(static_cast<std::size_t>(j))
                                 + eps(static_cast<std::size_t>(k))
                                 - eps(static_cast<std::size_t>(nocc + a));
                double acc = D * r[static_cast<std::size_t>(p)];
                for (int i = 0; i < nocc; ++i)
                    acc += ovoo(j, a, k, i) * r[static_cast<std::size_t>(i)];
                s[static_cast<std::size_t>(p)] = acc;
            }
        }
    }

    // PySCF flips the sign so eigenvalues are positive IPs.
    for (double& v : s)
        v = -v;
}

/// Dense eigenproblem for IP-ADC(2); returns lowest nroots real positive IPs.
std::vector<double> solve_ip_adc_(const xt::xtensor<double, 2>& Mij,
                                  const xt::xtensor<double, 4>& ovoo,
                                  const RealVector& eps, int nocc, int nvir,
                                  int nroots) {
    const int ns = nocc;
    const int nd = nvir * nocc * nocc;
    const int dim = ns + nd;

    xt::xtensor<double, 2> H = xt::zeros<double>(
        {static_cast<std::size_t>(dim), static_cast<std::size_t>(dim)});
    std::vector<double> e(static_cast<std::size_t>(dim), 0.0);
    std::vector<double> s;
    for (int k = 0; k < dim; ++k) {
        e[static_cast<std::size_t>(k)] = 1.0;
        matvec_ip_(Mij, ovoo, eps, nocc, nvir, e, s);
        for (int i = 0; i < dim; ++i)
            H(i, k) = s[static_cast<std::size_t>(i)];
        e[static_cast<std::size_t>(k)] = 0.0;
    }
    auto w = dgeev_real_sorted_(H);
    if (static_cast<int>(w.size()) < nroots)
        throw std::runtime_error(
            "ADC::ip_adc2: fewer real eigenvalues than n_states");
    w.resize(static_cast<std::size_t>(nroots));
    return w;
}

template <typename Method>
IPADC2Result ADC<Method>::ip_adc2(int n_states) const {
    const Method& mf = *mf_;
    if (!mf.has_result())
        throw std::runtime_error(
            "ADC::ip_adc2: no SCF result — call run(frame) first");

    const SlaterDeterminant& det = mf.determinant();
    const internal::IntegralView pack = mf.integral_view_();
    const xt::xtensor<double, 4> eri_ao =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();

    const RealMatrix& C = det.coefficients(SlaterDeterminant::Spin::alpha);
    const RealVector& f = det.occupations(SlaterDeterminant::Spin::alpha);
    const RealVector& eps =
        det.orbital_energies(SlaterDeterminant::Spin::alpha);
    const int n_mo = static_cast<int>(C.shape()[1]);
    int n_occ = 0;
    for (int p = 0; p < n_mo; ++p)
        if (f(static_cast<std::size_t>(p)) > 0.0)
            ++n_occ;
    const int n_vir = n_mo - n_occ;
    if (n_occ == 0 || n_vir == 0)
        throw std::invalid_argument(
            "ADC::ip_adc2: no virtual orbitals (e.g. He/STO-3G — no "
            "fallback)");
    if (n_states < 1 || n_states > n_occ)
        throw std::invalid_argument(
            "ADC::ip_adc2: n_states must be in [1, n_occ="
            + std::to_string(n_occ) + "]");

    const xt::xtensor<double, 4> eri_mo = ao2mo_full(eri_ao, C, n_mo);
    const MoBlocks bl = mo_blocks_(eri_mo, n_occ, n_vir);
    const xt::xtensor<double, 4> t2 = mp2_t2_(eri_mo, eps, n_occ, n_vir);
    const xt::xtensor<double, 2> Mij =
        compute_mij_ip_(t2, bl.ovvo, eps, n_occ, n_vir);

    const std::vector<double> ip_ha =
        solve_ip_adc_(Mij, bl.ovoo, eps, n_occ, n_vir, n_states);

    IPADC2Result out;
    out.e_corr_mp2_ha = mp2_corr_ha_(t2, bl.ovvo, n_occ, n_vir);
    out.energy_ev.resize(static_cast<std::size_t>(n_states));
    for (int s = 0; s < n_states; ++s)
        out.energy_ev[static_cast<std::size_t>(s)] =
            ip_ha[static_cast<std::size_t>(s)] * units::kHartreeInEv;
    return out;
}

template <typename Method>
EAADC2Result ADC<Method>::ea_adc2(int n_states) const {
    const Method& mf = *mf_;
    if (!mf.has_result())
        throw std::runtime_error(
            "ADC::ea_adc2: no SCF result — call run(frame) first");

    const SlaterDeterminant& det = mf.determinant();
    const internal::IntegralView pack = mf.integral_view_();
    const xt::xtensor<double, 4> eri_ao =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();

    const RealMatrix& C = det.coefficients(SlaterDeterminant::Spin::alpha);
    const RealVector& f = det.occupations(SlaterDeterminant::Spin::alpha);
    const RealVector& eps =
        det.orbital_energies(SlaterDeterminant::Spin::alpha);
    const int n_mo = static_cast<int>(C.shape()[1]);
    int n_occ = 0;
    for (int p = 0; p < n_mo; ++p)
        if (f(static_cast<std::size_t>(p)) > 0.0)
            ++n_occ;
    const int n_vir = n_mo - n_occ;
    if (n_occ == 0 || n_vir == 0)
        throw std::invalid_argument(
            "ADC::ea_adc2: no virtual orbitals (e.g. He/STO-3G — no fallback)");
    // Full EA space: 1p singles + 2p–1h doubles
    const int dim = n_vir + n_occ * n_vir * n_vir;
    if (n_states < 1 || n_states > dim)
        throw std::invalid_argument(
            "ADC::ea_adc2: n_states must be in [1, n_vir+n_occ*n_vir^2="
            + std::to_string(dim) + "]");

    const xt::xtensor<double, 4> eri_mo = ao2mo_full(eri_ao, C, n_mo);
    const MoBlocks bl = mo_blocks_(eri_mo, n_occ, n_vir);
    const xt::xtensor<double, 4> t2 = mp2_t2_(eri_mo, eps, n_occ, n_vir);
    const xt::xtensor<double, 2> Mab =
        compute_mab_ea_(t2, bl.ovvo, eps, n_occ, n_vir);

    const std::vector<double> omega_ha =
        davidson_ea_adc_(Mab, bl, eps, n_occ, n_vir, n_states);

    EAADC2Result out;
    out.e_corr_mp2_ha = mp2_corr_ha_(t2, bl.ovvo, n_occ, n_vir);
    out.energy_ev.resize(static_cast<std::size_t>(n_states));
    for (int s = 0; s < n_states; ++s)
        out.energy_ev[static_cast<std::size_t>(s)] =
            omega_ha[static_cast<std::size_t>(s)] * units::kHartreeInEv;
    return out;
}

template class ADC<RHF>;

} // namespace scf
} // namespace cpu
} // namespace ATV

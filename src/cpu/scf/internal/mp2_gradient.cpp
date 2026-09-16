/// @file cpu/scf/internal/mp2_gradient.cpp
/// @brief Closed-shell RMP2 analytic nuclear gradient (PySCF-compatible).
///
/// Algorithm mirrors pyscf.grad.mp2.grad_elec (no frozen orbitals):
///   T₂ → unrelaxed 1-RDM (doo,dvv) + non-separable 2-RDM
///   I-matrix + L_vo Lagrangian → Z-vector on ¹(A+B)
///   contract relaxed densities with skeleton integral derivatives.

#include "atomiverse/cpu/scf/internal/mp2_gradient.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/internal/ao2mo.h"
#include "atomiverse/cpu/scf/internal/integral_derivatives.h"
#include "atomiverse/cpu/scf/internal/orbital_hessian.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/internal/spin_orbital_eri.h"
#include "atomiverse/cpu/scf/rys_eri.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

namespace {

/// Closed-shell G[X]_{μν} = Σ_{λσ} X_{λσ} [(μν|λσ) − ½(μλ|νσ)].
RealMatrix g_contraction(const xt::xtensor<double, 4>& eri,
                         const RealMatrix& X) {
    const int n = static_cast<int>(X.shape()[0]);
    RealMatrix G = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    acc += X(k, l) * (eri(i, j, k, l) - 0.5 * eri(i, k, j, l));
            G(i, j) = acc;
        }
    return G;
}

} // namespace

GradientResult mp2_analytic_forces(const Frame& frame, const internal::IntegralView& pack,
                                   const SlaterDeterminant& det,
                                   double e_scf_ha, double e_corr_ha) {
    if (!det.restricted())
        throw std::invalid_argument(
            "mp2_analytic_forces: closed-shell RHF reference required");
    if (pack.ERI.size() == 0)
        throw std::invalid_argument(
            "mp2_analytic_forces: dense ERI required (compute_eri=true)");
    if (pack.spherical())
        throw std::invalid_argument(
            "mp2_analytic_forces: spherical-convention packs are not "
            "supported in this link — use a Cartesian basis (STO-3G / Pople)");

    using Spin = SlaterDeterminant::Spin;
    const RealMatrix C = det.coefficients(Spin::alpha);
    const RealVector eps = det.orbital_energies(Spin::alpha);
    const int n = static_cast<int>(C.shape()[0]);
    const int n_mo = static_cast<int>(C.shape()[1]);
    const int nocc = det.n_electrons(Spin::alpha);
    const int nvir = n_mo - nocc;
    const int natm = static_cast<int>(frame.nlocal());
    if (nvir <= 0)
        throw std::invalid_argument(
            "mp2_analytic_forces: no virtual orbitals (n_virt=0)");
    if (n != pack.n_basis)
        throw std::invalid_argument(
            "mp2_analytic_forces: AO dimension mismatch between det and pack");

    const xt::xtensor<double, 4>& eri = pack.ERI;
    const RealMatrix Co = xt::view(C, xt::all(), xt::range(0, nocc));
    const RealMatrix Cv = xt::view(C, xt::all(), xt::range(nocc, n_mo));

    // t2[i,j,a,b] = (ia|jb) / (ε_i + ε_j − ε_a − ε_b)
    const xt::xtensor<double, 4> ovov = ao2mo_ovov(eri, C, nocc, n_mo);
    xt::xtensor<double, 4> t2 = xt::zeros<double>({nocc, nocc, nvir, nvir});
    for (int i = 0; i < nocc; ++i)
        for (int j = 0; j < nocc; ++j)
            for (int a = 0; a < nvir; ++a)
                for (int b = 0; b < nvir; ++b) {
                    const double denom = mp_denominator(
                        eps(i), eps(j), eps(nocc + a), eps(nocc + b));
                    t2(i, j, a, b) = ovov(i, a, j, b) / denom;
                }

    // Unrelaxed 1-RDM (PySCF _gamma1_intermediates; returns −dm1occ, +dm1vir)
    RealMatrix doo = xt::zeros<double>({nocc, nocc});
    RealMatrix dvv = xt::zeros<double>({nvir, nvir});
    for (int i = 0; i < nocc; ++i) {
        // dm1occ(p,q) += Σ_ab [2 t[i,p,a,b] t[i,q,a,b] − t[i,p,a,b] t[i,q,b,a]]
        for (int p = 0; p < nocc; ++p)
            for (int q = 0; q < nocc; ++q) {
                double s2 = 0.0, sx = 0.0;
                for (int a = 0; a < nvir; ++a)
                    for (int b = 0; b < nvir; ++b) {
                        s2 += t2(i, p, a, b) * t2(i, q, a, b);
                        sx += t2(i, p, a, b) * t2(i, q, b, a);
                    }
                doo(p, q) += 2.0 * s2 - sx;
            }
        // dm1vir: einsum('jca,jcb->ba') → out(b,a) += t[i,j,c,a] t[i,j,c,b]
        for (int a = 0; a < nvir; ++a)
            for (int b = 0; b < nvir; ++b) {
                double s2 = 0.0, sx = 0.0;
                for (int j = 0; j < nocc; ++j)
                    for (int c = 0; c < nvir; ++c) {
                        s2 += t2(i, j, c, a) * t2(i, j, c, b);
                        sx += t2(i, j, c, a) * t2(i, j, b, c);
                    }
                dvv(b, a) += 2.0 * s2 - sx;
            }
    }
    doo = -doo;

    // part[i,j,u,v] = Σ_ab t[i,j,a,b] Cv[u,a] Cv[v,b]
    xt::xtensor<double, 4> part = xt::zeros<double>({nocc, nocc, n, n});
    for (int i = 0; i < nocc; ++i)
        for (int j = 0; j < nocc; ++j)
            for (int u = 0; u < n; ++u)
                for (int v = 0; v < n; ++v) {
                    double s = 0.0;
                    for (int a = 0; a < nvir; ++a)
                        for (int b = 0; b < nvir; ++b)
                            s += t2(i, j, a, b) * Cv(u, a) * Cv(v, b);
                    part(i, j, u, v) = s;
                }
    // part_dm2[i,u,v,j] = 4 part[i,j,u,v] − 2 part[i,j,v,u]
    xt::xtensor<double, 4> part_dm2 = xt::zeros<double>({nocc, n, n, nocc});
    for (int i = 0; i < nocc; ++i)
        for (int j = 0; j < nocc; ++j)
            for (int u = 0; u < n; ++u)
                for (int v = 0; v < n; ++v)
                    part_dm2(i, u, v, j) =
                        4.0 * part(i, j, u, v) - 2.0 * part(i, j, v, u);

    // Non-separable AO 2-RDM G[p,q,r,s]
    xt::xtensor<double, 4> Gamma = xt::zeros<double>({n, n, n, n});
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q)
            for (int r = 0; r < n; ++r)
                for (int s = 0; s < n; ++s) {
                    double acc = 0.0;
                    for (int i = 0; i < nocc; ++i)
                        for (int j = 0; j < nocc; ++j) {
                            acc += Co(p, i) * part_dm2(i, q, r, j) * Co(s, j);
                            acc += Co(q, i) * part_dm2(i, p, r, j) * Co(s, j);
                        }
                    Gamma(p, q, r, s) = acc;
                }
    {
        // (r,s) symmetrization as in PySCF dm2buf; full-ERI contraction then
        // needs the factor ½ that the s2kl pack path applies (diag×½ / tril
        // pairing) — otherwise Imat and de_ns are exactly 2× the PySCF values.
        const xt::xtensor<double, 4> Gs = Gamma;
        for (int p = 0; p < n; ++p)
            for (int q = 0; q < n; ++q)
                for (int r = 0; r < n; ++r)
                    for (int s = 0; s < n; ++s)
                        Gamma(p, q, r, s) =
                            0.5 * (Gs(p, q, r, s) + Gs(p, q, s, r));
    }

    // Imat[α,β] = Σ_{p,r,s} (p α | r s) Gamma[p, β, r, s]
    RealMatrix Imat = xt::zeros<double>({n, n});
    for (int p = 0; p < n; ++p)
        for (int alpha = 0; alpha < n; ++alpha)
            for (int beta = 0; beta < n; ++beta) {
                double acc = 0.0;
                for (int r = 0; r < n; ++r)
                    for (int s = 0; s < n; ++s)
                        acc += eri(p, alpha, r, s) * Gamma(p, beta, r, s);
                Imat(alpha, beta) += acc;
            }
    xt::xtensor<double, 2> de = xt::zeros<double>({natm, 3});
    const RealMatrix P_hf = det.density_total();

    // First-index dERI loop (int2e_ip1 analogue): non-separable 2-RDM
    // contribution + PySCF-style vhf1 = G-like skeleton from P_HF.
    // vhf1[A,d] is n×n, built exactly as pyscf.grad.mp2 (both index blocks).
    xt::xtensor<double, 4> vhf1 = xt::zeros<double>({natm, 3, n, n});
    {
        const ShellBlockPairList bpl{pack, /*thr=*/1e-14};
        const int n_blocks = static_cast<int>(bpl.blocks.size());
        std::vector<int> block_to_atom(bpl.blocks.size(), -1);
        for (std::size_t b = 0; b < bpl.blocks.size(); ++b) {
            const auto& Bl = bpl.blocks[b];
            for (int a = 0; a < natm; ++a) {
                const double dx = Bl.center[0] - pack.atoms[a].pos[0];
                const double dy = Bl.center[1] - pack.atoms[a].pos[1];
                const double dz = Bl.center[2] - pack.atoms[a].pos[2];
                if (dx * dx + dy * dy + dz * dz < 1e-20) {
                    block_to_atom[b] = a;
                    break;
                }
            }
        }
        for (int bi = 0; bi < n_blocks; ++bi) {
            const int A = block_to_atom[static_cast<std::size_t>(bi)];
            if (A < 0)
                continue;
            const auto& BA = bpl.blocks[static_cast<std::size_t>(bi)];
            for (int bj = 0; bj < n_blocks; ++bj) {
                const auto& BB = bpl.blocks[static_cast<std::size_t>(bj)];
                for (int bk = 0; bk < n_blocks; ++bk) {
                    const auto& BC = bpl.blocks[static_cast<std::size_t>(bk)];
                    for (int bl = 0; bl < n_blocks; ++bl) {
                        const auto& BD =
                            bpl.blocks[static_cast<std::size_t>(bl)];
                        const int na = BA.n_cart, nb = BB.n_cart,
                                  nc = BC.n_cart, nd = BD.n_cart;
                        const int block_n = na * nb * nc * nd;
                        std::vector<double> deriv(
                            static_cast<std::size_t>(3 * block_n), 0.0);
                        native::rys_eri_block_deriv(BA, BB, BC, BD,
                                                    /*deriv_center=*/0,
                                                    deriv.data());
                        for (int ia = 0; ia < na; ++ia) {
                            const int p = BA.ao_start + ia;
                            for (int ib = 0; ib < nb; ++ib) {
                                const int q = BB.ao_start + ib;
                                for (int ic = 0; ic < nc; ++ic) {
                                    const int r = BC.ao_start + ic;
                                    for (int id = 0; id < nd; ++id) {
                                        const int s = BD.ao_start + id;
                                        const int ao =
                                            ((ia * nb + ib) * nc + ic) * nd
                                            + id;
                                        const double g = Gamma(p, q, r, s);
                                        for (int ax = 0; ax < 3; ++ax) {
                                            // rys_eri_block_deriv ∂/∂A matches
                                            // the HF 2e-gradient engine; PySCF
                                            // int2e_ip1 is −∇_bra. Flip once so
                                            // the MP2 assembly matches PySCF's
                                            // nabla-on-bra sign convention.
                                            const double der =
                                                -deriv[static_cast<std::size_t>(
                                                    ax * block_n + ao)];
                                            // non-separable 2-RDM · dERI
                                            de(A, ax) -= 2.0 * der * g;
                                            // vhf1 from P_HF (PySCF mp2.py)
                                            vhf1(A, ax, r, s) +=
                                                der * P_hf(p, q);
                                            vhf1(A, ax, r, q) -=
                                                0.5 * der * P_hf(p, s);
                                            vhf1(A, ax, p, q) +=
                                                der * P_hf(r, s);
                                            vhf1(A, ax, p, s) -=
                                                0.5 * der * P_hf(q, r);
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

    IntegralDerivPack dpack{pack};

    // Unrelaxed dm1_mo
    RealMatrix dm1mo = xt::zeros<double>({n_mo, n_mo});
    for (int i = 0; i < nocc; ++i)
        for (int j = 0; j < nocc; ++j)
            dm1mo(i, j) = doo(i, j) + doo(j, i);
    for (int a = 0; a < nvir; ++a)
        for (int b = 0; b < nvir; ++b)
            dm1mo(nocc + a, nocc + b) = dvv(a, b) + dvv(b, a);

    // Imat_mo = − Cᵀ Imat S C
    const RealMatrix Imat_S = xt::linalg::dot(Imat, pack.S);
    RealMatrix Imat_mo =
        -xt::linalg::dot(xt::transpose(C), xt::linalg::dot(Imat_S, C));

    // Xvo = C_vᵀ (2 G[dm1_ao]) C_o + I_ovᵀ − I_vo
    RealMatrix dm1_ao =
        xt::linalg::dot(C, xt::linalg::dot(dm1mo, xt::transpose(C)));
    const RealMatrix vhf = g_contraction(eri, dm1_ao) * 2.0;
    RealMatrix Xvo =
        xt::linalg::dot(xt::transpose(Cv), xt::linalg::dot(vhf, Co));
    for (int a = 0; a < nvir; ++a)
        for (int i = 0; i < nocc; ++i)
            Xvo(a, i) += Imat_mo(i, nocc + a) - Imat_mo(nocc + a, i);

    // Z-vector: ¹(A+B) Z = −Xvo  (PySCF cphf.solve sign for h1=Xvo)
    OrbitalHessian oh{det, eri};
    const int nov = nocc * nvir;
    const RealMatrix Aplus = oh.dense_plus(Channel::singlet);
    RealVector rhs = xt::zeros<double>({static_cast<std::size_t>(nov)});
    for (int a = 0; a < nvir; ++a)
        for (int i = 0; i < nocc; ++i)
            rhs(a * nocc + i) = -Xvo(a, i);
    const RealVector zvec = xt::linalg::solve(Aplus, rhs);
    for (int a = 0; a < nvir; ++a)
        for (int i = 0; i < nocc; ++i) {
            dm1mo(nocc + a, i) = zvec(a * nocc + i);
            dm1mo(i, nocc + a) = zvec(a * nocc + i);
        }

    for (int i = 0; i < nocc; ++i)
        for (int a = 0; a < nvir; ++a)
            Imat_mo(nocc + a, i) = Imat_mo(i, nocc + a);
    const RealMatrix im1 =
        xt::linalg::dot(C, xt::linalg::dot(Imat_mo, xt::transpose(C)));

    // zeta_mo = ζ ⊙ dm1mo
    RealMatrix zeta_mo = xt::zeros<double>({n_mo, n_mo});
    for (int p = 0; p < n_mo; ++p)
        for (int q = 0; q < n_mo; ++q)
            zeta_mo(p, q) = 0.5 * (eps(p) + eps(q));
    for (int a = 0; a < nvir; ++a)
        for (int i = 0; i < nocc; ++i) {
            zeta_mo(nocc + a, i) = eps(i);
            zeta_mo(i, nocc + a) = eps(i);
        }
    for (int p = 0; p < n_mo; ++p)
        for (int q = 0; q < n_mo; ++q)
            zeta_mo(p, q) *= dm1mo(p, q);
    RealMatrix zeta =
        xt::linalg::dot(C, xt::linalg::dot(zeta_mo, xt::transpose(C)));

    dm1_ao = xt::linalg::dot(C, xt::linalg::dot(dm1mo, xt::transpose(C)));
    const RealMatrix P_occ = xt::linalg::dot(Co, xt::transpose(Co));
    const RealMatrix dm1_sym = dm1_ao + xt::transpose(dm1_ao);
    const RealMatrix g_dm1 = g_contraction(eri, dm1_sym);
    const RealMatrix vhf_s1occ =
        xt::linalg::dot(P_occ, xt::linalg::dot(g_dm1, P_occ));

    const RealMatrix dm1p = P_hf + dm1_ao * 2.0;
    const RealMatrix dm1_tot = P_hf + dm1_ao;

    // HF energy-weighted density (closed-shell factor-2 occupations)
    RealMatrix W_hf = xt::zeros<double>({n, n});
    for (int i = 0; i < nocc; ++i)
        for (int mu = 0; mu < n; ++mu)
            for (int nu = 0; nu < n; ++nu)
                W_hf(mu, nu) += 2.0 * eps(i) * C(mu, i) * C(nu, i);
    zeta += W_hf;

    // Final assembly. IntegralDerivPack dS/dH are FULL ∂/∂R_A (bra+ket).
    for (int A = 0; A < natm; ++A) {
        for (int d = 0; d < 3; ++d) {
            const RealMatrix dS =
                xt::view(dpack.dS, A, d, xt::all(), xt::all());
            const RealMatrix dH =
                xt::view(dpack.dT, A, d, xt::all(), xt::all())
                + xt::view(dpack.dV, A, d, xt::all(), xt::all());
            const RealMatrix v1 = xt::view(vhf1, A, d, xt::all(), xt::all());

            de(A, d) += xt::sum(dS * im1)();
            de(A, d) += xt::sum(dH * xt::transpose(dm1_tot))();
            de(A, d) -= xt::sum(dS * zeta)();
            de(A, d) -= xt::sum(dS * vhf_s1occ)();
            de(A, d) -= xt::sum(v1 * dm1p)();
            de(A, d) += dpack.dV_nuc(A, d);
        }
    }

    return GradientResult{FromHartreeBohr{}, de, e_scf_ha + e_corr_ha};
}

namespace {

/// Four-index AO→MO transform with independent coefficient blocks (chemist
/// (C0 C1 | C2 C3)). Same pattern as mp2.cpp ao2mo_mixed_.
xt::xtensor<double, 4> ao2mo_mixed(const xt::xtensor<double, 4>& eri_ao,
                                   const xt::xtensor<double, 2>& C0,
                                   const xt::xtensor<double, 2>& C1,
                                   const xt::xtensor<double, 2>& C2,
                                   const xt::xtensor<double, 2>& C3) {
    xt::xtensor<double, 4> cur = eri_ao;
    const xt::xtensor<double, 2>* Cs[4] = {&C3, &C2, &C1, &C0};
    for (int round = 0; round < 4; ++round) {
        const auto sh = cur.shape();
        auto flat = xt::reshape_view(cur, {sh[0] * sh[1] * sh[2], sh[3]});
        xt::xtensor<double, 2> h = xt::linalg::dot(flat, *Cs[round]);
        xt::xtensor<double, 4> next =
            xt::reshape_view(h, {sh[0], sh[1], sh[2], Cs[round]->shape()[1]});
        cur = xt::transpose(next, {3, 0, 1, 2});
    }
    return cur;
}

/// Unrestricted Coulomb+exchange: G^σ = J[Pα+Pβ] − K[P^σ].
void g_uhf(const xt::xtensor<double, 4>& eri, const RealMatrix& Pa,
           const RealMatrix& Pb, RealMatrix& Ga, RealMatrix& Gb) {
    const int n = static_cast<int>(Pa.shape()[0]);
    RealMatrix J = xt::zeros<double>({n, n});
    RealMatrix Ka = xt::zeros<double>({n, n});
    RealMatrix Kb = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double jv = 0.0, ka = 0.0, kb = 0.0;
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l) {
                    jv += eri(i, j, k, l) * (Pa(k, l) + Pb(k, l));
                    ka += eri(i, k, j, l) * Pa(k, l);
                    kb += eri(i, k, j, l) * Pb(k, l);
                }
            J(i, j) = jv;
            Ka(i, j) = ka;
            Kb(i, j) = kb;
        }
    Ga = J - Ka;
    Gb = J - Kb;
}

} // namespace

GradientResult ump2_analytic_forces(const Frame& frame,
                                    const internal::IntegralView& pack,
                                    const SlaterDeterminant& det,
                                    double e_scf_ha, double e_corr_ha) {
    if (det.restricted())
        throw std::invalid_argument(
            "ump2_analytic_forces: unrestricted UHF reference required");
    if (det.generalized())
        throw std::invalid_argument(
            "ump2_analytic_forces: GHF reference is not supported");
    if (pack.ERI.size() == 0)
        throw std::invalid_argument(
            "ump2_analytic_forces: dense ERI required (compute_eri=true)");
    if (pack.spherical())
        throw std::invalid_argument(
            "ump2_analytic_forces: spherical-convention packs are not "
            "supported — use a Cartesian basis (STO-3G / Pople)");

    using Spin = SlaterDeterminant::Spin;
    const RealMatrix Ca = det.coefficients(Spin::alpha);
    const RealMatrix Cb = det.coefficients(Spin::beta);
    const RealVector ea = det.orbital_energies(Spin::alpha);
    const RealVector eb = det.orbital_energies(Spin::beta);
    const int n = static_cast<int>(Ca.shape()[0]);
    const int nmoa = static_cast<int>(Ca.shape()[1]);
    const int nmob = static_cast<int>(Cb.shape()[1]);
    const int nocca = det.n_electrons(Spin::alpha);
    const int noccb = det.n_electrons(Spin::beta);
    const int nvira = nmoa - nocca;
    const int nvirb = nmob - noccb;
    const int natm = static_cast<int>(frame.nlocal());
    if (nvira <= 0 || nvirb <= 0)
        throw std::invalid_argument(
            "ump2_analytic_forces: each spin needs n_virt > 0");
    if (n != pack.n_basis || static_cast<int>(Cb.shape()[0]) != n)
        throw std::invalid_argument(
            "ump2_analytic_forces: AO dimension mismatch between det and pack");

    const xt::xtensor<double, 4>& eri = pack.ERI;
    const RealMatrix Coa = xt::view(Ca, xt::all(), xt::range(0, nocca));
    const RealMatrix Cva = xt::view(Ca, xt::all(), xt::range(nocca, nmoa));
    const RealMatrix Cob = xt::view(Cb, xt::all(), xt::range(0, noccb));
    const RealMatrix Cvb = xt::view(Cb, xt::all(), xt::range(noccb, nmob));

    // --- T2 amplitudes (PySCF ump2.kernel convention) --------------------
    // t2aa[i,j,a,b] = [(ia|jb) − (ib|ja)] / Δ
    // t2ab[i,j,a,b] = (ia|jb) / Δ_αβ
    // t2bb likewise.
    xt::xtensor<double, 4> t2aa =
        xt::zeros<double>({nocca, nocca, nvira, nvira});
    xt::xtensor<double, 4> t2bb =
        xt::zeros<double>({noccb, noccb, nvirb, nvirb});
    xt::xtensor<double, 4> t2ab =
        xt::zeros<double>({nocca, noccb, nvira, nvirb});

    if (nocca >= 1 && nvira >= 1) {
        const xt::xtensor<double, 4> ovov_aa =
            ao2mo_mixed(eri, Coa, Cva, Coa, Cva);
        for (int i = 0; i < nocca; ++i)
            for (int j = 0; j < nocca; ++j)
                for (int a = 0; a < nvira; ++a)
                    for (int b = 0; b < nvira; ++b) {
                        const double denom =
                            ea(i) + ea(j) - ea(nocca + a) - ea(nocca + b);
                        if (denom >= 0.0)
                            throw std::invalid_argument(
                                "ump2_analytic_forces: non-negative αα "
                                "orbital-energy denominator");
                        const double iajb = ovov_aa(i, a, j, b);
                        const double ibja = ovov_aa(i, b, j, a);
                        t2aa(i, j, a, b) = (iajb - ibja) / denom;
                    }
    }
    if (noccb >= 1 && nvirb >= 1) {
        const xt::xtensor<double, 4> ovov_bb =
            ao2mo_mixed(eri, Cob, Cvb, Cob, Cvb);
        for (int i = 0; i < noccb; ++i)
            for (int j = 0; j < noccb; ++j)
                for (int a = 0; a < nvirb; ++a)
                    for (int b = 0; b < nvirb; ++b) {
                        const double denom =
                            eb(i) + eb(j) - eb(noccb + a) - eb(noccb + b);
                        if (denom >= 0.0)
                            throw std::invalid_argument(
                                "ump2_analytic_forces: non-negative ββ "
                                "orbital-energy denominator");
                        const double iajb = ovov_bb(i, a, j, b);
                        const double ibja = ovov_bb(i, b, j, a);
                        t2bb(i, j, a, b) = (iajb - ibja) / denom;
                    }
    }
    if (nocca >= 1 && noccb >= 1 && nvira >= 1 && nvirb >= 1) {
        const xt::xtensor<double, 4> ovov_ab =
            ao2mo_mixed(eri, Coa, Cva, Cob, Cvb);
        for (int i = 0; i < nocca; ++i)
            for (int j = 0; j < noccb; ++j)
                for (int a = 0; a < nvira; ++a)
                    for (int b = 0; b < nvirb; ++b) {
                        const double denom =
                            ea(i) + eb(j) - ea(nocca + a) - eb(noccb + b);
                        if (denom >= 0.0)
                            throw std::invalid_argument(
                                "ump2_analytic_forces: non-negative αβ "
                                "orbital-energy denominator");
                        t2ab(i, j, a, b) = ovov_ab(i, a, j, b) / denom;
                    }
    }

    // --- Unrelaxed 1-RDM intermediates (PySCF _gamma1_intermediates) ----
    // dooa = −½ Σ t2aa t2aa − Σ t2ab t2ab;  dvva = +½ Σ t2aa t2aa + Σ t2ab
    RealMatrix dooa = xt::zeros<double>({nocca, nocca});
    RealMatrix doob = xt::zeros<double>({noccb, noccb});
    RealMatrix dvva = xt::zeros<double>({nvira, nvira});
    RealMatrix dvvb = xt::zeros<double>({nvirb, nvirb});
    for (int i = 0; i < nocca; ++i)
        for (int j = 0; j < nocca; ++j) {
            double s = 0.0;
            for (int m = 0; m < nocca; ++m)
                for (int e = 0; e < nvira; ++e)
                    for (int f = 0; f < nvira; ++f)
                        s += t2aa(i, m, e, f) * t2aa(j, m, e, f);
            dooa(i, j) -= 0.5 * s;
            s = 0.0;
            for (int m = 0; m < noccb; ++m)
                for (int e = 0; e < nvira; ++e)
                    for (int f = 0; f < nvirb; ++f)
                        s += t2ab(i, m, e, f) * t2ab(j, m, e, f);
            dooa(i, j) -= s;
        }
    for (int i = 0; i < noccb; ++i)
        for (int j = 0; j < noccb; ++j) {
            double s = 0.0;
            for (int m = 0; m < noccb; ++m)
                for (int e = 0; e < nvirb; ++e)
                    for (int f = 0; f < nvirb; ++f)
                        s += t2bb(i, m, e, f) * t2bb(j, m, e, f);
            doob(i, j) -= 0.5 * s;
            s = 0.0;
            for (int m = 0; m < nocca; ++m)
                for (int e = 0; e < nvira; ++e)
                    for (int f = 0; f < nvirb; ++f)
                        s += t2ab(m, i, e, f) * t2ab(m, j, e, f);
            doob(i, j) -= s;
        }
    for (int a = 0; a < nvira; ++a)
        for (int b = 0; b < nvira; ++b) {
            double s = 0.0;
            for (int m = 0; m < nocca; ++m)
                for (int n_ = 0; n_ < nocca; ++n_)
                    for (int e = 0; e < nvira; ++e)
                        s += t2aa(m, n_, a, e) * t2aa(m, n_, b, e);
            dvva(b, a) += 0.5 * s;
            s = 0.0;
            for (int m = 0; m < nocca; ++m)
                for (int n_ = 0; n_ < noccb; ++n_)
                    for (int e = 0; e < nvirb; ++e)
                        s += t2ab(m, n_, a, e) * t2ab(m, n_, b, e);
            dvva(b, a) += s;
        }
    for (int a = 0; a < nvirb; ++a)
        for (int b = 0; b < nvirb; ++b) {
            double s = 0.0;
            for (int m = 0; m < noccb; ++m)
                for (int n_ = 0; n_ < noccb; ++n_)
                    for (int e = 0; e < nvirb; ++e)
                        s += t2bb(m, n_, a, e) * t2bb(m, n_, b, e);
            dvvb(b, a) += 0.5 * s;
            s = 0.0;
            for (int m = 0; m < nocca; ++m)
                for (int n_ = 0; n_ < noccb; ++n_)
                    for (int e = 0; e < nvira; ++e)
                        s += t2ab(m, n_, e, a) * t2ab(m, n_, e, b);
            dvvb(b, a) += s;
        }

    // part_dm2aa[i,u,v,j] = ½ (part[i,j,u,v] − part[i,j,v,u])
    // part[i,j,u,v] = Σ_ab t2aa[i,j,a,b] Cva[u,a] Cva[v,b]
    auto make_part_ss = [&](const xt::xtensor<double, 4>& t2,
                            const RealMatrix& Cv, int no, int nv) {
        xt::xtensor<double, 4> part = xt::zeros<double>({no, no, n, n});
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int u = 0; u < n; ++u)
                    for (int v = 0; v < n; ++v) {
                        double s = 0.0;
                        for (int a = 0; a < nv; ++a)
                            for (int b = 0; b < nv; ++b)
                                s += t2(i, j, a, b) * Cv(u, a) * Cv(v, b);
                        part(i, j, u, v) = s;
                    }
        xt::xtensor<double, 4> part_dm2 = xt::zeros<double>({no, n, n, no});
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int u = 0; u < n; ++u)
                    for (int v = 0; v < n; ++v)
                        part_dm2(i, u, v, j) =
                            0.5 * (part(i, j, u, v) - part(i, j, v, u));
        return part_dm2;
    };
    const xt::xtensor<double, 4> part_dm2aa =
        make_part_ss(t2aa, Cva, nocca, nvira);
    const xt::xtensor<double, 4> part_dm2bb =
        make_part_ss(t2bb, Cvb, noccb, nvirb);
    // part_dm2ab[i,p,q,j] = Σ_ab t2ab[i,j,a,b] Cva[p,a] Cvb[q,b]
    xt::xtensor<double, 4> part_dm2ab = xt::zeros<double>({nocca, n, n, noccb});
    for (int i = 0; i < nocca; ++i)
        for (int j = 0; j < noccb; ++j)
            for (int p = 0; p < n; ++p)
                for (int q = 0; q < n; ++q) {
                    double s = 0.0;
                    for (int a = 0; a < nvira; ++a)
                        for (int b = 0; b < nvirb; ++b)
                            s += t2ab(i, j, a, b) * Cva(p, a) * Cvb(q, b);
                    part_dm2ab(i, p, q, j) = s;
                }

    // Full AO non-separable 2-RDM Gamma_a / Gamma_b (chemist pqrs).
    // Alpha buffer (PySCF dm2bufa before pack_tril):
    //   Coa[p]*part_aa[i,q,r,j]*Coa[s] + Coa[q]*part_aa[i,p,r,j]*Coa[s]
    // + Coa[p]*part_ab[i,q,r,j]*Cob[s] + Coa[q]*part_ab[i,p,r,j]*Cob[s]
    // then (r,s) symmetrization × ½ (s2kl diagonal packing).
    xt::xtensor<double, 4> Gamma_a = xt::zeros<double>({n, n, n, n});
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q)
            for (int r = 0; r < n; ++r)
                for (int s = 0; s < n; ++s) {
                    double acc = 0.0;
                    for (int i = 0; i < nocca; ++i)
                        for (int j = 0; j < nocca; ++j) {
                            acc +=
                                Coa(p, i) * part_dm2aa(i, q, r, j) * Coa(s, j);
                            acc +=
                                Coa(q, i) * part_dm2aa(i, p, r, j) * Coa(s, j);
                        }
                    for (int i = 0; i < nocca; ++i)
                        for (int j = 0; j < noccb; ++j) {
                            acc +=
                                Coa(p, i) * part_dm2ab(i, q, r, j) * Cob(s, j);
                            acc +=
                                Coa(q, i) * part_dm2ab(i, p, r, j) * Cob(s, j);
                        }
                    Gamma_a(p, q, r, s) = acc;
                }
    {
        const xt::xtensor<double, 4> Gs = Gamma_a;
        for (int p = 0; p < n; ++p)
            for (int q = 0; q < n; ++q)
                for (int r = 0; r < n; ++r)
                    for (int s = 0; s < n; ++s)
                        Gamma_a(p, q, r, s) =
                            0.5 * (Gs(p, q, r, s) + Gs(p, q, s, r));
    }

    // Beta buffer (PySCF dm2bufb):
    //   ss: Cob[p]*part_bb[i,q,r,j]*Cob[s] + Cob[q]*part_bb[i,p,r,j]*Cob[s]
    //   os: tmp_iqrs = part_ab[i,q,r,j]*Cob[s] + part_ab[i,q,s,j]*Cob[r]
    //       dm2bufb(s,r,p,q) += Coa[p,i]*tmp_iqrs
    // then (r,s) symmetrization × ½.
    xt::xtensor<double, 4> Gamma_b = xt::zeros<double>({n, n, n, n});
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q)
            for (int r = 0; r < n; ++r)
                for (int s = 0; s < n; ++s) {
                    double acc = 0.0;
                    for (int i = 0; i < noccb; ++i)
                        for (int j = 0; j < noccb; ++j) {
                            acc +=
                                Cob(p, i) * part_dm2bb(i, q, r, j) * Cob(s, j);
                            acc +=
                                Cob(q, i) * part_dm2bb(i, p, r, j) * Cob(s, j);
                        }
                    Gamma_b(p, q, r, s) = acc;
                }
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q)
            for (int r = 0; r < n; ++r)
                for (int s = 0; s < n; ++s) {
                    double acc = 0.0;
                    for (int i = 0; i < nocca; ++i)
                        for (int j = 0; j < noccb; ++j) {
                            acc +=
                                Coa(p, i) * part_dm2ab(i, q, r, j) * Cob(s, j);
                            acc +=
                                Coa(p, i) * part_dm2ab(i, q, s, j) * Cob(r, j);
                        }
                    Gamma_b(s, r, p, q) += acc;
                }
    {
        const xt::xtensor<double, 4> Gs = Gamma_b;
        for (int p = 0; p < n; ++p)
            for (int q = 0; q < n; ++q)
                for (int r = 0; r < n; ++r)
                    for (int s = 0; s < n; ++s)
                        Gamma_b(p, q, r, s) =
                            0.5 * (Gs(p, q, r, s) + Gs(p, q, s, r));
    }

    // Imat_σ[α,β] = Σ_{p,r,s} (p α | r s) Gamma_σ[p, β, r, s]
    RealMatrix Imata = xt::zeros<double>({n, n});
    RealMatrix Imatb = xt::zeros<double>({n, n});
    for (int p = 0; p < n; ++p)
        for (int alpha = 0; alpha < n; ++alpha)
            for (int beta = 0; beta < n; ++beta) {
                double aa = 0.0, bb = 0.0;
                for (int r = 0; r < n; ++r)
                    for (int s = 0; s < n; ++s) {
                        aa += eri(p, alpha, r, s) * Gamma_a(p, beta, r, s);
                        bb += eri(p, alpha, r, s) * Gamma_b(p, beta, r, s);
                    }
                Imata(alpha, beta) += aa;
                Imatb(alpha, beta) += bb;
            }

    xt::xtensor<double, 2> de = xt::zeros<double>({natm, 3});
    const RealMatrix P_hf_a = det.density(Spin::alpha);
    const RealMatrix P_hf_b = det.density(Spin::beta);
    const RealMatrix P_hf = P_hf_a + P_hf_b;

    // vhf1[A,spin,d,μ,ν] from skeleton int2e_ip1 (bra nabla, sign-flipped
    // to match PySCF) + non-separable 2-RDM · dERI.
    xt::xtensor<double, 5> vhf1 = xt::zeros<double>({natm, 2, 3, n, n});
    {
        const ShellBlockPairList bpl{pack, /*thr=*/1e-14};
        const int n_blocks = static_cast<int>(bpl.blocks.size());
        std::vector<int> block_to_atom(bpl.blocks.size(), -1);
        for (std::size_t b = 0; b < bpl.blocks.size(); ++b) {
            const auto& Bl = bpl.blocks[b];
            for (int a = 0; a < natm; ++a) {
                const double dx = Bl.center[0] - pack.atoms[a].pos[0];
                const double dy = Bl.center[1] - pack.atoms[a].pos[1];
                const double dz = Bl.center[2] - pack.atoms[a].pos[2];
                if (dx * dx + dy * dy + dz * dz < 1e-20) {
                    block_to_atom[b] = a;
                    break;
                }
            }
        }
        for (int bi = 0; bi < n_blocks; ++bi) {
            const int A = block_to_atom[static_cast<std::size_t>(bi)];
            if (A < 0)
                continue;
            const auto& BA = bpl.blocks[static_cast<std::size_t>(bi)];
            for (int bj = 0; bj < n_blocks; ++bj) {
                const auto& BB = bpl.blocks[static_cast<std::size_t>(bj)];
                for (int bk = 0; bk < n_blocks; ++bk) {
                    const auto& BC = bpl.blocks[static_cast<std::size_t>(bk)];
                    for (int bl = 0; bl < n_blocks; ++bl) {
                        const auto& BD =
                            bpl.blocks[static_cast<std::size_t>(bl)];
                        const int na = BA.n_cart, nb = BB.n_cart,
                                  nc = BC.n_cart, nd = BD.n_cart;
                        const int block_n = na * nb * nc * nd;
                        std::vector<double> deriv(
                            static_cast<std::size_t>(3 * block_n), 0.0);
                        native::rys_eri_block_deriv(BA, BB, BC, BD,
                                                    /*deriv_center=*/0,
                                                    deriv.data());
                        for (int ia = 0; ia < na; ++ia) {
                            const int p = BA.ao_start + ia;
                            for (int ib = 0; ib < nb; ++ib) {
                                const int q = BB.ao_start + ib;
                                for (int ic = 0; ic < nc; ++ic) {
                                    const int r = BC.ao_start + ic;
                                    for (int id = 0; id < nd; ++id) {
                                        const int s = BD.ao_start + id;
                                        const int ao =
                                            ((ia * nb + ib) * nc + ic) * nd
                                            + id;
                                        const double ga = Gamma_a(p, q, r, s);
                                        const double gb = Gamma_b(p, q, r, s);
                                        for (int ax = 0; ax < 3; ++ax) {
                                            const double der =
                                                -deriv[static_cast<std::size_t>(
                                                    ax * block_n + ao)];
                                            de(A, ax) -= 2.0 * der * ga;
                                            de(A, ax) -= 2.0 * der * gb;
                                            // UHF vhf1 (PySCF): J from P_tot,
                                            // −K from P_σ (no ½).
                                            vhf1(A, 0, ax, r, s) +=
                                                der * P_hf(p, q);
                                            vhf1(A, 1, ax, r, s) +=
                                                der * P_hf(p, q);
                                            vhf1(A, 0, ax, r, q) -=
                                                der * P_hf_a(p, s);
                                            vhf1(A, 1, ax, r, q) -=
                                                der * P_hf_b(p, s);
                                            vhf1(A, 0, ax, p, q) +=
                                                der * P_hf(r, s);
                                            vhf1(A, 1, ax, p, q) +=
                                                der * P_hf(r, s);
                                            vhf1(A, 0, ax, p, s) -=
                                                der * P_hf_a(q, r);
                                            vhf1(A, 1, ax, p, s) -=
                                                der * P_hf_b(q, r);
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

    IntegralDerivPack dpack{pack};

    // Unrelaxed dm1_mo (PySCF: (doo+doo.T)*0.5)
    RealMatrix dm1a = xt::zeros<double>({nmoa, nmoa});
    RealMatrix dm1b = xt::zeros<double>({nmob, nmob});
    for (int i = 0; i < nocca; ++i)
        for (int j = 0; j < nocca; ++j)
            dm1a(i, j) = 0.5 * (dooa(i, j) + dooa(j, i));
    for (int a = 0; a < nvira; ++a)
        for (int b = 0; b < nvira; ++b)
            dm1a(nocca + a, nocca + b) = 0.5 * (dvva(a, b) + dvva(b, a));
    for (int i = 0; i < noccb; ++i)
        for (int j = 0; j < noccb; ++j)
            dm1b(i, j) = 0.5 * (doob(i, j) + doob(j, i));
    for (int a = 0; a < nvirb; ++a)
        for (int b = 0; b < nvirb; ++b)
            dm1b(noccb + a, noccb + b) = 0.5 * (dvvb(a, b) + dvvb(b, a));

    // Imat_mo = − C† Imat S C
    auto to_mo = [&](const RealMatrix& Imat, const RealMatrix& C) {
        const RealMatrix IS = xt::linalg::dot(Imat, pack.S);
        return -xt::linalg::dot(xt::transpose(C), xt::linalg::dot(IS, C));
    };
    RealMatrix Imata_mo = to_mo(Imata, Ca);
    RealMatrix Imatb_mo = to_mo(Imatb, Cb);

    // Xvo = C_v† G[dm1_ao] C_o + I_ov† − I_vo
    RealMatrix dm1_ao_a =
        xt::linalg::dot(Ca, xt::linalg::dot(dm1a, xt::transpose(Ca)));
    RealMatrix dm1_ao_b =
        xt::linalg::dot(Cb, xt::linalg::dot(dm1b, xt::transpose(Cb)));
    RealMatrix Ga, Gb;
    g_uhf(eri, dm1_ao_a, dm1_ao_b, Ga, Gb);
    RealMatrix Xvo =
        xt::linalg::dot(xt::transpose(Cva), xt::linalg::dot(Ga, Coa));
    RealMatrix XVO =
        xt::linalg::dot(xt::transpose(Cvb), xt::linalg::dot(Gb, Cob));
    for (int a = 0; a < nvira; ++a)
        for (int i = 0; i < nocca; ++i)
            Xvo(a, i) += Imata_mo(i, nocca + a) - Imata_mo(nocca + a, i);
    for (int a = 0; a < nvirb; ++a)
        for (int i = 0; i < noccb; ++i)
            XVO(a, i) += Imatb_mo(i, noccb + a) - Imatb_mo(noccb + a, i);

    // Z-vector: (A+B) Z = −X  via dense solve on the unrestricted Hessian.
    const int rota = nvira * nocca;
    const int rotb = nvirb * noccb;
    const int nrot = rota + rotb;
    // MO-ERI blocks for (A+B)
    const xt::xtensor<double, 4> ovov_aa =
        ao2mo_mixed(eri, Cva, Coa, Cva, Coa); // (ai|bj)
    const xt::xtensor<double, 4> oovv_aa =
        ao2mo_mixed(eri, Coa, Coa, Cva, Cva); // (ij|ab)
    const xt::xtensor<double, 4> ovov_bb = ao2mo_mixed(eri, Cvb, Cob, Cvb, Cob);
    const xt::xtensor<double, 4> oovv_bb = ao2mo_mixed(eri, Cob, Cob, Cvb, Cvb);
    const xt::xtensor<double, 4> ovov_ab = ao2mo_mixed(eri, Cva, Coa, Cvb, Cob);

    auto apply_aplus = [&](const RealVector& v) {
        RealVector y = xt::zeros<double>({static_cast<std::size_t>(nrot)});
        for (int a = 0; a < nvira; ++a)
            for (int i = 0; i < nocca; ++i) {
                const int p = a * nocca + i;
                double acc = (ea(nocca + a) - ea(i)) * v(p);
                for (int b = 0; b < nvira; ++b)
                    for (int j = 0; j < nocca; ++j)
                        acc += (2.0 * ovov_aa(a, i, b, j) - oovv_aa(i, j, a, b)
                                - ovov_aa(a, j, b, i))
                               * v(b * nocca + j);
                for (int b = 0; b < nvirb; ++b)
                    for (int j = 0; j < noccb; ++j)
                        acc +=
                            2.0 * ovov_ab(a, i, b, j) * v(rota + b * noccb + j);
                y(p) = acc;
            }
        for (int a = 0; a < nvirb; ++a)
            for (int i = 0; i < noccb; ++i) {
                const int p = rota + a * noccb + i;
                double acc = (eb(noccb + a) - eb(i)) * v(p);
                for (int b = 0; b < nvirb; ++b)
                    for (int j = 0; j < noccb; ++j)
                        acc += (2.0 * ovov_bb(a, i, b, j) - oovv_bb(i, j, a, b)
                                - ovov_bb(a, j, b, i))
                               * v(rota + b * noccb + j);
                for (int b = 0; b < nvira; ++b)
                    for (int j = 0; j < nocca; ++j)
                        acc += 2.0 * ovov_ab(b, j, a, i) * v(b * nocca + j);
                y(p) = acc;
            }
        return y;
    };
    RealMatrix Aplus = xt::zeros<double>({nrot, nrot});
    for (int p = 0; p < nrot; ++p) {
        RealVector e = xt::zeros<double>({static_cast<std::size_t>(nrot)});
        e(p) = 1.0;
        const RealVector col = apply_aplus(e);
        for (int q = 0; q < nrot; ++q)
            Aplus(q, p) = col(q);
    }
    RealVector rhs = xt::zeros<double>({static_cast<std::size_t>(nrot)});
    for (int a = 0; a < nvira; ++a)
        for (int i = 0; i < nocca; ++i)
            rhs(a * nocca + i) = -Xvo(a, i);
    for (int a = 0; a < nvirb; ++a)
        for (int i = 0; i < noccb; ++i)
            rhs(rota + a * noccb + i) = -XVO(a, i);
    const RealVector zvec = xt::linalg::solve(Aplus, rhs);
    for (int a = 0; a < nvira; ++a)
        for (int i = 0; i < nocca; ++i) {
            dm1a(nocca + a, i) = zvec(a * nocca + i);
            dm1a(i, nocca + a) = zvec(a * nocca + i);
        }
    for (int a = 0; a < nvirb; ++a)
        for (int i = 0; i < noccb; ++i) {
            dm1b(noccb + a, i) = zvec(rota + a * noccb + i);
            dm1b(i, noccb + a) = zvec(rota + a * noccb + i);
        }

    // Hermitianize I_vo blocks
    for (int i = 0; i < nocca; ++i)
        for (int a = 0; a < nvira; ++a)
            Imata_mo(nocca + a, i) = Imata_mo(i, nocca + a);
    for (int i = 0; i < noccb; ++i)
        for (int a = 0; a < nvirb; ++a)
            Imatb_mo(noccb + a, i) = Imatb_mo(i, noccb + a);
    const RealMatrix im1 =
        xt::linalg::dot(Ca, xt::linalg::dot(Imata_mo, xt::transpose(Ca)))
        + xt::linalg::dot(Cb, xt::linalg::dot(Imatb_mo, xt::transpose(Cb)));

    // zeta_σ
    auto make_zeta = [&](const RealMatrix& C, const RealVector& eps,
                         const RealMatrix& dm1mo, int nocc, int nmo) {
        RealMatrix zeta_mo = xt::zeros<double>({nmo, nmo});
        for (int p = 0; p < nmo; ++p)
            for (int q = 0; q < nmo; ++q)
                zeta_mo(p, q) = 0.5 * (eps(p) + eps(q));
        for (int a = 0; a < nmo - nocc; ++a)
            for (int i = 0; i < nocc; ++i) {
                zeta_mo(nocc + a, i) = eps(i);
                zeta_mo(i, nocc + a) = eps(i);
            }
        for (int p = 0; p < nmo; ++p)
            for (int q = 0; q < nmo; ++q)
                zeta_mo(p, q) *= dm1mo(p, q);
        return xt::linalg::dot(C, xt::linalg::dot(zeta_mo, xt::transpose(C)));
    };
    RealMatrix zeta_a = make_zeta(Ca, ea, dm1a, nocca, nmoa);
    RealMatrix zeta_b = make_zeta(Cb, eb, dm1b, noccb, nmob);

    dm1_ao_a = xt::linalg::dot(Ca, xt::linalg::dot(dm1a, xt::transpose(Ca)));
    dm1_ao_b = xt::linalg::dot(Cb, xt::linalg::dot(dm1b, xt::transpose(Cb)));
    const RealMatrix dm1_sym_a = dm1_ao_a + xt::transpose(dm1_ao_a);
    const RealMatrix dm1_sym_b = dm1_ao_b + xt::transpose(dm1_ao_b);
    RealMatrix G_s1a, G_s1b;
    g_uhf(eri, dm1_sym_a, dm1_sym_b, G_s1a, G_s1b);
    const RealMatrix P_occ_a = xt::linalg::dot(Coa, xt::transpose(Coa));
    const RealMatrix P_occ_b = xt::linalg::dot(Cob, xt::transpose(Cob));
    // PySCF: 0.5 * (Pocc_a G_a Pocc_a + Pocc_b G_b Pocc_b)
    const RealMatrix vhf_s1occ =
        0.5
        * (xt::linalg::dot(P_occ_a, xt::linalg::dot(G_s1a, P_occ_a))
           + xt::linalg::dot(P_occ_b, xt::linalg::dot(G_s1b, P_occ_b)));

    const RealMatrix dm1pa = P_hf_a + dm1_ao_a * 2.0;
    const RealMatrix dm1pb = P_hf_b + dm1_ao_b * 2.0;
    const RealMatrix dm1_tot = P_hf + dm1_ao_a + dm1_ao_b;

    // HF energy-weighted density (factor-1 occupations per spin)
    RealMatrix W_hf = xt::zeros<double>({n, n});
    for (int i = 0; i < nocca; ++i)
        for (int mu = 0; mu < n; ++mu)
            for (int nu = 0; nu < n; ++nu)
                W_hf(mu, nu) += ea(i) * Ca(mu, i) * Ca(nu, i);
    for (int i = 0; i < noccb; ++i)
        for (int mu = 0; mu < n; ++mu)
            for (int nu = 0; nu < n; ++nu)
                W_hf(mu, nu) += eb(i) * Cb(mu, i) * Cb(nu, i);
    RealMatrix zeta = zeta_a + zeta_b + W_hf;

    for (int A = 0; A < natm; ++A) {
        for (int d = 0; d < 3; ++d) {
            const RealMatrix dS =
                xt::view(dpack.dS, A, d, xt::all(), xt::all());
            const RealMatrix dH =
                xt::view(dpack.dT, A, d, xt::all(), xt::all())
                + xt::view(dpack.dV, A, d, xt::all(), xt::all());
            const RealMatrix v1a =
                xt::view(vhf1, A, 0, d, xt::all(), xt::all());
            const RealMatrix v1b =
                xt::view(vhf1, A, 1, d, xt::all(), xt::all());

            de(A, d) += xt::sum(dS * im1)();
            de(A, d) += xt::sum(dH * xt::transpose(dm1_tot))();
            de(A, d) -= xt::sum(dS * zeta)();
            de(A, d) -= xt::sum(dS * vhf_s1occ)();
            de(A, d) -= xt::sum(v1a * dm1pa)();
            de(A, d) -= xt::sum(v1b * dm1pb)();
            de(A, d) += dpack.dV_nuc(A, d);
        }
    }

    return GradientResult{FromHartreeBohr{}, de, e_scf_ha + e_corr_ha};
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

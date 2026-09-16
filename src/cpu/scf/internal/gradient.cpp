/// @file cpu/scf/internal/gradient.cpp
/// @brief Unified analytic nuclear gradient for all mean-field methods.
///
/// `internal::ScfGradient::compute()` handles RHF, UHF, RKS, and UKS.
/// Method dispatch:
///   HF: one-electron + Coulomb + exchange + Pulay + nuclear
///   KS: one-electron + Coulomb + XC_grid  + Pulay + nuclear

#include "atomiverse/cpu/scf/internal/gradient.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/basis_eval.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/internal/dispersion.h"
#include "atomiverse/cpu/scf/internal/ecp_integrals.h"
#include "atomiverse/cpu/scf/internal/external_field.h"
#include "atomiverse/cpu/scf/internal/hamiltonian_builder.h"
#include "atomiverse/cpu/scf/internal/integral_derivatives.h"
#include "atomiverse/cpu/scf/internal/pcm.h"
#include "atomiverse/cpu/scf/internal/point_charges.h"
#include "atomiverse/cpu/scf/xc_functional.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>

namespace ATV {
namespace cpu {
namespace scf {

// ── Shared helpers ───────────────────────────────────────────────────

static std::pair<double, double>
gradient_metrics(const xt::xtensor<double, 2>& grad) {
    const int natm = static_cast<int>(grad.shape()[0]);
    double max_abs = 0.0;
    double max_atom_norm = 0.0;
    for (int A = 0; A < natm; ++A) {
        double atom_norm2 = 0.0;
        for (int d = 0; d < 3; ++d) {
            const double v = grad(A, d);
            max_abs = std::max(max_abs, std::abs(v));
            atom_norm2 += v * v;
        }
        max_atom_norm = std::max(max_atom_norm, std::sqrt(atom_norm2));
    }
    return {max_abs, max_atom_norm};
}

static void accumulate_W(xt::xtensor<double, 2>& W,
                         const xt::xtensor<double, 2>& C,
                         const xt::xtensor<double, 1>& eigenvalues, int n_occ,
                         int /*n_basis*/) {
    // W += Σ_i ε_i · c_i · c_i^T   (energy-weighted density matrix)
    for (int i = 0; i < n_occ; ++i)
        for (int mu = 0; mu < static_cast<int>(C.shape()[0]); ++mu)
            for (int nu = 0; nu < static_cast<int>(C.shape()[0]); ++nu)
                W(mu, nu) += eigenvalues(i) * C(mu, i) * C(nu, i);
}

static void accumulate_common_gradient(xt::xtensor<double, 2>& grad,
                                       const xt::xtensor<double, 2>& P,
                                       const xt::xtensor<double, 2>& W,
                                       const IntegralDerivPack& dpack,
                                       int /*n_basis*/, int natm) {
    // grad(A,d) = Tr[P · (dT + dV)] − Tr[W · dS] + dV_nuc
    // Each (A, d) slice of dpack.dT/dV/dS is an [n×n] matrix.
    for (int A = 0; A < natm; ++A) {
        for (int d = 0; d < 3; ++d) {
            auto dH = xt::view(dpack.dT, A, d, xt::all(), xt::all())
                      + xt::view(dpack.dV, A, d, xt::all(), xt::all());
            auto dS = xt::view(dpack.dS, A, d, xt::all(), xt::all());
            double val = xt::sum(P * dH)() - xt::sum(W * dS)();
            val += dpack.dV_nuc(A, d);
            grad(A, d) += val;
        }
    }
}

/// @brief Accumulate the exchange-correlation contribution dE_xc/dR_A.
///
/// For the LDA-style (vrho) part:
///   dE_xc/dR_{A,d} = sum_g w * [vrho_a * drho_a/dR_{A,d}
///                             + vrho_b * drho_b/dR_{A,d}]
///   drho_a/dR_{A,d} = -2 * sum_{mu in A} sum_nu P_a(mu,nu) dphi_mu_d phi_nu
/// In matrix form: drho_a/dR_{A,d}[g] = -2 * sum_{mu in A} T_a(g,mu) dPhi_d(g,mu)
/// with T_a = Phi @ P_a (precomputed via GEMM). For a GGA the per-point block
/// additionally contracts vsigma with d(grad_rho)/dR_A, whose nuclear
/// derivative needs the basis-function Hessian (Term2 below).
///
/// The GGA route is selected by `is_gga(config.xc_type)`, true for `PBE`,
/// `B3LYP` and `PBE0`. The vsigma chain rule is functional-agnostic given the
/// `vsigma_*` arrays returned by `evaluate_xc_batch`, so the B3LYP B88 / LYP
/// and PBE0 0.75*PBE-X semi-local exchange-correlation pieces flow through
/// unchanged — only the exact-exchange admixture (handled in the two-electron
/// gradient, not here) distinguishes a hybrid from the pure GGA.
///
/// @par Diagnosed defect (spec phase1-productization-02-ks-dshell-gradient)
/// The KS gradient was wrong by ~4 orders of magnitude. The defect was NOT in
/// the d-shell polynomial-lowering term — the `poly_grad`/`poly_hess` tables in
/// `basis_eval.h` are FD-verified correct (`test_basis_eval_dshell`), and it
/// was not l-dependent at all. Two independent root causes were found, both
/// restricted-KS-wide; they only surfaced on d-bearing bases because the sole
/// restricted-KS gradient exercise (`default_config`) happened to use one:
///
///  1. **Spin-density convention.** The XC functional is spin-resolved and
///     must be fed the per-spin factor-1 densities (restricted:
///     `rho_alpha = rho_beta = rho_total/2`). The old result bag stored the
///     factor-2 total density in its alpha slot for restricted runs and this
///     gradient assembly once fed it unhalved into the density build, so
///     `vrho`/`vsigma` were evaluated at twice the density — the dominant
///     ~24 eV/Å error. `SlaterDeterminant::density(Spin)` returns the
///     per-spin factor-1 density in BOTH spin treatments, which is exactly
///     what this assembly needs — no scale factor remains.
///
///  2. **GGA XC-potential factor of 2.** The GGA contribution to the XC
///     potential matrix in `xc_grid_engine.h` carried a spurious `* 2.0`,
///     so the SCF converged at a density where `Vxc != dE_xc/dP`. The
///     analytic nuclear gradient relies on the variational identity
///     `dE/dP = 0` (occ-virt) to drop the density-relaxation terms; with an
///     inconsistent `Vxc` that cancellation failed, leaving a residual
///     ~0.15 eV/Å on the KS gradient even after fix (1). Fixed in
///     `xc_grid_engine.h` (the `ga_d`/`gb_d` coefficients). LSDA was never
///     affected (no `vsigma`), which is how the GGA-only nature was isolated.
///
/// @param grad_out      [natm,3] gradient accumulator, units Hartree/Bohr.
/// @param integrals     Integral pack at the current geometry.
/// @param Pa            Per-spin factor-1 alpha density (dimensionless).
/// @param Pb            Per-spin factor-1 beta density (dimensionless).
/// @param xc_type       Exchange-correlation functional.
/// @param grid_level    DFT integration-grid quality level.
/// @param n_radial      Radial grid points per atom (SG0 grid only).
/// @param n_angular     Angular grid points per atom (SG0 grid only).
/// @param shell_to_atom Map from basis-function index to owning atom.
/// @param atom_shells   Per-atom list of owned basis-function indices.
/// @param natm          Number of atoms.
/// @param n             Number of basis functions.

static void accumulate_xc_gradient(
    xt::xtensor<double, 2>& grad_out, const internal::IntegralView& integrals,
    const xt::xtensor<double, 2>& Pa, const xt::xtensor<double, 2>& Pb,
    XCType xc_type, GridLevel grid_level, int n_radial, int n_angular,
    const std::vector<int>& shell_to_atom,
    const std::vector<std::vector<int>>& atom_shells, int natm, int n) {
    // GGA semi-local XC gradient route. True for PBE and the B3LYP / PBE0
    // global hybrids — all carry a gradient-dependent (vsigma) semi-local
    // term whose nuclear-derivative chain rule is functional-agnostic given
    // the vsigma_* arrays from evaluate_xc_batch. Note: the local name
    // `is_gga` deliberately shadows the free function below; the
    // ATV::cpu::scf:: qualifier disambiguates the one call site.
    const bool is_gga = ATV::cpu::scf::is_gga(xc_type);
    const int N = static_cast<int>(n);
    const size_t Ns = static_cast<size_t>(N);

    const MolecularGrid grid(integrals.atoms.data(), natm, grid_level, n_radial,
                             n_angular);
    const int G = static_cast<int>(grid.points.size());
    const size_t Gs = static_cast<size_t>(G);

    // ── 1. Build Phi[G×N], dPhi[G×N×3], (Hess[G×N×9] for GGA) ─────
    xt::xtensor<double, 2> Phi = xt::zeros<double>({Gs, Ns});
    xt::xtensor<double, 3> dPhi = xt::zeros<double>({Gs, Ns, size_t(3)});
    xt::xtensor<double, 3> Hess;
    if (is_gga)
        Hess = xt::zeros<double>({Gs, Ns, size_t(9)});

    for (int mu = 0; mu < N; ++mu) {
        const auto& sh = integrals.shells[mu];
        double min_alpha = sh.alpha[0];
        for (int k = 1; k < sh.n_prim; ++k)
            if (sh.alpha[k] < min_alpha)
                min_alpha = sh.alpha[k];

        for (int g = 0; g < G; ++g) {
            double rx = static_cast<double>(grid.points[g].x);
            double ry = static_cast<double>(grid.points[g].y);
            double rz = static_cast<double>(grid.points[g].z);
            double dx = rx - sh.center[0], dy = ry - sh.center[1],
                   dz = rz - sh.center[2];
            double r2 = dx * dx + dy * dy + dz * dz;
            if (min_alpha * r2 > 36.0)
                continue;

            if (is_gga) {
                double val, gx, gy, gz, hxx, hxy, hxz, hyy, hyz, hzz;
                eval_basis_all(sh, rx, ry, rz, val, gx, gy, gz, hxx, hxy, hxz,
                               hyy, hyz, hzz);
                Phi(g, mu) = val;
                dPhi(g, mu, 0) = gx;
                dPhi(g, mu, 1) = gy;
                dPhi(g, mu, 2) = gz;
                // Hess stored as flat [xx,xy,xz, yx,yy,yz, zx,zy,zz]
                Hess(g, mu, 0) = hxx;
                Hess(g, mu, 1) = hxy;
                Hess(g, mu, 2) = hxz;
                Hess(g, mu, 3) = hxy;
                Hess(g, mu, 4) = hyy;
                Hess(g, mu, 5) = hyz;
                Hess(g, mu, 6) = hxz;
                Hess(g, mu, 7) = hyz;
                Hess(g, mu, 8) = hzz;
            } else {
                double val, gx, gy, gz;
                eval_basis_val_grad(sh, rx, ry, rz, val, gx, gy, gz);
                Phi(g, mu) = val;
                dPhi(g, mu, 0) = gx;
                dPhi(g, mu, 1) = gy;
                dPhi(g, mu, 2) = gz;
            }
        }
    }

    // ── 2. Density via GEMM ────────────────────────────────────────────
    //
    // Spin-density convention (must match hamiltonian_builder.cpp's XC path):
    // the spin-resolved XC functional needs the per-spin FACTOR-1 densities —
    // exactly what `SlaterDeterminant::density(Spin)` returns for both spin
    // treatments (restricted: P^alpha = P^beta = P_total/2; unrestricted: one
    // spin each), and what the caller passes in as `Pa`/`Pb` (row-major, to
    // match the GEMM operands below). Feeding the unhalved P_total into a
    // restricted-KS XC gradient evaluates vrho/vsigma at twice the correct
    // density and doubles drho/dR — the ~4-orders-of-magnitude defect the
    // per-spin convention prevents.
    auto T_a = xt::linalg::dot(Phi, Pa); // [G × N]
    auto T_b = xt::linalg::dot(Phi, Pb);

    xt::xtensor<double, 1> rho_a_arr =
        xt::maximum(xt::sum(T_a * Phi, {1}), 0.0);
    xt::xtensor<double, 1> rho_b_arr =
        xt::maximum(xt::sum(T_b * Phi, {1}), 0.0);

    // Gradient density for GGA.
    xt::xtensor<double, 2> grad_rho_a, grad_rho_b; // [G × 3]
    xt::xtensor<double, 1> sigma_aa_arr, sigma_ab_arr, sigma_bb_arr;
    if (is_gga) {
        grad_rho_a = xt::zeros<double>({Gs, size_t(3)});
        grad_rho_b = xt::zeros<double>({Gs, size_t(3)});
        for (int d = 0; d < 3; ++d) {
            auto dPhi_d = xt::view(dPhi, xt::all(), xt::all(), d);
            xt::view(grad_rho_a, xt::all(), d) =
                2.0 * xt::sum(T_a * dPhi_d, {1});
            xt::view(grad_rho_b, xt::all(), d) =
                2.0 * xt::sum(T_b * dPhi_d, {1});
        }
        sigma_aa_arr = xt::sum(grad_rho_a * grad_rho_a, {1});
        sigma_ab_arr = xt::sum(grad_rho_a * grad_rho_b, {1});
        sigma_bb_arr = xt::sum(grad_rho_b * grad_rho_b, {1});
    }

    // ── 3. Batch XC evaluation ─────────────────────────────────────────
    xt::xtensor<double, 1> exc_arr = xt::zeros<double>({Gs});
    xt::xtensor<double, 1> vrho_a_arr = xt::zeros<double>({Gs});
    xt::xtensor<double, 1> vrho_b_arr = xt::zeros<double>({Gs});
    xt::xtensor<double, 1> vsigma_aa_arr, vsigma_ab_arr, vsigma_bb_arr;
    if (is_gga) {
        vsigma_aa_arr = xt::zeros<double>({Gs});
        vsigma_ab_arr = xt::zeros<double>({Gs});
        vsigma_bb_arr = xt::zeros<double>({Gs});
    }

    evaluate_xc_batch(xc_type, G, rho_a_arr.data(), rho_b_arr.data(),
                      is_gga ? sigma_aa_arr.data() : nullptr,
                      is_gga ? sigma_ab_arr.data() : nullptr,
                      is_gga ? sigma_bb_arr.data() : nullptr, exc_arr.data(),
                      vrho_a_arr.data(), vrho_b_arr.data(),
                      is_gga ? vsigma_aa_arr.data() : nullptr,
                      is_gga ? vsigma_ab_arr.data() : nullptr,
                      is_gga ? vsigma_bb_arr.data() : nullptr);

    // ── 4. Weights ─────────────────────────────────────────────────────
    xt::xtensor<double, 1> w_arr = xt::zeros<double>({Gs});
    for (int g = 0; g < G; ++g)
        w_arr(g) = static_cast<double>(grid.points[g].weight);

    // ── 5. XC gradient accumulation (per atom, per direction) ──────────
    //
    // The KS XC nuclear gradient is the sum of three additive pieces:
    //
    //   (a) AO-derivative term  -- the basis-function centers move with
    //       their atoms (grid held fixed). Johnson-Gill-Pople convention,
    //       JCP 98, 5612 (1993).
    //   (b) grid-coordinate term -- every grid point rides rigidly with its
    //       parent atom; dr_g/dR_parent = I. Reuses the AO-gradient
    //       contraction but sums over ALL basis functions and attributes
    //       the result to each grid point's parent atom. PySCF's
    //       `excsum[atm_id] += einsum('xij,ji->x', vtmp, dms) * 2`.
    //   (c) weight-derivative term -- the Becke fuzzy-cell partition weight
    //       w_n(R) depends on the nuclear positions.
    //       dE_xc/dR_A|weight = sum_g (dW_g/dR_A) * exc(g) * rho(g).
    //
    // The density-relaxation contribution cancels because the SCF is
    // variational and Vxc == dE_xc/dP. With (a)+(b)+(c) the absolute
    // per-atom XC gradient is correct from first principles and
    // sum_A dE_xc/dR_A vanishes without any translational-invariance
    // projection.
    //
    // LDA AO-derivative term:
    //   dE_xc/dR_{A,d} = -2 * sum_g w * (vrho_a * sum_{mu in A} T_a(g,mu)*dPhi_d
    //                                    + vrho_b * sum_{mu in A} T_b(g,mu)*dPhi_d)
    // Pre-weight T by w*vrho: WT_a(g,mu) = w(g)*vrho_a(g)*T_a(g,mu).
    auto w_vrho_a_col = xt::reshape_view(w_arr * vrho_a_arr, {Gs, size_t(1)});
    auto w_vrho_b_col = xt::reshape_view(w_arr * vrho_b_arr, {Gs, size_t(1)});
    xt::xtensor<double, 2> WT_a = T_a * w_vrho_a_col;
    xt::xtensor<double, 2> WT_b = T_b * w_vrho_b_col;

    // Per-(g,d) full GGA sigma-response contraction. `sign` is +1 when the
    // active basis functions move with the grid point (grid-coordinate
    // term) and -1 when they move with their own atom (AO-derivative term);
    // `restrict_to_A` selects the AO-derivative term (only basis functions
    // on atom A move) versus the grid-coordinate term (all basis functions
    // move because the grid point itself moves).
    auto gga_contraction =
        [&](int g, int dd, double sign,
            const std::vector<int>* restrict_to_A) -> double {
        const double w = w_arr(g);
        const double vsa = vsigma_aa_arr(g);
        const double vsab = vsigma_ab_arr(g);
        const double vsb = vsigma_bb_arr(g);
        double dgrad_rho_a[3] = {0, 0, 0}, dgrad_rho_b[3] = {0, 0, 0};

        // Term1: derivative of the Phi(mu) factor.
        auto term1 = [&](int mu) {
            const double dphi_mu_d = sign * dPhi(g, mu, dd);
            for (int nu = 0; nu < N; ++nu) {
                const double pa = Pa(mu, nu), pb = Pb(mu, nu);
                for (int i = 0; i < 3; ++i) {
                    dgrad_rho_a[i] += 2.0 * pa * dphi_mu_d * dPhi(g, nu, i);
                    dgrad_rho_b[i] += 2.0 * pb * dphi_mu_d * dPhi(g, nu, i);
                }
            }
        };
        // Term2: derivative of the dPhi(nu)/dr_i factor.
        auto term2 = [&](int nu) {
            for (int mu = 0; mu < N; ++mu) {
                const double pa = Pa(mu, nu), pb = Pb(mu, nu);
                const double phi_mu = Phi(g, mu);
                for (int i = 0; i < 3; ++i) {
                    const double h_nu_di = sign * Hess(g, nu, dd * 3 + i);
                    dgrad_rho_a[i] += 2.0 * pa * phi_mu * h_nu_di;
                    dgrad_rho_b[i] += 2.0 * pb * phi_mu * h_nu_di;
                }
            }
        };
        if (restrict_to_A) {
            for (int mu : *restrict_to_A)
                term1(mu);
            for (int nu : *restrict_to_A)
                term2(nu);
        } else {
            for (int mu = 0; mu < N; ++mu)
                term1(mu);
            for (int nu = 0; nu < N; ++nu)
                term2(nu);
        }

        double dsigma_aa = 0, dsigma_ab = 0, dsigma_bb = 0;
        for (int i = 0; i < 3; ++i) {
            dsigma_aa += 2.0 * grad_rho_a(g, i) * dgrad_rho_a[i];
            dsigma_bb += 2.0 * grad_rho_b(g, i) * dgrad_rho_b[i];
            dsigma_ab += grad_rho_a(g, i) * dgrad_rho_b[i]
                         + grad_rho_b(g, i) * dgrad_rho_a[i];
        }
        return w * (vsa * dsigma_aa + vsab * dsigma_ab + vsb * dsigma_bb);
    };

    // ── 5a. AO-derivative term (basis centers move with their atoms) ───
    for (int A = 0; A < natm; ++A) {
        const auto& shells_A = atom_shells[A];
        if (shells_A.empty())
            continue;

        for (int dd = 0; dd < 3; ++dd) {
            double xc_grad = 0;

            // LDA contribution via pre-weighted T; basis on atom A moving
            // contributes -dPhi (nabla_R = -nabla_r).
            for (int mu : shells_A) {
                for (int g = 0; g < G; ++g) {
                    double dphi_d = dPhi(g, mu, dd);
                    xc_grad -= 2.0 * (WT_a(g, mu) + WT_b(g, mu)) * dphi_d;
                }
            }

            // GGA contribution (per-point for correctness).
            if (is_gga) {
                for (int g = 0; g < G; ++g) {
                    double rho = rho_a_arr(g) + rho_b_arr(g);
                    if (rho < 1e-20)
                        continue;
                    // sign = -1: basis on atom A moves, nabla_R = -nabla_r.
                    xc_grad += gga_contraction(g, dd, -1.0, &shells_A);
                }
            }
            grad_out(A, dd) += static_cast<double>(xc_grad);
        }
    }

    // ── 5b. Grid-coordinate term (grid points ride with parent atom) ───
    //
    // Each grid point's electron-coordinate r_g rides rigidly with its
    // parent atom P (dr_g/dR_P = I). The XC integrand's r_g-dependence
    // therefore contributes a force on P. This reuses the AO-gradient
    // contraction with ALL basis functions active (every phi_mu(r_g) moves
    // because r_g moves) and sign = +1 (nabla_{R_P} = +nabla_{r_g}).
    for (int g = 0; g < G; ++g) {
        const int P = grid.points[g].parent_atom;
        if (P < 0 || P >= natm)
            continue;
        for (int dd = 0; dd < 3; ++dd) {
            // LDA part: +2 * sum_{mu ALL} (WT_a + WT_b) * dPhi.
            double gc = 0;
            for (int mu = 0; mu < N; ++mu) {
                gc += 2.0 * (WT_a(g, mu) + WT_b(g, mu)) * dPhi(g, mu, dd);
            }
            if (is_gga) {
                double rho = rho_a_arr(g) + rho_b_arr(g);
                if (rho >= 1e-20)
                    gc += gga_contraction(g, dd, +1.0, nullptr);
            }
            grad_out(P, dd) += static_cast<double>(gc);
        }
    }

    // ── 5c. Becke weight-derivative term ───────────────────────────────
    //
    // dE_xc/dR_A|weight = sum_g (dW_g/dR_A) * exc(g) * rho(g),  where
    // dW_g/dR_A = (W_g / w_parent) * d(w_parent_normalized)/dR_A and the
    // `* rho` factor is mandatory: Atomiverse stores exc PER ELECTRON
    // (xc_functional.h), so the weight-response carries the density (matches
    // PySCF's einsum('r,r,nxr->nx', exc, rho, weight1)).
    {
        std::vector<double> dw(static_cast<size_t>(natm) * 3);
        for (int g = 0; g < G; ++g) {
            const int P = grid.points[g].parent_atom;
            if (P < 0 || P >= natm)
                continue;
            const double rho = rho_a_arr(g) + rho_b_arr(g);
            if (rho < 1e-20)
                continue;
            const double w_parent = becke_weight_derivative(
                integrals.atoms.data(), natm, grid.points[g].x,
                grid.points[g].y, grid.points[g].z, P, grid_level, dw.data());
            if (w_parent < 1e-300)
                continue;
            // dW_g/dR_A = (W_g / w_parent) * d(w_parent_normalized)/dR_A.
            const double scale = w_arr(g) / w_parent;
            const double exc_rho = exc_arr(g) * rho;
            for (int A = 0; A < natm; ++A)
                for (int d = 0; d < 3; ++d)
                    grad_out(A, d) += scale * dw[A * 3 + d] * exc_rho;
        }
    }
}

// ── Unified gradient ─────────────────────────────────────────────────

namespace internal {

ScfGradient::ScfGradient(MeanFieldType method, SpinType spin, XCType xc_type,
                         GridLevel grid_level, int n_radial, int n_angular,
                         bool xc_grad_project_translation, bool d3bj, bool d4)
    : method_(method), spin_(spin), xc_type_(xc_type), grid_level_(grid_level),
      n_radial_(n_radial), n_angular_(n_angular),
      xc_grad_project_translation_(xc_grad_project_translation), d3bj_(d3bj),
      d4_(d4) {}

GradientResult ScfGradient::compute(const Frame& frame,
                                    const internal::IntegralView& integrals,
                                    const SlaterDeterminant& det,
                                    double e_scf_ha) const {
    // This entry answers exactly one question — what are P and W at a
    // CONVERGED determinant — and hands them to the shared assembly. The
    // Car-Parrinello entry (`compute_at_density`) answers the same question at
    // a propagated, non-stationary state and reuses everything below it.
    const int n = integrals.n_basis;
    const bool is_ks = (method_ == MeanFieldType::KS);
    const bool generalized = (spin_ == SpinType::Generalized);
    if (generalized && is_ks)
        throw std::runtime_error(
            "ScfGradient::compute: GKS (generalized KS) is not implemented.");

    // P_total = P^alpha + P^beta in both spin treatments (restricted:
    // density_total() is the factor-2 closed-shell density). Hoisted ONCE —
    // density_total() returns by value. Generalized: n×n charge density
    // P^αα + P^ββ (spin-off-diagonal blocks carry no charge).
    const xt::xtensor<double, 2> P = det.density_total();

    // Per-spin factor-1 densities for the unrestricted exchange contraction
    // and the KS XC grid. `density(Spin)` answers a restricted β query with
    // the α channel, so both are P_total/2 there. Skipped for GHF, whose
    // exchange needs all four spin blocks (it keeps the determinant route).
    xt::xtensor<double, 2> Pa, Pb;
    if (!generalized) {
        Pa = det.density(SlaterDeterminant::Spin::alpha);
        Pb = det.density(SlaterDeterminant::Spin::beta);
    }

    // Row-major copies of the MO coefficients for accumulate_W (restricted
    // determinants answer the beta query with the alpha channel, matching
    // the old C_beta == C_alpha convention). Generalized: C_a is the 2n×N_mo
    // spinor coefficient matrix.
    using Spin = SlaterDeterminant::Spin;
    const xt::xtensor<double, 2> C_a = det.coefficients(Spin::alpha);
    const xt::xtensor<double, 2> C_b = det.coefficients(Spin::beta);

    xt::xtensor<double, 2> W = xt::zeros<double>({n, n});
    if (generalized) {
        // Spinor W (2n×2n), then charge-block sum Waa + Wbb for Pulay with
        // S⊗1₂: Tr[W_spinor · (dS⊗I)] = Tr[(Waa+Wbb) · dS].
        if (!det.generalized())
            throw std::invalid_argument(
                "ScfGradient::compute: SpinType::Generalized requires a "
                "generalized (spinor) SlaterDeterminant.");
        const int two_n = static_cast<int>(C_a.shape()[0]);
        if (two_n != 2 * n)
            throw std::invalid_argument(
                "ScfGradient::compute: GHF spinor C rows ("
                + std::to_string(two_n) + ") != 2 * n_basis ("
                + std::to_string(2 * n) + ").");
        xt::xtensor<double, 2> W_spinor = xt::zeros<double>({two_n, two_n});
        accumulate_W(W_spinor, C_a, det.orbital_energies(Spin::alpha),
                     det.n_electrons_total(), two_n);
        W = xt::view(W_spinor, xt::range(0, n), xt::range(0, n))
            + xt::view(W_spinor, xt::range(n, two_n), xt::range(n, two_n));
    } else if (spin_ == SpinType::RestrictedOpen) {
        // ROHF / ROKS energy-weighted density (PySCF grad/rohf.py
        // `make_rdm1e`, shared by dft/rks_grad for ROKS):
        //   W = O_a·F_a·O_a + O_b·F_b·O_b
        // where F_a/F_b are the SPIN Fock matrices at the converged density
        // (NOT the effective Roothaan Fock, NOT Σ ε_i C C^T — that is ~7×
        // wrong for ROHF and likewise wrong for ROKS), and O_a/O_b are the
        // occupied-α / doubly-occupied MO projectors. Rebuild spin Focks
        // here: HF for ROHF, KS (J + Vxc_σ ± hybrid K) for ROKS.
        // hybrid_alpha_override = -1 → method default exchange fraction.
        HamiltonianBuilder hb;
        hb.prepare(integrals, /*is_hf=*/!is_ks, grid_level_, n_radial_,
                   n_angular_, /*fixed_grid=*/nullptr);
        const FockResult fock =
            hb.step(integrals, /*restricted=*/false, /*is_hf=*/!is_ks, xc_type_,
                    /*hybrid_alpha_override=*/-1.0, Pa, Pb);
        // Occupied-MO projector O = Σ_{i<nocc} c_i c_iᵀ (C_a is the shared
        // RO MO set; the first n_alpha columns are occupied-α, the first
        // n_beta are doubly-occupied).
        const auto projector = [&](int nocc) {
            xt::xtensor<double, 2> O = xt::zeros<double>({n, n});
            for (int i = 0; i < nocc; ++i)
                for (int mu = 0; mu < n; ++mu)
                    for (int nu = 0; nu < n; ++nu)
                        O(mu, nu) += C_a(mu, i) * C_a(nu, i);
            return O;
        };
        const xt::xtensor<double, 2> Oa =
            projector(det.n_electrons(Spin::alpha));
        const xt::xtensor<double, 2> Ob =
            projector(det.n_electrons(Spin::beta));
        W = xt::linalg::dot(Oa, xt::linalg::dot(fock.F_alpha, Oa))
            + xt::linalg::dot(Ob, xt::linalg::dot(fock.F_beta, Ob));
    } else {
        accumulate_W(W, C_a, det.orbital_energies(Spin::alpha),
                     det.n_electrons(Spin::alpha), n);
        accumulate_W(W, C_b, det.orbital_energies(Spin::beta),
                     det.n_electrons(Spin::beta), n);
    }

    return assemble_(frame, integrals, P, Pa, Pb, W, e_scf_ha,
                     generalized ? &det : nullptr);
}

GradientResult ScfGradient::compute_at_density(
    const Frame& frame, const internal::IntegralView& integrals,
    const RealMatrix& C_occ_alpha, const RealMatrix& F_alpha,
    const RealMatrix& C_occ_beta, const RealMatrix& F_beta, double e_ha) const {
    const int n = integrals.n_basis;
    const bool restricted = (spin_ == SpinType::Restricted);
    if (spin_ == SpinType::Generalized || spin_ == SpinType::RestrictedOpen)
        throw std::invalid_argument(
            "ScfGradient::compute_at_density: only Restricted and "
            "Unrestricted spin treatments carry the multiplier weight "
            "W = occ·C(CᵀFC)Cᵀ — ROHF/ROKS use projected spin Focks and GHF "
            "a spinor weight, neither of which is this formula.");

    // Row-major copies: the assembly below (and the ERI-derivative
    // contraction) index row-major AO matrices, while the SCF matrices are
    // column-major `RealMatrix`.
    const auto to_rm = [](const RealMatrix& M) {
        return xt::xtensor<double, 2>(M);
    };
    const auto check = [&](const RealMatrix& C, const RealMatrix& F,
                           const char* which) {
        if (static_cast<int>(C.shape()[0]) != n)
            throw std::invalid_argument(
                std::string("ScfGradient::compute_at_density: ") + which
                + " coefficient rows (" + std::to_string(C.shape()[0])
                + ") != n_basis (" + std::to_string(n) + ").");
        if (static_cast<int>(F.shape()[0]) != n
            || static_cast<int>(F.shape()[1]) != n)
            throw std::invalid_argument(
                std::string("ScfGradient::compute_at_density: ") + which
                + " Fock matrix must be n_basis × n_basis (" + std::to_string(n)
                + ").");
    };

    // W_σ = occ·C_σ (C_σᵀ F_σ C_σ) C_σᵀ, the Lagrange-multiplier Pulay weight
    // of a state that need not diagonalize F. `occ` is folded exactly as the
    // converged path folds it: the restricted channel is counted twice
    // (factor 2), each unrestricted channel once.
    const auto weight = [](const xt::xtensor<double, 2>& C_occ,
                           const xt::xtensor<double, 2>& F, double occ) {
        const xt::xtensor<double, 2> FC = xt::linalg::dot(F, C_occ);
        const xt::xtensor<double, 2> eps_mo =
            xt::linalg::dot(xt::transpose(C_occ), FC);
        return xt::xtensor<double, 2>(
            occ
            * xt::linalg::dot(C_occ,
                              xt::linalg::dot(eps_mo, xt::transpose(C_occ))));
    };

    xt::xtensor<double, 2> P, Pa, Pb, W;
    if (restricted) {
        if (C_occ_beta.size() != 0 || F_beta.size() != 0)
            throw std::invalid_argument(
                "ScfGradient::compute_at_density: a restricted method has one "
                "channel — pass empty beta matrices (the alpha channel "
                "carries occ = 2).");
        check(C_occ_alpha, F_alpha, "alpha");
        const xt::xtensor<double, 2> Ca = to_rm(C_occ_alpha);
        const xt::xtensor<double, 2> Fa = to_rm(F_alpha);
        Pa = xt::linalg::dot(Ca, xt::transpose(Ca)); // factor-1 per spin
        Pb = Pa;
        P = 2.0 * Pa;
        W = weight(Ca, Fa, 2.0);
    } else {
        check(C_occ_alpha, F_alpha, "alpha");
        const xt::xtensor<double, 2> Ca = to_rm(C_occ_alpha);
        const xt::xtensor<double, 2> Fa = to_rm(F_alpha);
        Pa = xt::linalg::dot(Ca, xt::transpose(Ca));
        W = weight(Ca, Fa, 1.0);
        // A channel with no electrons (e.g. the β side of a doublet) is an
        // empty coefficient block, not an error: it contributes a zero
        // density and no weight.
        if (C_occ_beta.size() != 0) {
            check(C_occ_beta, F_beta, "beta");
            const xt::xtensor<double, 2> Cb = to_rm(C_occ_beta);
            const xt::xtensor<double, 2> Fb = to_rm(F_beta);
            Pb = xt::linalg::dot(Cb, xt::transpose(Cb));
            W += weight(Cb, Fb, 1.0);
        } else {
            Pb = xt::zeros<double>({n, n});
        }
        P = Pa + Pb;
    }

    return assemble_(frame, integrals, P, Pa, Pb, W, e_ha,
                     /*generalized_det=*/nullptr);
}

GradientResult ScfGradient::assemble_(
    const Frame& frame, const internal::IntegralView& integrals,
    const xt::xtensor<double, 2>& P, const xt::xtensor<double, 2>& Pa,
    const xt::xtensor<double, 2>& Pb, const xt::xtensor<double, 2>& W,
    double e_scf_ha, const SlaterDeterminant* generalized_det) const {
    // Spherical-convention packs (def2 family) are supported: the derivative
    // integrals are sandwiched through the stored c2s in IntegralDerivPack, and
    // the two-electron path back-transforms the density (both reuse ipack.c2s).
    // ECP packs are supported via a semi-numerical dU/dR term added to the
    // gradient below (the analytic non-ECP terms already use Z_val).
    // ROHF reuses the unrestricted J−K derivative contraction (its α/β
    // densities come from shared spatial orbitals; the ROHF energy equals the
    // constrained-UHF energy at the same density). The only ROHF-specific piece
    // is the energy-weighted density W, built below from the spin Fock matrices.
    //
    // GHF: charge density P^αα+P^ββ for the 1e/J/nuclear terms; spinor
    // energy-weighted density W_spinor = Σ_i ε_i c_i c_iᵀ contracted as
    // Waa+Wbb against dS (because S_spinor = S⊗1₂). Exchange uses all four
    // spin blocks via accumulate_two_electron_gradient.
    const auto t_total = LogClock::now();
    int natm = static_cast<int>(frame.nlocal());
    int n = integrals.n_basis;
    bool is_ks = (method_ == MeanFieldType::KS);
    bool restricted = (spin_ == SpinType::Restricted);
    const bool generalized = (spin_ == SpinType::Generalized);
    if (generalized && !generalized_det)
        throw std::invalid_argument(
            "ScfGradient: SpinType::Generalized requires the spinor "
            "determinant (its exchange contraction needs all four spin "
            "blocks) — reachable only through compute().");
    // ROKS (RestrictedOpen + KS) is supported: same ROHF W construction with
    // KS spin Focks (F_σ = H + J + Vxc_σ ± hybrid K), plus the KS XC grid
    // gradient below. PySCF: dft.roks.Gradients → uks_grad.grad_elec +
    // rohf.make_rdm1e on the KS spin Fock pair.
    const char* method_label = is_ks ? "KS" : "HF";
    const char* spin_label =
        generalized ? "generalized"
                    : (spin_ == SpinType::RestrictedOpen
                           ? "restricted_open"
                           : (restricted ? "restricted" : "unrestricted"));

    ATV_INFO("[gradient] start method={} spin={} natm={} basis={}",
             method_label, spin_label, natm, n);

    const auto t_dpack = LogClock::now();
    auto dpack = IntegralDerivPack{integrals};
    ATV_DEBUG("[gradient] derivative_pack_ms={:.1f}", elapsed_ms(t_dpack));

    xt::xtensor<double, 2> grad = xt::zeros<double>({(int)natm, (int)3});

    const auto t_common = LogClock::now();
    xt::xtensor<double, 2> grad_common = xt::zeros<double>({(int)natm, (int)3});
    accumulate_common_gradient(grad_common, P, W, dpack, n, natm);
    grad += grad_common;
    const auto [common_max_abs, common_max_atom_norm] =
        gradient_metrics(grad_common);
    ATV_DEBUG("[gradient] common_terms_ms={:.1f}", elapsed_ms(t_common));
    ATV_INFO(
        "[gradient] common_terms max_abs_hb={:.6f} max_atom_norm_hb={:.6f}",
        common_max_abs, common_max_atom_norm);

    // Two-electron term: contracted on the fly to avoid materializing dERI.
    // GHF keeps the determinant route — its unsymmetrized four-block exchange
    // needs P^{αβ}/P^{βα}, which the two spin densities do not carry.
    const auto t_two_e = LogClock::now();
    xt::xtensor<double, 2> grad_two_e = xt::zeros<double>({(int)natm, (int)3});
    if (generalized)
        accumulate_two_electron_gradient(grad_two_e, integrals, P,
                                         *generalized_det, method_, spin_,
                                         xc_type_);
    else
        accumulate_two_electron_gradient(grad_two_e, integrals, P, Pa, Pb,
                                         method_, spin_, xc_type_);
    grad += grad_two_e;
    const auto [two_e_max_abs, two_e_max_atom_norm] =
        gradient_metrics(grad_two_e);

    // ── ECP nuclear-gradient term (semi-numerical dU/dR) ─────────────────
    // The only ECP-specific gradient term is Tr[P·dU/dR]; U is a cheap 1e
    // matrix (built once per geometry, not O(N⁴)), so dU/dR is obtained by a
    // central finite difference. Displacing atom A moves its shells AND its
    // ECP center together, so dU captures ∂U/∂R_A (basis-function part +
    // ECP-center part) to FD order; the Pulay −W·S^X term already handles the
    // orthonormality/density response and is ECP-agnostic. U is sandwiched to
    // spherical by the SAME geometry-independent c2s the energy path used.
    // ponytail: FD of the ECP 1e block; write analytic dU/dR recurrences only
    // if a hot ECP-force path ever needs it.
    if (integrals.has_ecp) {
        const double h = 1e-4; // Bohr — central diff of a smooth 1e matrix
        const bool sph = integrals.spherical();
        const auto& c2s = integrals.c2s;
        xt::xtensor<double, 2> ct;
        if (sph)
            ct = xt::transpose(c2s);
        const auto build_U_sph = [&](int A, int d, double sign) {
            std::vector<native::Shell> sh = integrals.shells;
            std::vector<internal::EcpCenter> ce = integrals.ecp_centers;
            const double* ap = integrals.atoms[A].pos;
            const auto on_A = [&](const double* c) {
                return c[0] == ap[0] && c[1] == ap[1] && c[2] == ap[2];
            };
            for (auto& s : sh)
                if (on_A(s.center))
                    s.center[d] += sign * h;
            for (auto& c : ce)
                if (on_A(c.pos))
                    c.pos[d] += sign * h;
            internal::EcpIntegrals e{sh, std::move(ce)};
            xt::xtensor<double, 2> U = e.matrix();
            if (sph)
                U = xt::linalg::dot(ct, xt::linalg::dot(U, c2s));
            return U;
        };
        for (int A = 0; A < natm; ++A)
            for (int d = 0; d < 3; ++d) {
                const xt::xtensor<double, 2> dU =
                    (build_U_sph(A, d, +1.0) - build_U_sph(A, d, -1.0))
                    / (2.0 * h);
                grad(A, d) += xt::sum(P * dU)();
            }
        ATV_INFO("[gradient] ecp semi-numerical dU/dR term added (h={:.0e})",
                 h);
    }

    // ── Point-charge embedding (QM/MM) gradient ──────────────────────────
    // Analytic both legs (esmm-qmmm-01):
    //   QM nuclei: Tr[P·dV/dR_A] + classical nucleus–MM Coulomb
    //   MM sites:  Tr[P·dV/dR_C] + classical (third law) + optional MM–MM
    // No central-difference core_matrix rebuild.
    xt::xtensor<double, 2> site_grad_ha; // empty unless embedding active
    if (!integrals.embed_charges.empty()) {
        if (!integrals.ecp_centers.empty())
            throw std::runtime_error(
                "ScfGradient: point-charge embedding + ECP is not supported "
                "(analytic embedding path refuses the combination; no silent "
                "FD fallback)");
        const internal::PointChargeEmbedding emb{integrals.embed_charges};
        const int n_cart = integrals.n_ao_cart();
        // Density in Cartesian AO basis for on-the-fly contraction.
        xt::xtensor<double, 2> P_cart = P;
        if (integrals.spherical()) {
            // P^cart = c2s · P^sph · c2sᵀ
            P_cart = xt::linalg::dot(
                integrals.c2s,
                xt::linalg::dot(P, xt::transpose(integrals.c2s)));
        }
        if (static_cast<int>(P_cart.shape(0)) != n_cart
            || static_cast<int>(P_cart.shape(1)) != n_cart)
            throw std::runtime_error(
                "ScfGradient: embedding density cart shape mismatch");

        std::vector<int> shell_to_atom(static_cast<std::size_t>(n_cart), -1);
        for (int i = 0; i < n_cart; ++i) {
            const double* c =
                integrals.shells[static_cast<std::size_t>(i)].center;
            for (int a = 0; a < natm; ++a) {
                const double* p =
                    integrals.atoms[static_cast<std::size_t>(a)].pos;
                if (c[0] == p[0] && c[1] == p[1] && c[2] == p[2]) {
                    shell_to_atom[static_cast<std::size_t>(i)] = a;
                    break;
                }
            }
            if (shell_to_atom[static_cast<std::size_t>(i)] < 0)
                throw std::runtime_error(
                    "ScfGradient: embed shell center has no matching atom");
        }

        emb.accumulate_qm_center_gradient(grad, integrals.shells, P_cart,
                                          shell_to_atom, integrals.atoms);

        const int nc = emb.n_charges();
        site_grad_ha = xt::zeros<double>({nc, 3});
        emb.accumulate_site_gradient(site_grad_ha, integrals.shells, P_cart,
                                     integrals.atoms);
        if (integrals.embed_self_energy)
            emb.accumulate_self_gradient(site_grad_ha);

        ATV_INFO("[gradient] embed (QM/MM) analytic gradient (n_charges={}, "
                 "self_energy={})",
                 nc, integrals.embed_self_energy);
    }

    // ── Uniform external electric field gradient ─────────────────────────
    // The field 1e operator V_field = Σ_a F_a⟨μ|(r−R₀)_a|ν⟩ is folded into
    // Hcore and the classical nuclear field energy −Σ_A Z_A F·(R_A−R₀) into
    // V_nuc (both density-independent). dE_field/dR_A adds the explicit
    // derivative of both (the density response is already in P/W), obtained by
    // central difference — displacing atom A moves its shells AND the nuclear-
    // charge center R₀, both recomputed by rebuilding ExternalField.
    const std::array<double, 3>& fld = integrals.external_field;
    if (fld[0] != 0.0 || fld[1] != 0.0 || fld[2] != 0.0) {
        const bool sph = integrals.spherical();
        const auto& c2s = integrals.c2s;
        xt::xtensor<double, 2> ct;
        if (sph)
            ct = xt::transpose(c2s);
        const double h = 1e-4; // Bohr
        for (int A = 0; A < natm; ++A) {
            const double ax = integrals.atoms[A].pos[0];
            const double ay = integrals.atoms[A].pos[1];
            const double az = integrals.atoms[A].pos[2];
            const auto on_A = [&](const double* c) {
                return c[0] == ax && c[1] == ay && c[2] == az;
            };
            for (int d = 0; d < 3; ++d) {
                std::vector<native::Atom> atoms_p = integrals.atoms;
                std::vector<native::Atom> atoms_m = integrals.atoms;
                atoms_p[A].pos[d] += h;
                atoms_m[A].pos[d] -= h;
                std::vector<native::Shell> shp = integrals.shells;
                std::vector<native::Shell> shm = integrals.shells;
                for (auto& s : shp)
                    if (on_A(s.center))
                        s.center[d] += h;
                for (auto& s : shm)
                    if (on_A(s.center))
                        s.center[d] -= h;
                const internal::ExternalField fp{fld, atoms_p};
                const internal::ExternalField fm{fld, atoms_m};
                xt::xtensor<double, 2> Vp = fp.core_matrix(shp);
                xt::xtensor<double, 2> Vm = fm.core_matrix(shm);
                if (sph) {
                    Vp = xt::linalg::dot(ct, xt::linalg::dot(Vp, c2s));
                    Vm = xt::linalg::dot(ct, xt::linalg::dot(Vm, c2s));
                }
                const double d_enuc =
                    (fp.nuclear_energy(atoms_p) - fm.nuclear_energy(atoms_m))
                    / (2.0 * h);
                grad(A, d) += xt::sum(P * (Vp - Vm))() / (2.0 * h) + d_enuc;
            }
        }
        ATV_INFO("[gradient] external-field gradient term added");
    }

    // ── C-PCM / COSMO free-energy gradient (fixed-P semi-numerical) ─────
    // Free energy G = E_HF_vac[P] + ΔG_el[P,R]. The vacuum HF terms above use
    // the solvent-polarized density P and energy-weighted W from the full
    // (vacuum+V_pcm) Fock eigenvalues — orbital response cancels by the
    // Brillouin condition of the solvent Fock. The remaining piece is
    // ∂ΔG_el/∂R|_P, obtained by central difference: rebuild the cavity +
    // unit-charge AO potentials at R±h with the FIXED reference density.
    // Hard bury-delete makes the surface discontinuous; h is small enough
    // that equilibrium H₂O stays off the topology jumps.
    // Refs: Barone & Cossi, J. Phys. Chem. A 102, 1995 (1998); free-energy
    // form ΔG = ½ q·V with q = −f S^{-1} V ⇒ ∂ΔG/∂R|_P = q·∂V/∂R − ½ q·∂S·q.
    if (integrals.solvent_eps > 0.0) {
        const double h = 1e-4; // Bohr
        const bool sph = integrals.spherical();
        const auto& c2s = integrals.c2s;
        const double eps = integrals.solvent_eps;
        const double scale = integrals.solvent_radii_scale;
        const double x = integrals.solvent_x;
        const SolventModel model = integrals.solvent_model;
        constexpr int kLeb = 302;
        const auto delta_g_at = [&](std::vector<native::Atom> atoms,
                                    std::vector<native::Shell> shells) {
            internal::Pcm pcm(std::move(atoms), eps, scale, x, kLeb, model);
            if (sph)
                pcm.prepare(shells, c2s);
            else
                pcm.prepare(shells);
            return pcm.update(P).delta_g_el;
        };
        for (int A = 0; A < natm; ++A) {
            const double ax =
                integrals.atoms[static_cast<std::size_t>(A)].pos[0];
            const double ay =
                integrals.atoms[static_cast<std::size_t>(A)].pos[1];
            const double az =
                integrals.atoms[static_cast<std::size_t>(A)].pos[2];
            const auto on_A = [&](const double* c) {
                return c[0] == ax && c[1] == ay && c[2] == az;
            };
            for (int d = 0; d < 3; ++d) {
                std::vector<native::Atom> atoms_p = integrals.atoms;
                std::vector<native::Atom> atoms_m = integrals.atoms;
                atoms_p[static_cast<std::size_t>(A)].pos[d] += h;
                atoms_m[static_cast<std::size_t>(A)].pos[d] -= h;
                std::vector<native::Shell> shp = integrals.shells;
                std::vector<native::Shell> shm = integrals.shells;
                for (auto& s : shp)
                    if (on_A(s.center))
                        s.center[d] += h;
                for (auto& s : shm)
                    if (on_A(s.center))
                        s.center[d] -= h;
                const double d_dg =
                    (delta_g_at(std::move(atoms_p), std::move(shp))
                     - delta_g_at(std::move(atoms_m), std::move(shm)))
                    / (2.0 * h);
                grad(A, d) += d_dg;
            }
        }
        ATV_INFO("[gradient] C-PCM free-energy gradient term added "
                 "(eps={}, x={}, leb={})",
                 eps, x, kLeb);
    }
    ATV_DEBUG("[gradient] two_e_ms={:.1f}", elapsed_ms(t_two_e));
    ATV_INFO("[gradient] two_e_terms max_abs_hb={:.6f} max_atom_norm_hb={:.6f}",
             two_e_max_abs, two_e_max_atom_norm);

    // XC grid contribution (KS only)
    if (is_ks) {
        const auto t_xc = LogClock::now();
        std::vector<int> shell_to_atom(n);
        std::vector<std::vector<int>> atom_shells(natm);
        {
            int idx = 0;
            for (int a = 0; a < natm; ++a) {
                for (int mu = idx; mu < n; ++mu) {
                    double dx = integrals.shells[mu].center[0]
                                - integrals.atoms[a].pos[0];
                    double dy = integrals.shells[mu].center[1]
                                - integrals.atoms[a].pos[1];
                    double dz = integrals.shells[mu].center[2]
                                - integrals.atoms[a].pos[2];
                    if (dx * dx + dy * dy + dz * dz < 1e-20) {
                        shell_to_atom[mu] = a;
                        atom_shells[a].push_back(mu);
                    } else {
                        break;
                    }
                }
                idx += static_cast<int>(atom_shells[a].size());
            }
        }
        xt::xtensor<double, 2> grad_xc = xt::zeros<double>({(int)natm, (int)3});
        accumulate_xc_gradient(grad_xc, integrals, Pa, Pb, xc_type_,
                               grid_level_, n_radial_, n_angular_,
                               shell_to_atom, atom_shells, natm, n);

        // Optional debug: enforce translational invariance of the XC
        // gradient by projecting out the net (sum-over-atoms) component.
        //
        // The exact dE_xc/dR is translation-invariant (a rigid shift of the
        // molecule leaves E_xc unchanged), so sum_A dE_xc/dR_A = 0. With the
        // genuine Becke grid-response term now assembled in
        // accumulate_xc_gradient (weight-derivative + grid-coordinate), the
        // net translational force already vanishes from first principles
        // (<= 1e-8 Ha/Bohr), so this projection is a no-op and is OFF by
        // default. It is retained behind the `xc_grad_project_translation`
        // method member (ScfGradient::xc_grad_project_translation_)
        // purely as a debug / fixed-grid-comparison aid: with a grid held
        // fixed in space the AO-derivative term alone leaves a ~4e-4 Ha/Bohr
        // net residual that this projection removes.
        if (xc_grad_project_translation_ && natm > 0) {
            for (int d = 0; d < 3; ++d) {
                double net = 0.0;
                for (int A = 0; A < natm; ++A)
                    net += grad_xc(A, d);
                const double shift = net / static_cast<double>(natm);
                for (int A = 0; A < natm; ++A)
                    grad_xc(A, d) -= shift;
            }
        }
        grad += grad_xc;
        const auto [xc_max_abs, xc_max_atom_norm] = gradient_metrics(grad_xc);
        ATV_DEBUG("[gradient] xc_ms={:.1f}", elapsed_ms(t_xc));
        ATV_INFO(
            "[gradient] xc_terms max_abs_hb={:.6f} max_atom_norm_hb={:.6f}",
            xc_max_abs, xc_max_atom_norm);
    }

    // ── DFT-D3(BJ) / DFT-D4 dispersion gradient ──────────────────────────
    // The geometry-only twin of `e_total_ += e_disp_` in the method: E_disp
    // depends on the nuclei alone (no density, no Fock), but it depends on
    // them, so its gradient belongs to the reported force exactly as much as
    // its energy belongs to the reported energy. Without this leg a d3bj/d4
    // BOMD integrates a surface that is not the one it prints, and NVE energy
    // drifts by the dispersion work done along the trajectory.
    //
    // D3(BJ): closed-form analytic (two-body C6/C8 + the dC6/dCN·dCN/dR chain),
    // shared with GPW — `cpu/gpw/internal/dftd3.cpp` is the only copy of the D3
    // math in the tree. D4: central differences of the same D4 energy
    // functional (`scf::dftd4`, gated against dftd4 3.7.0 `-g`).
    // d4 wins when both flags are set — the SAME precedence the energy uses.
    if (d4_ || d3bj_) {
        const std::vector<std::array<double, 3>> f_disp =
            dispersion_forces(frame, d3bj_, d4_);
        if (static_cast<int>(f_disp.size()) != natm)
            throw std::runtime_error(
                "ScfGradient: the dispersion evaluator returned "
                + std::to_string(f_disp.size()) + " force triples for "
                + std::to_string(natm) + " atoms");
        double disp_max_abs = 0.0;
        for (int A = 0; A < natm; ++A)
            for (int d = 0; d < 3; ++d) {
                // stored gradient is dE/dR; the evaluator returns F = −dE/dR
                const double g = -f_disp[static_cast<std::size_t>(A)]
                                        [static_cast<std::size_t>(d)];
                grad(A, d) += g;
                disp_max_abs = std::max(disp_max_abs, std::abs(g));
            }
        ATV_INFO("[gradient] dispersion gradient term added ({}) "
                 "max_abs_hb={:.6f}",
                 d4_ ? "d4" : "d3bj", disp_max_abs);
    }

    GradientResult out{FromHartreeBohr{}, grad, e_scf_ha, site_grad_ha};
    double max_force = 0.0;
    for (int A = 0; A < natm; ++A)
        for (int d = 0; d < 3; ++d)
            max_force = std::max(max_force, std::abs(out.forces(A, d)));

    ATV_INFO("[gradient] done total_ms={:.1f} energy_eV={:.6f} "
             "max_force_component_eVA={:.6f}",
             elapsed_ms(t_total), out.energy, max_force);

    return out;
}

} // namespace internal

} // namespace scf
} // namespace cpu
} // namespace ATV

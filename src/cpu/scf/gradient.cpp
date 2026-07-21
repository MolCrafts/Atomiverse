/// @file cpu/scf/gradient.cpp
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
#include "atomiverse/cpu/scf/internal/integral_derivatives.h"
#include "atomiverse/cpu/scf/xc_functional.h"

#include <cmath>
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

static GradientResult make_result(const xt::xtensor<double, 2>& grad,
                                  double energy_hartree, int natm) {
    // SCF gradient lives in Hartree/Bohr; Frame / MD drivers expect eV/Å.
    using units::convert;
    GradientResult out;
    out.forces = xt::zeros<double>({(int)natm, (int)3});
    for (int A = 0; A < natm; ++A)
        for (int d = 0; d < 3; ++d)
            out.forces(A, d) =
                -convert(grad(A, d), units::Force::HartreePerBohr,
                         units::Force::eVPerAngstrom);
    out.energy =
        convert(energy_hartree, units::Energy::Hartree, units::Energy::eV);
    return out;
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
/// @param det           Converged mean-field wavefunction (per-spin factor-1
///                      densities via `density(Spin)`).
/// @param xc_type       Exchange-correlation functional.
/// @param grid_level    DFT integration-grid quality level.
/// @param n_radial      Radial grid points per atom (SG0 grid only).
/// @param n_angular     Angular grid points per atom (SG0 grid only).
/// @param shell_to_atom Map from basis-function index to owning atom.
/// @param atom_shells   Per-atom list of owned basis-function indices.
/// @param natm          Number of atoms.
/// @param n             Number of basis functions.

static void accumulate_xc_gradient(
    xt::xtensor<double, 2>& grad_out, const IntegralPack& integrals,
    const SlaterDeterminant& det, XCType xc_type,
    GridLevel grid_level, int n_radial, int n_angular,
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

    auto grid = build_molecular_grid(integrals.atoms.data(), natm, grid_level,
                                     n_radial, n_angular);
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
    // the spin-resolved XC functional needs the per-spin FACTOR-1 densities,
    // which is exactly what `SlaterDeterminant::density(Spin)` returns for
    // both spin treatments (restricted: P^alpha = P^beta = P_total/2;
    // unrestricted: one spin each). Feeding the unhalved P_total into a
    // restricted-KS XC gradient evaluates vrho/vsigma at twice the correct
    // density and doubles drho/dR — the ~4-orders-of-magnitude defect the
    // per-spin convention prevents. Hoisted ONCE (density() returns by
    // value); row-major copies to match the GEMM operands below.
    const xt::xtensor<double, 2> Pa =
        det.density(SlaterDeterminant::Spin::alpha);
    const xt::xtensor<double, 2> Pb =
        det.density(SlaterDeterminant::Spin::beta);

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
                         bool force_native_two_electron_deriv,
                         bool xc_grad_project_translation)
    : method_(method), spin_(spin), xc_type_(xc_type), grid_level_(grid_level),
      n_radial_(n_radial), n_angular_(n_angular),
      force_native_two_electron_deriv_(force_native_two_electron_deriv),
      xc_grad_project_translation_(xc_grad_project_translation) {}

GradientResult ScfGradient::compute(const Frame& frame,
                                    const IntegralPack& integrals,
                                    const SlaterDeterminant& det,
                                    double e_scf_ha) const {
    if (integrals.spherical())
        throw std::runtime_error(
            "ScfGradient::compute: analytic gradients are not implemented "
            "for spherical-harmonic basis sets (def2 family) yet — the "
            "derivative-integral c2s transform is a follow-up chain link. "
            "Use a Cartesian-convention basis or finite differences.");
    if (integrals.has_ecp)
        throw std::runtime_error(
            "ScfGradient::compute: analytic gradients are not implemented "
            "for ECP packs yet — dU/dR of the semi-local potential is a "
            "follow-up chain link. Use finite differences.");
    if (spin_ == SpinType::RestrictedOpen)
        throw std::runtime_error(
            "ScfGradient::compute: the ROHF analytic gradient is not "
            "implemented — the Lagrangian needs the Roothaan effective-Fock "
            "coupling terms (follow-up chain link); refusing to take the "
            "unrestricted branch silently. Use finite differences.");
    if (spin_ == SpinType::Generalized)
        throw std::runtime_error(
            "ScfGradient::compute: the GHF analytic gradient is not "
            "implemented — the spinor density has no per-spin channel this "
            "assembly understands; refusing to take the unrestricted branch "
            "silently. Use finite differences.");
    const auto t_total = LogClock::now();
    int natm = static_cast<int>(frame.nlocal());
    int n = integrals.n_basis;
    bool is_ks = (method_ == MeanFieldType::KS);
    bool restricted = (spin_ == SpinType::Restricted);
    const char* method_label = is_ks ? "KS" : "HF";
    const char* spin_label = restricted ? "restricted" : "unrestricted";

    ATV_INFO("[gradient] start method={} spin={} natm={} basis={}",
             method_label, spin_label, natm, n);

    const auto t_dpack = LogClock::now();
    auto dpack = IntegralDerivPack{integrals};
    ATV_DEBUG("[gradient] derivative_pack_ms={:.1f}", elapsed_ms(t_dpack));

    // P_total = P^alpha + P^beta in both spin treatments (restricted:
    // density_total() is the factor-2 closed-shell density). Hoisted ONCE —
    // density_total() returns by value.
    const xt::xtensor<double, 2> P = det.density_total();

    // Row-major copies of the MO coefficients for accumulate_W (restricted
    // determinants answer the beta query with the alpha channel, matching
    // the old C_beta == C_alpha convention).
    using Spin = SlaterDeterminant::Spin;
    const xt::xtensor<double, 2> C_a = det.coefficients(Spin::alpha);
    const xt::xtensor<double, 2> C_b = det.coefficients(Spin::beta);

    xt::xtensor<double, 2> W = xt::zeros<double>({n, n});
    accumulate_W(W, C_a, det.orbital_energies(Spin::alpha),
                 det.n_electrons(Spin::alpha), n);
    accumulate_W(W, C_b, det.orbital_energies(Spin::beta),
                 det.n_electrons(Spin::beta), n);

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
    const auto t_two_e = LogClock::now();
    xt::xtensor<double, 2> grad_two_e = xt::zeros<double>({(int)natm, (int)3});
    accumulate_two_electron_gradient(grad_two_e, frame, integrals, P, det,
                                     method_, spin_, xc_type_,
                                     force_native_two_electron_deriv_);
    grad += grad_two_e;
    const auto [two_e_max_abs, two_e_max_atom_norm] =
        gradient_metrics(grad_two_e);
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
        accumulate_xc_gradient(grad_xc, integrals, det, xc_type_,
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

    auto out = make_result(grad, e_scf_ha, natm);
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

/// @file cpu/scf/xc_grid_engine.h
/// @brief Batched XC grid integration engine — GEMM-based density/Vxc assembly.
///
/// Replaces per-point XC integration with matrix operations:
///   1. Build Phi[n_grid × n_basis] — batched basis function evaluation
///   2. rho = rowsum((Phi @ P) * Phi) — density via GEMM
///   3. XC batch evaluation — vectorized over grid points
///   4. Vxc = Phi_w^T @ Phi_w — potential matrix via GEMM
///
/// Shared by scf_solver.cpp (Fock build) and gradient.cpp (XC gradient).
///
/// The AO-on-grid matrices Phi (and, for GGA, dPhi_x/y/z) depend only on the
/// molecular geometry and the grid, not on the density, so they are geometry-
/// fixed across all SCF iterations. `XCGridBasis` holds them so the Fock
/// builder can build them once per geometry (in `HamiltonianBuilder`, a mutable
/// per-geometry cache) and reuse them across the ~15–30 SCF iterations,
/// avoiding a fresh G×N pass of transcendental basis evaluations every step.
///
/// Memory cost of the cache (unconditional — no thresholds, no config knobs):
/// the LSDA cache holds one `Phi[G × N]`; the GGA cache holds `Phi` plus three
/// gradient blocks `dPhi_x/y/z`, i.e. **4×G·N doubles**. For def2-SVP benzene
/// (N ≈ 120 Cartesian AOs, medium grid G ≈ 1.2×10⁵) that is ≈115 MB (LSDA) /
/// ≈460 MB (GGA). This is the same order as the `T_a/T_b` transients the
/// density GEMM already materializes each call; the cache simply keeps that
/// working set resident across iterations. If a future very large grid OOMs,
/// the escape hatch is a grid-point threshold — deliberately not introduced
/// here (no speculative configuration).
///
/// @see PySCF numint.py for the reference algorithm.
#pragma once

#include <type_traits>
#include <utility>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/cpu/scf/basis_eval.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/xc_functional.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Result of batched XC grid integration.
struct XCGridResult {
    xt::xtensor<double, 2>
        Vxc_alpha; ///< XC potential matrix (alpha) [n_basis × n_basis].
    xt::xtensor<double, 2>
        Vxc_beta; ///< XC potential matrix (beta) [n_basis × n_basis].
    double E_xc;  ///< Total XC energy [Hartree].
    double n_alpha_integrated; ///< Grid-integrated alpha electron count.
    double n_beta_integrated;  ///< Grid-integrated beta electron count.
};

/// @brief Geometry-fixed AO-on-grid basis workspace (internal, not a Config
///        bag): the value matrix Phi and, for GGA, the gradient matrices
///        dPhi_x/y/z on a fixed molecular grid.
///
/// These matrices depend only on the geometry + grid, so `build()` is a
/// per-geometry one-time cost the density-dependent `evaluate_xc_batched`
/// overload consumes across every SCF iteration. Owned as a private cache by
/// `HamiltonianBuilder`; it is not a user-facing type.
struct XCGridBasis {
    xt::xtensor<double, 2> Phi;    ///< AO values Phi[G × N].
    xt::xtensor<double, 2> dPhi_x; ///< d/dx Phi[G × N] (GGA only).
    xt::xtensor<double, 2> dPhi_y; ///< d/dy Phi[G × N] (GGA only).
    xt::xtensor<double, 2> dPhi_z; ///< d/dz Phi[G × N] (GGA only).
    bool gga = false;              ///< true → gradient blocks are populated.
    bool built = false; ///< true → build() has run for the current geometry.

    /// @brief Evaluate the AO value (and, for GGA, gradient) matrices on the
    ///        grid, reusing the same per-point basis evaluation as the cube
    ///        writers (`eval_basis` / `eval_basis_val_grad`).
    ///
    /// @param grid    Molecular quadrature grid.
    /// @param shells  Basis-function shell descriptors [n_basis].
    /// @param n_basis Number of basis functions.
    /// @param is_gga  true → also build dPhi_x/y/z (GGA functionals).
    void build(const MolecularGrid& grid,
               const std::vector<native::Shell>& shells, int n_basis,
               bool is_gga) {
        const int G = static_cast<int>(grid.points.size());
        const int N = n_basis;
        const std::size_t Gs = static_cast<std::size_t>(G);
        const std::size_t Ns = static_cast<std::size_t>(N);
        gga = is_gga;

        // Row-major for element-wise ops; will convert for BLAS as needed.
        Phi = xt::zeros<double>({Gs, Ns});
        if (is_gga) {
            dPhi_x = xt::zeros<double>({Gs, Ns});
            dPhi_y = xt::zeros<double>({Gs, Ns});
            dPhi_z = xt::zeros<double>({Gs, Ns});
        }

        // Loop: outer over grid points so each OpenMP thread owns a disjoint
        // set of grid rows -- Phi(g,:) and dPhi_*(g,:) are then written by
        // exactly one thread, race-free. The result is bitwise identical to a
        // serial mu-outer loop (each Phi(g,mu) is the same single eval); only
        // the iteration order changes. Delegates the per-point polynomial logic
        // to basis_eval.h so d-shells (and higher l) inherit the same code path
        // as the cube writers.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int g = 0; g < G; ++g) {
            const double rx = static_cast<double>(grid.points[g].x);
            const double ry = static_cast<double>(grid.points[g].y);
            const double rz = static_cast<double>(grid.points[g].z);
            for (int mu = 0; mu < N; ++mu) {
                const auto& sh = shells[mu];
                if (is_gga) {
                    double v, gx, gy, gz;
                    eval_basis_val_grad(sh, rx, ry, rz, v, gx, gy, gz);
                    Phi(g, mu) = v;
                    dPhi_x(g, mu) = gx;
                    dPhi_y(g, mu) = gy;
                    dPhi_z(g, mu) = gz;
                } else {
                    Phi(g, mu) = eval_basis(sh, rx, ry, rz);
                }
            }
        }
        built = true;
    }
};

/// @brief Parallel grid-contraction GEMM: C = A^T @ B.
///
/// Computes `C(mu,nu) = sum_g A(g,mu) * B(g,nu)` for two `[G x N]` row-major
/// grid matrices, the contraction pattern of every XC potential-matrix
/// assembly. OpenMP-parallel over the output column `nu`; the reduction
/// over grid points `g` runs serially per output element in ascending
/// order. Because each output element owns its full reduction, the result
/// is bitwise identical for any thread count and equal to the serial
/// ascending-`g` sum. This replaces the single-threaded `xt::linalg::dot`
/// fallback that dominated benzene-scale GGA Fock builds; the existing
/// LSDA/PBE grid and KS-gradient regression tests confirm the non-hybrid
/// KS results are unchanged within SCF tolerance.
///
/// @param A `[G x N]` row-major grid matrix (the left, transposed operand).
/// @param B `[G x N]` row-major grid matrix (the right operand).
/// @return `[N x N]` matrix C = A^T @ B.
inline xt::xtensor<double, 2>
grid_contract_tn(const xt::xtensor<double, 2>& A,
                 const xt::xtensor<double, 2>& B) {
    const std::size_t G = A.shape()[0];
    const std::size_t N = A.shape()[1];
    const auto Ni = static_cast<long>(N);
    const auto Gi = static_cast<long>(G);

    // Transpose A and B to [N x G] (column = grid point) so the per-output
    // reduction over g walks contiguous memory. A naive [G x N] access
    // strides by N doubles per g-step and thrashes the cache on a
    // benzene-scale grid; the O(N*G) transpose copy is negligible against
    // the O(N^2*G) contraction it accelerates.
    std::vector<double> At(static_cast<std::size_t>(N) * G);
    std::vector<double> Bt(static_cast<std::size_t>(N) * G);
    const double* Ap = A.data();
    const double* Bp = B.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long g = 0; g < Gi; ++g)
        for (long mu = 0; mu < Ni; ++mu) {
            At[static_cast<std::size_t>(mu) * G + g] = Ap[g * Ni + mu];
            Bt[static_cast<std::size_t>(mu) * G + g] = Bp[g * Ni + mu];
        }

    xt::xtensor<double, 2> C = xt::zeros<double>({N, N});
    double* Cp = C.data();
    // Parallel over the output column nu; the g-reduction for each output
    // element runs serially in ascending order, so the result is bitwise
    // identical for any thread count (and matches a serial g-order sum).
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long nu = 0; nu < Ni; ++nu) {
        const double* Brow = Bt.data() + static_cast<std::size_t>(nu) * G;
        for (long mu = 0; mu < Ni; ++mu) {
            const double* Arow = At.data() + static_cast<std::size_t>(mu) * G;
            double acc = 0.0;
            for (long g = 0; g < Gi; ++g)
                acc += Arow[g] * Brow[g];
            Cp[mu * Ni + nu] = acc;
        }
    }
    return C;
}

/// @brief Batched XC grid integration on a precomputed AO-on-grid basis.
///
/// Runs the density-dependent steps (density GEMM → XC batch → Vxc GEMM) on a
/// geometry-fixed `XCGridBasis`, so the geometry-invariant Phi/dPhi build is
/// hoisted out of the SCF iteration loop. Bit-for-bit identical to the
/// rebuild-every-call overload on the same inputs.
///
/// @param aog     Precomputed AO value/gradient matrices (built for this grid).
/// @param grid    Molecular quadrature grid (supplies the weights + point
///                count; must be the grid `aog` was built from).
/// @param P_alpha Alpha density matrix [n_basis × n_basis].
/// @param P_beta  Beta density matrix [n_basis × n_basis].
/// @param xc_type XC functional type (LSDA or PBE).
/// @return XCGridResult with Vxc matrices, E_xc, and integration diagnostics.
inline XCGridResult evaluate_xc_batched(const XCGridBasis& aog,
                                        const MolecularGrid& grid,
                                        const xt::xtensor<double, 2>& P_alpha,
                                        const xt::xtensor<double, 2>& P_beta,
                                        XCType xc_type) {
    const int G = static_cast<int>(grid.points.size());
    const int N = static_cast<int>(aog.Phi.shape()[1]);
    // GGA path is needed for PBE and for the B3LYP / PBE0 hybrids, whose
    // semi-local part carries a gradient-dependent term.
    const bool is_gga = ATV::cpu::scf::is_gga(xc_type);
    const size_t Gs = static_cast<size_t>(G);
    const size_t Ns = static_cast<size_t>(N);

    // ── 1. AO-on-grid basis is precomputed (aog). ────────────────────
    const xt::xtensor<double, 2>& Phi = aog.Phi;
    const xt::xtensor<double, 2>& dPhi_x = aog.dPhi_x;
    const xt::xtensor<double, 2>& dPhi_y = aog.dPhi_y;
    const xt::xtensor<double, 2>& dPhi_z = aog.dPhi_z;

    // ── 2. Density via GEMM: T = Phi @ P, rho = rowsum(T * Phi) ───────
    // When P is already double (default AccReal) contract against the input
    // directly, skipping the O(N²) elementwise copy; the float-storage path
    // (StorageReal=float) keeps the explicit conversion.
    xt::xtensor<double, 2> T_a, T_b;
    if constexpr (std::is_same_v<
                      typename std::decay_t<decltype(P_alpha)>::value_type,
                      double>) {
        T_a = xt::linalg::dot(Phi, P_alpha);
        T_b = xt::linalg::dot(Phi, P_beta);
    } else {
        xt::xtensor<double, 2> Pa({Ns, Ns}), Pb({Ns, Ns});
        for (int mu = 0; mu < N; ++mu)
            for (int nu = 0; nu < N; ++nu) {
                Pa(mu, nu) = static_cast<double>(P_alpha(mu, nu));
                Pb(mu, nu) = static_cast<double>(P_beta(mu, nu));
            }
        T_a = xt::linalg::dot(Phi, Pa);
        T_b = xt::linalg::dot(Phi, Pb);
    }

    // rho_a[g] = sum_nu T_a(g,nu) * Phi(g,nu)  — element-wise multiply + row sum
    xt::xtensor<double, 1> rho_a_arr = xt::sum(T_a * Phi, {1});
    xt::xtensor<double, 1> rho_b_arr = xt::sum(T_b * Phi, {1});

    // Clamp densities >= 0
    rho_a_arr = xt::maximum(rho_a_arr, 0.0);
    rho_b_arr = xt::maximum(rho_b_arr, 0.0);

    // ── 2b. Gradient density for GGA ───────────────────────────────────
    // grad_rho_a[d] = 2 * sum_nu T_a(g,nu) * dPhi_d(g,nu)
    xt::xtensor<double, 1> sigma_aa_arr, sigma_ab_arr, sigma_bb_arr;
    xt::xtensor<double, 2> grad_rho_ax, grad_rho_ay, grad_rho_az;
    xt::xtensor<double, 2> grad_rho_bx, grad_rho_by, grad_rho_bz;
    if (is_gga) {
        // grad_rho_a_d[g] = 2 * sum_nu T_a(g,nu) * dPhi_d(g,nu)
        auto gax = 2.0 * xt::sum(T_a * dPhi_x, {1});
        auto gay = 2.0 * xt::sum(T_a * dPhi_y, {1});
        auto gaz = 2.0 * xt::sum(T_a * dPhi_z, {1});
        auto gbx = 2.0 * xt::sum(T_b * dPhi_x, {1});
        auto gby = 2.0 * xt::sum(T_b * dPhi_y, {1});
        auto gbz = 2.0 * xt::sum(T_b * dPhi_z, {1});

        sigma_aa_arr = gax * gax + gay * gay + gaz * gaz;
        sigma_ab_arr = gax * gbx + gay * gby + gaz * gbz;
        sigma_bb_arr = gbx * gbx + gby * gby + gbz * gbz;

        // Store for GGA Vxc assembly
        grad_rho_ax =
            xt::xtensor<double, 2>(xt::reshape_view(gax, {Gs, size_t(1)}));
        grad_rho_ay =
            xt::xtensor<double, 2>(xt::reshape_view(gay, {Gs, size_t(1)}));
        grad_rho_az =
            xt::xtensor<double, 2>(xt::reshape_view(gaz, {Gs, size_t(1)}));
        grad_rho_bx =
            xt::xtensor<double, 2>(xt::reshape_view(gbx, {Gs, size_t(1)}));
        grad_rho_by =
            xt::xtensor<double, 2>(xt::reshape_view(gby, {Gs, size_t(1)}));
        grad_rho_bz =
            xt::xtensor<double, 2>(xt::reshape_view(gbz, {Gs, size_t(1)}));
    }

    // ── 3. Batch XC evaluation ─────────────────────────────────────────
    xt::xtensor<double, 1> exc_arr = xt::zeros<double>({Gs});
    xt::xtensor<double, 1> vrho_a_arr = xt::zeros<double>({Gs});
    xt::xtensor<double, 1> vrho_b_arr = xt::zeros<double>({Gs});
    xt::xtensor<double, 1> vsigma_aa_out, vsigma_ab_out, vsigma_bb_out;
    if (is_gga) {
        vsigma_aa_out = xt::zeros<double>({Gs});
        vsigma_ab_out = xt::zeros<double>({Gs});
        vsigma_bb_out = xt::zeros<double>({Gs});
    }

    evaluate_xc_batch(xc_type, G, rho_a_arr.data(), rho_b_arr.data(),
                      is_gga ? sigma_aa_arr.data() : nullptr,
                      is_gga ? sigma_ab_arr.data() : nullptr,
                      is_gga ? sigma_bb_arr.data() : nullptr, exc_arr.data(),
                      vrho_a_arr.data(), vrho_b_arr.data(),
                      is_gga ? vsigma_aa_out.data() : nullptr,
                      is_gga ? vsigma_ab_out.data() : nullptr,
                      is_gga ? vsigma_bb_out.data() : nullptr);

    // ── 4. Integrate E_xc and electron counts ──────────────────────────
    xt::xtensor<double, 1> w_arr = xt::zeros<double>({Gs});
    for (int g = 0; g < G; ++g)
        w_arr(g) = static_cast<double>(grid.points[g].weight);

    double E_xc = xt::sum(w_arr * exc_arr * (rho_a_arr + rho_b_arr))();
    double n_alpha_int = xt::sum(w_arr * rho_a_arr)();
    double n_beta_int = xt::sum(w_arr * rho_b_arr)();

    // ── 5. Vxc via GEMM ───────────────────────────────────────────────
    // LDA: Vxc = Phi_w^T @ Phi  where Phi_w(g,mu) = w(g)*vrho(g)*Phi(g,mu)
    // This gives: Vxc(mu,nu) = sum_g w(g) * vrho(g) * Phi(g,mu) * Phi(g,nu)
    //
    // For GGA, add gradient terms:
    //   Vxc += sum_d (Phi_gd_a)^T @ dPhi_d + dPhi_d^T @ (Phi_gd_a)
    // where Phi_gd_a(g,mu) = w(g) * ga_d(g) * Phi(g,mu)
    //   and ga_d = 2*(2*vsigma_aa*grad_rho_a_d + vsigma_ab*grad_rho_b_d)

    // LDA contribution (common to both LDA and GGA)
    // Phi_wa(g, mu) = w(g) * vrho_a(g) * Phi(g, mu)
    auto w_vrho_a = xt::reshape_view(w_arr * vrho_a_arr, {Gs, size_t(1)});
    auto w_vrho_b = xt::reshape_view(w_arr * vrho_b_arr, {Gs, size_t(1)});
    xt::xtensor<double, 2> Phi_wa = Phi * w_vrho_a;
    xt::xtensor<double, 2> Phi_wb = Phi * w_vrho_b;

    // Vxc_alpha = Phi_wa^T @ Phi  [N × N]. grid_contract_tn replaces the
    // single-threaded xt::linalg::dot fallback with an OpenMP-parallel,
    // bitwise-identical (same g-order reduction) contraction.
    xt::xtensor<double, 2> Vxc_a_d = grid_contract_tn(Phi_wa, Phi);
    xt::xtensor<double, 2> Vxc_b_d = grid_contract_tn(Phi_wb, Phi);

    if (is_gga) {
        // GGA contribution to the XC potential matrix.
        //
        //   Vxc_a^sigma(mu,nu) = sum_g sum_d ga_d(g)
        //                            * [Phi(mu)dPhi(nu,d) + dPhi(mu,d)Phi(nu)]
        //   ga_d = w * (2*vsigma_aa*grad_rho_a_d + vsigma_ab*grad_rho_b_d)
        //   gb_d = w * (2*vsigma_bb*grad_rho_b_d + vsigma_ab*grad_rho_a_d)
        //
        // E_xc = sum_g w f(sigma), sigma_aa = grad_rho_a . grad_rho_a, with the
        // P-symmetric form grad_rho_a_d = sum_kl P_a(kl)[dPhi(k,d)Phi(l) +
        // Phi(k)dPhi(l,d)]. Then dE/dP_a(mu,nu) = sum_g w [2*vsigma_aa*gra_d
        // + vsigma_ab*grb_d] * [dPhi(mu,d)Phi(nu)+Phi(mu)dPhi(nu,d)]. The
        // symmetric [Phi*dPhi + dPhi*Phi] pair is already supplied by the two
        // GEMMs below, so ga_d carries NO extra factor of 2. An earlier
        // spurious `* 2.0` here doubled the GGA potential, leaving the SCF
        // converged at a density where Vxc != dE_xc/dP; that broke the
        // density-relaxation cancellation in the analytic KS nuclear gradient
        // (spec phase1-productization-02-ks-dshell-gradient).
        auto vsa = xt::reshape_view(vsigma_aa_out, {Gs, size_t(1)});
        auto vsab = xt::reshape_view(vsigma_ab_out, {Gs, size_t(1)});
        auto vsb = xt::reshape_view(vsigma_bb_out, {Gs, size_t(1)});
        auto w_col = xt::reshape_view(w_arr, {Gs, size_t(1)});

        // For each Cartesian direction d:
        //   Vxc_a += w * (ga_d * Phi)^T @ dPhi_d + w * (ga_d * dPhi_d)^T @ Phi
        // which is: Vxc_a += (Phi * w*ga_d)^T @ dPhi_d + (dPhi_d * w*ga_d)^T @ Phi
        // Symmetrized: Vxc_a += (Phi * w*ga_d)^T @ dPhi_d + dPhi_d^T @ (Phi * w*ga_d)
        // But Phi * w*ga_d transposed-dotted with dPhi_d gives a non-symmetric matrix.
        // The correct formula for LDA+GGA combined Vxc(mu,nu) = sum_g w *
        //   [vrho * phi_mu * phi_nu + sum_d ga_d * (phi_mu * dphi_d_nu + dphi_d_mu * phi_nu)]
        // = LDA part + sum_d [ga_d * Phi]^T @ dPhi_d + sum_d [ga_d * dPhi_d]^T @ Phi

        const xt::xtensor<double, 2>* grad_rho_a[3] = {
            &grad_rho_ax, &grad_rho_ay, &grad_rho_az};
        const xt::xtensor<double, 2>* grad_rho_b[3] = {
            &grad_rho_bx, &grad_rho_by, &grad_rho_bz};
        const xt::xtensor<double, 2>* dPhi_arr[3] = {&dPhi_x, &dPhi_y, &dPhi_z};

        for (int d = 0; d < 3; ++d) {
            auto ga_d =
                w_col
                * (2.0 * vsa * (*grad_rho_a[d]) + vsab * (*grad_rho_b[d]));
            auto gb_d =
                w_col
                * (2.0 * vsb * (*grad_rho_b[d]) + vsab * (*grad_rho_a[d]));
            const auto& dPhi_d = *dPhi_arr[d];

            // Vxc_a += (Phi * ga_d)^T @ dPhi_d + dPhi_d^T @ (Phi * ga_d).
            // Phi*ga_d is materialized into a container so grid_contract_tn
            // sees a contiguous operand.
            xt::xtensor<double, 2> Phi_ga = Phi * ga_d;
            Vxc_a_d += grid_contract_tn(Phi_ga, dPhi_d);
            Vxc_a_d += grid_contract_tn(dPhi_d, Phi_ga);

            xt::xtensor<double, 2> Phi_gb = Phi * gb_d;
            Vxc_b_d += grid_contract_tn(Phi_gb, dPhi_d);
            Vxc_b_d += grid_contract_tn(dPhi_d, Phi_gb);
        }
    }

    // Symmetrize (GEMM may produce slight asymmetry from rounding)
    Vxc_a_d = 0.5 * (Vxc_a_d + xt::transpose(Vxc_a_d));
    Vxc_b_d = 0.5 * (Vxc_b_d + xt::transpose(Vxc_b_d));

    // Emit the result. When the working element type already matches the
    // output (double, the default AccReal) move the assembled matrices out —
    // skipping the O(N²) elementwise copy; the float-storage path keeps the
    // explicit conversion.
    XCGridResult result;
    if constexpr (std::is_same_v<
                      typename std::decay_t<decltype(P_alpha)>::value_type,
                      double>) {
        result.Vxc_alpha = std::move(Vxc_a_d);
        result.Vxc_beta = std::move(Vxc_b_d);
    } else {
        result.Vxc_alpha = xt::xtensor<double, 2>::from_shape({Ns, Ns});
        result.Vxc_beta = xt::xtensor<double, 2>::from_shape({Ns, Ns});
        for (int mu = 0; mu < N; ++mu)
            for (int nu = 0; nu < N; ++nu) {
                result.Vxc_alpha(mu, nu) = static_cast<double>(Vxc_a_d(mu, nu));
                result.Vxc_beta(mu, nu) = static_cast<double>(Vxc_b_d(mu, nu));
            }
    }
    result.E_xc = static_cast<double>(E_xc);
    result.n_alpha_integrated = static_cast<double>(n_alpha_int);
    result.n_beta_integrated = static_cast<double>(n_beta_int);
    return result;
}

/// @brief Batched XC grid integration engine (rebuild-every-call).
///
/// Thin wrapper that builds the AO value/gradient matrices for this geometry
/// and delegates to the precomputed-basis overload. Kept bit-identical for the
/// TD-DFT / gradient call sites that do not (yet) hoist the AO-on-grid cache.
///
/// @param grid    Molecular quadrature grid.
/// @param shells  Basis function shell descriptors [n_basis].
/// @param n_basis Number of basis functions.
/// @param P_alpha Alpha density matrix [n_basis × n_basis].
/// @param P_beta  Beta density matrix [n_basis × n_basis].
/// @param xc_type XC functional type (LSDA or PBE).
/// @return XCGridResult with Vxc matrices, E_xc, and integration diagnostics.
inline XCGridResult
evaluate_xc_batched(const MolecularGrid& grid,
                    const std::vector<native::Shell>& shells, int n_basis,
                    const xt::xtensor<double, 2>& P_alpha,
                    const xt::xtensor<double, 2>& P_beta, XCType xc_type) {
    XCGridBasis aog;
    aog.build(grid, shells, n_basis, ATV::cpu::scf::is_gga(xc_type));
    return evaluate_xc_batched(aog, grid, P_alpha, P_beta, xc_type);
}

} // namespace scf
} // namespace cpu
} // namespace ATV

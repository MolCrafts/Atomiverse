/// @file cpu/scf/internal/hamiltonian_builder.cpp
/// @brief Method-dispatched Fock build (RHF / UHF / RKS / UKS).

#include "atomiverse/cpu/scf/internal/hamiltonian_builder.h"

#include <stdexcept>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/reducers/xreducer.hpp>

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/xc_grid_engine.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

void HamiltonianBuilder::prepare(const IntegralPack& integrals, bool is_hf,
                                 GridLevel grid_level, int n_radial,
                                 int n_angular, const MolecularGrid* fixed_grid,
                                 RI ri, AuxBasis aux_basis) {
    if (!integrals.shells.empty()) {
        sp_block_ = ShellBlockPairList{integrals};
        sp_ = ShellPairList{integrals};
    }

    ri_ = ri;
    if (ri_ != RI::None) {
        if (integrals.shells.empty())
            throw std::runtime_error(
                "HamiltonianBuilder: RI requested but the integral pack "
                "carries no shell workspace.");
        const auto t_df = LogClock::now();
        df_ = std::make_unique<DensityFitting>(integrals.shells,
                                               integrals.atoms, aux_basis);
        ATV_INFO("[scf] density-fitting workspace built n_ao={} n_aux={} "
                 "ms={:.1f}",
                 df_->n_ao(), df_->n_aux(), elapsed_ms(t_df));
    } else {
        df_.reset();
    }

    if (!is_hf && !integrals.atoms.empty()) {
        if (fixed_grid != nullptr) {
            // Externally supplied grid: used by the fixed-grid FD KS
            // gradient oracle so the +/- displacements share one grid.
            grid_ = *fixed_grid;
            ATV_INFO(
                "[scf] grid build skipped: using supplied fixed grid points={}",
                grid_.points.size());
        } else {
            const auto t_grid = LogClock::now();
            ATV_INFO(
                "[scf] grid build start natm={} level={} radial={} angular={}",
                integrals.atoms.size(), static_cast<int>(grid_level), n_radial,
                n_angular);
            grid_ =
                build_molecular_grid(integrals.atoms.data(),
                                     static_cast<int>(integrals.atoms.size()),
                                     grid_level, n_radial, n_angular);
            ATV_INFO("[scf] grid build done points={} ms={:.1f}",
                     grid_.points.size(), elapsed_ms(t_grid));
        }
    }
}

FockResult HamiltonianBuilder::step(const IntegralPack& integrals,
                                    bool restricted, bool is_hf, XCType xc_type,
                                    double hybrid_alpha_override,
                                    const RealMatrix& P_alpha,
                                    const RealMatrix& P_beta) const {
    FockResult out;
    const RealMatrix& Hcore = integrals.Hcore;
    const int n = integrals.n_basis;

    // Spherical-basis bridge: the direct two-electron / XC engines are
    // Cartesian (they walk pack.shells). With a spherical pack the incoming
    // densities are converted to the Cartesian engine space and every
    // engine-built operator is sandwiched back with `c2s`. Both maps are the
    // same linear transform, so the bridge is mathematically exact; cost is
    // O(n^3) GEMMs per iteration, noise next to the Fock build. The stored-ERI
    // path needs no bridge — pack.ERI is already spherical.
    const bool sph = integrals.spherical();
    const auto& Csp = integrals.c2s;
    const auto to_eng = [&](const RealMatrix& M) {
        return sph ? RealMatrix(xt::linalg::dot(
                         Csp, RealMatrix(xt::linalg::dot(
                                  M, xt::transpose(Csp)))))
                   : M;
    };
    const auto to_ao = [&](auto& M) {
        if (sph)
            M = xt::linalg::dot(xt::transpose(Csp),
                                RealMatrix(xt::linalg::dot(M, Csp)));
    };
    RealMatrix Pa_scratch, Pb_scratch;
    if (sph) {
        Pa_scratch = to_eng(P_alpha);
        if (!restricted)
            Pb_scratch = to_eng(P_beta);
    }
    const RealMatrix& Pa_eng = sph ? Pa_scratch : P_alpha;
    const RealMatrix& Pb_eng =
        sph ? (restricted ? Pa_scratch : Pb_scratch) : P_beta;

    // RI (density-fitting) routing: when `ri_ != RI::None` J (and for
    // RI::JK also K) come from the DensityFitting workspace in the
    // Cartesian engine space and bridge back through `to_ao` exactly like
    // the direct builders. The `RI::None` branches below are untouched —
    // bit-identical to the pre-RI path. K normalization convention:
    // `k_fitted*` matches `ShellBlockPairList::k` (K[P], no 0.5), so the
    // Fock assembly lines mirror the exact-path arithmetic one-to-one.
    if (restricted && is_hf && ri_ != RI::None) {
        // RHF with RI-J / RI-JK: F = Hcore + J − 0.5·K on the total
        // density. RI::J fits only J (exchange stays exact via the
        // shell-block K builder); RI::JK fits both.
        out.J = df_->j_fitted(Pa_eng);
        RealMatrix K;
        if (ri_ == RI::JK)
            K = df_->k_fitted_from_density(Pa_eng);
        else
            sp_block_.k(Pa_eng, K);
        RealMatrix G = out.J - double(0.5) * K;
        to_ao(out.J);
        to_ao(G);
        out.F_alpha = Hcore + G;
        out.F_beta = out.F_alpha;
    } else if (restricted && is_hf) {
        // RHF: PySCF approach — store ERI once, then BLAS contraction.
        // Falls back to direct SCF when ERI is not stored.
        RealMatrix G;
        if (integrals.ERI.size() > 0) {
            integrals.coulomb_exchange_fused(P_alpha, G);
        } else {
            sp_block_.g_rhf(Pa_eng, G);
            to_ao(G);
        }
        out.F_alpha = Hcore + G;
        out.F_beta = out.F_alpha;
    } else if (!restricted && is_hf && ri_ != RI::None) {
        // UHF with RI-J / RI-JK: one fitted J from the total density,
        // per-spin exchange (fitted for RI::JK, exact K[P_σ] otherwise —
        // the shell-block K builder is linear in its density argument).
        RealMatrix P_total = Pa_eng + Pb_eng;
        out.J = df_->j_fitted(P_total);
        RealMatrix K_alpha, K_beta;
        if (ri_ == RI::JK) {
            K_alpha = df_->k_fitted_from_density(Pa_eng);
            K_beta = df_->k_fitted_from_density(Pb_eng);
        } else {
            sp_block_.k(Pa_eng, K_alpha);
            sp_block_.k(Pb_eng, K_beta);
        }
        to_ao(out.J);
        to_ao(K_alpha);
        to_ao(K_beta);
        out.F_alpha = Hcore + out.J - K_alpha;
        out.F_beta = Hcore + out.J - K_beta;
    } else if (!restricted && is_hf) {
        RealMatrix P_total = Pa_eng + Pb_eng;
        RealMatrix K_alpha, K_beta;
        sp_.jk_uhf(P_total, Pa_eng, Pb_eng, out.J, K_alpha, K_beta);
        to_ao(out.J);
        to_ao(K_alpha);
        to_ao(K_beta);
        out.F_alpha = Hcore + out.J - K_alpha;
        out.F_beta = Hcore + out.J - K_beta;
    } else {
        // KS branch (restricted or unrestricted).
        // Restricted convention: P_alpha is the FULL double-occupied density
        //                        (P_beta aliased). XC takes spin-resolved halves.
        // Unrestricted convention: P_alpha and P_beta each carry one spin.
        RealMatrix P_total_J, P_alpha_xc, P_beta_xc;
        if (restricted) {
            P_total_J = Pa_eng;
            P_alpha_xc = double(0.5) * Pa_eng;
            P_beta_xc = P_alpha_xc;
        } else {
            P_total_J = Pa_eng + Pb_eng;
            P_alpha_xc = Pa_eng;
            P_beta_xc = Pb_eng;
        }

        // Hybrid exact-exchange admixture. alpha_hf is taken from the
        // functional unless the regression override is set (>= 0). It is
        // applied at exactly ONE site below: the bare K matrix is scaled
        // by alpha_hf once, and both the Fock term (-alpha_hf*K) and the
        // energy term (alpha_hf*E_K) consume that single scaled matrix.
        const double alpha_hf = (hybrid_alpha_override >= double(0))
                                    ? hybrid_alpha_override
                                    : hybrid_exchange_fraction(xc_type);
        const bool is_hybrid =
            (alpha_hf > double(0)) || (hybrid_alpha_override >= double(0));

        auto xc = evaluate_xc_batched(grid_, integrals.shells,
                                      integrals.n_ao_cart(), P_alpha_xc,
                                      P_beta_xc, xc_type);
        to_ao(xc.Vxc_alpha);
        to_ao(xc.Vxc_beta);

        if (is_hybrid && restricted) {
            // Restricted KS global hybrid.
            //
            // Convention pin: ShellBlockPairList::k returns the full
            // exchange matrix K with the RHF normalization
            // F = Hcore + J - 0.5*K. The "bare K" (the matrix that, as
            // F = Hcore + J - K, reproduces pure HF) is K_bare = 0.5*K.
            // alpha_hf is applied at exactly ONE site — the scalar
            // 0.5*alpha_hf multiplying K — shared by the Fock term and the
            // energy term (E_K = -0.25*Tr[P*K]).
            const double half_alpha = double(0.5) * alpha_hf;

            if (ri_ != RI::None) {
                // Restricted KS hybrid with RI: fitted J; exchange fitted
                // for RI::JK, exact shell-block K for RI::J. Same single
                // application site of alpha_hf as the exact branch.
                out.J = df_->j_fitted(P_total_J);
                RealMatrix K;
                if (ri_ == RI::JK)
                    K = df_->k_fitted_from_density(P_total_J);
                else
                    sp_block_.k(P_total_J, K);
                const double E_K = double(-0.25) * xt::sum(P_total_J * K)();
                out.E_exact_exchange = alpha_hf * E_K;
                RealMatrix F2e = out.J - half_alpha * K;
                to_ao(out.J);
                to_ao(F2e);
                out.F_alpha = Hcore + F2e + xc.Vxc_alpha;
                out.F_beta = out.F_alpha;
            } else if (half_alpha == double(0)) {
                // alpha_hf == 0 regression path (ac-004). J is built with
                // the SAME ShellPairList::j call as the pure-KS branch and
                // the Fock / energy arithmetic is the byte-for-byte
                // identical expression — so the hybrid path collapses
                // exactly onto the pre-existing pure-DFT result. K is still
                // built (the hybrid branch is genuinely exercised) but the
                // 0.5*0*K term is omitted to avoid any -0.0/+0.0 artifact.
                sp_.j(P_total_J, out.J);
                to_ao(out.J);
                RealMatrix K;
                sp_block_.k(P_total_J, K);
                out.F_alpha = Hcore + out.J + xc.Vxc_alpha;
                out.F_beta = out.F_alpha;
                out.E_exact_exchange = double(0);
            } else {
                // Production B3LYP / PBE0. Both J and K come from the fast
                // shell-block builders: ShellBlockPairList::g_rhf gives
                // G = J - 0.5*K, ShellBlockPairList::k gives K, hence
                // J = G + 0.5*K. The Fock matrix is then
                //   F = Hcore + J - 0.5*alpha_hf*K + Vxc
                //     = Hcore + G + 0.5*(1 - alpha_hf)*K + Vxc.
                RealMatrix G, K;
                sp_block_.g_rhf(P_total_J, G);
                sp_block_.k(P_total_J, K);
                // E_K = -0.5*Tr[P*K_bare] = -0.25*Tr[P*K]; traces are
                // basis-invariant, so compute in engine space BEFORE the
                // spherical bridge. The hybrid contribution is alpha_hf*E_K.
                const double E_K = double(-0.25) * xt::sum(P_total_J * K)();
                out.E_exact_exchange = alpha_hf * E_K;
                to_ao(G);
                to_ao(K);
                out.J = G + double(0.5) * K; // Coulomb-only, for E_J
                out.F_alpha = Hcore + G
                              + double(0.5) * (double(1) - alpha_hf) * K
                              + xc.Vxc_alpha;
                out.F_beta = out.F_alpha;
            }
        } else if (is_hybrid) {
            // Unrestricted KS hybrid (open-shell). Out of the closed-shell
            // deliverable's scope, but kept correct: per-spin exchange via
            // the AO-level UHF builder. F^sigma = Hcore + J - alpha_hf*K^sigma.
            RealMatrix K_alpha, K_beta;
            if (ri_ != RI::None) {
                out.J = df_->j_fitted(P_total_J);
                if (ri_ == RI::JK) {
                    K_alpha = df_->k_fitted_from_density(P_alpha_xc);
                    K_beta = df_->k_fitted_from_density(P_beta_xc);
                } else {
                    sp_block_.k(P_alpha_xc, K_alpha);
                    sp_block_.k(P_beta_xc, K_beta);
                }
            } else {
                sp_.jk_uhf(P_total_J, P_alpha_xc, P_beta_xc, out.J, K_alpha,
                           K_beta);
            }
            const double E_K = double(-0.5)
                               * (xt::sum(P_alpha_xc * K_alpha)()
                                  + xt::sum(P_beta_xc * K_beta)());
            out.E_exact_exchange = alpha_hf * E_K;
            to_ao(out.J);
            to_ao(K_alpha);
            to_ao(K_beta);
            out.F_alpha = Hcore + out.J - alpha_hf * K_alpha + xc.Vxc_alpha;
            out.F_beta = Hcore + out.J - alpha_hf * K_beta + xc.Vxc_beta;
        } else {
            // Pure KS: Coulomb only, no exchange matrix. Byte-for-byte
            // identical to the pre-hybrid path when `ri_ == RI::None`.
            if (ri_ != RI::None)
                out.J = df_->j_fitted(P_total_J);
            else
                sp_.j(P_total_J, out.J);
            to_ao(out.J);
            out.F_alpha = Hcore + out.J + xc.Vxc_alpha;
            out.F_beta = Hcore + out.J + xc.Vxc_beta;
        }

        out.E_xc_grid = xc.E_xc;
        out.int_n_alpha = xc.n_alpha_integrated;
        out.int_n_beta = xc.n_beta_integrated;
    }

    return out;
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

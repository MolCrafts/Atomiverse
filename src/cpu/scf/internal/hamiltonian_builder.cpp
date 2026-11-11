/// @file cpu/scf/internal/hamiltonian_builder.cpp
/// @brief Method-dispatched Fock build (RHF / UHF / RKS / UKS).

#include "atomiverse/cpu/scf/internal/hamiltonian_builder.h"

#include <optional>
#include <stdexcept>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/reducers/xreducer.hpp>

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/xc_grid_engine.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

void HamiltonianBuilder::prepare(const internal::IntegralView& integrals,
                                 bool is_hf, GridLevel grid_level, int n_radial,
                                 int n_angular, const MolecularGrid* fixed_grid,
                                 RI ri, AuxBasis aux_basis, Coulomb coulomb,
                                 int cfmm_order, double cfmm_threshold,
                                 double max_memory_mb, bool mpi) {
    // New geometry: invalidate the AO-on-grid cache so the first KS step()
    // rebuilds Phi/dPhi against the grid built below.
    xc_basis_.built = false;
    memory_mb_ = max_memory_mb;
    mpi_ = mpi;
    exact_ = ExactJK{};
    exact_lr_ = ExactJK{};

    coulomb_ = coulomb;
    // Coulomb::RI with ri==None is treated as RI-J (member-driven seam).
    if (coulomb_ == Coulomb::RI && ri == RI::None)
        ri = RI::J;
    if (coulomb_ == Coulomb::CFMM && ri != RI::None)
        throw std::runtime_error(
            "HamiltonianBuilder: Coulomb::CFMM is mutually exclusive with "
            "RI::J/JK — pick one J route (CFMM multipole or density fitting).");

    ri_ = ri;
    // Every exact Coulomb build (Direct and the CFMM near field) uses the one
    // exact engine. Exchange-only use under RI::J is prepared on first need.
    if (ri_ == RI::None)
        exact_.prepare(integrals, memory_mb_, mpi_);
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

    if (coulomb_ == Coulomb::CFMM) {
        if (integrals.shells.empty())
            throw std::runtime_error(
                "HamiltonianBuilder: Coulomb::CFMM requested but the integral "
                "pack carries no shell workspace.");
        const auto t_cfmm = LogClock::now();
        cfmm_ = std::make_unique<CfmmJ>(integrals.shells, cfmm_order,
                                        cfmm_threshold);
        cfmm_->prepare();
        ATV_INFO("[scf] CFMM-J workspace built n_ao={} n_dist={} order={} "
                 "thr={:.1e} ms={:.1f}",
                 integrals.shells.size(), cfmm_->n_distributions(), cfmm_order,
                 cfmm_threshold, elapsed_ms(t_cfmm));
    } else {
        cfmm_.reset();
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
            grid_ = MolecularGrid(integrals.atoms.data(),
                                  static_cast<int>(integrals.atoms.size()),
                                  grid_level, n_radial, n_angular);
            ATV_INFO("[scf] grid build done points={} ms={:.1f}",
                     grid_.points.size(), elapsed_ms(t_grid));
        }
    }
}

const ExactJK&
HamiltonianBuilder::exact_engine_(const internal::IntegralView& integrals,
                                  double omega) const {
    ExactJK& engine = omega > 0 ? exact_lr_ : exact_;
    if (!engine.prepared())
        engine.prepare(integrals, memory_mb_, mpi_, omega);
    return engine;
}

FockResult HamiltonianBuilder::step(
    const internal::IntegralView& integrals, bool restricted, bool is_hf,
    xc::XCType xc_type, double hybrid_alpha_override, const RealMatrix& P_alpha,
    const RealMatrix& P_beta,
    const ATV::electronic::XcFunctional* functional) const {
    FockResult out;
    const RealMatrix& Hcore = integrals.Hcore;

    // Spherical-basis bridge: density fitting, CFMM and the XC grid work in
    // the Cartesian engine space (they walk pack.shells). Their densities are
    // converted to engine space and their operators sandwiched back with
    // `c2s`; the exact J/K engine takes and returns working-basis matrices.
    const bool sph = integrals.spherical();
    const auto& Csp = integrals.c2s;
    const auto to_eng = [&](const RealMatrix& M) {
        return sph ? RealMatrix(xt::linalg::dot(
                         Csp,
                         RealMatrix(xt::linalg::dot(M, xt::transpose(Csp)))))
                   : M;
    };
    const auto to_ao = [&](RealMatrix M) {
        return sph ? RealMatrix(
                         xt::linalg::dot(xt::transpose(Csp),
                                         RealMatrix(xt::linalg::dot(M, Csp))))
                   : M;
    };

    // Restricted convention: P_alpha is the FULL doubly occupied density
    // (P_beta aliased). Unrestricted: P_alpha and P_beta each carry one spin.
    const RealMatrix P_total =
        restricted ? P_alpha : RealMatrix(P_alpha + P_beta);

    // Exact-exchange weights. HF: unit exchange. KS: alpha_hf from the
    // functional unless the regression override is set (>= 0), plus the
    // erf-attenuated beta*K[erf(mu r)/r] of a range-separated hybrid.
    double alpha_hf = 1.0;
    xc::RshExchangeParams rsh;
    if (!is_hf) {
        rsh = xc::rsh_exchange_params(xc_type);
        alpha_hf = (hybrid_alpha_override >= double(0)) ? hybrid_alpha_override
                                                        : rsh.alpha;
    }
    const bool want_k = alpha_hf > double(0);
    const bool exact_j = ri_ == RI::None;
    const bool exact_k = want_k && ri_ != RI::JK;

    // ── One exact pass: J of the total density and every scaled exchange.
    // Restricted: K_s = (alpha/2) K[P_total] so F = H + J - K_s and
    // E_K = -1/2 Tr[P_total K_s]. Unrestricted: K_s = alpha K[P_sigma].
    RealMatrix J, K_alpha, K_beta;
    std::vector<JKRequest> requests;
    if (restricted) {
        if (exact_j || exact_k)
            requests.push_back({&P_total, exact_j ? &J : nullptr,
                                exact_k ? &K_alpha : nullptr,
                                double(0.5) * alpha_hf, 0});
    } else {
        if (exact_j)
            requests.push_back({&P_total, &J, nullptr, 1.0, 0});
        if (exact_k) {
            requests.push_back({&P_alpha, nullptr, &K_alpha, alpha_hf, 1});
            requests.push_back({&P_beta, nullptr, &K_beta, alpha_hf, 2});
        }
    }
    if (!requests.empty())
        exact_engine_(integrals, 0.0).step(requests);

    // ── Fitted / multipole replacements (engine space, bridged back).
    if (ri_ != RI::None || coulomb_ == Coulomb::CFMM || (want_k && !exact_k)) {
        const RealMatrix Pt_eng = to_eng(P_total);
        if (ri_ != RI::None)
            J = to_ao(df_->j_fitted(Pt_eng));
        if (coulomb_ == Coulomb::CFMM) {
            // Exact near field above; replace the far pairs' ERI scatter by
            // their multipole interaction (no-op when the far list is empty).
            RealMatrix far = xt::zeros<double>(Pt_eng.shape());
            cfmm_->apply_far_correction(Pt_eng, far);
            J += to_ao(far);
        }
        if (want_k && !exact_k) {
            if (restricted) {
                K_alpha = double(0.5) * alpha_hf
                          * to_ao(df_->k_fitted_from_density(Pt_eng));
            } else {
                K_alpha = alpha_hf
                          * to_ao(df_->k_fitted_from_density(to_eng(P_alpha)));
                K_beta = alpha_hf
                         * to_ao(df_->k_fitted_from_density(to_eng(P_beta)));
            }
        }
    }

    // ── Range-separated exchange: beta*K[erf(mu r)/r], same conventions.
    RealMatrix Klr_alpha, Klr_beta;
    if (rsh.beta != double(0)) {
        std::vector<JKRequest> lr;
        if (restricted) {
            lr.push_back(
                {&P_total, nullptr, &Klr_alpha, double(0.5) * rsh.beta, 0});
        } else {
            lr.push_back({&P_alpha, nullptr, &Klr_alpha, rsh.beta, 0});
            lr.push_back({&P_beta, nullptr, &Klr_beta, rsh.beta, 1});
        }
        exact_engine_(integrals, rsh.mu).step(lr);
    }

    out.J = J;
    out.F_alpha = Hcore + J;
    if (want_k) {
        if (restricted) {
            out.F_alpha -= K_alpha;
            out.E_exact_exchange = double(-0.5) * xt::sum(P_total * K_alpha)();
        } else {
            out.F_beta = out.F_alpha - K_beta;
            out.F_alpha -= K_alpha;
            out.E_exact_exchange =
                double(-0.5)
                * (xt::sum(P_alpha * K_alpha)() + xt::sum(P_beta * K_beta)());
        }
    } else if (!restricted) {
        out.F_beta = out.F_alpha;
    }
    if (rsh.beta != double(0)) {
        if (restricted) {
            out.F_alpha -= Klr_alpha;
            out.E_exact_exchange -=
                double(0.5) * xt::sum(P_total * Klr_alpha)();
        } else {
            out.F_alpha -= Klr_alpha;
            out.F_beta -= Klr_beta;
            out.E_exact_exchange -= double(0.5)
                                    * (xt::sum(P_alpha * Klr_alpha)()
                                       + xt::sum(P_beta * Klr_beta)());
        }
    }
    if (is_hf) {
        // HF carries its exchange inside F; the energy reads it from F.
        out.E_exact_exchange = 0;
        if (restricted)
            out.F_beta = out.F_alpha;
        return out;
    }

    // ── Kohn-Sham exchange-correlation on the cached AO-on-grid basis.
    // Geometry-fixed AO-on-grid cache: build Phi/dPhi once per geometry
    // (first KS iteration, or when the LDA/GGA flavor changes) and reuse it
    // across all SCF iterations.
    std::optional<xc::BuiltinXc> builtin;
    if (!functional) {
        builtin.emplace(xc_type);
        functional = &*builtin;
    }
    const bool gga =
        (static_cast<unsigned>(functional->ingredients()) & 2u) != 0;
    if (!xc_basis_.built || xc_basis_.gga != gga)
        xc_basis_.build(grid_, integrals.shells, integrals.n_ao_cart(), gga);
    const RealMatrix Pa_xc = restricted
                                 ? RealMatrix(double(0.5) * to_eng(P_alpha))
                                 : to_eng(P_alpha);
    const RealMatrix Pb_xc = restricted ? Pa_xc : to_eng(P_beta);
    auto xc =
        evaluate_xc_batched(xc_basis_, grid_, Pa_xc, Pb_xc, *functional,
                            restricted ? ATV::electronic::XcSpin::Restricted
                                       : ATV::electronic::XcSpin::Collinear);
    if (restricted) {
        out.F_alpha += to_ao(xc.Vxc_alpha);
        out.F_beta = out.F_alpha;
    } else {
        out.F_alpha += to_ao(xc.Vxc_alpha);
        out.F_beta += to_ao(xc.Vxc_beta);
    }
    out.E_xc_grid = xc.E_xc;
    out.int_n_alpha = xc.n_alpha_integrated;
    out.int_n_beta = xc.n_beta_integrated;
    return out;
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

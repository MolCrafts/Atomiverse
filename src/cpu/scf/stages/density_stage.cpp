/// @file cpu/scf/stages/density_stage.cpp
/// @brief DensityStage implementation — bit-identical to `scf_solver.cpp`
///        steps 4-6 (density build → damping → four-branch energy), plus
///        optional MOM/IMOM occupation selection (Gilbert–Besley–Gill).

#include "atomiverse/cpu/scf/stages/density_stage.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/reducers/xreducer.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/linalg/density.h"
#include "atomiverse/cpu/scf/internal/mean_field_energy.h"

namespace ATV {
namespace cpu {
namespace scf {

DensityStage::DensityStage(
    bool restricted, bool is_hf, double damping, const RealMatrix& C_alpha,
    const RealMatrix& C_beta, const RealMatrix& F_alpha,
    const RealMatrix& F_beta, const RealMatrix& Hcore, const RealMatrix& J,
    const int& n_alpha, const int& n_beta, const int& n_basis,
    const bool& diis_active, const double& e_xc_grid,
    const double& e_exact_exchange, const double& v_nuc, RealMatrix& P_alpha,
    RealMatrix& P_beta, double& e_total, double& e_one, double& e_coulomb,
    bool generalized, const double& e_solvent, const double& e_pcm_fock_half,
    const RealMatrix& S, const MomStyle& mom_style, RealMatrix& mom_ref_C_alpha,
    RealMatrix& mom_ref_C_beta)
    : restricted_(restricted), is_hf_(is_hf), generalized_(generalized),
      damping_(damping), C_alpha_(C_alpha), C_beta_(C_beta), F_alpha_(F_alpha),
      F_beta_(F_beta), Hcore_(Hcore), J_(J), S_(S), n_alpha_(n_alpha),
      n_beta_(n_beta), n_basis_(n_basis), diis_active_(diis_active),
      e_xc_grid_(e_xc_grid), e_exact_exchange_(e_exact_exchange), v_nuc_(v_nuc),
      e_solvent_(e_solvent), e_pcm_fock_half_(e_pcm_fock_half),
      mom_style_(mom_style), P_alpha_(P_alpha), P_beta_(P_beta),
      mom_ref_C_alpha_(mom_ref_C_alpha), mom_ref_C_beta_(mom_ref_C_beta),
      e_total_(e_total), e_one_(e_one), e_coulomb_(e_coulomb) {}

std::vector<DefaultBinding> DensityStage::default_bindings() const {
    return {ATV::make_binding(ATV::scf::Phase::density, 0)};
}

void DensityStage::build(const cpu::BuildContext& ctx) {
    (void)ctx;
    density_.prepare(n_basis_);
}

RealMatrix DensityStage::mom_density_(const RealMatrix& C, RealMatrix& C_ref,
                                      const RealMatrix& S, int n_occ,
                                      int n_basis, double factor,
                                      MomStyle style) {
    if (n_occ <= 0)
        return xt::zeros<double>({n_basis, n_basis});
    const int n_mo = static_cast<int>(C.shape()[1]);
    if (n_occ > n_mo)
        throw std::invalid_argument("DensityStage MOM: n_occ="
                                    + std::to_string(n_occ)
                                    + " exceeds n_mo=" + std::to_string(n_mo));
    if (static_cast<int>(C_ref.shape()[0]) != n_basis
        || static_cast<int>(C_ref.shape()[1]) != n_occ)
        throw std::invalid_argument("DensityStage MOM: reference C_ref is "
                                    + std::to_string(C_ref.shape()[0]) + "x"
                                    + std::to_string(C_ref.shape()[1])
                                    + " but expected " + std::to_string(n_basis)
                                    + "x" + std::to_string(n_occ));
    if (static_cast<int>(S.shape()[0]) != n_basis
        || static_cast<int>(S.shape()[1]) != n_basis)
        throw std::invalid_argument(
            "DensityStage MOM: overlap S dimension mismatch");

    // s_ij = (C_ref^T S C)_{ij}; score_j = Σ_i s_ij²  (PySCF mom_occ_).
    const RealMatrix SC = xt::linalg::dot(S, C);
    const RealMatrix proj = xt::linalg::dot(xt::transpose(C_ref), SC);
    std::vector<double> score(static_cast<std::size_t>(n_mo), 0.0);
    for (int j = 0; j < n_mo; ++j) {
        double s = 0.0;
        for (int i = 0; i < n_occ; ++i)
            s += proj(i, j) * proj(i, j);
        score[static_cast<std::size_t>(j)] = s;
    }
    std::vector<int> order(static_cast<std::size_t>(n_mo));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return score[static_cast<std::size_t>(a)]
               > score[static_cast<std::size_t>(b)];
    });

    RealMatrix C_occ = xt::zeros<double>({n_basis, n_occ});
    for (int k = 0; k < n_occ; ++k)
        xt::view(C_occ, xt::all(), k) =
            xt::view(C, xt::all(), order[static_cast<std::size_t>(k)]);

    // MOM updates the reference; IMOM keeps the seed fixed.
    if (style == MomStyle::MOM)
        C_ref = C_occ;

    return linalg::build_density_matrix(C_occ, n_occ, n_basis, factor);
}

void DensityStage::launch(const cpu::LaunchContext& ctx) {
    if (!ctx.at(ATV::scf::Phase::density))
        return;

    if (generalized_) {
        // GHF spinor path: single channel at dimension 2n. New density
        // P = C_occ C_occᵀ over the lowest n_alpha (= N) spinors, damping
        // toward the retained OLD spinor density, and the single-channel HF
        // energy E = ½Tr[P_old(H + F)] + E_nn on the OLD density (the Fock
        // wired in this iteration was built from it) — the unrestricted-HF
        // convention at spinor dimension.
        const RealMatrix P_old = P_alpha_; // OLD spinor density
        const int dim = 2 * n_basis_;
        RealMatrix P_new = density_.step(C_alpha_, n_alpha_, dim);
        if (!diis_active_)
            P_new = (double(1) - damping_) * P_new + damping_ * P_old;

        e_one_ = xt::sum(Hcore_ * P_old)();
        e_coulomb_ = 0; // folded into ½Tr[P(H+F)] for HF (never split out)
        // Free-energy convention: peel ½ Tr[V_pcm P] out of ½ Tr[(H+F)P]
        // and replace with ΔG_el (both zero when solvent is off).
        e_total_ = double(0.5) * xt::sum((Hcore_ + F_alpha_) * P_old)() + v_nuc_
                   - e_pcm_fock_half_ + e_solvent_;
        P_alpha_ = std::move(P_new);
        P_beta_ = P_alpha_;
        return;
    }

    // Local mirrors of the shipping solver's names (scf_solver.cpp:293-348).
    // The wired P references are old-in/new-out: cache the OLD density so the
    // HF energy branches read it after the reference has been overwritten.
    const RealMatrix& Hcore_cm = Hcore_;
    const RealMatrix& F_alpha = F_alpha_;
    const RealMatrix& F_beta = F_beta_;
    const RealMatrix P_alpha = P_alpha_; // OLD α density
    const RealMatrix P_beta = P_beta_;   // OLD β density
    const int n = n_basis_;
    const double damping = damping_;
    const bool restricted = restricted_;
    const bool is_hf = is_hf_;
    const bool diis_active = diis_active_;
    const MomStyle mom = mom_style_;
    const bool use_mom = (mom != MomStyle::Off);

    // 4. Build new density (Aufbau or MOM/IMOM projection scores).
    RealMatrix P_alpha_new, P_beta_new;
    if (use_mom) {
        if (restricted) {
            P_alpha_new = mom_density_(C_alpha_, mom_ref_C_alpha_, S_, n_alpha_,
                                       n, double(2), mom);
            P_beta_new = P_alpha_new;
        } else {
            P_alpha_new = mom_density_(C_alpha_, mom_ref_C_alpha_, S_, n_alpha_,
                                       n, double(1), mom);
            P_beta_new = mom_density_(C_beta_, mom_ref_C_beta_, S_, n_beta_, n,
                                      double(1), mom);
        }
    } else if (restricted) {
        P_alpha_new = density_.step(C_alpha_, n_alpha_, n, double(2));
        P_beta_new = P_alpha_new;
    } else {
        P_alpha_new = density_.step(C_alpha_, n_alpha_, n);
        P_beta_new = density_.step(C_beta_, n_beta_, n);
    }

    // 5. Damping (PySCF convention: damping = fraction of OLD retained).
    RealMatrix P_alpha_next, P_beta_next;
    if (!diis_active) {
        P_alpha_next = (double(1) - damping) * P_alpha_new + damping * P_alpha;
        if (!restricted)
            P_beta_next = (double(1) - damping) * P_beta_new + damping * P_beta;
        else
            P_beta_next = P_alpha_next;
    } else {
        P_alpha_next = P_alpha_new;
        P_beta_next = restricted ? P_alpha_new : P_beta_new;
    }

    // 6. Energy evaluation — the four-branch functional, shared verbatim with
    // the fixed-density (Car-Parrinello) evaluation. The HF branches read the
    // density the Fock was built from (`P_alpha`/`P_beta`, OLD); the KS
    // branches read this iteration's output (`P_alpha_new`/`P_beta_new`).
    const internal::MeanFieldEnergy e = internal::mean_field_energy_at_density(
        restricted, is_hf, Hcore_cm, F_alpha, F_beta, J_, P_alpha, P_beta,
        P_alpha_new, P_beta_new, e_xc_grid_, e_exact_exchange_, v_nuc_,
        e_solvent_, e_pcm_fock_half_);

    // Publish: new density (old-in/new-out) and the branch energies.
    P_alpha_ = P_alpha_next;
    P_beta_ = P_beta_next;
    e_total_ = e.total;
    e_one_ = e.one_electron;
    e_coulomb_ = e.coulomb;
}

} // namespace scf
} // namespace cpu
} // namespace ATV

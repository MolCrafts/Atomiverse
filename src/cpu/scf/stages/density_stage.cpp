/// @file cpu/scf/stages/density_stage.cpp
/// @brief DensityStage implementation — bit-identical to `scf_solver.cpp`
///        steps 4-6 (density build → damping → four-branch energy).

#include "atomiverse/cpu/scf/stages/density_stage.h"

#include <xtensor/core/xmath.hpp>
#include <xtensor/reducers/xreducer.hpp>

#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cpu {
namespace scf {

DensityStage::DensityStage(bool restricted, bool is_hf, double damping,
                           const RealMatrix& C_alpha, const RealMatrix& C_beta,
                           const RealMatrix& F_alpha, const RealMatrix& F_beta,
                           const RealMatrix& Hcore, const RealMatrix& J,
                           const int& n_alpha, const int& n_beta,
                           const int& n_basis, const bool& diis_active,
                           const double& e_xc_grid,
                           const double& e_exact_exchange, const double& v_nuc,
                           RealMatrix& P_alpha, RealMatrix& P_beta,
                           double& e_total, double& e_one, double& e_coulomb,
                           bool generalized, const double& e_solvent,
                           const double& e_pcm_fock_half)
    : restricted_(restricted), is_hf_(is_hf), generalized_(generalized),
      damping_(damping), C_alpha_(C_alpha), C_beta_(C_beta),
      F_alpha_(F_alpha), F_beta_(F_beta), Hcore_(Hcore), J_(J),
      n_alpha_(n_alpha), n_beta_(n_beta), n_basis_(n_basis),
      diis_active_(diis_active), e_xc_grid_(e_xc_grid),
      e_exact_exchange_(e_exact_exchange), v_nuc_(v_nuc),
      e_solvent_(e_solvent), e_pcm_fock_half_(e_pcm_fock_half),
      P_alpha_(P_alpha), P_beta_(P_beta), e_total_(e_total), e_one_(e_one),
      e_coulomb_(e_coulomb) {
}

std::vector<DefaultBinding> DensityStage::default_bindings() const {
    return {ATV::make_binding(ATV::scf::Phase::density, 0)};
}

void DensityStage::build(const cpu::BuildContext& ctx) {
    (void)ctx;
    density_.prepare(n_basis_);
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
        e_total_ =
            double(0.5) * xt::sum((Hcore_ + F_alpha_) * P_old)() + v_nuc_
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

    // 4. Build new density.
    RealMatrix P_alpha_new, P_beta_new;
    if (restricted) {
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

    // 6. Energy evaluation.
    double E_total;
    double E_one = 0, E_J = 0;

    if (restricted && is_hf) {
        E_total =
            double(0.5) * xt::sum((Hcore_cm + F_alpha) * P_alpha)() + v_nuc_
            - e_pcm_fock_half_ + e_solvent_;
        E_one = xt::sum(Hcore_cm * P_alpha)();
    } else if (restricted) {
        E_one = xt::sum(Hcore_cm * P_alpha_new)();
        E_J = double(0.5) * xt::sum(J_ * P_alpha_new)();
        // e_exact_exchange = alpha_hf * E_K, nonzero only for a KS global
        // hybrid (B3LYP / PBE0); zero for pure KS.
        // KS energy does not use F, so add ΔG_el directly (no half-trace peel).
        E_total = E_one + E_J + e_exact_exchange_ + e_xc_grid_ + v_nuc_
                  + e_solvent_;
    } else if (is_hf) {
        E_total = double(0.5)
                      * (xt::sum((Hcore_cm + F_alpha) * P_alpha)()
                         + xt::sum((Hcore_cm + F_beta) * P_beta)())
                  + v_nuc_ - e_pcm_fock_half_ + e_solvent_;
        E_one = xt::sum(Hcore_cm * P_alpha)() + xt::sum(Hcore_cm * P_beta)();
    } else {
        RealMatrix P_total_new = P_alpha_new + P_beta_new;
        E_one = xt::sum(Hcore_cm * P_alpha_new)()
                + xt::sum(Hcore_cm * P_beta_new)();
        E_J = double(0.5) * xt::sum(J_ * P_total_new)();
        // e_exact_exchange nonzero only for a UKS global hybrid.
        E_total = E_one + E_J + e_exact_exchange_ + e_xc_grid_ + v_nuc_
                  + e_solvent_;
    }

    // Publish: new density (old-in/new-out) and the branch energies.
    P_alpha_ = P_alpha_next;
    P_beta_ = P_beta_next;
    e_total_ = E_total;
    e_one_ = E_one;
    e_coulomb_ = E_J;
}

} // namespace scf
} // namespace cpu
} // namespace ATV

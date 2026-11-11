/// @file cpu/scf/internal/mean_field_energy.cpp
/// @brief The four-branch mean-field energy functional — lifted verbatim from
///        `DensityStage::launch` step 6 (bit-identical; the stage keeps its
///        old-P / new-P call convention).

#include "atomiverse/cpu/scf/internal/mean_field_energy.h"

#include <xtensor/core/xmath.hpp>
#include <xtensor/reducers/xreducer.hpp>

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

MeanFieldEnergy mean_field_energy_at_density(
    bool restricted, bool is_hf, const RealMatrix& Hcore,
    const RealMatrix& F_alpha, const RealMatrix& F_beta, const RealMatrix& J,
    const RealMatrix& P_alpha_fock, const RealMatrix& P_beta_fock,
    const RealMatrix& P_alpha_out, const RealMatrix& P_beta_out,
    double e_xc_grid, double e_exact_exchange, double v_nuc, double e_solvent,
    double e_pcm_fock_half) {
    // Local mirrors of the shipping stage's names (density_stage.cpp step 6),
    // so the four expressions below stay diff-identical to the loop's.
    const RealMatrix& Hcore_cm = Hcore;
    const RealMatrix& P_alpha = P_alpha_fock;
    const RealMatrix& P_beta = P_beta_fock;
    const RealMatrix& P_alpha_new = P_alpha_out;
    const RealMatrix& P_beta_new = P_beta_out;

    double E_total;
    double E_one = 0, E_J = 0;

    if (restricted && is_hf) {
        E_total = double(0.5) * xt::sum((Hcore_cm + F_alpha) * P_alpha)()
                  + v_nuc - e_pcm_fock_half + e_solvent;
        E_one = xt::sum(Hcore_cm * P_alpha)();
    } else if (restricted) {
        E_one = xt::sum(Hcore_cm * P_alpha_new)();
        E_J = double(0.5) * xt::sum(J * P_alpha_new)();
        // e_exact_exchange = alpha_hf * E_K, nonzero only for a KS global
        // hybrid (B3LYP / PBE0); zero for pure KS.
        // KS energy does not use F, so add ΔG_el directly (no half-trace peel).
        E_total =
            E_one + E_J + e_exact_exchange + e_xc_grid + v_nuc + e_solvent;
    } else if (is_hf) {
        E_total = double(0.5)
                      * (xt::sum((Hcore_cm + F_alpha) * P_alpha)()
                         + xt::sum((Hcore_cm + F_beta) * P_beta)())
                  + v_nuc - e_pcm_fock_half + e_solvent;
        E_one = xt::sum(Hcore_cm * P_alpha)() + xt::sum(Hcore_cm * P_beta)();
    } else {
        RealMatrix P_total_new = P_alpha_new + P_beta_new;
        E_one = xt::sum(Hcore_cm * P_alpha_new)()
                + xt::sum(Hcore_cm * P_beta_new)();
        E_J = double(0.5) * xt::sum(J * P_total_new)();
        // e_exact_exchange nonzero only for a UKS global hybrid.
        E_total =
            E_one + E_J + e_exact_exchange + e_xc_grid + v_nuc + e_solvent;
    }

    return MeanFieldEnergy{E_total, E_one, E_J};
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

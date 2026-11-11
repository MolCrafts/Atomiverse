/// @file cpu/scf/stages/fock_stage.cpp
/// @brief FockStage — wraps internal::HamiltonianBuilder at scf::Phase::fock.

#include "atomiverse/cpu/scf/stages/fock_stage.h"

#include <stdexcept>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/scf/internal/density_builder.h"

namespace ATV {
namespace cpu {
namespace scf {

FockStage::FockStage(
    bool restricted, bool is_hf, xc::XCType xc_type, GridLevel grid_level,
    int n_radial, int n_angular, double hybrid_alpha_override,
    const MolecularGrid* fixed_grid, internal::IntegralView integrals,
    const RealMatrix& P_alpha, const RealMatrix& P_beta, RealMatrix& F_alpha,
    RealMatrix& F_beta, RealMatrix& J, double& e_xc_grid,
    double& e_exact_exchange, double& int_n_alpha, double& int_n_beta, RI ri,
    AuxBasis aux_basis, bool generalized, internal::Pcm* pcm, double& e_solvent,
    double& e_pcm_fock_half, Coulomb coulomb, int cfmm_order,
    double cfmm_threshold, const ATV::electronic::XcFunctional* functional,
    double max_memory_mb, bool mpi)
    : restricted_(restricted), is_hf_(is_hf), generalized_(generalized),
      max_memory_mb_(max_memory_mb), mpi_(mpi), xc_type_(xc_type),
      functional_(functional), grid_level_(grid_level), n_radial_(n_radial),
      n_angular_(n_angular), hybrid_alpha_override_(hybrid_alpha_override),
      fixed_grid_(fixed_grid), ri_(ri), aux_basis_(aux_basis),
      coulomb_(coulomb), cfmm_order_(cfmm_order),
      cfmm_threshold_(cfmm_threshold), integrals_(integrals), P_alpha_(P_alpha),
      P_beta_(P_beta), F_alpha_(F_alpha), F_beta_(F_beta), J_(J),
      e_xc_grid_(e_xc_grid), e_exact_exchange_(e_exact_exchange),
      int_n_alpha_(int_n_alpha), int_n_beta_(int_n_beta), pcm_(pcm),
      e_solvent_(e_solvent), e_pcm_fock_half_(e_pcm_fock_half) {
    if (generalized_ && ri_ != RI::None)
        throw std::runtime_error(
            "FockStage: RI (density fitting) is not wired for the "
            "generalized (GHF) spinor Fock build. Use ri = RI::None with GHF.");
    if (generalized_ && coulomb_ == Coulomb::CFMM)
        throw std::runtime_error(
            "FockStage: Coulomb::CFMM is not wired for the generalized "
            "(GHF) spinor Fock build — use coulomb = Coulomb::Direct.");
    if (generalized_ && pcm_ != nullptr)
        throw std::runtime_error(
            "FockStage: C-PCM is not wired for the generalized (GHF) "
            "spinor Fock build in this link — clear solvent_eps or use "
            "RHF/UHF/RKS/UKS.");
    if (coulomb_ == Coulomb::CFMM && ri_ != RI::None)
        throw std::runtime_error(
            "FockStage: Coulomb::CFMM is mutually exclusive with RI::J/JK "
            "— pick one J route.");
}

std::vector<DefaultBinding> FockStage::default_bindings() const {
    return {ATV::make_binding(ATV::scf::Phase::fock, 0)};
}

void FockStage::build(const cpu::BuildContext& ctx) {
    (void)ctx;
    // One-time allocation of shell-pair lists + (KS) the integration grid,
    // via HamiltonianBuilder's config-free entry.
    hamiltonian_.prepare(integrals_, is_hf_, grid_level_, n_radial_, n_angular_,
                         fixed_grid_, ri_, aux_basis_, coulomb_, cfmm_order_,
                         cfmm_threshold_, max_memory_mb_, mpi_);
}

void FockStage::launch(const cpu::LaunchContext& ctx) {
    if (!ctx.at(ATV::scf::Phase::fock))
        return;

    if (generalized_) {
        // GHF spinor Fock: F^{σσ'} = δ_{σσ'}(h + J[P^αα+P^ββ]) − K[P^{σσ'}].
        // The common exact engine contracts each working-basis spin block;
        // off-diagonal spin densities need not be symmetric.
        using DB = internal::DensityBuilder;
        const std::size_t n = static_cast<std::size_t>(integrals_.n_basis);
        const RealMatrix Paa = DB::spin_block(P_alpha_, 0, 0);
        const RealMatrix Pab = DB::spin_block(P_alpha_, 0, 1);
        const RealMatrix Pba = DB::spin_block(P_alpha_, 1, 0);
        const RealMatrix Pbb = DB::spin_block(P_alpha_, 1, 1);

        // One integral pass: J of the spin-summed density, K of all four
        // spin blocks.
        const RealMatrix Pn = Paa + Pbb;
        RealMatrix Jn, Kaa, Kbb, Kab, Kba;
        const internal::JKRequest requests[] = {{&Pn, &Jn, nullptr, 1.0, 0},
                                                {&Paa, nullptr, &Kaa, 1.0, 1},
                                                {&Pbb, nullptr, &Kbb, 1.0, 2},
                                                {&Pab, nullptr, &Kab, 1.0, 3},
                                                {&Pba, nullptr, &Kba, 1.0, 4}};
        hamiltonian_.exact_jk(requests);

        const RealMatrix& h = integrals_.Hcore;
        RealMatrix F = xt::zeros<double>({2 * n, 2 * n});
        xt::view(F, xt::range(0, n), xt::range(0, n)) = h + Jn - Kaa;
        xt::view(F, xt::range(0, n), xt::range(n, 2 * n)) = -Kab;
        xt::view(F, xt::range(n, 2 * n), xt::range(0, n)) = -Kba;
        xt::view(F, xt::range(n, 2 * n), xt::range(n, 2 * n)) = h + Jn - Kbb;

        F_alpha_ = std::move(F);
        F_beta_ = F_alpha_;
        physical_alpha_ = F_alpha_;
        physical_beta_ = F_beta_;
        J_ = DB::spinor_embed(Jn);
        e_xc_grid_ = 0;
        e_exact_exchange_ = 0;
        int_n_alpha_ = 0;
        int_n_beta_ = 0;
        return;
    }

    // Same core, same order, no extra arithmetic: the discrete step() forwards
    // to the shared HamiltonianBuilder core, so the outputs are bit-identical to
    // the shipping solver path when pcm_ is null.
    const internal::FockResult result = hamiltonian_.step(
        integrals_, restricted_, is_hf_, xc_type_, hybrid_alpha_override_,
        P_alpha_, P_beta_, functional_);

    F_alpha_ = result.F_alpha;
    F_beta_ = result.F_beta;
    J_ = result.J;
    e_xc_grid_ = result.E_xc_grid;
    e_exact_exchange_ = result.E_exact_exchange;
    int_n_alpha_ = result.int_n_alpha;
    int_n_beta_ = result.int_n_beta;

    e_solvent_ = 0;
    e_pcm_fock_half_ = 0;
    if (pcm_ != nullptr) {
        // Total density for the reaction field: closed-shell P_alpha is already
        // the total (occ=2) density; open-shell / UKS uses Pα+Pβ.
        RealMatrix P_tot = P_alpha_;
        if (!restricted_)
            P_tot = P_alpha_ + P_beta_;
        const auto pcm = pcm_->update(P_tot);
        F_alpha_ += pcm.V_pcm;
        if (!restricted_)
            F_beta_ += pcm.V_pcm;
        else
            F_beta_ = F_alpha_;
        e_solvent_ = pcm.delta_g_el;
        // ½ Tr[V_pcm P_total] is the piece already present in the HF
        // ½ Tr[(H+F)P] energy once V_pcm sits on F; DensityStage peels it off
        // and replaces it with the free-energy ΔG_el = ½ q·V.
        e_pcm_fock_half_ = 0.5 * xt::sum(pcm.V_pcm * P_tot)();
    }
    physical_alpha_ = F_alpha_;
    physical_beta_ = F_beta_;
}

} // namespace scf
} // namespace cpu
} // namespace ATV

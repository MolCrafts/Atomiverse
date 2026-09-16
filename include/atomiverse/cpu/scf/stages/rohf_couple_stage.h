/// @file cpu/scf/stages/rohf_couple_stage.h
/// @brief cpu::scf::RohfCoupleStage — the ROHF Roothaan effective-Fock
///        coupling, bound at `scf::Phase::mix` order −1 (i.e. strictly before
///        the MixStage's DIIS and therefore before `scf::Phase::diagonalize`).
///
/// ROHF reuses the unrestricted Pople–Nesbet Fock build (FockStage with
/// independent F_α/F_β) and then, once per iteration, this stage couples the
/// two spin Focks into ONE Roothaan effective Fock F_eff so a single
/// restricted-style diagonalization yields a single MO set (closed / open /
/// virtual three-space solution). State flows through NAMED constructor
/// references (const& reads, & writes) — no config struct, no workspace bag —
/// exactly the `DiagonalizeStage` wiring style.
///
/// **Convention (binding, user-ruled):** the PySCF canonical-ROHF projector
/// assembly (`pyscf/scf/rohf.py get_roothaan_fock`, PySCF 2.14.0), i.e. the
/// Guest–Saunders family of coupling coefficients:
///
///   | block      | closed | open | virtual |
///   |------------|--------|------|---------|
///   | closed     |  F_c   | F_β  |  F_c    |
///   | open       |  F_β   | F_c  |  F_α    |
///   | virtual    |  F_c   | F_α  |  F_c    |
///
/// with F_c ≡ (F_α + F_β)/2, assembled in the AO basis via the density
/// projectors P_c = P_β·S (closed), P_o = (P_α − P_β)·S (open),
/// P_v = 1 − P_α·S (virtual):
///
///   F = ½ P_cᵀ F_c P_c + ½ P_oᵀ F_c P_o + ½ P_vᵀ F_c P_v
///     + P_oᵀ F_β P_c + P_oᵀ F_α P_v + P_vᵀ F_c P_c,     F_eff = F + Fᵀ.
///
/// **PITFALL (do not "fix"):** the Roothaan / Guest–Saunders /
/// McWeeny–Diercksen coupling-coefficient families all converge to the SAME
/// total energy (the energy functional is the coefficient-free Pople–Nesbet
/// one) but produce DIFFERENT orbital energies — ε are convention-bound
/// quantities. This engine pins the PySCF canonical (Guest–Saunders)
/// convention; orbital energies are only ever compared under it.
///
/// Side products for the surrounding stages:
///   - the TRUE spin Focks are preserved into `F_alpha_uncoupled` /
///     `F_beta_uncoupled` so the density stage evaluates the Pople–Nesbet
///     energy functional E = ½Tr[P_α(H+F_α)] + ½Tr[P_β(H+F_β)] + V_nn
///     [Hartree] on the physical Focks, never on F_eff;
///   - the total density P_tot = P_α + P_β is published so the MixStage's
///     single-channel DIIS error is the PySCF-matching commutator
///     Xᵀ(F_eff·P_tot·S − S·P_tot·F_eff)X.
///
/// References: Roothaan, Rev. Mod. Phys. 32, 179 (1960),
/// DOI 10.1103/RevModPhys.32.179; Guest & Saunders, Mol. Phys. 28, 819
/// (1974); Tsuchimochi & Scuseria, J. Chem. Phys. 133, 141102 (2010),
/// DOI 10.1063/1.3503173 (constrained-UHF view of ROHF).
#pragma once

#include <string>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/generators/xbuilder.hpp>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/scf/dense_matrix.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief ROHF coupling stage: assemble the Roothaan effective Fock (PySCF
///        canonical / Guest–Saunders convention) into the α-Fock channel,
///        preserving the true spin Focks and the total density for the
///        energy and DIIS consumers downstream.
class RohfCoupleStage : public cpu::Component {
  public:
    /// @brief Wire the coupling stage to the method driver's iteration state.
    /// @param S Overlap matrix (dimensionless), n × n [read].
    /// @param P_alpha α density matrix (dimensionless AO density) [read].
    /// @param P_beta β density matrix (dimensionless AO density) [read].
    /// @param F_alpha α Fock channel [Hartree]: read as the true F_α, then
    ///        overwritten with F_eff (the diagonalize stage reads it) [write].
    /// @param F_beta β Fock channel [Hartree]: read as the true F_β, then
    ///        mirrored to F_eff [write].
    /// @param F_alpha_uncoupled [out] preserved true F_α [Hartree] — the
    ///        density stage's energy functional input.
    /// @param F_beta_uncoupled [out] preserved true F_β [Hartree].
    /// @param P_total [out] total density P_α + P_β — the MixStage's
    ///        single-channel DIIS commutator density.
    RohfCoupleStage(const RealMatrix& S, const RealMatrix& P_alpha,
                    const RealMatrix& P_beta, RealMatrix& F_alpha,
                    RealMatrix& F_beta, RealMatrix& F_alpha_uncoupled,
                    RealMatrix& F_beta_uncoupled, RealMatrix& P_total)
        : S_(S), P_alpha_(P_alpha), P_beta_(P_beta), F_alpha_(F_alpha),
          F_beta_(F_beta), F_alpha_uncoupled_(F_alpha_uncoupled),
          F_beta_uncoupled_(F_beta_uncoupled), P_total_(P_total) {}

    /// @brief Component name.
    /// @return "scf_rohf_couple".
    std::string name() const override { return "scf_rohf_couple"; }

    /// @brief Default binding: `scf::Phase::mix` @ −1 — strictly before the
    ///        MixStage (order 0), so DIIS extrapolates F_eff, and therefore
    ///        strictly before `scf::Phase::diagonalize`.
    /// @return Single binding at the mix phase, order −1.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::scf::Phase::mix, -1)};
    }

    /// @brief Assemble F_eff (PySCF `get_roothaan_fock` algebra, replicated
    ///        term by term) into the α/β Fock channels; preserve the true
    ///        spin Focks and publish P_tot.
    /// @param ctx Launch context; self-guarded by `ctx.at(scf::Phase::mix)`.
    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::scf::Phase::mix))
            return;

        // Preserve the physical spin Focks for the energy functional, and
        // the total density for the single-channel DIIS error.
        F_alpha_uncoupled_ = F_alpha_;
        F_beta_uncoupled_ = F_beta_;
        P_total_ = P_alpha_ + P_beta_;

        // PySCF get_roothaan_fock, term by term (real symmetric case):
        //   fc = (focka + fockb) * .5
        //   pc = dmb·s ; po = (dma−dmb)·s ; pv = 1 − dma·s
        //   fock  = .5·pcᵀ·fc·pc + .5·poᵀ·fc·po + .5·pvᵀ·fc·pv
        //         + poᵀ·fockb·pc + poᵀ·focka·pv + pvᵀ·fc·pc
        //   fock += fockᵀ
        const std::size_t n = S_.shape()[0];
        const RealMatrix Fc = 0.5 * (F_alpha_uncoupled_ + F_beta_uncoupled_);
        const RealMatrix Pc = xt::linalg::dot(P_beta_, S_);
        const RealMatrix Po =
            xt::linalg::dot(RealMatrix(P_alpha_ - P_beta_), S_);
        const RealMatrix Pv =
            RealMatrix(xt::eye<double>(n)) - xt::linalg::dot(P_alpha_, S_);

        auto sandwich = [](const RealMatrix& L, const RealMatrix& M,
                           const RealMatrix& R) -> RealMatrix {
            return xt::linalg::dot(xt::transpose(L), xt::linalg::dot(M, R));
        };

        RealMatrix F =
            0.5 * sandwich(Pc, Fc, Pc) + 0.5 * sandwich(Po, Fc, Po)
            + 0.5 * sandwich(Pv, Fc, Pv) + sandwich(Po, F_beta_uncoupled_, Pc)
            + sandwich(Po, F_alpha_uncoupled_, Pv) + sandwich(Pv, Fc, Pc);

        F_alpha_ = F + xt::transpose(F);
        F_beta_ = F_alpha_;
    }

  private:
    const RealMatrix& S_;       ///< Overlap (dimensionless, read).
    const RealMatrix& P_alpha_; ///< α density (read).
    const RealMatrix& P_beta_;  ///< β density (read).
    RealMatrix& F_alpha_;       ///< α Fock channel → F_eff [Hartree] (write).
    RealMatrix& F_beta_;        ///< β Fock channel → F_eff [Hartree] (write).
    RealMatrix& F_alpha_uncoupled_; ///< Preserved true F_α [Hartree] (write).
    RealMatrix& F_beta_uncoupled_;  ///< Preserved true F_β [Hartree] (write).
    RealMatrix& P_total_;           ///< P_α + P_β (write).
};

} // namespace scf
} // namespace cpu
} // namespace ATV

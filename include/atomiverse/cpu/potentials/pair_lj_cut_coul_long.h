/// @file cpu/potentials/pair_lj_cut_coul_long.h
/// @brief Fused LJ 12-6 + real-space coul/long — one neighbor walk (CPU).
///
/// Matches LAMMPS `pair_style lj/cut/coul/long`. Multi-type LJ (Lorentz–
/// Berthelot fill), Amber specials (LJ and Coulomb scales may differ),
/// tabulated erfc. Reciprocal space is a separate `KSpacePME` / `KSpacePPPM`.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/component/long_range_protocol.h"
#include "atomiverse/cpu/potentials/erfc_table.h"
#include "atomiverse/cpu/potentials/pair.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/potentials/special_scale.h"

namespace ATV {
namespace cpu {

/// @brief Fused `lj/cut/coul/long` with multi-type LJ and Amber specials.
class PairLJCutCoulLong : public Pair, public LongRangeParticipant {
  public:
    /// @brief The fused kernel reads the neighbour charge (`q_ptr[j]`).
    /// @param out Schema being accumulated at build().
    void declare_fields(Schema& out) const override {
        out.require(kAtom_charge, "PairLJCutCoulLong");
    }

    /// @param coeffs  Diagonal (and optional off-diagonal) LJ type pairs.
    /// @param alpha   Ewald α [1/Å] — must match the KSpace component.
    /// @param cutoff  Shared LJ/Coul real-space cutoff [Å] (LAMMPS default).
    PairLJCutCoulLong(std::vector<LJTypeCoeff> coeffs, double alpha,
                      double cutoff);

    std::string name() const override { return "lj/cut/coul/long"; }

    const LongRangeParticipant* as_long_range_participant() const override {
        return this;
    }
    LongRangeContract long_range_contract() const override {
        return {LongRangeRole::RealSpace, alpha_, cutoff_, AccReal(0)};
    }

    /// @brief Amber `special_bonds amber`: LJ (0,0,0.5), coul (0,0,5/6).
    PairLJCutCoulLong& set_special_bonds_amber() &;
    PairLJCutCoulLong set_special_bonds_amber() &&;

    /// @brief Explicit LJ / Coulomb special scales (LAMMPS special_bonds).
    PairLJCutCoulLong& set_special_bonds(double lj12, double lj13, double lj14,
                                         double c12, double c13, double c14) &;
    PairLJCutCoulLong set_special_bonds(double lj12, double lj13, double lj14,
                                        double c12, double c13, double c14) &&;

    PairLJCutCoulLong& set_compute_stress(bool on) &;
    PairLJCutCoulLong set_compute_stress(bool on) &&;

    /// @brief LAMMPS `pair_modify shift` — subtract V_LJ(rc) so the LJ energy
    ///        is zero at the cutoff. Default on. LAMMPS production decks with
    ///        `tail yes` run shift off.
    PairLJCutCoulLong& set_energy_shift(bool on) &;
    PairLJCutCoulLong set_energy_shift(bool on) &&;

    /// @brief LAMMPS `pair_modify tail yes` — analytic LJ tail energy and
    ///        pressure beyond the cutoff (homogeneous g(r)=1 approximation).
    ///        Global per-type counts are gathered on first evaluation; the
    ///        correction tracks the current box volume every step.
    PairLJCutCoulLong& set_tail_correction(bool on) &;
    PairLJCutCoulLong set_tail_correction(bool on) &&;

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;
    std::size_t memory_bytes() const override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    void finalize_type_table_();
    void rebuild_filtered_list(const Frame& frame, const md::PairList& pairs);

    double alpha_ = 0.0;
    double cutoff_sq_ = 0.0;

    int n_types_ = 0;
    std::vector<double> eps_tab_, sigma_tab_, sig6_tab_, shift_tab_;
    /// LAMMPS-style force/energy coeffs: lj1=48εσ¹², lj2=24εσ⁶, lj3=4εσ¹², lj4=4εσ⁶
    /// used as r6inv=1/r⁶: f = r2inv*(lj1*r6inv² − lj2*r6inv), e = lj3*r6inv² − lj4*r6inv − shift
    std::vector<double> lj1_tab_, lj2_tab_, lj3_tab_, lj4_tab_;
    std::vector<char> set_tab_;

    double lj12_ = 1.0, lj13_ = 1.0, lj14_ = 1.0;
    double c12_ = 1.0, c13_ = 1.0, c14_ = 1.0;
    bool any_special_ = false;

    bool energy_shift_ = true;
    bool tail_ = false;
    bool tail_ready_ = false;
    /// E_tail = etail_pref_/V; P_tail = ptail_pref_/V² (see compute()).
    double etail_pref_ = 0.0, ptail_pref_ = 0.0;
    bool compute_stress_ = true;

    md::SpecialScaleTable special_lj_;
    md::SpecialScaleTable special_coul_;
    md::ErfcTable erfc_;

    int n_threads_ = 1;
    std::vector<std::vector<double>> fx_thr_, fy_thr_, fz_thr_;

    /// Parallel to PairList::neighbors[] (no compact copy). scale==0 → skip.
    std::vector<float> scale_lj_;
    std::vector<float> scale_coul_;
    unsigned long long scale_version_ = 0;
};

} // namespace cpu
} // namespace ATV

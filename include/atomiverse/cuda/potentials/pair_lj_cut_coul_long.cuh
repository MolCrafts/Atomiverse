/// @file cuda/potentials/pair_lj_cut_coul_long.cuh
/// @brief Fused multi-type LJ and Ewald real space with Amber exceptions.
#pragma once

#include "atomiverse/cuda/component/long_range_protocol.cuh"
#include "atomiverse/cuda/potentials/pair.cuh"
#include <memory>
#include <vector>

namespace ATV::cuda {

/// @brief Single-GPU Lorentz–Berthelot LJ + screened Coulomb tile evaluator.
/// Bond-graph distances 1, 2 and 3 define exceptions; reciprocal PME is separate.
/// Exceptions use minimum images and are evaluated without a distance cutoff.
class PairLJCutCoulLong : public Pair, public LongRangeParticipant {
  public:
    /// @brief Construct diagonal LJ coefficients; pair coefficients are mixed.
    /// @param epsilon Well depths indexed by atom type [eV].
    /// @param sigma Collision diameters indexed by atom type [Å].
    /// @param alpha Ewald splitting parameter [1/Å], matching reciprocal space.
    /// @param cutoff LJ and Coulomb real-space cutoff [Å]. LJ is unshifted.
    PairLJCutCoulLong(std::vector<double> epsilon, std::vector<double> sigma,
                      double alpha, double cutoff);
    ~PairLJCutCoulLong() override;
    PairLJCutCoulLong(PairLJCutCoulLong&&) noexcept;
    PairLJCutCoulLong& operator=(PairLJCutCoulLong&&) noexcept;

    std::string name() const override { return "lj/cut/coul/long"; }
    const LongRangeParticipant* as_long_range_participant() const override {
        return this;
    }
    LongRangeContract long_range_contract() const override {
        return {LongRangeRole::RealSpace, AccReal(alpha_), AccReal(cutoff_), 0};
    }
    /// @brief Set Amber LJ (0,0,1/2) and Coulomb (0,0,5/6) exceptions.
    /// @return This evaluator.
    PairLJCutCoulLong& set_special_bonds_amber() &;
    /// @brief Set Amber exceptions on a temporary evaluator.
    /// @return The configured evaluator by value.
    PairLJCutCoulLong set_special_bonds_amber() &&;
    void set_stress_required(bool required) override { stress_ = required; }
    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    std::vector<double> epsilon_, sigma_;
    double alpha_;
    bool amber_ = false, stress_ = true;
    struct State;
    std::unique_ptr<State> state_;
};
} // namespace ATV::cuda

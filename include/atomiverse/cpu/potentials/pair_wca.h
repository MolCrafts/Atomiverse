/// @file cpu/potentials/pair_wca.h
/// @brief Weeks–Chandler–Andersen repulsive LJ (CPU).
#pragma once

#include <string>

#include "atomiverse/cpu/potentials/pair.h"

namespace ATV {
namespace cpu {

/// @brief WCA pair: LJ 12-6 cut and shifted at \(r_c = 2^{1/6}\sigma\).
///
/// Purely repulsive; \(U(r_c)=0\), \(F(r_c)=0\).
class PairWCA : public Pair {
  public:
    /// @param epsilon Well depth of the parent LJ [eV].
    /// @param sigma   Length scale [Å].
    PairWCA(double epsilon, double sigma);

    std::string name() const override { return "wca"; }

    const NeighborConsumer* as_neighbor_consumer() const final {
        return nullptr;
    }

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    double epsilon_ = 0;
    double sigma_ = 0;
    double cutoff_sq_ = 0;
    double v_shift_ = 0; ///< 4ε[(σ/rc)¹²−(σ/rc)⁶] so U(rc)=0
};

} // namespace cpu
} // namespace ATV

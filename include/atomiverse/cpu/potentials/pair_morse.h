/// @file cpu/potentials/pair_morse.h
/// @brief Morse-style pair potential — V = D(1 − e^{−α(r − r₀)})² (CPU).
///
/// This class is the **live demonstration** that adding a new pair style
/// under the post-2026-05-19 `cpu::Pair` hierarchy costs exactly two new
/// files: this header + `src/cpu/potentials/pair_morse.cpp`. No CMake
/// edit (the GLOB picks it up); no Driver edit (users plug it in with
/// `sim.set_evaluator(cpu::PairMorse{...})`); no central registry. Per the
/// `pe-potential-abstraction.md` ac-009 acceptance.
///
/// Implementation note: the demo uses an O(N²) all-pairs walk instead of
/// the project's neighbor-list infrastructure, to keep the example
/// readable. A production Morse pair style would mirror PairLJCut's NL
/// walk; that's a downstream optimisation, not part of the "one new
/// file" demonstration.
#pragma once

#include <string>

#include "atomiverse/cpu/potentials/pair.h"

namespace ATV {
namespace cpu {

/// @brief Morse pair potential: `V(r) = D · (1 − exp(−α(r − r₀)))²`.
///
/// `D`  — well depth [eV].
/// `α`  — width parameter [Å⁻¹].
/// `r₀` — equilibrium distance [Å].
/// Cutoff is shifted (potential goes to zero at `cutoff`, no force tail).
class PairMorse : public Pair {
  public:
    /// @param D       Well depth [eV].
    /// @param alpha   Width parameter [Å⁻¹].
    /// @param r0      Equilibrium bond length [Å].
    /// @param cutoff  Pair cutoff [Å].
    PairMorse(double D, double alpha, double r0, double cutoff);

    std::string name() const override { return "morse"; }

    /// @brief Demo uses O(N²) all-pairs; opt out of the neighbor-list gate
    /// by withdrawing the consumer view the `PairEvaluator` base installs.
    const NeighborConsumer* as_neighbor_consumer() const final {
        return nullptr;
    }

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    double D_, alpha_, r0_;
    double cutoff_sq_, v_shift_;
};

} // namespace cpu
} // namespace ATV

/// @file cuda/potentials/bond_harmonic.cuh
/// @brief LAMMPS-style BondHarmonic — V = K(r − r₀)² (CUDA).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "atomiverse/cuda/potentials/bond.cuh"

namespace ATV {
namespace cuda {

class BondHarmonic : public Bond {
  public:
    /// @param K  Spring constants per bond type [eV/Å²].
    /// @param r0 Equilibrium distances per bond type [Å].
    BondHarmonic(std::vector<double> K, std::vector<double> r0);
    ~BondHarmonic() override;
    BondHarmonic(BondHarmonic&&) noexcept;
    BondHarmonic& operator=(BondHarmonic&&) noexcept;

    std::string name() const override { return "harmonic_bond"; }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    std::vector<double> K_, r0_;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace cuda
} // namespace ATV

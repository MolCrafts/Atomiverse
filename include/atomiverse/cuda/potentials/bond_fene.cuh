/// @file cuda/potentials/bond_fene.cuh
/// @brief CUDA BondFene. The energy and force are `fene_term` in core/fene.h.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "atomiverse/cuda/potentials/bond.cuh"

namespace ATV {
namespace cuda {

class BondFene : public Bond {
  public:
    /// @param K  Spring constants per bond type [eV/Å²].
    /// @param R0 Maximum extensions per bond type [Å].
    BondFene(std::vector<double> K, std::vector<double> R0);
    ~BondFene() override;
    BondFene(BondFene&&) noexcept;
    BondFene& operator=(BondFene&&) noexcept;

    std::string name() const override { return "fene_bond"; }
    bool graph_capturable() const override { return true; }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    std::vector<double> K_, R0_;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace cuda
} // namespace ATV

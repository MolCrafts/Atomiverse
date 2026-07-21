/// @file cuda/potentials/dihedral_multi_harmonic.cuh
/// @brief LAMMPS-style DihedralMultiHarmonic — V = Σₙ Cₙ cosⁿ(φ) (CUDA).
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "atomiverse/cuda/potentials/dihedral.cuh"

namespace ATV {
namespace cuda {

class DihedralMultiHarmonic : public Dihedral {
  public:
    /// @param C Per-type arrays of 5 coefficients {C0..C4} [eV].
    DihedralMultiHarmonic(std::vector<std::array<double, 5>> C);
    ~DihedralMultiHarmonic() override;
    DihedralMultiHarmonic(DihedralMultiHarmonic&&) noexcept;
    DihedralMultiHarmonic& operator=(DihedralMultiHarmonic&&) noexcept;

    std::string name() const override { return "multi_harmonic_dihedral"; }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    std::vector<std::array<double, 5>> C_;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace cuda
} // namespace ATV

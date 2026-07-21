/// @file cuda/potentials/angle_harmonic.cuh
/// @brief LAMMPS-style AngleHarmonic — V = K(θ − θ₀)² (CUDA).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "atomiverse/cuda/potentials/angle.cuh"

namespace ATV {
namespace cuda {

class AngleHarmonic : public Angle {
  public:
    /// @param K      Spring constants per angle type [eV/rad²].
    /// @param theta0 Equilibrium angles per type [RADIANS]. Used directly in
    ///        `theta - theta0_[type]` (matches the CPU sibling + the PE
    ///        benchmark, which pass `109.5 * pi/180`); NOT converted from
    ///        degrees — see angle_harmonic.cu's note on the removed kDegToRad.
    AngleHarmonic(std::vector<double> K, std::vector<double> theta0);
    ~AngleHarmonic() override;
    AngleHarmonic(AngleHarmonic&&) noexcept;
    AngleHarmonic& operator=(AngleHarmonic&&) noexcept;

    std::string name() const override { return "harmonic_angle"; }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    std::vector<double> K_, theta0_;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace cuda
} // namespace ATV

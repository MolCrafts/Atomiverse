/// @file cuda/potentials/angle_cosine.cuh
/// @brief LAMMPS-style AngleCosine — V = K(1 + cos θ) (CUDA).
///
/// Port of LAMMPS `src/MOLECULE/angle_cosine.cpp` (Sandia, GPL; checkout
/// SHA e8c4c278a8). Same per-tuple law as `cpu::AngleCosine`
/// (`core/angle_cosine.h`). CUDA-free header.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "atomiverse/cuda/potentials/angle.cuh"

namespace ATV {
namespace cuda {

/// @brief Cosine angle style \(V = K(1+\cos\theta)\), per angle type.
///
/// One thread per angle of the flat device angle list; energy and the six
/// virial components are block-reduced in FP64 and added to
/// `SystemScalars::energy` / `stress` only on steps that request them.
/// Graph-capturable: the launch depends on build-time state and the step's
/// energy/virial requests only.
class AngleCosine : public Angle {
  public:
    /// @param K Angle stiffness per angle type [eV]. Non-empty, finite.
    /// @throws std::invalid_argument on an empty or non-finite K.
    explicit AngleCosine(std::vector<double> K);
    ~AngleCosine() override;
    AngleCosine(AngleCosine&&) noexcept;
    AngleCosine& operator=(AngleCosine&&) noexcept;

    std::string name() const override { return "cosine_angle"; }
    bool graph_capturable() const override { return true; }

    /// @brief Upload K and check every angle type against it (one D2H of the
    ///        angle types, at build only).
    /// @throws std::invalid_argument on an angle type outside the K table.
    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    std::vector<double> K_;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace cuda
} // namespace ATV

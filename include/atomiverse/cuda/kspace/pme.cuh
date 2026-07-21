/// @file cuda/kspace/pme.cuh
/// @brief Smooth Particle-Mesh Ewald reciprocal sum — CUDA (cuFFT).
///        Requires `ATV_BUILD_KSPACE`.
///
/// GPU mirror of `cpu::KSpacePME` (Essmann smooth PME): B-spline charge
/// spreading → cuFFT → influence-function multiply → inverse cuFFT → force
/// interpolation. Drop-in reciprocal replacement for `cuda::KSpaceEwald`;
/// shares α with `cuda::CoulLong`. Single-GPU in this revision.
#pragma once

#include <array>
#include <string>

#include "atomiverse/cuda/kspace/kspace.cuh"

namespace ATV {
namespace cuda {

/// @brief Smooth Particle-Mesh Ewald reciprocal electrostatics (CUDA).
class KSpacePME : public KSpace {
  public:
    /// @param alpha        Ewald splitting parameter α [1/Å] (match CoulLong).
    /// @param grid_spacing Target real-space grid spacing [Å]. Default 1.0.
    /// @param order        B-spline order (≥ 3). Default 6.
    explicit KSpacePME(double alpha, double grid_spacing = 1.0, int order = 6);
    ~KSpacePME() override;

    std::string name() const override { return "pme"; }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    double alpha_;
    double spacing_;
    int order_;

    int K_[3] = {0, 0, 0};
    int ntot_ = 0;
    std::array<double, 9> inv_cell_{}; ///< host snapshot of cell⁻¹
    double vol_ = 0;

    // Device-side spline moduli per grid line and the complex work grid.
    double* d_bsp_[3] = {nullptr, nullptr, nullptr};
    void* d_grid_ = nullptr; ///< cufftDoubleComplex* (opaque here).
    int plan_ = -1;          ///< cufftHandle (int) — -1 when uncreated.
    bool have_plan_ = false;
};

} // namespace cuda
} // namespace ATV

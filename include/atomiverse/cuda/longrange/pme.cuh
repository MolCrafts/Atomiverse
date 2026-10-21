/// @file cuda/longrange/pme.cuh
/// @brief Smooth Particle-Mesh Ewald reciprocal sum — CUDA (cuFFT).
///        Requires `ATV_BUILD_KSPACE`.
///
/// GPU mirror of `cpu::KSpacePME` (Essmann smooth PME): B-spline charge
/// spreading → cuFFT → influence-function multiply → inverse cuFFT → force
/// interpolation. Drop-in reciprocal replacement for `cuda::KSpaceEwald`;
/// shares α with `cuda::CoulLong`. Single-GPU, fixed-cell/atom-set in this revision.
/// The reciprocal pipeline is captured once in build and replayed as a graph.
/// Spline/interpolation, real mesh and half-spectrum follow CUDA work_t precision;
/// geometry, influence coefficients and energy reductions remain double.
#pragma once

#include <array>
#include <string>

#include "atomiverse/core/precision.h"
#include "atomiverse/cuda/longrange/kspace.cuh"

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
    LongRangeContract long_range_contract() const override {
        return {LongRangeRole::ReciprocalSpace, AccReal(alpha_), AccReal(0),
                AccReal(0)};
    }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    void enqueue_(const LaunchContext& ctx);
    void* graph_exec_ = nullptr; ///< cudaGraphExec_t, allocated only in build.
    int built_atoms_ = 0;
    std::array<double, 9> built_cell_{};
    double alpha_;
    double spacing_;
    int order_;

    int K_[3] = {0, 0, 0};
    int ntot_ = 0;
    int nhalf_ = 0;
    std::array<double, 9> inv_cell_{}; ///< host snapshot of cell⁻¹
    double vol_ = 0;

    double* d_influence_ = nullptr; ///< Fixed-cell Hermitian influence, double.
    void* d_grid_ = nullptr;        ///< work_t real mesh, ntot entries.
    void* d_spectrum_ =
        nullptr;            ///< work_t complex half-spectrum, nhalf entries.
    int forward_plan_ = -1; ///< R2C/D2Z cufftHandle; -1 when uncreated.
    int inverse_plan_ = -1; ///< C2R/Z2D cufftHandle; -1 when uncreated.
};

} // namespace cuda
} // namespace ATV

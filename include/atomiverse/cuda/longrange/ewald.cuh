/// @file cuda/longrange/ewald.cuh
/// @brief Reciprocal-space Ewald summation (LAMMPS `kspace_style ewald`) — CUDA.
///
/// GPU mirror of `cpu::KSpaceEwald`. Adds the long-range reciprocal sum,
/// the self-energy, on top of the real-space `cuda::CoulLong`. Single-GPU in
/// this revision (the structure-factor sum is over `frame.n_local()` atoms;
/// a cross-GPU `S(k)` reduction is a multi-GPU follow-up — NCCL is available).
///
/// Cross-driver component: bind explicitly to the force phase. Shares the
/// splitting parameter α with `cuda::CoulLong`.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/cuda/longrange/kspace.cuh"

namespace ATV {
namespace cuda {

/// @brief Reciprocal-space Ewald summation for periodic electrostatics (CUDA).
class KSpaceEwald : public KSpace {
  public:
    /// @param alpha    Ewald splitting parameter α [1/Å] (match CoulLong).
    /// @param accuracy Relative accuracy target (sets the k-vector cutoff).
    explicit KSpaceEwald(double alpha, double accuracy = 1e-5);
    ~KSpaceEwald() override;

    std::string name() const override { return "ewald"; }
    LongRangeContract long_range_contract() const override {
        return {LongRangeRole::ReciprocalSpace, AccReal(alpha_), AccReal(0),
                AccReal(accuracy_)};
    }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    double alpha_;
    double accuracy_;

    int ng_ = 0; ///< Number of reciprocal vectors.
    // Device reciprocal vectors + kernel weight, and structure-factor scratch.
    double* d_kx_ = nullptr;
    double* d_ky_ = nullptr;
    double* d_kz_ = nullptr;
    double* d_kw_ = nullptr; ///< exp(-k²/4α²)/k².
    double* d_sr_ = nullptr; ///< Re S(k).
    double* d_si_ = nullptr; ///< Im S(k).
};

} // namespace cuda
} // namespace ATV

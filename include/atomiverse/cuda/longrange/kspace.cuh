/// @file cuda/longrange/kspace.cuh
/// @brief Abstract base for reciprocal-space (long-range) electrostatics — CUDA.
///
/// Mirror of `cpu::KSpace`. A first-class force *family*, sibling to
/// `cuda::Pair` / `cuda::Bond` / `cuda::Angle` / `cuda::Dihedral`: the force
/// engine composes one KSpace alongside them (LAMMPS `Force::kspace`,
/// `src/force.h:95`). Concrete styles (`KSpaceEwald`, `KSpacePME`) implement
/// only the reciprocal-space `compute()`; the short-range real-space coulomb is
/// a `Pair` (`cuda::CoulLong`). The two share the Ewald splitting parameter α.
///
/// The reciprocal sum needs a *global* structure factor `S(k) = Σ_i q_i e^{ik·r_i}`,
/// so a multi-GPU KSpace requires a cross-device reduction distinct from the
/// per-atom halo (force-infra-04 will add that hook on this base). CUDA-free
/// header except for the component base it extends.
#pragma once

#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/component/long_range_protocol.cuh"
#include "atomiverse/cuda/component/potential.cuh"

namespace ATV {
namespace cuda {

/// @brief Abstract base for reciprocal-space electrostatics styles (CUDA).
class KSpace : public cuda::Potential, public LongRangeParticipant {
  public:
    ~KSpace() override = default;

    /* role via Potential/kind base */
    const LongRangeParticipant* as_long_range_participant() const final {
        return this;
    }

    /// @brief Framework dispatch — `final`, calls `compute()` (mirrors
    ///        `cuda::Pair` / `cuda::Bond`).
    void launch(const LaunchContext& ctx) final { compute(ctx); }

  protected:
    KSpace() = default;

    /// @brief Hot-path override — derived class launches its reciprocal-space
    ///        kernels (direct k-sum for Ewald, FFT for PME).
    virtual void compute(const LaunchContext& ctx) = 0;
};

} // namespace cuda
} // namespace ATV

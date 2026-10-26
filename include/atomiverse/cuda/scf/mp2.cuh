/// @file cuda/scf/mp2.cuh
/// @brief Device-resident canonical closed-shell MP2 on a borrowed CUDA RHF.
#pragma once

#include <cstddef>
#include <memory>

#include "atomiverse/cpu/scf/mp2.h"
#include "atomiverse/cuda/driver/driver_node.h"

namespace ATV::cpu {
class Frame;
}

namespace ATV::cuda::scf {

class RHF;
struct SCFWorkspace;
namespace internal {
class MP2Device;
}

/// @brief Canonical RMP2 with occupied tiling and device-only HF handoff.
///
/// Construct with a CUDA RHF that outlives this object, then `run(frame)`.
/// Setup allocates a bounded O(n_AO^3) correlation workspace. The HF orbitals,
/// energies and AO integrals remain on device through correlation. Only the
/// final scalar result and requested Frame output are retrieved. Nuclear
/// forces are unavailable; mounting in a force-requesting phase throws.
class MP2 : public cuda::DriverNode {
  public:
    /// @brief Bind an RHF reference and bound additional correlation scratch.
    /// @param mf Borrowed CUDA RHF; must outlive this object.
    /// @param workspace_budget_bytes Maximum MP2 scratch [bytes], excluding HF.
    explicit MP2(RHF& mf,
                 std::size_t workspace_budget_bytes = 512u * 1024u * 1024u);
    /// @brief Release private device correlation storage.
    ~MP2() override;
    /// @brief Move the borrowed reference and private workspace.
    MP2(MP2&&) noexcept;
    MP2& operator=(MP2&&) = delete;
    MP2(const MP2&) = delete;
    MP2& operator=(const MP2&) = delete;

    /// @brief Lowest occupied spatial orbitals excluded from correlation.
    int frozen = 0;
    /// @brief Opposite-spin energy scale [dimensionless]; SCS uses 1.2.
    double os_scale = 1.0;
    /// @brief Same-spin energy scale [dimensionless]; SCS uses 1/3.
    double ss_scale = 1.0;
    /// @brief Maximum additional device correlation workspace [bytes].
    const std::size_t workspace_budget_bytes;

    /// @brief Solve RHF then MP2, overwriting Frame energy [eV].
    /// @param frame Geometry [Å], with nuclear atomic numbers in `kAtom_type`.
    /// @return Unscaled spin channels, correlation and total energy [Hartree].
    cpu::scf::MP2Result run(cpu::Frame& frame);
    /// @brief Solve RHF then MP2 without writing the input Frame.
    /// @param frame Geometry [Å], read only.
    /// @return Spin channels, correlation and total energy [Hartree].
    cpu::scf::MP2Result run(const cpu::Frame& frame);

    /// @brief This method contributes a system total energy.
    /// @return True.
    bool provides_energy() const { return true; }
    /// @brief Analytic MP2 forces are unavailable on CUDA.
    /// @return False.
    bool provides_force() const { return false; }
    /// @brief Reference SCF iteration budget [dimensionless].
    /// @return Bound RHF iteration cap.
    int default_budget() const override;
    /// @brief Prepare the borrowed RHF plan on the parent's execution owner.
    /// @param ctx Parent CUDA context.
    /// @param frame Parent device Frame.
    void build_on(cuda::Context& ctx, cuda::Frame& frame) override;
    /// @brief Solve on the parent Frame, accumulating MP2 total energy [eV].
    /// @param ctx Parent CUDA context; force requests are rejected.
    /// @param frame Parent device Frame.
    /// @param budget Reference SCF iteration limit; <=0 uses the RHF cap.
    /// @return Reference SCF iterations executed.
    int run_nested_erased(cuda::Context& ctx, cuda::Frame& frame,
                          int budget) override;
    /// @brief Release workspace and detach the borrowed RHF.
    /// @param ctx Parent CUDA context.
    /// @param frame Parent device Frame.
    void destroy_on(cuda::Context& ctx, cuda::Frame& frame) override;

  private:
    friend class RHF;
    void prepare_device_(const SCFWorkspace& ws, cuda::Context& ctx,
                         cuda::Frame& frame, bool accumulate);
    void evaluate_device_(const SCFWorkspace& ws, cuda::Context& ctx,
                          cuda::Frame& frame, bool retrieve);
    double retrieve_result_(cuda::Context& ctx);
    cpu::scf::MP2Result solve_(cpu::Frame& frame, bool write);

    RHF* mf_ = nullptr;
    std::unique_ptr<internal::MP2Device> device_;
    cpu::scf::MP2Result result_{};
};

} // namespace ATV::cuda::scf

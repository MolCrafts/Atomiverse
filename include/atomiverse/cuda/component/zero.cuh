/// @file cuda/component/zero.cuh
/// @brief Phase-independent accumulator initialization shared by all workflows.
#pragma once
#include "atomiverse/cuda/component/component.cuh"
namespace ATV::cuda {
/// @brief Zeros force arrays (fx, fy, fz) on device using cudaMemsetAsync.
///
/// Phase-agnostic infrastructure component — the driver binds it to its
/// own phase via explicit `on(phase).add()` (md::MD binds it at
/// pre_force, order -1000). Uses the DMA engine (cudaMemsetAsync), not a
/// compute kernel.
class ZeroForces : public Component {
  public:
    bool supports_input_changes(InputChange) const override { return true; }
    /// @brief Component name.
    /// @return "zero_forces".
    std::string name() const override { return "zero_forces"; }
    bool graph_capturable() const override { return true; }

    /// @brief Structural role — none. Zeroing an accumulator is not a
    ///        loop slot; the driver decides where it binds.
    /// @return ComponentRole::None.
    /* role default default role 0 */

    /// @brief Zero fx, fy, fz arrays via cudaMemsetAsync.
    /// @param ctx Launch context.
    void launch(const LaunchContext& ctx) override;
    void initialize_forces(const LaunchContext& ctx) override { launch(ctx); }
};

/// @brief Zeros potential energy scalar on device using cudaMemsetAsync.
///
/// Phase-agnostic infrastructure component — the driver binds it to its
/// own phase via explicit `on(phase).add()` (md::MD binds it at
/// pre_force, order -999). Uses the DMA engine (cudaMemsetAsync), not a
/// compute kernel.
class ZeroEnergy : public Component {
  public:
    bool supports_input_changes(InputChange) const override { return true; }
    /// @brief Component name.
    /// @return "zero_energy".
    std::string name() const override { return "zero_energy"; }
    bool graph_capturable() const override { return true; }

    /// @brief Structural role — none. Zeroing an accumulator is not a
    ///        loop slot; the driver decides where it binds.
    /// @return ComponentRole::None.
    /* role default default role 0 */

    /// @brief Zero energy scalar via cudaMemsetAsync.
    /// @param ctx Launch context.
    void launch(const LaunchContext& ctx) override;
    void initialize_forces(const LaunchContext& ctx) override { launch(ctx); }
};

/// @brief Zeros the pair/bonded virial tensor `stress[6]`.
///
/// Phase-agnostic — MD binds at `pre_force @ -998`, Minimize at
/// `force @ -998`, Evaluate at `prepare @ -998`. Mirrors `cpu::ZeroStress`.
class ZeroStress : public Component {
  public:
    bool supports_input_changes(InputChange) const override { return true; }
    std::string name() const override { return "zero_stress"; }
    bool graph_capturable() const override { return true; }
    void launch(const LaunchContext& ctx) override;
    void initialize_forces(const LaunchContext& ctx) override { launch(ctx); }
};

/// @brief Zeros every system scalar (energies, kinetic energy, conserved
///        quantity, virial and kinetic stress) in one cudaMemsetAsync.
///
/// MD binds it at pre_force (order -999): every accumulator a step fills is
/// produced after it (forces at force, KE and kinetic stress at end_of_step),
/// and everything read before it (initial_integrate) is the previous step's.
class ZeroScalars : public Component {
  public:
    bool supports_input_changes(InputChange) const override { return true; }
    std::string name() const override { return "zero_scalars"; }
    bool graph_capturable() const override { return true; }
    void launch(const LaunchContext& ctx) override;
    void initialize_forces(const LaunchContext& ctx) override { launch(ctx); }
};

} // namespace ATV::cuda

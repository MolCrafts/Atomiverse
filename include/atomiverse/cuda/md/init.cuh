/// @file cuda/md/init.cuh
/// @brief Initializer components for forces, energy, kinetic energy, and velocity.
///
/// Initializers prepare accumulator fields before the computation phase:
/// - ZeroForces / ZeroEnergy / ZeroKinetic: zero accumulators via cudaMemsetAsync.
/// - InitVelocity: Maxwell-Boltzmann velocity initialization (LAMMPS `velocity create`).
///
/// The Zero* components are **phase-agnostic**: they declare no
/// `default_bindings()`, so the concrete driver binds them to its own
/// phase enum via explicit `on(phase).add()`. `cuda::md::MD` binds them
/// at `md::Phase`; `cuda::minimizer` drivers reuse the same classes at
/// `minimizer::Phase`. This mirrors the CPU resolution in `cpu/md/init.h`.
/// CUDA-free header.
///
/// @see md::MD for the MD driver that explicitly binds the Zero* initializers
/// @see md::KineticEnergy for the component that accumulates kinetic energy
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/facade/component_role.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cuda {
namespace md {

/// @brief Zeros force arrays (fx, fy, fz) on device using cudaMemsetAsync.
///
/// Phase-agnostic infrastructure component — the driver binds it to its
/// own phase via explicit `on(phase).add()` (md::MD binds it at
/// pre_force, order -1000). Uses the DMA engine (cudaMemsetAsync), not a
/// compute kernel.
class ZeroForces : public Component {
  public:
    /// @brief Component name.
    /// @return "zero_forces".
    std::string name() const override { return "zero_forces"; }

    /// @brief Structural role — none. Zeroing an accumulator is not a
    ///        loop slot; the driver decides where it binds.
    /// @return ComponentRole::None.
    /* role default default role 0 */

    /// @brief Zero fx, fy, fz arrays via cudaMemsetAsync.
    /// @param ctx Launch context.
    void launch(const LaunchContext& ctx) override;
};

/// @brief Zeros potential energy scalar on device using cudaMemsetAsync.
///
/// Phase-agnostic infrastructure component — the driver binds it to its
/// own phase via explicit `on(phase).add()` (md::MD binds it at
/// pre_force, order -999). Uses the DMA engine (cudaMemsetAsync), not a
/// compute kernel.
class ZeroEnergy : public Component {
  public:
    /// @brief Component name.
    /// @return "zero_energy".
    std::string name() const override { return "zero_energy"; }

    /// @brief Structural role — none. Zeroing an accumulator is not a
    ///        loop slot; the driver decides where it binds.
    /// @return ComponentRole::None.
    /* role default default role 0 */

    /// @brief Zero energy scalar via cudaMemsetAsync.
    /// @param ctx Launch context.
    void launch(const LaunchContext& ctx) override;
};

/// @brief Zeros kinetic energy scalar on device using cudaMemsetAsync.
///
/// Phase-agnostic infrastructure component — the driver binds it to its
/// own phase via explicit `on(phase).add()` (md::MD binds it at
/// end_of_step, order -1000). Uses the DMA engine (cudaMemsetAsync), not
/// a compute kernel.
class ZeroKinetic : public Component {
  public:
    /// @brief Component name.
    /// @return "zero_kinetic".
    std::string name() const override { return "zero_kinetic"; }

    /// @brief Structural role — none. Zeroing an accumulator is not a
    ///        loop slot; the driver decides where it binds.
    /// @return ComponentRole::None.
    /* role default default role 0 */

    /// @brief Zero kinetic energy scalar via cudaMemsetAsync.
    /// @param ctx Launch context.
    void launch(const LaunchContext& ctx) override;
};

/// @brief Removes net linear COM momentum every `freq` steps.
///
/// CUDA sibling of `cpu::md::ZeroLinearMomentum` (`cpu::md::ZeroMomentum`
/// alias). Mirrors LAMMPS `fix momentum N linear 1 1 1`
/// (`velocity.cpp:738`): subtracts `P_cm / M_tot` from every atom's
/// velocity so the net translational drift stays zero — required on
/// stages 2/3/4 of the MSU PE protocol where Langevin + NPT coupling
/// otherwise lets the COM walk away.
///
/// Auto-binds to `end_of_step @ +25` (same slot as the CPU sibling, so
/// it runs after `KineticEnergy(-100)`, `NoseHooverNPT(-40)`, the
/// integrator's `end_of_step`, and the recorder/thermo writers at
/// `+20`, but before any user-added end_of_step observer that needs to
/// see drift-corrected velocities).
///
/// Single-GPU scope (no multi-GPU sync). The 4-double D2H of
/// `{Σ mv_x, Σ mv_y, Σ mv_z, Σ m}` at every active step is the
/// dominant cost; for cheap operation use `freq > 1`. The host scalars
/// are computed on the host stream synchronously and the subtract
/// kernel re-launched on the compute stream — this is a deliberate
/// host barrier and is excluded from the Phase H zero-sync recorder
/// contract (integrity components are exempt).
///
/// @see cpu::md::ZeroLinearMomentum for the CPU reference implementation.
/// @see .claude/specs/msu-pe-cuda-components-01-fidelity-and-momentum.md
class ZeroMomentum : public Component {
  public:
    /// @brief Construct momentum-removal component.
    /// @param freq Subtract every `freq` `end_of_step` invocations
    ///        (LAMMPS default 1). `freq=0` is treated as 1.
    explicit ZeroMomentum(int freq = 1) : freq_(freq <= 0 ? 1 : freq) {}

    /// @brief Component name.
    /// @return "zero_momentum".
    std::string name() const override { return "zero_momentum"; }

    /// @brief Structural role — none. This is an integrity / drift-
    ///        cancellation hook, not a loop slot
    ///        (Thermostat / Integrator / …).
    /// @return ComponentRole::None.
    /* role default default role 0 */
    /// @brief Write-set: subtracts the drift from every velocity.

    /// @brief Default binding: `end_of_step @ +25` — after thermo/recorder
    ///        writers (`+20`) and the integrator's `end_of_step` slot.
    /// @return Single binding.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +25)};
    }

    /// @brief Allocate the 4 device scratch words `{px, py, pz, M_tot}`
    ///        used to accumulate momentum + total mass each active step.
    /// @param ctx Build context.
    void build(const BuildContext& ctx) override;

    /// @brief Compute COM momentum on the device, D2H copy, broadcast
    ///        `v_cm = P_cm / M_tot` back as kernel args, subtract from
    ///        every velocity. No-op when `step % freq != 0`.
    /// @param ctx Launch context.
    void launch(const LaunchContext& ctx) override;

    /// @brief Free the device scratch buffer allocated in build().
    /// @param ctx Destroy context.
    void destroy(const DestroyContext& ctx) override;

    /// @brief Removal frequency (LAMMPS `fix momentum N`).
    /// @return Integer frequency (always >= 1).
    int frequency_steps() const { return freq_; }

  private:
    int freq_;
    long long step_ = 0;
    // 4 contiguous work_t = {Σ mv_x, Σ mv_y, Σ mv_z, Σ m}. cudaMalloc'd
    // in build(), zeroed in launch() before each accumulate, cudaFree'd
    // in destroy().
    void* d_scratch_ = nullptr; // work_t[4]
};

/// @brief Maxwell-Boltzmann velocity initialization (LAMMPS `velocity create`).
///
/// Generates Gaussian-distributed velocities at the target temperature
/// and removes center-of-mass velocity (zero total momentum).
///
/// One-time initialization in build(). launch() is a no-op.
///
/// `role()` stays `None` — it is a one-shot setup component, not a loop slot
/// (Integrator / Thermostat / …).
///
/// `capabilities()` is `Velocity`, mirroring `cpu::md::InitVelocity`. It seeds the
/// Maxwell distribution in `build()` rather than in `launch()`, and that used to be
/// the argument for declaring `None` — but Capability is the WRITE-SET, not a
/// per-step-writer flag. A component that writes `v` declares `v`, whichever
/// lifecycle hook it does the writing in. (Both backends declared `None` here while
/// writing the most fundamental field in the frame; that hole is what the
/// Position/Velocity bits closed.)
///
/// @par Units
/// Temperature in Kelvin. Velocities in Angstrom/fs.
/// v_i ~ N(0, sqrt(kB*T / (kMvv2e * m_i))) per component.
///
/// @par Usage
/// @code{.cpp}
/// sim.add(md::InitVelocity{300.0, 42});  // 300 K, seed 42
/// @endcode
///
/// @see LAMMPS velocity create: https://docs.lammps.org/velocity.html
class InitVelocity : public Component {
  public:
    /// @brief Construct velocity initializer.
    /// @param temperature Target temperature [K].
    /// @param seed Random number seed.
    InitVelocity(double temperature, unsigned long long seed)
        : temperature_(temperature), seed_(seed) {}

    /// @brief Component name.
    /// @return "init_velocity".
    std::string name() const override { return "init_velocity"; }

    /// @brief Write-set: seeds the Maxwell distribution into v (at build).

    /// @brief Default bindings: initial_integrate at order -2000 (before integrator).
    /// @return Single binding to initial_integrate.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, -2000)};
    }

    /// @brief Generate Maxwell-Boltzmann velocities on device, remove COM velocity.
    /// @param ctx Build context.
    void build(const BuildContext& ctx) override;

    /// @brief No-op (velocities initialized once in build).
    /// @param ctx Launch context.
    void launch(const LaunchContext& ctx) override;

    /// @brief Target temperature.
    /// @return Temperature [K].
    double temperature() const { return temperature_; }

    /// @brief Random seed.
    /// @return Seed value.
    unsigned long long seed() const { return seed_; }

  private:
    double temperature_;      ///< Target temperature [K].
    unsigned long long seed_; ///< RNG seed.
};

} // namespace md
} // namespace cuda
} // namespace ATV

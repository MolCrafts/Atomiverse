/// @file cuda/dynamics/gear_pc.cuh
/// @brief Fifth-order Gear predictor–corrector with the Gaussian and
///        configurational thermostats — the CUDA twin of
///        `cpu/dynamics/gear_pc.h` (Petravic & Delhommelle, J. Chem. Phys.
///        118, 7477 (2003)).
///
/// The equations of motion are integrated as the first-order system
/// \f[
///   \dot r_i = v_i + \frac{\xi}{T_0}\frac{\partial T_{C1}}{\partial r_i},
///   \qquad \dot v_i = \frac{F_i}{m_i} - \alpha\,w_i, \qquad
///   \dot\xi = -Q\,\frac{T_{C1}-T_0}{T_0}
/// \f]
/// with the six-value Nordsieck history of `core/gear_pc.h` per variable,
/// kept on the device. \(\alpha\) is the Gauss multiplier that holds
/// \(\sum_i m_i w_i^2\) of the peculiar velocities \(w\) fixed (kinetic,
/// flow-free or transverse definition, `cpu::md::GaussKind`); the corrector
/// is followed by a projection onto the constraint. The thermostat
/// parameters are the CPU backend's (`cpu/dynamics/gauss_thermostat.h`).
///
/// Every step launch only enqueues kernels: the multiplier, the projection
/// scale and \(\xi\) are device scalars (iron law 10). The Nordsieck start
/// runs at the backend's initial force evaluation, so a run on unchanged
/// inputs continues the history and \(\xi\) of the run before it (the
/// backend's continued-run rule; the CPU twin restarts at every run) and an
/// input edit restarts them. The diagnostics are read back when the run
/// returns.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/gauss_thermostat.h"
#include "atomiverse/cuda/component/curvature_protocol.cuh"
#include "atomiverse/cuda/component/integrator.cuh"
#include "atomiverse/cuda/component/timestep_protocol.cuh"
#include "atomiverse/cuda/dynamics/configurational_temperature.cuh"

namespace ATV {
namespace cuda {
namespace md {

/// @brief Gear predictor–corrector integrator, optionally thermostatted.
class GearPC : public cuda::Integrator,
               public cuda::TimestepProvider,
               public cuda::CurvatureConsumer {
  public:
    /// @brief Unthermostatted (NVE) integration.
    /// @param dt Time step [time unit of the run's unit system] (> 0).
    explicit GearPC(double dt);
    /// @brief Gaussian isokinetic integration.
    /// @param dt Time step (> 0).
    /// @param thermostat Constrained temperature definition and target.
    GearPC(double dt, cpu::md::GaussThermostat thermostat);
    /// @brief Configurational (Braga–Travis / Petravic) thermostat.
    /// @param dt Time step (> 0).
    /// @param thermostat Target temperature and feedback constant.
    GearPC(double dt, cpu::md::ConfigurationalThermostat thermostat);
    ~GearPC() override;
    GearPC(GearPC&&) noexcept;
    GearPC& operator=(GearPC&&) noexcept;

    std::string name() const override { return "gear_pc"; }
    const cuda::TimestepProvider* as_timestep_provider() const override {
        return this;
    }
    double timestep() const override { return dt_; }
    bool graph_capturable() const override { return true; }
    std::uint64_t graph_epoch() const override;
    CurvatureConsumer* as_curvature_consumer() override {
        return configurational_ ? this : nullptr;
    }
    void set_curvature_terms(std::vector<CurvatureSource*> sources,
                             std::vector<const ExternalDrive*> drives) override;

    /// Predictor, the Nordsieck start (after every force term of the
    /// initial evaluation), corrector.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 0),
                ATV::make_binding(ATV::md::Phase::post_force, 1000),
                ATV::make_binding(ATV::md::Phase::final_integrate, 0)};
    }

    void invalidate_inputs(InputChange changes,
                           const BuildContext& ctx) override;
    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    /// @brief Start the Nordsieck history from the frame's state and the
    ///        initial forces (first run, or after an input edit).
    /// @throws std::runtime_error when the constrained kinetic energy is
    ///         zero or the Laplacian of the potential is not positive.
    void initialize_forces(const LaunchContext& ctx) override;
    /// @brief Read the diagnostics back; reports a thermostat fault.
    void on_retrieve(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

    /// @brief \(|\sum m w^2 / K_0 - 1|\) the last corrector left before the
    ///        projection (0 without a Gaussian thermostat), as of the last
    ///        returned run.
    double constraint_drift() const { return drift_; }
    /// @brief Gauss multiplier \(\alpha\) of the last step [1/time].
    double gauss_multiplier() const { return alpha_; }
    /// @brief Configurational thermostat variable \(\xi\).
    double xi() const { return xi_; }
    /// @brief \(T_{C1}\) of the last corrector evaluation [K].
    double configurational_temperature() const { return t_c1_; }

  private:
    void start_(const LaunchContext& ctx);
    void measure_(const LaunchContext& ctx, bool with_force, bool projection);
    void project_(const LaunchContext& ctx);
    void throw_on_fault_(int status, double laplacian) const;

    double dt_;
    std::optional<cpu::md::GaussThermostat> gauss_;
    std::optional<cpu::md::ConfigurationalThermostat> configurational_;
    CurvatureAssembly assembly_;
    bool started_ = false;

    double drift_ = 0.0;
    double alpha_ = 0.0;
    double xi_ = 0.0;
    double t_c1_ = 0.0;

    /// Device history, thermostat scalars and the species table.
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace md
} // namespace cuda
} // namespace ATV

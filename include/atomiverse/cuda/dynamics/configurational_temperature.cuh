/// @file cuda/dynamics/configurational_temperature.cuh
/// @brief First-order configurational temperature on the device — the CUDA
///        twin of `cpu/dynamics/configurational_temperature.h`.
///
/// \f[
///   T_{C1} = \frac{\sum_i |F_i|^2}{k_B \sum_i \nabla_i^2 U}
/// \f]
/// (Petravic & Delhommelle, J. Chem. Phys. 118, 7477 (2003), eq. 7), with
/// \(F\) the interaction force (the frame force minus every `ExternalDrive`)
/// and \(U\) the interaction energy of every `CurvatureSource`.
/// `CurvatureAssembly` enqueues one evaluation and keeps its per-atom rows
/// and the two sums on the device; the observer accumulates run averages
/// there and reads them back when the run returns.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/component/curvature_protocol.cuh"
#include "atomiverse/cuda/component/observer.cuh"

namespace ATV {
namespace cuda {

/// @brief One configurational-temperature evaluation.
struct ConfigurationalSample {
    double force_sq = 0.0;    ///< \(N = \sum_i |F_i|^2\) [eV²/Å²]
    double laplacian = 0.0;   ///< \(D = \sum_i \nabla_i^2 U\) [eV/Å²]
    double temperature = 0.0; ///< \(T_{C1} = N/(k_B D)\) [K]
};

/// @brief Device rows and sums of one curvature evaluation, shared by the
///        observer and the configurational thermostat of `cuda::md::GearPC`.
///
/// `evaluate()` only enqueues kernels: the interaction force into `u`, every
/// source's \(H u\), Laplacian and Laplacian gradient, and the reduction of
/// \(N\) and \(D\) into `sums()`.
class CurvatureAssembly {
  public:
    CurvatureAssembly() = default;
    CurvatureAssembly(const CurvatureAssembly&) = delete;
    CurvatureAssembly& operator=(const CurvatureAssembly&) = delete;
    CurvatureAssembly(CurvatureAssembly&& other) noexcept;
    CurvatureAssembly& operator=(CurvatureAssembly&& other) noexcept;
    ~CurvatureAssembly();

    /// @brief The run's force terms (`CurvatureConsumer::set_curvature_terms`).
    void set_terms(std::vector<CurvatureSource*> sources,
                   std::vector<const ExternalDrive*> drives);

    /// @brief Allocate the rows for the frame capacity.
    /// @throws std::runtime_error without a `CurvatureSource`, or with
    ///         ghost rows (single GPU only).
    void build(const BuildContext& ctx);
    /// @brief Free the device rows.
    void destroy();

    /// @brief Enqueue one evaluation at the frame's current positions and
    ///        forces (after the force phase of the same configuration).
    void evaluate(const LaunchContext& ctx) const;

    /// @brief Device pair {N, D} of the last enqueued evaluation.
    const double* sums() const { return sums_; }
    /// @brief Device rows of the last enqueued evaluation.
    const CurvatureRows& rows() const { return rows_; }
    /// @brief Combined `drive_epoch()` of the run's drives.
    std::uint64_t epoch() const;

  private:
    std::vector<CurvatureSource*> sources_;
    std::vector<const ExternalDrive*> drives_;
    double* store_ = nullptr; ///< 10 rows of the frame capacity + {N, D}.
    double* sums_ = nullptr;
    CurvatureRows rows_{};
};

/// @brief Samples \(T_{C1}\), \(N\) and \(D\) every `every` steps, after
///        the force phase, and keeps their run averages.
///
/// The averaged temperature is the ratio of the averaged sums,
/// \(\langle N\rangle / (k_B \langle D\rangle)\). Values are read back from
/// the device when `run()` returns (iron law 10).
class ConfigurationalTemperature : public Observer, public CurvatureConsumer {
  public:
    /// @param every Sampling stride in steps (>= 1).
    explicit ConfigurationalTemperature(int every = 1);
    ~ConfigurationalTemperature() override;
    ConfigurationalTemperature(ConfigurationalTemperature&&) noexcept;
    ConfigurationalTemperature&
    operator=(ConfigurationalTemperature&&) noexcept;

    std::string name() const override { return "configurational_temperature"; }
    int frequency() const override { return every_; }
    bool graph_capturable() const override { return true; }
    std::uint64_t graph_epoch() const override { return assembly_.epoch(); }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::post_force, 200)};
    }
    CurvatureConsumer* as_curvature_consumer() override { return this; }
    void set_curvature_terms(std::vector<CurvatureSource*> sources,
                             std::vector<const ExternalDrive*> drives) override;
    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    /// @brief Start of a run: a new average (the CPU observer's rule).
    void setup_output(const LaunchContext& ctx) override;
    void on_retrieve(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

    /// @brief Most recent sample, as of the last returned run.
    const ConfigurationalSample& last() const { return last_; }
    /// @brief Averages over the last returned run.
    const ConfigurationalSample& mean() const { return mean_; }
    /// @brief Samples accumulated in the last returned run.
    long long samples() const { return samples_; }
    /// @brief The device rows of the last evaluation.
    const CurvatureAssembly& assembly() const { return assembly_; }

  private:
    int every_;
    CurvatureAssembly assembly_;
    double* stats_ = nullptr; ///< Device {N, D, ΣN, ΣD, samples}.
    ConfigurationalSample last_{};
    ConfigurationalSample mean_{};
    long long samples_ = 0;
};

} // namespace cuda
} // namespace ATV

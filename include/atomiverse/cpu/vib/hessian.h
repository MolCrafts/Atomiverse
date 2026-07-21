/// @file cpu/vib/hessian.h
/// @brief cpu::vib::Hessian — concrete numerical-Hessian driver.
///
/// `cpu::vib::Hessian` is the user-facing CPU numerical-Hessian preset. It
/// auto-registers the four structural components that assemble a
/// finite-difference (central-difference) molecular Hessian and exposes the
/// result via `run()` / `last_result()`.
///
/// Usage — register any force/energy kind (`Potential` / `Bias` / `HF` /
/// `Semiempirical` / `PlaneWave` / `GPW`) on `vib::Phase::force` (an FF
/// evaluator or a concrete SCF method such as `cpu::scf::RHF`), exactly as
/// AIMD registers a force evaluator into `cpu::md::MD`:
///
/// @code{.cpp}
///   cpu::scf::RHF rhf;
///   rhf.basis = cpu::scf::BasisSet::STO3G;
///
///   cpu::vib::Hessian driver;
///   driver.on(ATV::vib::Phase::force)
///         .add(std::move(rhf));
///   driver.build();
///   cpu::vib::HessianResult H = driver.run(frame);   // [3N,3N] eV/Angstrom^2
/// @endcode
///
/// Layering: this header includes no `cpu/scf/` header. The force
/// evaluator is supplied entirely by user code through
/// `on(vib::Phase::force).add(...)`; no domain-to-domain header edge is
/// created.
#pragma once

#include <memory>
#include <string>

#include "atomiverse/cpu/component/bias.h"
#include "atomiverse/cpu/component/gpw.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/component/path_coupling.h"
#include "atomiverse/cpu/component/plane_wave.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/driver/cpu_driver.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/vib/displace_step.h"
#include "atomiverse/cpu/vib/hessian_accumulator.h"
#include "atomiverse/cpu/vib/hessian_result.h"
#include "atomiverse/cpu/vib/hessian_workspace.h"
#include "atomiverse/cpu/vib/reference_geometry.h"
#include "atomiverse/cpu/vib/symmetrize.h"
#include "atomiverse/cpu/vib/thermochem.h"
#include "atomiverse/facade/component_role.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cpu {
namespace vib {

/// @brief Finite-difference numerical-Hessian preset.
///
/// Termination is a **fixed `6N`-displacement budget** followed by one finalize pass —
/// every displacement must complete for the Hessian to be defined, so there is no
/// convergence path and no early stop. `6N` is derived from the frame inside `run()`,
/// never passed in. The budget is the loop bound; no criterion is attached.
///
/// Inheritance: Hessian → CPUDriver<vib::Phase> → Driver<vib::Phase, cpu::device_tag>.
/// There is no `ResponseDriver` category in between any more.
class Hessian : public CPUDriver<ATV::vib::Phase> {
  public:
    /// @brief Tag type to construct a driver WITHOUT the HessianAccumulator.
    ///
    /// Exists only so tests can verify `validate_components()` throws when no
    /// `HessianAccumulator` is present. Production code uses the default
    /// constructor.
    struct WithoutAccumulator {};

    /// @brief Output paths for the two auto-injected Recorder components.
    ///
    /// `frequency_tsv` is the `FrequencyReport` TSV path;
    /// `mode_xyz_prefix` is the `NormalModeWriter` file-name prefix
    /// (mode `k` is written to `"<prefix>_mode<k>.xyz"`).
    struct OutputConfig {
        std::string frequency_tsv = "frequencies.tsv"; ///< TSV path.
        std::string mode_xyz_prefix = "mode";          ///< XYZ prefix.
    };

    /// @brief Construct a fully-wired numerical-Hessian driver.
    ///
    /// Auto-registers `ReferenceGeometry`, `DisplaceStep`,
    /// `HessianAccumulator`, `Symmetrize`, `MassWeighting`,
    /// `HessianDiagonalizer`, plus the thermochemistry chain (`Thermochem`,
    /// `FrequencyReport`, `NormalModeWriter`), all bound to a
    /// driver-owned shared `HessianWorkspace`. The caller must still
    /// register a force/energy kind on `vib::Phase::force`.
    ///
    /// Thermochemistry uses standard conditions (298.15 K, 1 atm,
    /// closed-shell singlet, symmetry number 1). Use the
    /// `Thermochem::Config` overload to override.
    Hessian();

    /// @brief Construct a numerical-Hessian driver with custom
    ///        thermochemistry inputs.
    ///
    /// Identical wiring to the default constructor, but the auto-injected
    /// `Thermochem` component uses `tc` (temperature / pressure /
    /// multiplicity / symmetry number) instead of standard conditions.
    /// @param tc Thermochemistry inputs forwarded to `cpu::vib::Thermochem`.
    explicit Hessian(const Thermochem::Config& tc);

    /// @brief Construct with custom thermochemistry inputs and custom
    ///        Recorder output paths.
    ///
    /// Identical wiring to the default constructor; the auto-injected
    /// `Thermochem` uses `tc` and the `FrequencyReport` /
    /// `NormalModeWriter` Recorders use the paths in `out`.
    /// @param tc  Thermochemistry inputs forwarded to `cpu::vib::Thermochem`.
    /// @param out Output paths for the FrequencyReport / NormalModeWriter.
    Hessian(const Thermochem::Config& tc, const OutputConfig& out);

    /// @brief Construct a driver WITHOUT the HessianAccumulator (test-only).
    ///
    /// Registers only `ReferenceGeometry`, `DisplaceStep`, and
    /// `Symmetrize`. `build()` then throws from
    /// `Hessian::validate_components()` for the missing
    /// `HessianAccumulator`.
    /// @param tag Disambiguating tag (`WithoutAccumulator{}`).
    explicit Hessian(WithoutAccumulator tag);

    /// @brief Run the `6N`-displacement finite-difference Hessian assembly.
    ///
    /// Sizes the shared workspace from `frame.nlocal()`, runs the fixed
    /// budget (`6N` displacement passes + one finalize pass), and returns the
    /// symmetrized Hessian. The result is also
    /// cached for `last_result()`. On return the frame carries the
    /// equilibrium geometry again.
    ///
    /// @param frame Host-resident system state (positions in Angstrom).
    /// @return `HessianResult` with the `[3N,3N]` Hessian in eV/Angstrom^2.
    HessianResult run(cpu::Frame& frame);

    /// @brief Most recent `HessianResult` produced by `run()`.
    /// @pre `run()` has completed at least once.
    /// @return The cached Hessian result.
    const HessianResult& last_result() const { return last_result_; }

  protected:
    /// @brief A force evaluator to displace against, and an accumulator to build into.
    void validate_components() const override {
        bool has_force_writer = false;
        bool has_hessian_writer = false;
        for_each_pending_kind(
            [&](const cpu::Component& c) {
                if (dynamic_cast<const Potential*>(&c)
                    || dynamic_cast<const Bias*>(&c)
                    || dynamic_cast<const PathCoupling*>(&c)
                    || dynamic_cast<const HF*>(&c)
                    || dynamic_cast<const Semiempirical*>(&c)
                    || dynamic_cast<const PlaneWave*>(&c)
                    || dynamic_cast<const GPW*>(&c))
                    has_force_writer = true;
                if (c.name() == "vib_hessian_accumulator")
                    has_hessian_writer = true;
            },
            [&](const cpu::Driver* d) {
                if (dynamic_cast<const HF*>(d))
                    has_force_writer = true;
            });
        if (!has_force_writer)
            throw_missing(
                "Potential / Bias / HF / Semiempirical / PlaneWave / GPW");
        if (!has_hessian_writer)
            throw_missing("vib_hessian_accumulator");
    }

  private:
    /// @brief Shared scratch for the four structural components.
    ///
    /// Heap-allocated so its address is stable across any driver move —
    /// the components hold raw pointers into it.
    std::unique_ptr<HessianWorkspace> workspace_{
        std::make_unique<HessianWorkspace>()};

    /// @brief Cached result of the most recent `run()`.
    HessianResult last_result_;
};

} // namespace vib
} // namespace cpu
} // namespace ATV

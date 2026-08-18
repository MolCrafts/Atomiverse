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

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/bias.h"
#include "atomiverse/cpu/component/gpw.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/component/path_coupling.h"
#include "atomiverse/cpu/component/plane_wave.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/vibration/displace_step.h"
#include "atomiverse/cpu/vibration/hessian_accumulator.h"
#include "atomiverse/cpu/vibration/hessian_result.h"
#include "atomiverse/cpu/vibration/hessian_workspace.h"
#include "atomiverse/cpu/vibration/reference_geometry.h"
#include "atomiverse/cpu/vibration/symmetrize.h"
#include "atomiverse/cpu/vibration/thermochem.h"

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
/// Inheritance: Hessian → Driver<vib::Phase> → Driver<vib::Phase, cpu::device_tag>.
/// There is no `ResponseDriver` category in between any more.
class Hessian : public Driver<ATV::vib::Phase> {
  public:
    /// @brief Tag type to construct a driver WITHOUT the HessianAccumulator.
    ///
    /// Exists only so tests can verify `validate_components()` throws when no
    /// `HessianAccumulator` is present. Production code uses the default
    /// constructor.
    struct WithoutAccumulator {};

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
    /// closed-shell singlet, symmetry number 1). Override via the
    /// multi-argument constructor (no nested Config bag).
    Hessian();

    /// @brief Construct with custom thermochemistry inputs.
    ///
    /// @param temperature Temperature [K].
    /// @param pressure Pressure [Pa].
    /// @param multiplicity Spin multiplicity `2S+1`.
    /// @param symmetry_number Rotational sigma.
    Hessian(double temperature, double pressure, int multiplicity,
            int symmetry_number);

    /// @brief Construct with thermochemistry inputs and Recorder paths.
    ///
    /// @param temperature Temperature [K].
    /// @param pressure Pressure [Pa].
    /// @param multiplicity Spin multiplicity `2S+1`.
    /// @param symmetry_number Rotational sigma.
    /// @param frequency_tsv Path for the `FrequencyReport` TSV.
    /// @param mode_xyz_prefix Prefix for `NormalModeWriter` XYZ files
    ///        (mode `k` → `"<prefix>_mode<k>.xyz"`).
    Hessian(double temperature, double pressure, int multiplicity,
            int symmetry_number, std::string frequency_tsv,
            std::string mode_xyz_prefix);

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

    /// @brief Frequencies / modes from a **precomputed** Cartesian Hessian.
    ///
    /// Same mass-weight + TR-project + diagonalize finalize as after a FD
    /// `run()`, without the 6N force loop. Use when the matrix already exists
    /// (e.g. `cpu::scf::RHF::hessian`). Frame supplies masses (`kAtom_type`)
    /// and equilibrium geometry for the TR projector.
    ///
    /// Does not require a force evaluator on this driver. Result is cached in
    /// `last_result()` like `run()`.
    ///
    /// @param frame Equilibrium geometry [Å].
    /// @param hessian_ev_ang2 Dense symmetric `[3N,3N]` Hessian [eV/Å²].
    /// @return Frequencies [cm⁻¹], normal modes, etc. on `HessianResult`
    ///         (Cartesian `hessian` field copies the input). This path runs
    ///         no `thermochem` pass, so EVERY thermochemistry field comes
    ///         back 0.0 — `zpe`, `u_tot`, `h_tot`, `g_tot`, `g_thermal` and
    ///         all four entropies. Those are not physical zeros; use the
    ///         full `run()` (or a driver constructed with thermochemistry
    ///         inputs) when they are wanted.
    /// @throws std::invalid_argument on empty frame or shape mismatch.
    HessianResult run_from_cartesian(cpu::Frame& frame,
                                     xt::xtensor<double, 2> hessian_ev_ang2);

    /// @brief Most recent `HessianResult` produced by `run()` /
    ///        `run_from_cartesian()`.
    /// @pre A run method has completed at least once.
    /// @return The cached Hessian result.
    const HessianResult& last_result() const { return last_result_; }

  private:
    /// @brief Register the full vib component chain via bind_defaults / on().
    void wire_full_chain_(HessianWorkspace* ws, double temperature,
                          double pressure, int multiplicity,
                          int symmetry_number, const std::string& frequency_tsv,
                          const std::string& mode_xyz_prefix);

  protected:
    /// @brief A force evaluator to displace against, and an accumulator to build into.
    void validate_components() const override {
        this->require_force();
        this->require_hessian();
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

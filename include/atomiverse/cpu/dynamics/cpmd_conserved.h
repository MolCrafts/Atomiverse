/// @file cpu/dynamics/cpmd_conserved.h
/// @brief cpu::md::CpmdConserved — Car–Parrinello conserved energy assembler.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief End-of-step assembler publishing the Car–Parrinello conserved
///        energy onto `frame.meta[kConserved]` [eV].
///
/// \f[
///   E_{\mathrm{cons}} = E_{\mathrm{KS}} + T_{\mathrm{nuc}}
///                     + T_e + E_{\mathrm{therm}}^{e}
/// \f]
///
/// i.e. `kConserved = kEnergy + kKinetic + kEnergyElectronKinetic
/// + kEnergyElectronThermostat`, all four terms in eV. `kConserved` is
/// **replaced**, never accumulated into: it is a state channel with exactly
/// one owner per run.
///
/// ## Why end_of_step @ -90 (the only slot where all four terms are current)
///
/// - `KineticEnergy` runs at `end_of_step @ -100`, so `kKinetic` is
///   half-step stale anywhere earlier — in particular at `force` phase, which
///   is why the CPMD evaluator cannot assemble \(E_{\mathrm{cons}}\) itself
///   and only publishes its two electronic terms.
/// - `mpi::Reduce` (`@ -50`), the thermo/record components and any user
///   `Criterion` reading the frame all run *after* -90, so the Thermo
///   "Conserved" column and a user-space drift criterion see this step's
///   value.
///
/// ## Fail-fast contract
///
/// `launch()` requires **both** electronic keys to be present and throws
/// `std::runtime_error` naming the missing key when either is absent (no
/// CPMD-enabled evaluator ran). Treating an absent key as 0 would publish a
/// plausible-looking \(E_{\mathrm{cons}}\) with the drifting fictitious term
/// silently dropped — a conserved quantity that looks conserved because the
/// interesting part is gone. A refused launch writes nothing.
///
/// ## Wiring: explicit, and do not combine with a `kConserved` owner
///
/// This component is user-wired, never auto-injected:
///
/// ```cpp
/// cpu::md::MD cpmd_md;
/// cpmd_md.set_integrator(cpu::md::VelocityVerlet{0.25});
/// cpmd_md.on(md::Phase::force).add(std::move(gpw_cpmd_evaluator));
/// cpmd_md.add(cpu::md::CpmdConserved{});   // default binding: end_of_step @ -90
/// ```
///
/// (`add()` uses the default binding above; `on(end_of_step).add(...)` would
/// bind at order 0 — after `mpi::Reduce @ -50` — defeating the rationale.)
///
/// The extended-system integrators/thermostats `NoseHooverNPT`,
/// `NphIntegrator`, `ParrinelloRahmanNPT` and `FixCSVR` each own
/// `kConserved` themselves; combining one of them with `CpmdConserved`
/// makes two components fight over the same channel and the last writer
/// wins. CPMD + NPT is not a product path — run CPMD under NVE nuclei (an
/// electronic Nosé–Hoover thermostat contributes through
/// `kEnergyElectronThermostat`).
class CpmdConserved : public cpu::Component {
  public:
    /// @brief Plan / timing label.
    /// @return `"cpmd_conserved"`.
    std::string name() const override { return "cpmd_conserved"; }

    /// @brief One binding: `md::Phase::end_of_step @ -90` (see class doc).
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, -90)};
    }

    /// @brief Sum the four energy terms onto `kConserved` [eV].
    /// @param ctx Launch context; only `ctx.frame.meta` is touched.
    /// @throws std::runtime_error when `kEnergyElectronKinetic` or
    ///         `kEnergyElectronThermostat` is absent from the frame meta.
    void launch(const cpu::LaunchContext& ctx) override;
};

} // namespace md
} // namespace cpu
} // namespace ATV

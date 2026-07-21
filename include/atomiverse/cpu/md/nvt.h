/// @file cpu/md/nvt.h
/// @brief cpu::md::NVT — LAMMPS `fix nvt`-style canonical-ensemble setup that
///        bundles velocity-Verlet integration with a selectable thermostat.
///
/// In LAMMPS, `fix nvt` is a single user-facing fix that performs NVE time
/// integration AND thermostatting, instead of requiring a separate `fix nve`
/// + thermostat fix. `NVT` mirrors that ergonomic: one call,
/// `sim.set_nvt(cpu::md::NVT{dt, T, tau, NVT::Thermostat::CSVR})`, configures
/// the canonical ensemble — the user no longer wires `set_integrator` +
/// `set_thermostat` separately, and CSVR is selected *under* NVT rather than as
/// a free-floating component.
///
/// `NVT` is a configuration carrier, not a component: `MD::set_nvt` expands it
/// into the existing `VelocityVerlet` integrator + the chosen thermostat
/// (`NoseHoover` or `FixCSVR`), each registered with its own validated phase
/// bindings. This is deliberate — a single bundled component cannot reproduce
/// Nose-Hoover's operator splitting (its thermostat half-steps bind to BOTH
/// `initial_integrate` and `final_integrate` at order +100, interleaved with
/// the integrator at order 0; a lone component sees only the phase, not the
/// order, so it cannot drive integrator-then-thermostat within one phase).
/// Composing the proven components keeps each thermostat's physics exactly as
/// validated standalone, while presenting the bundled LAMMPS ergonomic.
///
/// @note LAMMPS itself ships CSVR as the separate `fix temp/csvr` (layered on
/// `fix nve`), not under `fix nvt`. `NVT` unifies both thermostat choices under
/// one canonical abstraction by request — cleaner than the LAMMPS split, and
/// the standalone `VelocityVerlet` / `NoseHoover` / `FixCSVR` components plus the
/// `set_integrator` + `set_thermostat` model remain available unchanged.
#pragma once

#include <cstdint>

namespace ATV {
namespace cpu {
namespace md {

/// @brief Canonical-ensemble (NVT) configuration: velocity-Verlet + a chosen
///        thermostat, applied via `MD::set_nvt`.
struct NVT {
    /// @brief Thermostat strategy bundled under NVT.
    enum class Thermostat { NoseHoover, CSVR };

    double dt;  ///< Timestep [fs].
    double T;   ///< Target temperature [K].
    double tau; ///< Coupling time [fs] (CP2K TIMECON).
    Thermostat thermo = Thermostat::CSVR; ///< Thermostat kind.
    std::uint32_t seed = 0u;              ///< RNG seed (CSVR; ignored by NH).

    /// @brief Optional linear-ramp end temperature [K]. NaN (default) = no ramp
    /// (constant T). When set, also set @ref ramp_first / @ref ramp_last.
    double T_stop = nan_();
    std::int64_t ramp_first = -1; ///< Ramp start step (inclusive).
    std::int64_t ramp_last = -1;  ///< Ramp end step (inclusive).

    /// @brief True when a `T_start → T_stop` ramp over [ramp_first, ramp_last]
    /// is configured.
    bool has_ramp() const {
        return T_stop == T_stop /*not NaN*/ && ramp_first >= 0
               && ramp_last >= 0;
    }

  private:
    static constexpr double nan_() {
        return __builtin_nan(""); // header-only sentinel; replaced by has_ramp
    }
};

} // namespace md
} // namespace cpu
} // namespace ATV

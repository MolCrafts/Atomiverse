/// @file cpu/vib/thermochem.h
/// @brief cpu::vib::Thermochem — harmonic-oscillator / rigid-rotor
///        thermochemistry, a faithful port of PySCF
///        `pyscf/hessian/thermo.py::thermo()`.
///
/// Bound at `vib::Phase::thermochem`, after `HessianDiagonalizer` (which runs at
/// `vib::Phase::diagonalize`) has populated the normal-mode frequencies.
/// On the finalize pass it computes the electronic / translational /
/// rotational / vibrational contributions to the zero-point energy,
/// internal energy U, enthalpy H, entropy S and Gibbs free energy G,
/// each component separately (mirroring PySCF's component-wise `dict`),
/// then aggregates the totals.
///
/// All thermodynamic quantities are reported in Hartree (Eh) and Eh/K,
/// consistent with PySCF `thermo()`. The acceptance compares the
/// electronic-energy-independent thermal Gibbs correction `G_tot - E0`,
/// so the component reports U/H/G with the electronic energy set to
/// zero — `g_thermal` is exactly that correction.
///
/// The rotational symmetry number sigma is **supplied** per molecule
/// through `Thermochem::Config` — point-group / sigma auto-detection is
/// out of scope for this link.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/component/observer.h"
#include "atomiverse/cpu/vib/hessian_workspace.h"
#include "atomiverse/facade/component_role.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cpu {
namespace vib {

/// @brief Observer component computing harmonic thermochemistry.
///
/// `role() == ComponentRole::Observer` — it derives thermodynamic
/// quantities from the normal-mode frequencies and writes them into the
/// shared `HessianWorkspace`; it performs no I/O and writes no
/// `cpu::Frame` field, so `capabilities()` is empty.
class Thermochem : public cpu::Observer {
  public:
    /// @brief Per-molecule thermochemistry inputs.
    ///
    /// `temperature` and `pressure` default to standard conditions
    /// (298.15 K, 1 atm). `multiplicity` is the electronic spin
    /// multiplicity `2S+1` (1 for a closed-shell singlet).
    /// `symmetry_number` is the rotational symmetry number sigma,
    /// supplied verbatim — it is NOT auto-detected.
    struct Config {
        double temperature = 298.15; ///< Temperature [K].
        double pressure = 101325.0;  ///< Pressure [Pa].
        int multiplicity = 1;        ///< Spin multiplicity `2S+1`.
        int symmetry_number = 1;     ///< Rotational sigma.
    };

    /// @brief Construct bound to a driver-owned workspace.
    /// @param ws  Non-owning pointer to the shared Hessian workspace.
    /// @param cfg Thermochemistry inputs (temperature / pressure /
    ///            multiplicity / symmetry number).
    Thermochem(HessianWorkspace* ws, const Config& cfg) : ws_(ws), cfg_(cfg) {}

    /// @brief Construct without a workspace (test-only role check).
    ///
    /// Used by tests that only assert `role()`; `launch()` is a no-op
    /// when `ws_` is null.
    /// @param cfg Thermochemistry inputs.
    explicit Thermochem(const Config& cfg) : ws_(nullptr), cfg_(cfg) {}

    std::string name() const override { return "vib_thermochem"; }

    /// @brief Structural role — Observer (computes derived quantities).
    /// @return `ComponentRole::Observer`.
    /* role via Observer base */

    /// @brief Default binding: `vib::Phase::thermochem` @ 0.
    /// @return Single binding placing this at the thermochem phase.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::vib::Phase::thermochem, 0)};
    }

    /// @brief Compute the harmonic thermochemistry.
    ///
    /// Active only on the finalize pass (`ctx.step >= 6N`), after
    /// `HessianDiagonalizer` has produced the normal-mode frequencies. Writes
    /// `zpe`, `u_tot`, `h_tot`, `s_tot` and its components, `g_tot`,
    /// `g_thermal` into the shared workspace; `cpu::vib::Hessian::run()`
    /// copies them into `HessianResult`.
    ///
    /// @param ctx Launch context; `ctx.step` is the displacement index.
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    HessianWorkspace* ws_; ///< Non-owning; driver-owned (may be null).
    Config cfg_;           ///< Per-molecule thermochemistry inputs.
};

} // namespace vib
} // namespace cpu
} // namespace ATV

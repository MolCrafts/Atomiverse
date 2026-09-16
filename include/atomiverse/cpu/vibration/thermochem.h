/// @file cpu/vibration/thermochem.h
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
/// The rotational symmetry number sigma is **supplied** per molecule as a
/// member (`symmetry_number`) — point-group / sigma auto-detection is
/// out of scope for this link. Parameters are constructor arguments /
/// public members — no nested Config bag.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/observer.h"
#include "atomiverse/cpu/vibration/hessian_workspace.h"

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
    /// @brief Temperature [K]. Default: standard ambient 298.15 K.
    double temperature = 298.15;
    /// @brief Pressure [Pa]. Default: 1 atm.
    double pressure = 101325.0;
    /// @brief Electronic spin multiplicity `2S+1` (1 = closed-shell singlet).
    int multiplicity = 1;
    /// @brief Rotational symmetry number sigma (not auto-detected).
    int symmetry_number = 1;

    /// @brief Construct bound to a driver-owned workspace (default thermo).
    /// @param ws Non-owning pointer to the shared Hessian workspace (may be
    ///        null for test-only role checks).
    explicit Thermochem(HessianWorkspace* ws = nullptr) : ws_(ws) {}

    /// @brief Construct with thermochemistry inputs as constructor args.
    /// @param ws Non-owning workspace pointer (may be null).
    /// @param temperature Temperature [K].
    /// @param pressure Pressure [Pa].
    /// @param multiplicity Spin multiplicity `2S+1`.
    /// @param symmetry_number Rotational sigma.
    Thermochem(HessianWorkspace* ws, double temperature, double pressure,
               int multiplicity, int symmetry_number)
        : temperature(temperature), pressure(pressure),
          multiplicity(multiplicity), symmetry_number(symmetry_number),
          ws_(ws) {}

    std::string name() const override { return "vib_thermochem"; }

    /// @brief Default binding: `vib::Phase::thermochem` @ 0.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::vib::Phase::thermochem, 0)};
    }

    /// @brief Compute the harmonic thermochemistry on the finalize pass.
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    HessianWorkspace* ws_; ///< Non-owning; driver-owned (may be null).
};

} // namespace vib
} // namespace cpu
} // namespace ATV

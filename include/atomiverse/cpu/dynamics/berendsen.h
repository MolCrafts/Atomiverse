/// @file cpu/dynamics/berendsen.h
/// @brief CPU Berendsen weak-coupling thermostat.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/thermostat.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Berendsen thermostat — weak coupling to an external bath.
///
/// Berendsen, Postma, van Gunsteren, DiNola, Haak, "Molecular dynamics with
/// coupling to an external bath", *J. Chem. Phys.* **81**, 3684 (1984),
/// DOI [10.1063/1.448118](https://doi.org/10.1063/1.448118). LAMMPS
/// `fix temp/berendsen`; DFTB+ `Thermostat = Berendsen`.
///
/// Once per step every velocity is scaled by the single global factor
///
/// \f[
///   \lambda = \sqrt{1 + \frac{\Delta t}{\tau}\left(\frac{T_0}{T} - 1\right)}
/// \f]
///
/// with \f$T\f$ the instantaneous temperature from the global kinetic energy
/// over \f$N_f\f$ degrees of freedom.
///
/// ## This is not a canonical thermostat
///
/// Berendsen suppresses the kinetic-energy fluctuations that the canonical
/// ensemble requires, and the suppression is not uniform across degrees of
/// freedom — the "flying ice cube", where energy drains into translation.
/// **`FixCSVR` is the same weak-coupling idea done right** (Bussi's stochastic
/// term restores the correct fluctuations and yields a genuine conserved
/// quantity) and is the better choice for new work.
///
/// This class exists because published protocols specify Berendsen and a
/// reproduction has to run the thermostat the protocol names, not a better
/// one — reproducing DFTB+ / CP2K trajectories, or the DFTB+ NPT/NVT
/// equilibration in Hooven et al., *J. Chem. Theory Comput.* **22**, 6917
/// (2026). Use it to match a protocol; do not use it to sample.
///
/// ## Units
///
/// \f$\tau\f$ is [fs] on this surface, as everywhere else in `cpu/dynamics`.
/// DFTB+ and CP2K spell the same quantity in **ps** — a DFTB+
/// `Timescale [ps] = 0.1` is `tau = 100.0` here.
///
/// ## Bookkeeping
///
/// The running \f$-\sum \Delta K\f$ is accumulated and published to
/// `kConserved` alongside the potential and kinetic energy, matching what
/// LAMMPS folds into `econserve`. Read it as the thermostat's cumulative work,
/// **not** as a conserved quantity: Berendsen dynamics does not come from a
/// Hamiltonian, so a flat sum here is a diagnostic, not evidence of a good
/// ensemble.
class Berendsen : public cpu::Thermostat {
  public:
    /// @param T_start Target / ramp-start temperature [K]; must be > 0.
    /// @param tau     Coupling time constant [fs]; must be > 0. (DFTB+ and
    ///                CP2K quote this in ps.)
    /// @param dt      Timestep [fs]; must be > 0.
    /// @throws std::invalid_argument on a non-positive argument.
    Berendsen(double T_start, double tau, double dt);

    std::string name() const override { return "berendsen"; }

    /// @brief Applied once per step, after the velocity-Verlet update — the
    ///        same slot `FixCSVR` occupies.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +100)};
    }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Set the ramp end temperature [K]; enables the linear ramp over
    ///        the span set by `set_run_span`. Default `= T_start`.
    /// @return `*this`, for chaining before registration.
    Berendsen& set_T_stop(double T_stop) {
        T_stop_ = T_stop;
        return *this;
    }

    /// @brief Set the ramp run span `[nfirst, nlast]` in MD step indices.
    /// @return `*this`, for chaining before registration.
    Berendsen& set_run_span(std::int64_t nfirst, std::int64_t nlast) {
        nfirst_ = nfirst;
        nlast_ = nlast;
        return *this;
    }

    /// @return Live thermostat setpoint [K] — for diagnostics / observers.
    double T_target() const noexcept { return T_live_; }

    /// @brief Accumulated thermostat work \f$-\sum\Delta K\f$ [eV].
    ///        Bookkeeping, not a conserved quantity — see the class note.
    double thermostat_energy() const noexcept { return thermostat_energy_; }

  private:
    /// @brief Interpolate the live setpoint for `step` (no-ramp default
    ///        returns `T_start_`), mirroring `FixCSVR::compute_temp_target`.
    double compute_temp_target(std::int64_t step) const;

    double T_start_;               ///< Ramp start / target temperature [K].
    double T_stop_;                ///< Ramp end temperature [K].
    double tau_;                   ///< Coupling time constant [fs].
    double dt_;                    ///< Timestep [fs].
    int n_dof_global_ = 0;         ///< Constraint-aware global DOF count.
    double thermostat_energy_ = 0; ///< Accumulated −Σ ΔK [eV].

    std::int64_t nfirst_ = 0; ///< First MD step of the run (ramp span).
    std::int64_t nlast_ = 0;  ///< Last MD step of the run (ramp span).
    double T_live_ = 0.0;     ///< Live setpoint [K], interpolated each step.
};

} // namespace md
} // namespace cpu
} // namespace ATV

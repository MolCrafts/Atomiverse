/// @file core/thermo_style.h
/// @brief LAMMPS `thermo_style custom` keywords shared by the CPU and CUDA
///        thermo recorders, and the energy/virial demand they imply.
///
/// A thermo line prints only what its keywords name, and the keywords are
/// the single source of what the force field must evaluate on a thermo step:
/// `pe`/`etotal`/`econserve` need the potential energy, the pressure
/// keywords need the virial (LAMMPS `Integrate::ev_set`).
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/precision.h"

namespace ATV {

/// @brief One `thermo_style custom` column.
enum class ThermoKeyword {
    Step,      ///< `step` — timestep index.
    Temp,      ///< `temp` — 2 KE / (Ndof boltz) [temperature].
    Pe,        ///< `pe` — potential energy [energy].
    Ke,        ///< `ke` — kinetic energy [energy].
    Etotal,    ///< `etotal` — pe + ke [energy].
    Econserve, ///< `econserve` — integrator conserved quantity [energy].
    Press,     ///< `press` — isotropic pressure [pressure].
    Pxx,       ///< `pxx` — pressure tensor xx [pressure].
    Pyy,       ///< `pyy` — pressure tensor yy [pressure].
    Pzz,       ///< `pzz` — pressure tensor zz [pressure].
    Pxy,       ///< `pxy` — pressure tensor xy [pressure].
    Pxz,       ///< `pxz` — pressure tensor xz [pressure].
    Pyz,       ///< `pyz` — pressure tensor yz [pressure].
    Vol,       ///< `vol` — cell volume [volume].
};

/// @brief One sampled step, as the recorder captured it, in the run's
///        LAMMPS `units` style (every column is printed in that style).
struct ThermoSample {
    BigInt step = 0;
    double pe = 0;          ///< [energy]
    double ke = 0;          ///< [energy]
    double conserved = 0;   ///< [energy]
    double volume = 0;      ///< [volume]
    double virial[6] = {};  ///< Extensive pair/bonded virial, Voigt [energy].
    double kinetic[6] = {}; ///< Extensive kinetic stress, Voigt [energy].
    BigInt ndof = 0;
    double boltz = 1.0;  ///< The run's boltz (`temp`).
    double nktv2p = 1.0; ///< The run's nktv2p (pressure columns).
};

/// @brief Parsed `thermo_style custom` keyword list.
class ThermoStyle {
  public:
    /// @brief LAMMPS columns printed when no style is given:
    ///        `step temp pe ke etotal`.
    ThermoStyle();

    /// @brief Parse keywords; throws `std::invalid_argument` naming an
    ///        unknown or repeated keyword, or an empty list.
    explicit ThermoStyle(const std::vector<std::string>& keywords);

    const std::vector<ThermoKeyword>& columns() const { return columns_; }

    /// @brief A printed column needs the potential energy this step.
    bool needs_energy() const;
    /// @brief A printed column needs the virial this step.
    bool needs_virial() const;

    /// @brief LAMMPS column headers (`Step Temp PotEng ...`).
    std::string header() const;
    /// @brief One formatted line for a sample.
    std::string line(const ThermoSample& sample) const;

  private:
    std::vector<ThermoKeyword> columns_;
};

} // namespace ATV

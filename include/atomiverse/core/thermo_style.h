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
    Temp,      ///< `temp` — 2 KE / (Ndof kB) [K].
    Pe,        ///< `pe` — potential energy [eV].
    Ke,        ///< `ke` — kinetic energy [eV].
    Etotal,    ///< `etotal` — pe + ke [eV].
    Econserve, ///< `econserve` — integrator conserved quantity [eV].
    Press,     ///< `press` — isotropic pressure [bar].
    Pxx,       ///< `pxx` — pressure tensor xx [bar].
    Pyy,       ///< `pyy` — pressure tensor yy [bar].
    Pzz,       ///< `pzz` — pressure tensor zz [bar].
    Pxy,       ///< `pxy` — pressure tensor xy [bar].
    Pxz,       ///< `pxz` — pressure tensor xz [bar].
    Pyz,       ///< `pyz` — pressure tensor yz [bar].
    Vol,       ///< `vol` — cell volume [Å³].
};

/// @brief One sampled step, as the recorder captured it.
struct ThermoSample {
    BigInt step = 0;
    double pe = 0;          ///< [eV]
    double ke = 0;          ///< [eV]
    double conserved = 0;   ///< [eV]
    double volume = 0;      ///< [Å³]
    double virial[6] = {};  ///< Extensive pair/bonded virial, Voigt [eV].
    double kinetic[6] = {}; ///< Extensive kinetic stress, Voigt [eV].
    BigInt ndof = 0;
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

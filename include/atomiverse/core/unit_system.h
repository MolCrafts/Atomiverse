/// @file core/unit_system.h
/// @brief LAMMPS `units` styles: the run-level unit system of classical MD.
///
/// The factor table is `Update::set_units` from LAMMPS `src/update.cpp`
/// (2 Sep 2026): lj, real, metal, si, cgs, electron, micro, nano, with every
/// `Force` factor that command sets (`boltz`, `hplanck`, `mvv2e`, `ftm2v`,
/// `mv2d`, `nktv2p`, `qqr2e`, `qe2f`, `vxmu2f`, `xxt2kmu`, `e_mass`,
/// `hhmrr2e`, `mvh2r`, `angstrom`, `femtosecond`, `qelectron`) and the two
/// defaults it resets (`update->dt`, `neighbor->skin`). `qqrd2e = qqr2e /
/// dielectric` is `Force::init`; `two_charge_force` is `KSpace::two_charge`.
///
/// A run selects one style, exactly like the `units` command: the driver
/// owns it (`Driver::set_units`, default lj as in LAMMPS), publishes it on
/// the execution Context before any component builds, and refuses a change
/// once the run has bound a system (LAMMPS refuses `units` after the box
/// exists). Every classical factor a pair style, k-space solver,
/// integrator, thermostat, barostat or observer multiplies is read from that
/// Context (`ctx.runtime.units`); CUDA kernels receive the factors as launch
/// arguments. There is no process-global style.
///
/// The CODATA eV·Å·fs identities in `units.h` (`coulomb`, `kB_ev_per_k`,
/// `h_planck_ev_fs`, ...) belong to the electronic-structure Hamiltonians
/// (SCF, PW, DFTB, NDDO), which work in their own units; they are not the
/// LAMMPS metal row and classical MD never multiplies them.
#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ATV::units {

/// @brief LAMMPS `units` style. The byte value is the task-wire encoding.
enum class Style : std::uint8_t {
    lj = 0,
    real = 1,
    metal = 2,
    si = 3,
    cgs = 4,
    electron = 5,
    micro = 6,
    nano = 7,
};

/// Number of styles (wire range check).
inline constexpr int kStyleCount = 8;

/// LAMMPS spelling of each style, indexed by `Style`.
inline constexpr std::array<const char*, kStyleCount> kStyleNames = {
    "lj", "real", "metal", "si", "cgs", "electron", "micro", "nano"};

/// @brief LAMMPS name of `s` (e.g. `"real"`).
[[nodiscard]] constexpr const char* style_name(Style s) {
    return kStyleNames[static_cast<std::uint8_t>(s)];
}

/// @brief Style named `name`, spelled as in the LAMMPS `units` command.
/// @throws std::invalid_argument on an unknown name.
[[nodiscard]] inline Style parse_style(std::string_view name) {
    for (int i = 0; i < kStyleCount; ++i)
        if (name == kStyleNames[i])
            return static_cast<Style>(i);
    throw std::invalid_argument(
        "units: unknown LAMMPS style '" + std::string(name)
        + "' (lj, real, metal, si, cgs, electron, micro, nano)");
}

/// @brief One LAMMPS `units` style, in that style's own factors.
///
/// Field names and values are LAMMPS `Force` members after
/// `Update::set_units`. `e_mass`, `hhmrr2e` and `mvh2r` are 0 where LAMMPS
/// leaves them "not yet set".
struct UnitSystem {
    Style style = Style::lj;
    double boltz = 1.0;       ///< Boltzmann constant [energy/temperature].
    double hplanck = 1.0;     ///< Planck constant [energy·time].
    double mvv2e = 1.0;       ///< mass·velocity² → energy.
    double ftm2v = 1.0;       ///< force/mass·time → velocity.
    double mv2d = 1.0;        ///< mass/volume → density.
    double nktv2p = 1.0;      ///< N·k·T/V → pressure.
    double qqr2e = 1.0;       ///< q·q/r → energy.
    double qe2f = 1.0;        ///< q·E-field → force.
    double vxmu2f = 1.0;      ///< velocity × dipole → force (magnetic).
    double xxt2kmu = 1.0;     ///< length²/time → kinematic viscosity·μ.
    double e_mass = 0.0;      ///< Electron mass [mass].
    double hhmrr2e = 0.0;     ///< ħ²/(m·r²) → energy.
    double mvh2r = 0.0;       ///< m·v/ħ → 1/length.
    double angstrom = 1.0;    ///< 1 Å in the style's length.
    double femtosecond = 1.0; ///< 1 fs in the style's time.
    double qelectron = 1.0;   ///< Elementary charge in the style's charge.
    double dielectric = 1.0;  ///< LAMMPS `dielectric` (default 1).
    double dt = 0.005;        ///< Default `timestep` [style time].
    double skin = 0.3;        ///< Default `neighbor` skin [style length].

    /// @brief LAMMPS name of this style.
    [[nodiscard]] constexpr const char* name() const {
        return style_name(style);
    }

    /// @brief LAMMPS `qqrd2e = qqr2e / dielectric` (`Force::init`).
    [[nodiscard]] constexpr double qqrd2e() const { return qqr2e / dielectric; }

    /// @brief LAMMPS `KSpace::two_charge`: force between two unit charges one
    ///        Å apart, in the style's force unit.
    [[nodiscard]] constexpr double two_charge_force() const {
        return qqr2e * qelectron * qelectron / (angstrom * angstrom);
    }

    [[nodiscard]] static constexpr UnitSystem lj() { return UnitSystem{}; }
};

/// @brief LAMMPS `Update::set_units(style)`.
[[nodiscard]] constexpr UnitSystem from_style(Style s) {
    UnitSystem u;
    u.style = s;
    switch (s) {
    case Style::lj:
        break;
    case Style::real:
        u.boltz = 0.0019872067;
        u.hplanck = 95.306976368;
        u.mvv2e = 48.88821291 * 48.88821291;
        u.ftm2v = 1.0 / 48.88821291 / 48.88821291;
        u.mv2d = 1.0 / 0.602214129;
        u.nktv2p = 68568.415;
        u.qqr2e = 332.06371;
        u.qe2f = 23.060549;
        u.vxmu2f = 1.4393264316e4;
        u.xxt2kmu = 0.1;
        u.e_mass = 1.0 / 1836.1527556560675;
        u.hhmrr2e = 0.0957018663603261;
        u.mvh2r = 1.5339009481951;
        u.angstrom = 1.0;
        u.femtosecond = 1.0;
        u.qelectron = 1.0;
        u.dt = 1.0;
        u.skin = 2.0;
        break;
    case Style::metal:
        u.boltz = 8.617343e-5;
        u.hplanck = 4.135667403e-3;
        u.mvv2e = 1.0364269e-4;
        u.ftm2v = 1.0 / 1.0364269e-4;
        u.mv2d = 1.0 / 0.602214129;
        u.nktv2p = 1.6021765e6;
        u.qqr2e = 14.399645;
        u.qe2f = 1.0;
        u.vxmu2f = 0.6241509647;
        u.xxt2kmu = 1.0e-4;
        u.angstrom = 1.0;
        u.femtosecond = 1.0e-3;
        u.qelectron = 1.0;
        u.dt = 0.001;
        u.skin = 2.0;
        break;
    case Style::si:
        u.boltz = 1.3806504e-23;
        u.hplanck = 6.62606896e-34;
        u.qqr2e = 8.9876e9;
        u.angstrom = 1.0e-10;
        u.femtosecond = 1.0e-15;
        u.qelectron = 1.6021765e-19;
        u.dt = 1.0e-8;
        u.skin = 0.001;
        break;
    case Style::cgs:
        u.boltz = 1.3806504e-16;
        u.hplanck = 6.62606896e-27;
        u.qqr2e = 1.0;
        u.angstrom = 1.0e-8;
        u.femtosecond = 1.0e-15;
        u.qelectron = 4.8032044e-10;
        u.dt = 1.0e-8;
        u.skin = 0.1;
        break;
    case Style::electron:
        u.boltz = 3.16681534e-6;
        u.hplanck = 0.1519829846;
        u.mvv2e = 1.06657236;
        u.ftm2v = 0.937582899;
        u.mv2d = 1.0;
        u.nktv2p = 2.94210108e13;
        u.qqr2e = 1.0;
        u.qe2f = 1.94469051e-10;
        u.vxmu2f = 3.39893149e1;
        u.xxt2kmu = 3.13796367e-2;
        u.angstrom = 1.88972612;
        u.femtosecond = 1.0;
        u.qelectron = 1.0;
        u.dt = 0.001;
        u.skin = 2.0;
        break;
    case Style::micro:
        u.boltz = 1.3806504e-8;
        u.hplanck = 6.62606896e-13;
        u.qqr2e = 8.987556e6;
        u.angstrom = 1.0e-4;
        u.femtosecond = 1.0e-9;
        u.qelectron = 1.6021765e-7;
        u.dt = 2.0;
        u.skin = 0.1;
        break;
    case Style::nano:
        u.boltz = 0.013806504;
        u.hplanck = 6.62606896e-4;
        u.qqr2e = 230.7078669;
        u.angstrom = 1.0e-1;
        u.femtosecond = 1.0e-6;
        u.qelectron = 1.0;
        u.dt = 0.00045;
        u.skin = 0.1;
        break;
    }
    return u;
}

/// @brief LAMMPS `Update::set_units` for a style spelled by name.
/// @throws std::invalid_argument on an unknown name.
[[nodiscard]] inline UnitSystem from_style(std::string_view name) {
    return from_style(parse_style(name));
}

static_assert(UnitSystem::lj().two_charge_force() == 1.0);
static_assert(UnitSystem::lj().qqrd2e() == 1.0);
static_assert(from_style(Style::real).qqr2e == 332.06371);

} // namespace ATV::units

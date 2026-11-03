/// @file core/unit_system.h
/// @brief LAMMPS `units` styles. Classical MD multiplies the lj row.
///
/// The factor table is `Update::set_units` from LAMMPS `src/update.cpp`
/// (lj, real, metal, si, cgs, electron, micro, nano): `boltz`, `mvv2e`,
/// `ftm2v`, `qqr2e`, `angstrom`, `femtosecond`, `qelectron`, and
/// `KSpace::two_charge`. `qqrd2e = qqr2e / dielectric` with dielectric 1,
/// matching a run that never called `pair_modify dielectric`.
///
/// Classical pair Coulomb, PPPM/PME/Ewald, the slab correction,
/// electrodes, thermostats, integrators, PIMD and the thermal wavelength
/// multiply `md_qqr2e`, `md_mvv2e`, `md_boltz`, `md_ftm2v` and
/// `md_hplanck`. Those five are the lj row (all 1). They are not the
/// metal CODATA constants in `units.h` (`coulomb`, `kB_ev_per_k`,
/// `mass_v2_to_energy`, `force_to_accel`, `h_planck_ev_fs`), which stay
/// the eV·Å·fs identities used by electronic-structure Hamiltonians
/// (SCF, PW, DFTB, NDDO, Hartree).
///
/// `set_style` accepts only `"lj"`. Other rows are built with
/// `from_style` and passed explicitly into the PPPM estimator so a
/// `units metal` LAMMPS log can still be reproduced. The mesh that
/// estimator returns is LAMMPS `PPPM::init`. A later volume change
/// refreshes the Green's function at those grid counts (LAMMPS
/// `PPPM::setup` from `fix npt`) and does not call `set_grid_global`.
#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace ATV::units {

/// @brief One LAMMPS `units` style, in that style's own factors.
struct UnitSystem {
    const char* style = "lj";
    double boltz = 1.0;
    double hplanck = 1.0;
    double mvv2e = 1.0;
    double ftm2v = 1.0;
    double mv2d = 1.0;
    double nktv2p = 1.0;
    double qqr2e = 1.0;
    double qe2f = 1.0;
    double vxmu2f = 1.0;
    double xxt2kmu = 1.0;
    double angstrom = 1.0;
    double femtosecond = 1.0;
    double qelectron = 1.0;
    double dielectric = 1.0;
    /// LAMMPS default `timestep` after `units <style>` [style time].
    double dt = 0.005;
    /// LAMMPS default `neighbor` skin after `units <style>` [style length].
    double skin = 0.3;

    /// LAMMPS `qqrd2e = qqr2e / dielectric`.
    [[nodiscard]] constexpr double qqrd2e() const { return qqr2e / dielectric; }

    /// LAMMPS `KSpace::two_charge`: force between two unit charges one
    /// length-unit apart, in the style's force unit.
    [[nodiscard]] constexpr double two_charge_force() const {
        return qqr2e * qelectron * qelectron / (angstrom * angstrom);
    }

    [[nodiscard]] static constexpr UnitSystem lj() {
        return UnitSystem{"lj", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   1.0, 1.0,
                          1.0,  1.0, 1.0, 1.0, 1.0, 1.0, 0.005, 0.3};
    }
};

/// Classical-engine coupling. Equal to `UnitSystem::lj()`; kernels read
/// these so the lj row is the value, not a literal typed at the call.
inline constexpr double md_qqr2e = 1.0;
inline constexpr double md_mvv2e = 1.0;
inline constexpr double md_boltz = 1.0;
inline constexpr double md_ftm2v = 1.0;
inline constexpr double md_hplanck = 1.0;

static_assert(UnitSystem::lj().qqr2e == md_qqr2e);
static_assert(UnitSystem::lj().mvv2e == md_mvv2e);
static_assert(UnitSystem::lj().boltz == md_boltz);
static_assert(UnitSystem::lj().ftm2v == md_ftm2v);
static_assert(UnitSystem::lj().hplanck == md_hplanck);
static_assert(UnitSystem::lj().two_charge_force() == 1.0);
static_assert(UnitSystem::lj().qqrd2e() == 1.0);

/// @brief LAMMPS `Update::set_units` for `style`.
/// @throws std::invalid_argument on an unknown style.
[[nodiscard]] inline UnitSystem from_style(std::string_view style) {
    if (style == "lj")
        return UnitSystem::lj();
    if (style == "real")
        return UnitSystem{"real",
                          0.0019872067,
                          95.306976368,
                          48.88821291 * 48.88821291,
                          1.0 / 48.88821291 / 48.88821291,
                          1.0 / 0.602214129,
                          68568.415,
                          332.06371,
                          23.060549,
                          1.4393264316e4,
                          0.1,
                          1.0,
                          1.0,
                          1.0,
                          1.0,
                          1.0,
                          2.0};
    if (style == "metal")
        return UnitSystem{"metal",
                          8.617343e-5,
                          4.135667403e-3,
                          1.0364269e-4,
                          1.0 / 1.0364269e-4,
                          1.0 / 0.602214129,
                          1.6021765e6,
                          14.399645,
                          1.0,
                          0.6241509647,
                          1.0e-4,
                          1.0,
                          1.0e-3,
                          1.0,
                          1.0,
                          0.001,
                          2.0};
    if (style == "si")
        return UnitSystem{
            "si",   1.3806504e-23, 6.62606896e-34, 1.0,           1.0,
            1.0,    1.0,           8.9876e9,       1.0,           1.0,
            1.0,    1.0e-10,       1.0e-15,        1.6021765e-19, 1.0,
            1.0e-8, 0.001};
    if (style == "cgs")
        return UnitSystem{"cgs",
                          1.3806504e-16,
                          6.62606896e-27,
                          1.0,
                          1.0,
                          1.0,
                          1.0,
                          1.0,
                          1.0,
                          1.0,
                          1.0,
                          1.0e-8,
                          1.0e-15,
                          4.8032044e-10,
                          1.0,
                          1.0e-8,
                          0.1};
    if (style == "electron")
        return UnitSystem{"electron",
                          3.16681534e-6,
                          0.1519829846,
                          1.06657236,
                          0.937582899,
                          1.0,
                          2.94210108e13,
                          1.0,
                          1.94469051e-10,
                          3.39893149e1,
                          3.13796367e-2,
                          1.88972612,
                          1.0,
                          1.0,
                          1.0,
                          0.001,
                          2.0};
    if (style == "micro")
        return UnitSystem{
            "micro", 1.3806504e-8, 6.62606896e-13, 1.0,          1.0,
            1.0,     1.0,          8.987556e6,     1.0,          1.0,
            1.0,     1.0e-4,       1.0e-9,         1.6021765e-7, 1.0,
            2.0,     0.1};
    if (style == "nano")
        return UnitSystem{"nano",  0.013806504, 6.62606896e-4, 1.0, 1.0,
                          1.0,     1.0,         230.7078669,   1.0, 1.0,
                          1.0,     1.0e-1,      1.0e-6,        1.0, 1.0,
                          0.00045, 0.1};
    throw std::invalid_argument(
        "units: unknown LAMMPS style '" + std::string(style)
        + "' (lj, real, metal, si, cgs, electron, micro, nano)");
}

inline UnitSystem& active_slot() {
    static UnitSystem slot = UnitSystem::lj();
    return slot;
}

/// Style selected for the process. The classical engine multiplies the lj
/// row regardless; this is lj unless `set_style("lj")` ran.
[[nodiscard]] inline const UnitSystem& active() { return active_slot(); }

/// @brief LAMMPS `units` command for the classical engine.
/// @throws std::invalid_argument when `style` is not `lj`. The other rows
///         exist for an explicit `from_style` estimate. The running pair,
///         k-space and integrator multiply the lj row.
inline void set_style(std::string_view style) {
    if (style != "lj")
        throw std::invalid_argument(
            "classical MD runs in LAMMPS lj units; '" + std::string(style)
            + "' is a from_style row for an explicit PPPM estimate, not the "
              "engine coupling");
    active_slot() = from_style(style);
}

} // namespace ATV::units

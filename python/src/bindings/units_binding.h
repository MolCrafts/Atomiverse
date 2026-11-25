/// @file python/src/bindings/units_binding.h
/// @brief `set_units` / `units_style` on every bound driver (LAMMPS `units`).
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <optional>

#include "atomiverse/core/unit_system.h"

#include <stdexcept>
#include <string>

namespace ATV::python {

/// @brief `style`, or — when None — `atomiverse.units_style()`, the style a
///        driver built without `units=` adopts (the Python `units` command).
/// @throws nanobind::value_error on an unknown LAMMPS style name.
inline std::string resolve_units(const std::optional<std::string>& style) {
    std::string name = style
                           ? *style
                           : nanobind::cast<std::string>(
                                 nanobind::module_::import_("atomiverse._units")
                                     .attr("units_style")());
    try {
        (void)units::parse_style(name);
    } catch (const std::invalid_argument& e) {
        throw nanobind::value_error(e.what());
    }
    return name;
}

/// @brief Bind LAMMPS `units <style>` on a driver class.
/// @param cls The nanobind class of the handle.
/// @param driver_of `Handle& -> Driver&` (the driver that owns the style).
/// @param guard `const Handle&, const char* setter` refusing after a run.
template <class Class, class DriverOf, class Guard>
void bind_units(Class& cls, DriverOf driver_of, Guard guard) {
    using Handle = typename Class::Type;
    cls.def(
           "set_units",
           [driver_of, guard](Handle& self, const std::string& style) {
               guard(self, "set_units");
               try {
                   driver_of(self).set_units(style);
               } catch (const std::invalid_argument& e) {
                   throw nanobind::value_error(e.what());
               }
           },
           nanobind::arg("style"),
           "LAMMPS `units <style>` of this run: lj, real, metal, si, cgs, "
           "electron, micro or nano (default lj). Every parameter, the "
           "timestep, temperatures and pressures use that style. Annotated "
           "Frames are converted on import and results restored to their "
           "original units. Fixed once the run has bound its system.")
        .def_prop_ro(
            "units_style",
            [driver_of](Handle& self) {
                return std::string(driver_of(self).units().name());
            },
            "LAMMPS `units` style of this run.");
}

} // namespace ATV::python

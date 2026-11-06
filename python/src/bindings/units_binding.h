/// @file python/src/bindings/units_binding.h
/// @brief `set_units` / `units_style` on every bound driver (LAMMPS `units`).
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <stdexcept>
#include <string>

namespace ATV::python {

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
           "timestep, temperatures, pressures and the frame are taken in "
           "that style and every result is reported in it; nothing is "
           "converted. Fixed once the run has bound its system.")
        .def_prop_ro(
            "units_style",
            [driver_of](Handle& self) {
                return std::string(driver_of(self).units().name());
            },
            "LAMMPS `units` style of this run.");
}

} // namespace ATV::python

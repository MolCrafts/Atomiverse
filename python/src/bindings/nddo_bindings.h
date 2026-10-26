/// @brief Shared Python interface for concrete native CPU/CUDA NDDO methods.
#pragma once
#include "atomiverse/cpu/frame/molpy_exchange.h"
#include "atomiverse/cpu/task/launcher_gate.h"
#include "force_array.h"
#include "molpy_frame_bridge.h"
#include <nanobind/nanobind.h>

// Only Python marshaling is shared: every operation invokes the concrete
// method's public Frame boundary. No solver, parameters or runtime are copied.
// DriverNode is a nonzero-offset base of these multiply inherited methods.
// Nanobind cannot adjust that cast; mount them through concrete typed setters.
template <class Method>
void bind_nddo_method(nb::module_& m, const char* name, const char* device,
                      bool local_forces_in_launcher = false) {
    nb::class_<Method>(m, name)
        .def(nb::init<int, int, int, double, double>(), nb::arg("charge") = 0,
             nb::arg("multiplicity") = 1, nb::arg("max_iter") = 200,
             nb::arg("density_tol") = 1e-8, nb::arg("energy_tol") = 1e-10)
        .def_ro("charge", &Method::charge)
        .def_ro("multiplicity", &Method::multiplicity)
        .def_rw("max_iter", &Method::max_iter)
        .def_rw("density_tol", &Method::density_tol)
        .def_rw("energy_tol", &Method::energy_tol)
        .def_prop_ro("device", [device](const Method&) { return device; })
        .def(
            "run",
            [](Method& method, nb::object frame_obj) {
                if (ATV::cpu::task::LauncherGate{}.active())
                    throw std::runtime_error(
                        "NDDO.run has no MPI task payload; use a local run");
                auto frame = borrow_molpy_frame(frame_obj);
                auto work = ATV::cpu::from_molpy(*frame);
                {
                    nb::gil_scoped_release release;
                    method.run(work);
                }
                ATV::cpu::to_molpy(work, *frame);
            },
            nb::arg("frame"),
            "Overwrite energy and charges; preserve existing forces.")
        .def(
            "forces",
            [local_forces_in_launcher](Method& method, nb::object frame_obj) {
                // AM1/RM1 retain their explicitly local legacy CPU force entry.
                // Newly exposed methods and CUDA have no MPI force payload.
                if (!local_forces_in_launcher
                    && ATV::cpu::task::LauncherGate{}.active())
                    throw std::runtime_error(
                        "NDDO forces have no MPI task payload; run locally");
                auto frame = borrow_molpy_frame(frame_obj);
                auto work = ATV::cpu::from_molpy(*frame);
                {
                    nb::gil_scoped_release release;
                    method.forces(work);
                }
                ATV::cpu::to_molpy(work, *frame);
                return to_force_array(method.last_result().forces_ev_a);
            },
            nb::arg("frame"),
            "Overwrite energy, charges and analytic forces; return an owned "
            "(N,3) force array [eV/A].")
        .def("last_result", &Method::last_result,
             nb::rv_policy::reference_internal,
             "Current method results; CUDA downloads current device results at "
             "this explicit read boundary.")
        .def("has_density", &Method::has_density)
        .def("shutdown", &Method::shutdown);
}

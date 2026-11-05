/// @file python/bind.cpp
/// @brief Top-level nanobind module definition for _atomiverse.
///
/// Atomiverse binds the driver / component API only — never a Frame data
/// type. Simulation state crosses the boundary as a `molpy.Frame` (from
/// molpy's own `molpy._core` extension); each `run()` borrows it via a PyCapsule.
///
/// Per-domain init functions are compiled only when the matching
/// ATV_BUILD_* product module is ON (see cmake/AtvProductModules.cmake).

#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/driver/driver_node.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/molpy_exchange.h"
#include "atomiverse/cpu/task/launcher_gate.h"
#include "molpy_frame_bridge.h"
#include <nanobind/nanobind.h>
#ifdef ATV_HAS_SEMIEMPIRICAL
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/cpu/semiempirical/mndo_solver.h"
#include "atomiverse/cpu/semiempirical/pm3_solver.h"
#include "atomiverse/cpu/semiempirical/rm1_solver.h"
#endif

namespace nb = nanobind;

#ifdef ATV_HAS_POTENTIALS
void bind_potentials(nb::module_& m);
#endif
#ifdef ATV_HAS_MD
void bind_md(nb::module_& m);
#endif
#ifdef ATV_HAS_MC
void bind_mc(nb::module_& m);
#endif
#ifdef ATV_HAS_MINIMIZER
void bind_minimizer(nb::module_& m);
#endif
#ifdef ATV_HAS_VIB
void bind_vib(nb::module_& m);
#endif
#ifdef ATV_HAS_PW
void bind_pw(nb::module_& m);
#endif
#ifdef ATV_HAS_SEMIEMPIRICAL
void bind_semiempirical(nb::module_& m);
#ifdef ATV_HAS_CUDA
void bind_cuda(nb::module_& m);
#endif
#endif
#ifdef ATV_HAS_HF
void bind_scf(nb::module_& m);
#ifdef ATV_HAS_CUDA
void bind_cuda_scf(nb::module_& m);
#endif
#endif

NB_MODULE(_atomiverse, m) {
    m.doc() = "Atomiverse: molecular simulation compute backend";
    nb::class_<ATV::cpu::Component>(m, "_Component");
    nb::class_<ATV::cpu::DriverNode>(m, "_DriverNode");
    m.def("_single_point_component",
          [](nb::object frame_obj, ATV::cpu::Component& evaluator) {
              if (ATV::cpu::task::LauncherGate{}.active())
                  throw std::runtime_error(
                      "this evaluator has no MPI single-point task payload; "
                      "run without the launcher");
              auto frame = borrow_molpy_frame(frame_obj);
              auto work = ATV::cpu::from_molpy(*frame);
              {
                  nb::gil_scoped_release release;
                  ATV::cpu::single_point(work, evaluator);
              }
              ATV::cpu::to_molpy(work, *frame);
          });

    nb::class_<ATV::cpu::Evaluate>(m, "_Evaluate")
        .def(nb::init<>())
#ifdef ATV_HAS_SEMIEMPIRICAL
        .def(
            "set_evaluator",
            [](ATV::cpu::Evaluate& self, ATV::cpu::semiempirical::AM1& method) {
                self.set_evaluator_ref(method);
            },
            nb::arg("method"), nb::keep_alive<1, 2>())
        .def(
            "set_evaluator",
            [](ATV::cpu::Evaluate& self, ATV::cpu::semiempirical::RM1& method) {
                self.set_evaluator_ref(method);
            },
            nb::arg("method"), nb::keep_alive<1, 2>())
        .def(
            "set_evaluator",
            [](ATV::cpu::Evaluate& self,
               ATV::cpu::semiempirical::MNDO& method) {
                self.set_evaluator_ref(method);
            },
            nb::arg("method"), nb::keep_alive<1, 2>())
        .def(
            "set_evaluator",
            [](ATV::cpu::Evaluate& self, ATV::cpu::semiempirical::PM3& method) {
                self.set_evaluator_ref(method);
            },
            nb::arg("method"), nb::keep_alive<1, 2>())
#endif
        .def(
            "set_evaluator",
            [](ATV::cpu::Evaluate& self, ATV::cpu::DriverNode& method) {
                self.set_evaluator_ref(method);
            },
            nb::arg("method"), nb::keep_alive<1, 2>())
        .def(
            "set_evaluator",
            [](ATV::cpu::Evaluate& self, ATV::cpu::Component& method) {
                self.set_evaluator_ref(method);
            },
            nb::arg("method"), nb::keep_alive<1, 2>())
        .def("build", [](ATV::cpu::Evaluate& self) { self.build(); })
        .def(
            "run",
            [](ATV::cpu::Evaluate& self, nb::object obj) {
                if (ATV::cpu::task::LauncherGate{}.active())
                    throw std::runtime_error(
                        "Evaluate has no MPI task payload; run locally");
                auto frame = borrow_molpy_frame(obj);
                auto work = ATV::cpu::from_molpy(*frame);
                {
                    nb::gil_scoped_release release;
                    self.run(work);
                }
                ATV::cpu::to_molpy(work, *frame);
            },
            nb::arg("frame"));

    // Force-field parameter types first: every driver binding takes them.
#ifdef ATV_HAS_POTENTIALS
    bind_potentials(m);
#endif
#ifdef ATV_HAS_MD
    bind_md(m);
#endif
#ifdef ATV_HAS_MC
    bind_mc(m);
#endif
#ifdef ATV_HAS_MINIMIZER
    bind_minimizer(m);
#endif
#ifdef ATV_HAS_VIB
    bind_vib(m);
#endif
#ifdef ATV_HAS_PW
    bind_pw(m);
#endif
#ifdef ATV_HAS_SEMIEMPIRICAL
    bind_semiempirical(m);
#ifdef ATV_HAS_CUDA
    bind_cuda(m);
#endif
#endif
#ifdef ATV_HAS_HF
    bind_scf(m);
#ifdef ATV_HAS_CUDA
    bind_cuda_scf(m);
#endif
#endif
}

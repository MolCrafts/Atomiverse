/// @file python/bind.cpp
/// @brief Top-level nanobind module definition for _atomiverse.
///
/// Atomiverse binds the driver / component API only — never a Frame data
/// type. Simulation state crosses the boundary as a `molpy.Frame` (from the
/// separate `molrs` extension); each `run()` borrows it via a PyCapsule.
///
/// Per-domain init functions are compiled only when the matching
/// ATV_BUILD_* product module is ON (see cmake/AtvProductModules.cmake).

#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/driver/driver_node.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/molrs_exchange.h"
#include "atomiverse/cpu/task/launcher_gate.h"
#include "molrs_frame_bridge.h"
#include <nanobind/nanobind.h>

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
#endif
#ifdef ATV_HAS_HF
void bind_scf(nb::module_& m);
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
              auto frame = borrow_molrs_frame(frame_obj);
              auto work = ATV::cpu::from_molrs(*frame);
              {
                  nb::gil_scoped_release release;
                  ATV::cpu::single_point(work, evaluator);
              }
              ATV::cpu::to_molrs(work, *frame);
          });

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
#endif
#ifdef ATV_HAS_HF
    bind_scf(m);
#endif
}

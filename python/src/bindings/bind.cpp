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
#include "atomiverse/cpu/units/frame.h"
#include "atv_source_stamp.h"
#include "molpy_frame_bridge.h"
#include "units_binding.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#ifdef ATV_HAS_SEMIEMPIRICAL
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"
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
#endif
#if defined(ATV_HAS_CUDA)                                                      \
    && (defined(ATV_HAS_SEMIEMPIRICAL) || defined(ATV_HAS_MD)                  \
        || defined(ATV_HAS_MINIMIZER))
void bind_cuda(nb::module_& m);
#endif
#ifdef ATV_HAS_HF
void bind_scf(nb::module_& m);
#ifdef ATV_HAS_CUDA
void bind_cuda_scf(nb::module_& m);
#endif
#endif

NB_MODULE(_atomiverse, m) {
    nb::class_<ATV::units::UnitSystem>(m, "_UnitSystem")
        .def_prop_ro("style",
                     [](const ATV::units::UnitSystem& u) {
                         return std::string(u.name());
                     })
        .def_prop_ro("boltzmann_constant",
                     &ATV::units::UnitSystem::boltzmann_constant)
        .def_prop_ro("planck_constant",
                     &ATV::units::UnitSystem::planck_constant)
        .def_prop_ro("coulomb_constant",
                     &ATV::units::UnitSystem::coulomb_constant)
        .def("thermal_energy", &ATV::units::UnitSystem::thermal_energy,
             nb::arg("temperature"));
    m.def("_units_from_style", [](const std::string& style) {
        return ATV::units::from_style(style);
    });
    const auto reference =
        [](const std::optional<std::array<double, 3>>& values,
           const std::string& identity)
        -> std::optional<ATV::units::ReducedReference> {
        if (!values)
            return {};
        ATV::units::ReducedReference ref{identity, (*values)[0], (*values)[1],
                                         (*values)[2]};
        ref.validate();
        return ref;
    };
    m.def(
        "_unit_scales",
        [reference](const std::string& source, const std::string& target,
                    const std::optional<std::array<double, 3>>& scales,
                    const std::string& identity) {
            if (source != "electronic")
                (void)ATV::units::parse_style(source);
            if (target != "electronic")
                (void)ATV::units::parse_style(target);
            if (source == target)
                return std::array<double, 5>{1., 1., 1., 1., 1.};
            const auto ref = reference(scales, identity);
            const auto a = ATV::units::PhysicalUnits::from_style(source, ref);
            const auto b = ATV::units::PhysicalUnits::from_style(target, ref);
            using ATV::units::convert;
            return std::array<double, 5>{convert(1., a.energy(), b.energy()),
                                         convert(1., a.length(), b.length()),
                                         convert(1., a.charge(), b.charge()),
                                         convert(1., a.mass(), b.mass()),
                                         convert(1., a.time(), b.time())};
        },
        nb::arg("source"), nb::arg("target"), nb::arg("reference") = nb::none(),
        nb::arg("identity") = "");
    m.def(
        "_convert_frame_units",
        [reference](nb::object obj, const std::string& target,
                    const std::optional<std::array<double, 3>>& scales,
                    const std::string& identity) {
            auto frame = borrow_molpy_frame(obj);
            auto work =
                ATV::units::convert(ATV::cpu::from_molpy(*frame), target,
                                    reference(scales, identity));
            ATV::cpu::to_molpy(work, *frame, false);
        },
        nb::arg("frame"), nb::arg("target"), nb::arg("reference") = nb::none(),
        nb::arg("identity") = "");
    m.def("_validate_unit_reference",
          [reference](const std::optional<std::array<double, 3>>& scales,
                      const std::string& identity) {
              (void)reference(scales, identity);
          });

    m.def("_frame_run_style", [](nb::object obj) {
        auto frame = borrow_molpy_frame(obj);
        const auto entry = molrs::frame_get_meta(*frame, "units_style");
        if (entry.dtype != molrs::MetaType::String)
            throw std::invalid_argument("units_style must be a string");
        const std::string style(entry.string_value.data(),
                                entry.string_value.size());
        if (style == "electronic")
            return std::string("metal");
        (void)ATV::units::parse_style(style);
        return style;
    });
    m.doc() = "Atomiverse: molecular simulation compute backend";
    // The source revision this extension was built from
    // (cmake/AtvSourceStamp.cmake): a 40-hex commit or "unknown", and
    // whether tracked files differed from it (None when unknown).
    m.attr("__git_commit__") = ATV_SOURCE_COMMIT;
    if (ATV_SOURCE_DIRTY < 0)
        m.attr("__git_dirty__") = nb::none();
    else
        m.attr("__git_dirty__") = nb::bool_(ATV_SOURCE_DIRTY == 1);
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

    nb::class_<ATV::cpu::Evaluate> evaluate(m, "_Evaluate");
    ATV::python::bind_units(
        evaluate,
        [](ATV::cpu::Evaluate& d) -> ATV::cpu::Evaluate& { return d; },
        [](const ATV::cpu::Evaluate&, const char*) {});
    evaluate
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
        .def(
            "set_evaluator",
            [](ATV::cpu::Evaluate& self,
               ATV::cpu::semiempirical::DFTB2& method) {
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
                auto work =
                    ATV::cpu::from_molpy_for_run(*frame, self.units().name());
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
#endif
#if defined(ATV_HAS_CUDA)                                                      \
    && (defined(ATV_HAS_SEMIEMPIRICAL) || defined(ATV_HAS_MD)                  \
        || defined(ATV_HAS_MINIMIZER))
    bind_cuda(m);
#endif
#ifdef ATV_HAS_HF
    bind_scf(m);
#ifdef ATV_HAS_CUDA
    bind_cuda_scf(m);
#endif
#endif
}

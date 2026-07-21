/// @file python/bind_pw.cpp
/// @brief Plane-wave DFT bindings backed directly by cpu::pw::PWSolver.

#include "molrs_frame_bridge.h"

#include <nanobind/stl/string.h>

#ifdef ATV_HAS_PW
#include "atomiverse/cpu/frame/materialize.h"
#include "atomiverse/cpu/pw/pw_solver.h"
#endif

void bind_pw(nb::module_& m) {

#ifdef ATV_HAS_PW
    using namespace ATV;
    using namespace ATV::cpu::pw;

    nb::class_<PWConfig>(m, "PWConfig",
                         "Configuration for the explicit plane-wave solver.")
        .def(nb::init<>())
        .def_rw("upf_path", &PWConfig::upf_path)
        .def_rw("ecut_ry", &PWConfig::ecut_ry)
        .def_rw("k_mesh", &PWConfig::k_mesh)
        .def_rw("tol", &PWConfig::tol)
        .def_rw("max_iter", &PWConfig::max_iter)
        .def_rw("mix_beta", &PWConfig::mix_beta)
        .def_rw("pulay_depth", &PWConfig::pulay_depth);

    nb::class_<PWResult>(m, "PWResult",
                         "Result of a plane-wave DFT calculation.")
        .def_prop_ro(
            "energy", [](const PWResult& r) { return r.energy; },
            "Total energy [Ry].")
        .def_prop_ro(
            "energy_ha", [](const PWResult& r) { return r.energy_ha; },
            "Total energy [Hartree].")
        .def_prop_ro(
            "converged", [](const PWResult& r) { return r.converged; },
            "Whether SCF converged within max_iter.")
        .def_prop_ro(
            "iterations", [](const PWResult& r) { return r.iterations; },
            "Number of SCF iterations taken.")
        .def("__repr__", [](const PWResult& r) {
            std::string conv = r.converged ? "converged" : "NOT converged";
            return "PWResult(" + conv + ", E=" + std::to_string(r.energy)
                   + " Ry, iters=" + std::to_string(r.iterations) + ")";
        });

    nb::class_<PWSolver>(m, "PWSolver",
                         "Explicit plane-wave SCF solver (build then run).")
        .def(nb::init<PWConfig>(), nb::arg("config"))
        .def(
            "build",
            [](PWSolver& solver, nb::object frame_obj) {
                auto fref = borrow_molrs_frame(frame_obj);
                nb::gil_scoped_release release;
                cpu::Frame work = cpu::materialize(*fref);
                solver.build(work);
            },
            nb::arg("frame"),
            "Validate the reference frame and perform one-time setup.")
        .def(
            "run",
            [](PWSolver& solver, nb::object frame_obj) {
                auto fref = borrow_molrs_frame(frame_obj);
                PWResult result;
                {
                    nb::gil_scoped_release release;
                    cpu::Frame work = cpu::materialize(*fref);
                    result = solver.run(work);
                }
                return result;
            },
            nb::arg("frame"),
            "Run SCF at the frame's current geometry. build() is required.")
        .def_prop_ro(
            "last_result",
            [](const PWSolver& solver) { return solver.last_result(); },
            "Copy of the most recent result.");
#endif
}

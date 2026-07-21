/// @file python/src/bindings/bind_semiempirical.cpp
/// @brief AM1 semiempirical single-point bindings: AM1Config, AM1Evaluator,
///        AM1Result, and the `single_point` entry.
///
/// Single-point only — `AM1Evaluator` carries `Energy | Charge |
/// DensityMatrix` (no Force), so it binds to `cpu::single_point` and never
/// to the MD driver. Energy [eV] lands on the frame's scalar channel;
/// Mulliken charges [e] write back into the borrowed molrs.Frame per the
/// pinned write_back round-trip contract (charge is in the write-back set).

#include "molrs_frame_bridge.h"

#include <utility>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "atomiverse/cpu/driver/single_point.h"
#include "atomiverse/cpu/frame/materialize.h"
#include "atomiverse/cpu/semiempirical/am1_evaluator.h"
#include "atomiverse/cpu/task/launcher_gate.h"
#include "atomiverse/cpu/task/task_spec.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>

#include "atomiverse/cpu/task/task_protocol.h"
#endif

using namespace ATV;

void bind_semiempirical(nb::module_& m) {
    using cpu::semiempirical::AM1Config;
    using cpu::semiempirical::AM1Evaluator;
    using cpu::semiempirical::AM1Result;

    // ── AM1Config (= NDDOConfig; `method` is pinned to AM1, not exposed) ──
    nb::class_<AM1Config>(m, "AM1Config",
                          "AM1 SCF configuration (method pinned to AM1).")
        .def(
            "__init__",
            [](AM1Config* cfg, int charge, int multiplicity, int max_iter,
               double density_tol, double energy_tol) {
                new (cfg) AM1Config{};
                cfg->charge = charge;
                cfg->multiplicity = multiplicity;
                cfg->max_iter = max_iter;
                cfg->density_tol = density_tol;
                cfg->energy_tol = energy_tol;
            },
            nb::arg("charge") = 0, nb::arg("multiplicity") = 1,
            nb::arg("max_iter") = 200, nb::arg("density_tol") = 1.0e-8,
            nb::arg("energy_tol") = 1.0e-10,
            "Construct with total charge [e], spin multiplicity, SCF "
            "iteration cap, density-norm tolerance (dimensionless), and "
            "energy tolerance [Hartree].")
        .def_rw("charge", &AM1Config::charge, "Total molecular charge [e].")
        .def_rw("multiplicity", &AM1Config::multiplicity, "Spin multiplicity.")
        .def_rw("max_iter", &AM1Config::max_iter, "SCF iteration cap.")
        .def_rw("density_tol", &AM1Config::density_tol,
                "Density-norm convergence threshold (dimensionless).")
        .def_rw("energy_tol", &AM1Config::energy_tol,
                "Energy convergence threshold [Hartree].");

    // ── AM1Result (= NDDOResult) ──
    nb::class_<AM1Result>(m, "AM1Result",
                          "Converged AM1 solve results (read-only).")
        .def_ro("converged", &AM1Result::converged,
                "True when the SCF met both tolerances.")
        .def_ro("iterations", &AM1Result::iterations,
                "SCF iterations performed.")
        .def_ro("electronic_energy_hartree",
                &AM1Result::electronic_energy_hartree,
                "Electronic energy [Hartree].")
        .def_ro("heat_of_formation_kcal_mol",
                &AM1Result::heat_of_formation_kcal_mol,
                "Heat of formation [kcal/mol].")
        .def_ro("mulliken_charges", &AM1Result::mulliken_charges,
                "Per-atom Mulliken charges [e] (double precision).")
        .def_ro("reference", &AM1Result::reference,
                "Literature reference for the parameterization.");

    // ── AM1Evaluator ──
    nb::class_<AM1Evaluator>(
        m, "AM1Evaluator",
        "AM1 single-point Evaluator (Energy | Charge | DensityMatrix; no "
        "forces — plugs into single_point, never into MD).")
        .def(nb::init<AM1Config>(), nb::arg("config") = AM1Config{},
             "Construct with an AM1 configuration.")
        .def("last_result", &AM1Evaluator::last_result,
             nb::rv_policy::reference_internal,
             "Results of the most recent solve (AM1Result).")
        .def("config", &AM1Evaluator::config, nb::rv_policy::reference_internal,
             "The configuration this evaluator was constructed with.");

    // ── single_point: one Evaluate pass over a borrowed molrs.Frame ──
    m.def(
        "single_point",
        [](nb::object frame_obj, AM1Evaluator& evaluator) {
            const cpu::task::LauncherGate gate;
            auto fref = borrow_molrs_frame(frame_obj);
            cpu::Frame work = cpu::materialize(*fref);
            if (!gate.active()) {
                {
                    nb::gil_scoped_release release;
                    cpu::single_point(work, evaluator);
                }
                // Unlike the PW path, AM1 declares Capability::Charge —
                // the per-atom charges must return to the SAME molrs.Frame
                // via the pinned write_back round-trip contract.
                cpu::write_back(work, *fref);
                return;
            }
#ifdef ATV_HAS_MPI
            // Launcher-hosted rank 0: dispatch a SinglePoint task. Results
            // (energy [eV] + charges [e]) come back through the Frame;
            // NOTE the borrowed evaluator's last_result() is NOT populated
            // on this path — the solve ran inside the per-rank
            // materializers.
            cpu::task::AM1EvaluatorParams p{};
            const AM1Config& cfg = evaluator.config();
            p.charge = cfg.charge;
            p.multiplicity = cfg.multiplicity;
            p.max_iter = cfg.max_iter;
            p.density_tol = cfg.density_tol;
            p.energy_tol = cfg.energy_tol;
            cpu::task::TaskSpec spec{cpu::task::SinglePointTask{p}};
            cpu::Frame result(0);
            {
                nb::gil_scoped_release release;
                cpu::task::TaskDispatcher dispatcher(
                    MPI_COMM_WORLD, cpu::task::kProtocolVersion,
                    /*handshake=*/false); // main's dispatcher handshook
                result = dispatcher.dispatch(spec, std::move(work)).frame;
            }
            cpu::write_back(result, *fref);
#endif
        },
        nb::arg("frame"), nb::arg("evaluator"),
        "Run one AM1 single-point on a molrs.Frame: energy [eV] onto the "
        "frame scalars, Mulliken charges [e] onto the atoms charge column. "
        "GIL released during compute; read evaluator.last_result() for "
        "Hartree/kcal-mol detail (serial path only — under the MPI "
        "launcher the results arrive via the Frame).");
}

/// @file python/src/bindings/bind_minimizer.cpp
/// @brief CPU Minimize + LBFGS bindings over the typed TaskSpec v3 path.

#include "molrs_frame_bridge.h"

#include <nanobind/stl/string.h>

#include <optional>
#include <utility>
#include <vector>

#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/materialize.h"
#include "atomiverse/cpu/minimizer/lbfgs.h"
#include "atomiverse/cpu/minimizer/minimize.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/task/launcher_gate.h"
#include "atomiverse/cpu/task/task_spec.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>

#include "atomiverse/cpu/task/task_protocol.h"
#endif

using namespace ATV;

namespace {

using cpu::task::ForceConvergedParams;
using cpu::task::MinimizeBinding;
using cpu::task::MinimizeComponentEntry;
using cpu::task::TaskResult;

struct MinimizeHandle {
    explicit MinimizeHandle(ForceConvergedParams convergence_)
        : convergence(std::move(convergence_)) {
        // Validate through the exact wire-boundary type used by launcher runs.
        (void)cpu::task::TaskSpec{cpu::task::MinimizeTask{{}, 1, convergence}};
        driver.add_criterion(cpu::ForceConverged{
            convergence.ftol, convergence.rms_ftol, convergence.max_dtol,
            convergence.rms_dtol, convergence.etol});
    }

    cpu::minimizer::Minimize driver;
    std::vector<MinimizeComponentEntry> mirror;
    ForceConvergedParams convergence;

    template <typename Params, typename Component>
    void add(const Params& p, Component component,
             std::optional<MinimizeBinding> binding) {
        if (binding)
            driver.on(binding->phase).add(std::move(component), binding->order);
        else
            driver.add(std::move(component));
        mirror.push_back(MinimizeComponentEntry{p, binding});
    }

    TaskResult run(nb::object frame_obj, int max_iter) {
        const cpu::task::TaskSpec spec{cpu::task::MinimizeTask{
            mirror, static_cast<std::int64_t>(max_iter), convergence}};
        const cpu::task::LauncherGate gate;
        if (!gate.active()) {
            auto frame = borrow_molrs_frame(frame_obj);
            int iterations = 0;
            {
                nb::gil_scoped_release release;
                iterations = driver.run(*frame, max_iter);
            }
            return TaskResult{iterations, driver.converged(),
                              driver.stop_reason()};
        }

#ifdef ATV_HAS_MPI
        auto frame = borrow_molrs_frame(frame_obj);
        cpu::Frame work = cpu::materialize(*frame);
        cpu::task::DispatchResult dispatched{cpu::Frame(0), {}};
        {
            nb::gil_scoped_release release;
            cpu::task::TaskDispatcher dispatcher(MPI_COMM_WORLD,
                                                 cpu::task::kProtocolVersion,
                                                 /*do_handshake=*/false);
            dispatched = dispatcher.dispatch(spec, std::move(work));
        }
        cpu::write_back(dispatched.frame, *frame);
        return std::move(dispatched.control);
#else
        throw std::runtime_error(
            "launcher mode requires an MPI-enabled Atomiverse build");
#endif
    }
};

struct MinimizePhaseBuilder {
    MinimizeHandle* handle;
    ATV::minimizer::Phase phase;
};

} // namespace

void bind_minimizer(nb::module_& m) {
    nb::enum_<minimizer::Phase>(m, "MinimizePhase",
                                "Geometry-minimization phase sequence.")
        .value("search_direction", minimizer::Phase::search_direction)
        .value("line_search", minimizer::Phase::line_search)
        .value("force", minimizer::Phase::force)
        .value("convergence_check", minimizer::Phase::convergence_check)
        .value("end_of_step", minimizer::Phase::end_of_step);

    nb::class_<cpu::task::LBFGSParams>(m, "LBFGS",
                                       "L-BFGS optimizer configuration.")
        .def(
            "__init__",
            [](cpu::task::LBFGSParams* p, int memory, double alpha0,
               int max_ls) {
                new (p) cpu::task::LBFGSParams{memory, alpha0, max_ls};
            },
            nb::arg("memory") = 5, nb::arg("alpha0") = 0.01,
            nb::arg("max_ls") = 10);

    nb::class_<ForceConvergedParams>(m, "ForceConverged",
                                     "Five-criterion minimization stop policy.")
        .def(
            "__init__",
            [](ForceConvergedParams* p, double ftol, double rms_ftol,
               double max_dtol, double rms_dtol, double etol) {
                new (p) ForceConvergedParams{ftol, rms_ftol, max_dtol, rms_dtol,
                                             etol};
            },
            nb::arg("ftol") = 0.05, nb::arg("rms_ftol") = 0.03,
            nb::arg("max_dtol") = 0.002, nb::arg("rms_dtol") = 0.001,
            nb::arg("etol") = 1e-5);

    nb::class_<TaskResult>(m, "MinimizeResult",
                           "Loop-control result; state is written to Frame.")
        .def_ro("iterations", &TaskResult::iterations)
        .def_ro("converged", &TaskResult::converged)
        .def_ro("stop_reason", &TaskResult::stop_reason);

    nb::class_<MinimizeHandle>(m, "Minimize",
                               "CPU geometry-minimization driver.")
        .def(nb::init<ForceConvergedParams>(), nb::arg("convergence"))
        .def(
            "add_lbfgs",
            [](MinimizeHandle& self, const cpu::task::LBFGSParams& p) {
                self.add(p, cpu::minimizer::LBFGS{p.memory, p.alpha0, p.max_ls},
                         std::nullopt);
            },
            nb::arg("component"))
        .def(
            "on",
            [](MinimizeHandle& self, minimizer::Phase phase) {
                return MinimizePhaseBuilder{&self, phase};
            },
            nb::arg("phase"))
        .def("build", [](MinimizeHandle& self) { self.driver.build(); })
        .def("run", &MinimizeHandle::run, nb::arg("frame"),
             nb::arg("max_iter"));

    nb::class_<MinimizePhaseBuilder>(
        m, "MinimizePhaseBuilder",
        "Fluent builder for typed minimizer phase bindings.")
        .def(
            "add_pair_lj_cut",
            [](MinimizePhaseBuilder& self, const cpu::task::PairLJCutParams& p,
               int order) -> MinimizePhaseBuilder& {
                self.handle->add(p,
                                 cpu::PairLJCut{p.epsilon, p.sigma, p.cutoff},
                                 MinimizeBinding{self.phase, order});
                return self;
            },
            nb::arg("component"), nb::arg("order") = 0,
            nb::rv_policy::reference)
        .def(
            "add_neighbor_list",
            [](MinimizePhaseBuilder& self,
               const cpu::task::NeighborListParams& p,
               int order) -> MinimizePhaseBuilder& {
                self.handle->add(p, cpu::md::NeighborList<>{p.cutoff, p.skin},
                                 MinimizeBinding{self.phase, order});
                return self;
            },
            nb::arg("component"), nb::arg("order") = 0,
            nb::rv_policy::reference);
}

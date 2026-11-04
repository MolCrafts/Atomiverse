/// @brief Native CUDA Python boundaries; scientific scheduling stays in Drivers.
#include "atomiverse/cpu/task/task_spec.h"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/semiempirical/am1_solver.cuh"
#include "atomiverse/cuda/semiempirical/mndo_solver.cuh"
#include "atomiverse/cuda/semiempirical/pm3_solver.cuh"
#include "atomiverse/cuda/semiempirical/rm1_solver.cuh"
#include "nddo_bindings.h"
#include <memory>
#include <nanobind/stl/string.h>
#include <optional>
#include <utility>
#ifdef ATV_HAS_MD
#include "atomiverse/core/sllod_flow.h"
#include "atomiverse/cuda/dynamics/barostat.cuh"
#include "atomiverse/cuda/dynamics/fix_langevin.cuh"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/nvt_sllod.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/angle_harmonic.cuh"
#include "atomiverse/cuda/potentials/bond_fene.cuh"
#include "atomiverse/cuda/potentials/coul_long.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "atomiverse/cuda/record/pressure_series.cuh"
#ifdef ATV_HAS_KSPACE
#include "atomiverse/cuda/longrange/pme.cuh"
#endif
#endif
#ifdef ATV_HAS_MINIMIZER
#include "atomiverse/cuda/minimize/minimize.cuh"
#include "atomiverse/cuda/minimize/sd.cuh"
#endif

using namespace ATV;
namespace {
void require_local_cuda() {
    if (cpu::task::LauncherGate{}.active())
        throw std::runtime_error(
            "CUDA Python drivers do not support MPI launcher tasks");
}
struct CudaEvaluateHandle {
    cuda::Evaluate driver;
    bool ran = false;
    void require_configurable() const {
        if (ran)
            throw std::runtime_error("Evaluate has already run; construct a "
                                     "new driver to change its plan");
    }
    void run(nb::object obj) {
        require_local_cuda();
        auto frame = borrow_molpy_frame(obj);
        auto work = cpu::from_molpy(*frame);
        {
            nb::gil_scoped_release release;
            driver.run(work);
        }
        cpu::to_molpy(work, *frame);
        ran = true;
    }
};
// Keep concrete types at the mount boundary: CUDA output capabilities must
// survive registration (the erased CUDA DriverNode does not expose them).
template <class Handle, class Method>
void bind_cuda_evaluator(nb::class_<Handle>& cls) {
    cls.def(
        "set_evaluator",
        [](Handle& self, Method& method) {
            self.require_configurable();
            self.driver.set_evaluator_ref(method);
        },
        nb::arg("method"), nb::keep_alive<1, 2>());
}
template <class Handle>
void bind_cuda_evaluators(nb::class_<Handle>& cls) {
    bind_cuda_evaluator<Handle, cuda::semiempirical::AM1>(cls);
    bind_cuda_evaluator<Handle, cuda::semiempirical::RM1>(cls);
    bind_cuda_evaluator<Handle, cuda::semiempirical::MNDO>(cls);
    bind_cuda_evaluator<Handle, cuda::semiempirical::PM3>(cls);
}
#ifdef ATV_HAS_MD
template <class Handle>
void bind_cuda_classical(nb::class_<Handle>& cls) {
    cls.def(
        "set_evaluator",
        [](Handle& self, const cpu::task::PairLJCutParams& p) {
            self.require_configurable();
            if (!p.type_i.empty() || !p.energy_shift)
                throw std::invalid_argument(
                    "CUDA PairLJCut is single-type and always shifted at the "
                    "cutoff: the per-type table and energy_shift=False are "
                    "CPU-only");
            auto lj = cuda::PairLJCut{p.epsilon, p.sigma, p.cutoff};
            lj.set_special_bonds(p.s12, p.s13, p.s14);
            self.driver.set_evaluator(std::move(lj));
        },
        nb::arg("evaluator"));
    cls.def(
        "set_evaluator",
        [](Handle& self, const cpu::task::CoulLongParams& p) {
            self.require_configurable();
            // Real-space half. No special-bond weights: charged beads are
            // not 1-2 or 1-3 of each other, and 1-4 Coulomb stays on.
            // LAMMPS matches with `special_bonds coul 1 1 1`.
            self.driver.set_evaluator(cuda::CoulLong{p.alpha, p.cutoff});
        },
        nb::arg("evaluator"));
#ifdef ATV_HAS_KSPACE
    cls.def(
        "set_evaluator",
        [](Handle& self, const cpu::task::KSpacePmeParams& p) {
            self.require_configurable();
            self.driver.set_evaluator(
                cuda::KSpacePME{p.alpha, p.grid_spacing, p.order});
        },
        nb::arg("evaluator"));
    cls.def(
        "set_evaluator",
        [](Handle& self, const cpu::task::KSpacePppmParams& p) {
            self.require_configurable();
            // ik differentiation, factors 2/3/5. Same alpha as CoulLong.
            self.driver.set_evaluator(
                cuda::KSpacePPPM{p.alpha, p.grid_spacing, p.order});
        },
        nb::arg("evaluator"));
#endif
    cls.def(
        "set_neighbor",
        [](Handle& self, const cpu::task::NeighborListParams& p) {
            self.require_configurable();
            self.driver.set_neighbor(cuda::NeighborList{p.cutoff, p.skin});
        },
        nb::arg("neighbor"));
}
#endif
#ifdef ATV_HAS_MD
struct CudaMdHandle {
    // Native MD retains a pointer to its host boundary across chunks. Preserve
    // that Frame address and its Python owner until after the driver dies.
    // Langevin is declared before the driver so the driver releases its
    // borrow first.
    nb::object frame_owner;
    std::optional<cpu::Frame> work;
    std::optional<cuda::md::FixLangevin> langevin;
    bool span_fixed = false;
    cuda::md::MD driver;
    bool ran = false;
    void require_configurable() const {
        if (ran)
            throw std::runtime_error("MD has already run; construct a new "
                                     "driver to change its plan");
    }
    void run(nb::object obj, BigInt steps) {
        require_local_cuda();
        if (steps < 0)
            throw std::invalid_argument("MD step budget must be nonnegative");
        if (frame_owner.ptr() != nullptr && frame_owner.ptr() != obj.ptr())
            throw std::invalid_argument(
                "CUDA MD continuation requires the same molpy.Frame; use a new "
                "MD for another frame");
        auto frame = borrow_molpy_frame(obj);
        auto incoming = cpu::from_molpy(*frame);
        if (!work) {
            work.emplace(std::move(incoming));
            frame_owner = obj;
        } else {
            // Update the existing boundary object; native MD detects host
            // input edits and invalidates device/force/electronic state.
            *work = std::move(incoming);
        }
        // A caller that already set the cool window keeps it. Otherwise the
        // first run owns the ramp, and a later chunk on this same driver
        // continues that window instead of collapsing onto T_stop.
        if (langevin && !span_fixed) {
            langevin->set_run_span(0, static_cast<std::int64_t>(steps));
            span_fixed = true;
        }
        {
            nb::gil_scoped_release release;
            driver.run(*work, steps);
        }
        cpu::to_molpy(*work, *frame);
        ran = true;
    }
};
#endif
#ifdef ATV_HAS_MINIMIZER
struct CudaMinimizeHandle {
    cuda::minimizer::Minimize driver;
    bool ran = false;
    void require_configurable() const {
        if (ran)
            throw std::runtime_error("Minimize has already run; construct a "
                                     "new driver to change its plan");
    }
    cpu::task::TaskResult run(nb::object obj, BigInt steps) {
        require_local_cuda();
        auto frame = borrow_molpy_frame(obj);
        auto work = cpu::from_molpy(*frame);
        BigInt iterations;
        {
            nb::gil_scoped_release release;
            iterations = driver.run(work, steps);
        }
        cpu::to_molpy(work, *frame);
        ran = true;
        return {iterations, driver.converged(), driver.stop_reason()};
    }
};
#endif
} // namespace

void bind_cuda(nb::module_& m) {
    nb::class_<cuda::Component>(m, "_CudaComponent");
    bind_nddo_method<cuda::semiempirical::AM1>(m, "_CudaAM1", "cuda");
    bind_nddo_method<cuda::semiempirical::RM1>(m, "_CudaRM1", "cuda");
    bind_nddo_method<cuda::semiempirical::MNDO>(m, "_CudaMNDO", "cuda");
    bind_nddo_method<cuda::semiempirical::PM3>(m, "_CudaPM3", "cuda");
    nb::class_<CudaEvaluateHandle> eval(m, "_CudaEvaluate");
    eval.def(nb::init<>())
        .def("build", [](CudaEvaluateHandle& self) { self.driver.build(); })
        .def("run", &CudaEvaluateHandle::run, nb::arg("frame"));
    bind_cuda_evaluators(eval);
#ifdef ATV_HAS_MD
    bind_cuda_classical(eval);
    // Evaluate has no bond/angle slot. The force phase is set_evaluator.
    eval.def(
        "set_evaluator",
        [](CudaEvaluateHandle& self, const cpu::task::BondFeneParams& p) {
            self.require_configurable();
            self.driver.set_evaluator(cuda::BondFene{p.K, p.R0});
        },
        nb::arg("evaluator"));
    eval.def(
        "set_evaluator",
        [](CudaEvaluateHandle& self, const cpu::task::AngleHarmonicParams& p) {
            self.require_configurable();
            self.driver.set_evaluator(cuda::AngleHarmonic{p.K, p.theta0});
        },
        nb::arg("evaluator"));
    nb::class_<CudaMdHandle> md(m, "_CudaMD");
    md.def(nb::init<>())
        .def("build", [](CudaMdHandle& self) { self.driver.build(); })
        .def(
            "set_integrator",
            [](CudaMdHandle& self, const cpu::task::VelocityVerletParams& p) {
                self.require_configurable();
                self.driver.set_integrator(cuda::md::VelocityVerlet{p.dt});
            },
            nb::arg("integrator"))
        .def(
            "set_integrator",
            [](CudaMdHandle& self, const ATV::md::NvtSllodSpec& p) {
                self.require_configurable();
                self.driver.set_integrator(
                    cuda::md::NvtSllod{p.temperature_K, p.tau_fs, p.dt_fs,
                                       p.rate_per_fs, p.direction});
            },
            nb::arg("integrator"))
        .def(
            "set_pressure_series",
            [](CudaMdHandle& self, const std::string& path, int every,
               double epsilon_ev, int direction) {
                self.require_configurable();
                self.driver.on(ATV::md::Phase::end_of_step)
                    .add(cuda::md::PressureSeries{path, every, epsilon_ev,
                                                  direction},
                         +20);
            },
            nb::arg("path"), nb::arg("every"), nb::arg("epsilon_ev"),
            nb::arg("direction"),
            "Write −P* every N end-of-step updates. One continuous run.");
    bind_cuda_classical(md);
    md.def(
          "set_evaluator",
          [](CudaMdHandle& self, const cpu::task::BondFeneParams& p) {
              self.require_configurable();
              self.driver.set_bond(cuda::BondFene{p.K, p.R0});
          },
          nb::arg("evaluator"))
        .def(
            "set_evaluator",
            [](CudaMdHandle& self, const cpu::task::AngleHarmonicParams& p) {
                self.require_configurable();
                self.driver.set_angle(cuda::AngleHarmonic{p.K, p.theta0});
            },
            nb::arg("evaluator"))
        .def(
            "set_run_span",
            [](CudaMdHandle& self, std::int64_t nfirst, std::int64_t nlast) {
                self.require_configurable();
                if (!self.langevin)
                    throw std::runtime_error(
                        "set_run_span requires a Langevin thermostat");
                if (nlast <= nfirst)
                    throw std::invalid_argument(
                        "run span nlast must be greater than nfirst");
                self.langevin->set_run_span(nfirst, nlast);
                self.span_fixed = true;
            },
            nb::arg("nfirst"), nb::arg("nlast"))
        .def(
            "set_thermostat",
            [](CudaMdHandle& self, const cpu::task::FixLangevinParams& p) {
                self.require_configurable();
                if (self.langevin)
                    throw std::runtime_error(
                        "CUDA MD accepts one Langevin thermostat");
                self.langevin.emplace(p.T_start, p.T_stop, p.damp, p.dt,
                                      p.seed);
                self.driver.set_thermostat_ref(*self.langevin);
            },
            nb::arg("thermostat"))
        .def(
            "set_barostat",
            [](CudaMdHandle& self,
               const cpu::task::BerendsenBarostatParams& p) {
                self.require_configurable();
                self.driver.set_barostat(cuda::md::Barostat{
                    p.P_target, p.tau_P, p.dt, p.bulk_modulus});
            },
            nb::arg("barostat"))
        .def("run", &CudaMdHandle::run, nb::arg("frame"), nb::arg("n_steps"));
    bind_cuda_evaluators(md);
#endif
#ifdef ATV_HAS_MINIMIZER
    nb::class_<cuda::minimizer::SteepestDescent, cuda::Component>(
        m, "_CudaSteepestDescent")
        .def(nb::init<double>(), nb::arg("alpha"));
    nb::class_<CudaMinimizeHandle> opt(m, "_CudaMinimize");
    opt.def(nb::init<>())
        .def("build", [](CudaMinimizeHandle& self) { self.driver.build(); })
        .def(
            "set_optimizer",
            [](CudaMinimizeHandle& self,
               cuda::minimizer::SteepestDescent& optimizer) {
                self.require_configurable();
                self.driver.set_optimizer_ref(optimizer);
            },
            nb::arg("optimizer"), nb::keep_alive<1, 2>())
        .def(
            "add_criterion",
            [](CudaMinimizeHandle& self,
               const cpu::task::ForceConvergedParams& p) {
                self.require_configurable();
                self.driver.add_criterion(cuda::ForceConverged{
                    p.ftol, p.rms_ftol, p.max_dtol, p.rms_dtol, p.etol});
            },
            nb::arg("criterion"))
        .def("run", &CudaMinimizeHandle::run, nb::arg("frame"),
             nb::arg("max_iter"));
    bind_cuda_evaluators(opt);
#endif
}

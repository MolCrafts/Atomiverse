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
#ifdef ATV_HAS_MD
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
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
        auto frame = borrow_molrs_frame(obj);
        auto work = cpu::from_molrs(*frame);
        {
            nb::gil_scoped_release release;
            driver.run(work);
        }
        cpu::to_molrs(work, *frame);
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
struct CudaMdHandle {
    // Native MD retains a pointer to its host boundary across chunks. Preserve
    // that Frame address and its Python owner until after the driver dies.
    nb::object frame_owner;
    std::optional<cpu::Frame> work;
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
        auto frame = borrow_molrs_frame(obj);
        auto incoming = cpu::from_molrs(*frame);
        if (!work) {
            work.emplace(std::move(incoming));
            frame_owner = obj;
        } else {
            // Update the existing boundary object; native MD detects host
            // input edits and invalidates device/force/electronic state.
            *work = std::move(incoming);
        }
        {
            nb::gil_scoped_release release;
            driver.run(*work, steps);
        }
        cpu::to_molrs(*work, *frame);
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
        auto frame = borrow_molrs_frame(obj);
        auto work = cpu::from_molrs(*frame);
        BigInt iterations;
        {
            nb::gil_scoped_release release;
            iterations = driver.run(work, steps);
        }
        cpu::to_molrs(work, *frame);
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

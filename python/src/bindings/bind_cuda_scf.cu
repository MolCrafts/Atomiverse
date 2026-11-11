/// @file python/src/bindings/bind_cuda_scf.cu
/// @brief Synchronous Python boundaries for the public CUDA HF methods.
#include "basis_names.h"
#include "force_array.h"
#include "molpy_frame_bridge.h"

#include "atomiverse/cpu/frame/molpy_exchange.h"
#include "atomiverse/cpu/task/launcher_gate.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/scf/rhf.cuh"
#include "atomiverse/cuda/scf/uhf.cuh"

#include <nanobind/stl/string.h>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

namespace {

template <class Method>
class CudaHfHandle {
  public:
    Method method;
    template <class... Args>
        requires std::is_constructible_v<Method, Args...>
    explicit CudaHfHandle(Args&&... args)
        : method(std::forward<Args>(args)...) {
        method.with_gradient(false);
    }

    void run(nb::object frame_obj) {
        require_local_();
        auto frame = borrow_molpy_frame(frame_obj);
        auto work = ATV::cpu::from_molpy(*frame);
        {
            nb::gil_scoped_release release;
            method.run(work);
            ATV_CUDA_CHECK(cudaDeviceSynchronize());
        }
        ATV::cpu::to_molpy(work, *frame);
    }

    ForceArray forces(nb::object frame_obj) {
        require_local_();
        auto frame = borrow_molpy_frame(frame_obj);
        auto work = ATV::cpu::from_molpy(*frame);
        ATV::cpu::scf::GradientResult result;
        {
            nb::gil_scoped_release release;
            result = method.forces(work);
            ATV_CUDA_CHECK(cudaDeviceSynchronize());
        }
        ATV::cpu::to_molpy(work, *frame);
        return to_force_array(std::span<const double>(result.forces.data(),
                                                      result.forces.size()));
    }

  private:
    static void require_local_() {
        if (ATV::cpu::task::LauncherGate{}.active())
            throw std::runtime_error(
                "CUDA HF methods do not support MPI launcher tasks");
    }
};

template <class Method>
void bind_method(nb::class_<CudaHfHandle<Method>>& cls) {
    using Handle = CudaHfHandle<Method>;
    cls.def("run", &Handle::run, nb::arg("frame"),
            "Energy-only public C++ solve. Writes Frame energy [eV], preserves "
            "forces, and completes cudaDeviceSynchronize before returning.")
        .def("forces", &Handle::forces, nb::arg("frame"),
             "Public C++ analytic forces [eV/Angstrom], returned as an owned "
             "(N,3) array and written to Frame; synchronized before returning.")
        .def(
            "energy", [](const Handle& self) { return self.method.energy(); },
            "Last successful standalone energy [Hartree]; throws before "
            "success.")
        .def("converged",
             [](const Handle& self) { return self.method.converged(); })
        .def("iterations",
             [](const Handle& self) { return self.method.iterations(); })
        .def(
            "last_scf_on_device",
            [](const Handle& self) { return self.method.last_scf_on_device(); })
        .def("last_force_on_device",
             [](const Handle& self) {
                 return self.method.last_force_on_device();
             })
        .def_prop_ro("basis",
                     [](const Handle& self) {
                         return std::string(
                             ATV::cpu::scf::basis_set_name(self.method.basis));
                     })
        .def_prop_ro("charge",
                     [](const Handle& self) { return self.method.charge; })
        .def_prop_ro("tol", [](const Handle& self) { return self.method.tol; })
        .def_prop_ro("max_iter",
                     [](const Handle& self) { return self.method.max_iter; })
        .def_prop_ro("level_shift", [](const Handle& self) {
            return self.method.level_shift;
        });
}

void synchronize() {
    nb::gil_scoped_release release;
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace

void bind_cuda_scf(nb::module_& m) {
    using ATV::cuda::scf::RHF;
    using ATV::cuda::scf::UHF;
    nb::class_<CudaHfHandle<RHF>> rhf(m, "CudaRHF",
                                      "CUDA-only restricted Hartree-Fock; "
                                      "unsupported device envelopes throw.");
    rhf.def(
        "__init__",
        [](CudaHfHandle<RHF>* self, const std::string& basis, int charge,
           double tol, int max_iter) {
            new (self) CudaHfHandle<RHF>(basis_from_name(basis), charge, tol,
                                         max_iter);
        },
        nb::arg("basis"), nb::arg("charge") = 0, nb::arg("tol") = 1e-9,
        nb::arg("max_iter") = 100);
    bind_method(rhf);
    nb::class_<CudaHfHandle<UHF>> uhf(
        m, "CudaUHF",
        "CUDA-only unrestricted Hartree-Fock; supports STO-3G s/p systems.");
    uhf.def(
           "__init__",
           [](CudaHfHandle<UHF>* self, const std::string& basis, int n_alpha,
              int n_beta, int charge, double tol, int max_iter) {
               new (self) CudaHfHandle<UHF>(basis_from_name(basis), n_alpha,
                                            n_beta, charge, tol, max_iter);
           },
           nb::arg("basis"), nb::arg("n_alpha"), nb::arg("n_beta"),
           nb::arg("charge") = 0, nb::arg("tol") = 1e-8,
           nb::arg("max_iter") = 100)
        .def_prop_ro(
            "n_alpha",
            [](const CudaHfHandle<UHF>& self) { return self.method.n_alpha; })
        .def_prop_ro("n_beta", [](const CudaHfHandle<UHF>& self) {
            return self.method.n_beta;
        });
    bind_method(uhf);

    m.def("cuda_hf_synchronize", &synchronize,
          "Complete all preceding work on the current CUDA device.");
    m.def(
        "cuda_hf_backend_info",
        [] {
            int count = 0, driver = 0, runtime = 0;
            ATV_CUDA_CHECK(cudaGetDeviceCount(&count));
            ATV_CUDA_CHECK(cudaDriverGetVersion(&driver));
            ATV_CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
            nb::dict info;
            info["backend"] = "cuda";
            info["device_count"] = count;
            info["driver_version"] = driver;
            info["runtime_version"] = runtime;
            info["synchronized"] = false;
            info["run_forces_synchronize"] = true;
            nb::list devices;
            for (int device = 0; device < count; ++device) {
                cudaDeviceProp properties{};
                ATV_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
                nb::dict item;
                item["index"] = device;
                item["name"] = properties.name;
                item["total_memory_bytes"] = properties.totalGlobalMem;
                item["compute_capability_major"] = properties.major;
                item["compute_capability_minor"] = properties.minor;
                devices.append(item);
            }
            info["devices"] = devices;
            if (count > 0) {
                int device = 0;
                std::size_t free = 0, total = 0;
                ATV_CUDA_CHECK(cudaGetDevice(&device));
                cudaDeviceProp properties{};
                ATV_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
                synchronize();
                ATV_CUDA_CHECK(cudaMemGetInfo(&free, &total));
                info["current_device"] = device;
                info["device_name"] = properties.name;
                info["free_memory_bytes"] = free;
                info["total_memory_bytes"] = total;
                info["synchronized"] = true;
            }
            return info;
        },
        "GPU identities, CUDA versions, current-device memory snapshot and "
        "completed synchronization.");
}

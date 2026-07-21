/// @file torch/frame/transfer.cu
/// @brief Transfer function implementations for torch::Frame.
///
/// The system scalars travel as the cross-backend `SystemScalars` bundle:
/// `cuda::Frame` carries one device-resident `SystemScalars*`, while
/// `torch::Frame` exposes its fields as individual CUDA Tensors. Since every
/// `SystemScalars` field is a `double`, the struct is a contiguous block of
/// 14 doubles -- energy, kinetic, conserved, stress[6], stress_kinetic[6] --
/// which lets `to_torch` wrap each field as a non-owning Tensor view.

#ifdef ATV_HAS_TORCH

#include "atomiverse/torch/frame/transfer.cuh"

#include <c10/cuda/CUDAStream.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>
#include <torch/torch.h>

#include "atomiverse/core/system_scalars.h"

namespace ATV {
namespace torch {

namespace {

/// @brief SystemScalars is a flat run of doubles -- assert the layout the
/// to_torch / to_cuda wrappers below rely on.
/// @{
static_assert(sizeof(SystemScalars) == 14 * sizeof(double),
              "SystemScalars must be a contiguous block of 14 doubles");
static_assert(offsetof(SystemScalars, energy) == 0,
              "SystemScalars.energy must be the first field");
static_assert(offsetof(SystemScalars, kinetic) == 1 * sizeof(double),
              "SystemScalars.kinetic offset");
static_assert(offsetof(SystemScalars, conserved) == 2 * sizeof(double),
              "SystemScalars.conserved offset");
static_assert(offsetof(SystemScalars, stress) == 3 * sizeof(double),
              "SystemScalars.stress offset");
static_assert(offsetof(SystemScalars, stress_kinetic) == 9 * sizeof(double),
              "SystemScalars.stress_kinetic offset");
/// @}

/// @brief Wrap a device cuda::real_t* as a non-owning 1D Tensor.
Tensor wrap_real(cuda::real_t* ptr, int n) {
    if (!ptr || n <= 0)
        return Tensor();
    auto opts =
        ::torch::TensorOptions().dtype(kRealDtype).device(::torch::kCUDA);
    return ::torch::from_blob(ptr, {static_cast<int64_t>(n)}, opts);
}

/// @brief Wrap a device int* as a non-owning 1D Tensor.
Tensor wrap_int(int* ptr, int n) {
    if (!ptr || n <= 0)
        return Tensor();
    auto opts =
        ::torch::TensorOptions().dtype(kIntDtype).device(::torch::kCUDA);
    return ::torch::from_blob(ptr, {static_cast<int64_t>(n)}, opts);
}

/// @brief Wrap a device std::int64_t* as a non-owning 1D Tensor (atom tag).
Tensor wrap_long(std::int64_t* ptr, int n) {
    if (!ptr || n <= 0)
        return Tensor();
    auto opts =
        ::torch::TensorOptions().dtype(kLongDtype).device(::torch::kCUDA);
    return ::torch::from_blob(ptr, {static_cast<int64_t>(n)}, opts);
}

/// @brief Wrap a flat device std::int8_t* [3n] as a non-owning [n,3] Tensor.
Tensor wrap_image(std::int8_t* ptr, int n) {
    if (!ptr || n <= 0)
        return Tensor();
    auto opts =
        ::torch::TensorOptions().dtype(kImageDtype).device(::torch::kCUDA);
    return ::torch::from_blob(ptr, {static_cast<int64_t>(n), 3}, opts);
}

/// @brief Wrap `count` doubles starting at field index `offset` of a device
///        SystemScalars struct as a non-owning Tensor (Double dtype).
Tensor wrap_scalars_field(const SystemScalars* scalars, int offset, int count) {
    if (!scalars)
        return Tensor();
    auto* base = reinterpret_cast<double*>(const_cast<SystemScalars*>(scalars));
    auto opts = ::torch::TensorOptions()
                    .dtype(::at::ScalarType::Double)
                    .device(::torch::kCUDA);
    return ::torch::from_blob(base + offset, {static_cast<int64_t>(count)},
                              opts);
}

/// @brief Extract cuda::real_t* from a Tensor. Validates CUDA, contiguous, correct dtype.
cuda::real_t* unwrap_real(const Tensor& t, const char* field_name) {
    if (!t.defined() || t.numel() == 0)
        return nullptr;
    if (!t.is_cuda())
        throw std::invalid_argument(std::string(field_name)
                                    + ": expected CUDA tensor");
    if (t.scalar_type() != kRealDtype)
        throw std::invalid_argument(std::string(field_name) + ": wrong dtype");
    if (!t.is_contiguous())
        throw std::invalid_argument(std::string(field_name)
                                    + ": not contiguous");
    return t.data_ptr<cuda::real_t>();
}

/// @brief Extract int* from a Tensor.
int* unwrap_int(const Tensor& t, const char* field_name) {
    if (!t.defined() || t.numel() == 0)
        return nullptr;
    if (!t.is_cuda())
        throw std::invalid_argument(std::string(field_name)
                                    + ": expected CUDA tensor");
    if (t.scalar_type() != kIntDtype)
        throw std::invalid_argument(std::string(field_name) + ": wrong dtype");
    if (!t.is_contiguous())
        throw std::invalid_argument(std::string(field_name)
                                    + ": not contiguous");
    return t.data_ptr<int>();
}

/// @brief Extract std::int64_t* from a Tensor (atom tag).
std::int64_t* unwrap_long(const Tensor& t, const char* field_name) {
    if (!t.defined() || t.numel() == 0)
        return nullptr;
    if (!t.is_cuda())
        throw std::invalid_argument(std::string(field_name)
                                    + ": expected CUDA tensor");
    if (t.scalar_type() != kLongDtype)
        throw std::invalid_argument(std::string(field_name) + ": wrong dtype");
    if (!t.is_contiguous())
        throw std::invalid_argument(std::string(field_name)
                                    + ": not contiguous");
    return t.data_ptr<std::int64_t>();
}

/// @brief Extract std::int8_t* from a Tensor (flat PBC image buffer).
std::int8_t* unwrap_image(const Tensor& t, const char* field_name) {
    if (!t.defined() || t.numel() == 0)
        return nullptr;
    if (!t.is_cuda())
        throw std::invalid_argument(std::string(field_name)
                                    + ": expected CUDA tensor");
    if (t.scalar_type() != kImageDtype)
        throw std::invalid_argument(std::string(field_name) + ": wrong dtype");
    if (!t.is_contiguous())
        throw std::invalid_argument(std::string(field_name)
                                    + ": not contiguous");
    return t.data_ptr<std::int8_t>();
}

/// @brief Copy a 1D xtensor<T> to a new CUDA Tensor.
template <typename T, ScalarType dtype>
Tensor xtensor_to_cuda(const xt::xtensor<T, 1>& arr) {
    int64_t n = static_cast<int64_t>(arr.size());
    if (n == 0)
        return Tensor();
    auto opts = ::torch::TensorOptions().dtype(dtype);
    auto cpu_t = ::torch::from_blob(const_cast<T*>(arr.data()), {n}, opts);
    return cpu_t.to(::torch::kCUDA);
}

/// @brief Copy a 2D xtensor<T> to a new CUDA Tensor with the same shape.
template <typename T, ScalarType dtype>
Tensor xtensor2_to_cuda(const xt::xtensor<T, 2>& arr) {
    int64_t r = static_cast<int64_t>(arr.shape()[0]);
    int64_t c = static_cast<int64_t>(arr.shape()[1]);
    if (r == 0 || c == 0)
        return Tensor();
    auto opts = ::torch::TensorOptions().dtype(dtype);
    auto cpu_t = ::torch::from_blob(const_cast<T*>(arr.data()), {r, c}, opts);
    return cpu_t.to(::torch::kCUDA);
}

/// @brief Copy a CUDA Tensor to a 1D xtensor<T>.
template <typename T>
void cuda_to_xtensor(const Tensor& t, xt::xtensor<T, 1>& arr) {
    if (!t.defined() || t.numel() == 0)
        return;
    auto cpu_t = t.to(::torch::kCPU).contiguous();
    int n = static_cast<int>(cpu_t.numel());
    std::memcpy(arr.data(), cpu_t.data_ptr<T>(),
                static_cast<size_t>(n) * sizeof(T));
}

/// @brief Copy a CUDA Tensor to a 2D xtensor<T> (shapes must match).
template <typename T>
void cuda_to_xtensor2(const Tensor& t, xt::xtensor<T, 2>& arr) {
    if (!t.defined() || t.numel() == 0)
        return;
    auto cpu_t = t.to(::torch::kCPU).contiguous();
    std::memcpy(arr.data(), cpu_t.data_ptr<T>(),
                static_cast<size_t>(cpu_t.numel()) * sizeof(T));
}

} // anonymous namespace

Frame to_torch(const cuda::Frame& src) {
    Frame dst;
    int na = src.n_atoms();
    dst.n_atoms_ = na;
    dst.n_bonds_ = src.n_bonds();
    dst.n_angles_ = src.n_angles();
    dst.n_dihedrals_ = src.n_dihedrals();

    // Per-atom arrays
    dst.atoms.x = wrap_real(src.atoms.x, na);
    dst.atoms.y = wrap_real(src.atoms.y, na);
    dst.atoms.z = wrap_real(src.atoms.z, na);
    dst.atoms.vx = wrap_real(src.atoms.vx, na);
    dst.atoms.vy = wrap_real(src.atoms.vy, na);
    dst.atoms.vz = wrap_real(src.atoms.vz, na);
    dst.atoms.fx = wrap_real(src.atoms.fx, na);
    dst.atoms.fy = wrap_real(src.atoms.fy, na);
    dst.atoms.fz = wrap_real(src.atoms.fz, na);
    dst.atoms.mass = wrap_real(src.atoms.mass, na);
    dst.atoms.charge = wrap_real(src.atoms.charge, na);
    dst.atoms.type = wrap_int(src.atoms.type, na);
    dst.atoms.tag = wrap_long(src.atoms.tag, na);
    dst.atoms.image = wrap_image(src.atoms.image, na);

    // System scalars: views into the device-resident SystemScalars struct
    // (energy=field 0, kinetic=1, conserved=2, stress=3..8, kinetic=9..14).
    dst.energy = wrap_scalars_field(src.scalars, 0, 1);
    dst.kinetic = wrap_scalars_field(src.scalars, 1, 1);
    dst.conserved = wrap_scalars_field(src.scalars, 2, 1);
    dst.stress = wrap_scalars_field(src.scalars, 3, 6);
    dst.stress_kinetic = wrap_scalars_field(src.scalars, 9, 6);

    if (src.box) {
        auto opts =
            ::torch::TensorOptions().dtype(kPrecDtype).device(::torch::kCUDA);
        dst.box = ::torch::from_blob(src.box, {9}, opts);
    }

    // Topology
    int nb = src.n_bonds();
    dst.bonds.atomi = wrap_int(src.bonds.atomi, nb);
    dst.bonds.atomj = wrap_int(src.bonds.atomj, nb);
    dst.bonds.type = wrap_int(src.bonds.type, nb);

    int nang = src.n_angles();
    dst.angles.atomi = wrap_int(src.angles.atomi, nang);
    dst.angles.atomj = wrap_int(src.angles.atomj, nang);
    dst.angles.atomk = wrap_int(src.angles.atomk, nang);
    dst.angles.type = wrap_int(src.angles.type, nang);

    int ndih = src.n_dihedrals();
    dst.dihedrals.atomi = wrap_int(src.dihedrals.atomi, ndih);
    dst.dihedrals.atomj = wrap_int(src.dihedrals.atomj, ndih);
    dst.dihedrals.atomk = wrap_int(src.dihedrals.atomk, ndih);
    dst.dihedrals.atoml = wrap_int(src.dihedrals.atoml, ndih);
    dst.dihedrals.type = wrap_int(src.dihedrals.type, ndih);

    return dst;
}

void to_cuda(const Frame& src, cuda::Frame& dst) {
    dst.n_atoms_ = src.n_atoms_;
    dst.n_bonds_ = src.n_bonds_;
    dst.n_angles_ = src.n_angles_;
    dst.n_dihedrals_ = src.n_dihedrals_;

    dst.atoms.x = unwrap_real(src.atoms.x, "x");
    dst.atoms.y = unwrap_real(src.atoms.y, "y");
    dst.atoms.z = unwrap_real(src.atoms.z, "z");
    dst.atoms.vx = unwrap_real(src.atoms.vx, "vx");
    dst.atoms.vy = unwrap_real(src.atoms.vy, "vy");
    dst.atoms.vz = unwrap_real(src.atoms.vz, "vz");
    dst.atoms.fx = unwrap_real(src.atoms.fx, "fx");
    dst.atoms.fy = unwrap_real(src.atoms.fy, "fy");
    dst.atoms.fz = unwrap_real(src.atoms.fz, "fz");
    dst.atoms.mass = unwrap_real(src.atoms.mass, "mass");
    dst.atoms.charge = unwrap_real(src.atoms.charge, "charge");
    dst.atoms.type = unwrap_int(src.atoms.type, "type");
    dst.atoms.tag = unwrap_long(src.atoms.tag, "tag");
    dst.atoms.image = unwrap_image(src.atoms.image, "image");

    // System scalars: copy the torch field Tensors (Double) into the
    // device-resident SystemScalars struct, which must already be allocated.
    if (dst.scalars) {
        auto* base = reinterpret_cast<double*>(dst.scalars);
        cudaStream_t stream = ::c10::cuda::getCurrentCUDAStream().stream();
        auto copy_field = [&](const Tensor& t, int offset, int count) {
            if (!t.defined() || t.numel() != count)
                return;
            auto src_d = t.to(::at::ScalarType::Double).contiguous();
            cudaMemcpyAsync(base + offset, src_d.data_ptr<double>(),
                            static_cast<size_t>(count) * sizeof(double),
                            cudaMemcpyDeviceToDevice, stream);
        };
        copy_field(src.energy, 0, 1);
        copy_field(src.kinetic, 1, 1);
        copy_field(src.conserved, 2, 1);
        copy_field(src.stress, 3, 6);
        copy_field(src.stress_kinetic, 9, 6);
    }

    dst.box = src.box.defined() ? src.box.data_ptr<cuda::prec_t>() : nullptr;

    dst.bonds.atomi = unwrap_int(src.bonds.atomi, "bonds.atomi");
    dst.bonds.atomj = unwrap_int(src.bonds.atomj, "bonds.atomj");
    dst.bonds.type = unwrap_int(src.bonds.type, "bonds.type");

    dst.angles.atomi = unwrap_int(src.angles.atomi, "angles.atomi");
    dst.angles.atomj = unwrap_int(src.angles.atomj, "angles.atomj");
    dst.angles.atomk = unwrap_int(src.angles.atomk, "angles.atomk");
    dst.angles.type = unwrap_int(src.angles.type, "angles.type");

    dst.dihedrals.atomi = unwrap_int(src.dihedrals.atomi, "dihedrals.atomi");
    dst.dihedrals.atomj = unwrap_int(src.dihedrals.atomj, "dihedrals.atomj");
    dst.dihedrals.atomk = unwrap_int(src.dihedrals.atomk, "dihedrals.atomk");
    dst.dihedrals.atoml = unwrap_int(src.dihedrals.atoml, "dihedrals.atoml");
    dst.dihedrals.type = unwrap_int(src.dihedrals.type, "dihedrals.type");
}

void to_cpu(const Frame& src, cpu::Frame& dst) {
    cuda_to_xtensor<double>(src.atoms.x, dst.atoms.x);
    cuda_to_xtensor<double>(src.atoms.y, dst.atoms.y);
    cuda_to_xtensor<double>(src.atoms.z, dst.atoms.z);
    cuda_to_xtensor<double>(src.atoms.vx, dst.atoms.vx);
    cuda_to_xtensor<double>(src.atoms.vy, dst.atoms.vy);
    cuda_to_xtensor<double>(src.atoms.vz, dst.atoms.vz);
    cuda_to_xtensor<double>(src.atoms.fx, dst.atoms.fx);
    cuda_to_xtensor<double>(src.atoms.fy, dst.atoms.fy);
    cuda_to_xtensor<double>(src.atoms.fz, dst.atoms.fz);
    cuda_to_xtensor<double>(src.atoms.mass, dst.atoms.mass);
    cuda_to_xtensor<double>(src.atoms.charge, dst.atoms.charge);
    cuda_to_xtensor<int>(src.atoms.type, dst.atoms.type);
    cuda_to_xtensor<std::int64_t>(src.atoms.tag, dst.atoms.tag);
    cuda_to_xtensor2<std::int8_t>(src.atoms.image, dst.atoms.image);

    // System scalars
    if (src.energy.defined() && src.energy.numel() > 0) {
        auto cpu_e = src.energy.to(::torch::kCPU);
        dst.scalars.energy = cpu_e.item<double>();
    }
    if (src.kinetic.defined() && src.kinetic.numel() > 0) {
        auto cpu_k = src.kinetic.to(::torch::kCPU);
        dst.scalars.kinetic = cpu_k.item<double>();
    }
    if (src.conserved.defined() && src.conserved.numel() > 0) {
        auto cpu_c = src.conserved.to(::torch::kCPU);
        dst.scalars.conserved = cpu_c.item<double>();
    }
    if (src.stress.defined() && src.stress.numel() == 6) {
        auto cpu_s = src.stress.to(::at::ScalarType::Double)
                         .to(::torch::kCPU)
                         .contiguous();
        std::memcpy(dst.scalars.stress, cpu_s.data_ptr<double>(),
                    6 * sizeof(double));
    }
    if (src.stress_kinetic.defined() && src.stress_kinetic.numel() == 6) {
        auto cpu_sk = src.stress_kinetic.to(::at::ScalarType::Double)
                          .to(::torch::kCPU)
                          .contiguous();
        std::memcpy(dst.scalars.stress_kinetic, cpu_sk.data_ptr<double>(),
                    6 * sizeof(double));
    }
    if (src.box.defined() && src.box.numel() == 9) {
        auto cpu_b =
            src.box.to(::at::ScalarType::Double).to(::torch::kCPU).contiguous();
        auto* bp = cpu_b.data_ptr<double>();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                dst.box(i, j) = bp[i * 3 + j];
    }
}

Frame from_cpu(const cpu::Frame& src) {
    Frame dst;
    int na = static_cast<int>(src.atoms.size());
    dst.n_atoms_ = na;

    dst.atoms.x = xtensor_to_cuda<double, kRealDtype>(src.atoms.x);
    dst.atoms.y = xtensor_to_cuda<double, kRealDtype>(src.atoms.y);
    dst.atoms.z = xtensor_to_cuda<double, kRealDtype>(src.atoms.z);
    dst.atoms.vx = xtensor_to_cuda<double, kRealDtype>(src.atoms.vx);
    dst.atoms.vy = xtensor_to_cuda<double, kRealDtype>(src.atoms.vy);
    dst.atoms.vz = xtensor_to_cuda<double, kRealDtype>(src.atoms.vz);
    dst.atoms.fx = xtensor_to_cuda<double, kRealDtype>(src.atoms.fx);
    dst.atoms.fy = xtensor_to_cuda<double, kRealDtype>(src.atoms.fy);
    dst.atoms.fz = xtensor_to_cuda<double, kRealDtype>(src.atoms.fz);
    dst.atoms.mass = xtensor_to_cuda<double, kRealDtype>(src.atoms.mass);
    dst.atoms.charge = xtensor_to_cuda<double, kRealDtype>(src.atoms.charge);
    dst.atoms.type = xtensor_to_cuda<int, kIntDtype>(src.atoms.type);
    dst.atoms.tag = xtensor_to_cuda<std::int64_t, kLongDtype>(src.atoms.tag);
    dst.atoms.image =
        xtensor2_to_cuda<std::int8_t, kImageDtype>(src.atoms.image);

    // System scalars: fresh zero Double Tensors on the device. The torch
    // frame mirrors core/system_scalars.h's SystemScalars field set.
    auto scalar_opts = ::torch::TensorOptions()
                           .dtype(::at::ScalarType::Double)
                           .device(::torch::kCUDA);
    dst.energy = ::torch::full({1}, src.scalars.energy, scalar_opts);
    dst.kinetic = ::torch::full({1}, src.scalars.kinetic, scalar_opts);
    dst.conserved = ::torch::full({1}, src.scalars.conserved, scalar_opts);
    {
        auto cpu_opts =
            ::torch::TensorOptions().dtype(::at::ScalarType::Double);
        dst.stress = ::torch::from_blob(const_cast<double*>(src.scalars.stress),
                                        {6}, cpu_opts)
                         .to(::torch::kCUDA);
        dst.stress_kinetic =
            ::torch::from_blob(const_cast<double*>(src.scalars.stress_kinetic),
                               {6}, cpu_opts)
                .to(::torch::kCUDA);
    }

    // Box: copy Mat3 (9 doubles, row-major)
    double box_data[9] = {src.box(0, 0), src.box(0, 1), src.box(0, 2),
                          src.box(1, 0), src.box(1, 1), src.box(1, 2),
                          src.box(2, 0), src.box(2, 1), src.box(2, 2)};
    auto box_opts = ::torch::TensorOptions().dtype(kPrecDtype);
    dst.box = ::torch::from_blob(box_data, {9}, box_opts).to(::torch::kCUDA);

    // Topology
    if (src.bonds.atomi.size() > 0) {
        dst.n_bonds_ = static_cast<int>(src.bonds.atomi.size());
        dst.bonds.atomi = xtensor_to_cuda<int, kIntDtype>(src.bonds.atomi);
        dst.bonds.atomj = xtensor_to_cuda<int, kIntDtype>(src.bonds.atomj);
        dst.bonds.type = xtensor_to_cuda<int, kIntDtype>(src.bonds.type);
    }
    if (src.angles.atomi.size() > 0) {
        dst.n_angles_ = static_cast<int>(src.angles.atomi.size());
        dst.angles.atomi = xtensor_to_cuda<int, kIntDtype>(src.angles.atomi);
        dst.angles.atomj = xtensor_to_cuda<int, kIntDtype>(src.angles.atomj);
        dst.angles.atomk = xtensor_to_cuda<int, kIntDtype>(src.angles.atomk);
        dst.angles.type = xtensor_to_cuda<int, kIntDtype>(src.angles.type);
    }
    if (src.dihedrals.atomi.size() > 0) {
        dst.n_dihedrals_ = static_cast<int>(src.dihedrals.atomi.size());
        dst.dihedrals.atomi =
            xtensor_to_cuda<int, kIntDtype>(src.dihedrals.atomi);
        dst.dihedrals.atomj =
            xtensor_to_cuda<int, kIntDtype>(src.dihedrals.atomj);
        dst.dihedrals.atomk =
            xtensor_to_cuda<int, kIntDtype>(src.dihedrals.atomk);
        dst.dihedrals.atoml =
            xtensor_to_cuda<int, kIntDtype>(src.dihedrals.atoml);
        dst.dihedrals.type =
            xtensor_to_cuda<int, kIntDtype>(src.dihedrals.type);
    }

    return dst;
}

void destroy(Frame& frame) { frame = Frame{}; }

} // namespace torch
} // namespace ATV

#endif // ATV_HAS_TORCH

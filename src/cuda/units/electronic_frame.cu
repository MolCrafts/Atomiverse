#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/units/electronic_frame.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace ATV::cuda::unit_boundary {
namespace {
struct Scales {
    double length, mass, velocity, charge, force, energy;
};
__host__ __device__ Box scaled_box(Box box, double length) {
    box.resident = nullptr;
    for (int k = 0; k < 9; ++k) {
        box.cell[k] *= length;
        box.inv_cell[k] /= length;
    }
    for (int k = 0; k < 3; ++k) {
        box.half_l[k] *= length;
        box.inv_l[k] /= length;
    }
    return box;
}
__device__ void scale_scalars(SystemScalars& s, double factor) {
    s.energy *= factor;
    s.energy_pair *= factor;
    s.energy_bond *= factor;
    s.energy_angle *= factor;
    s.energy_dihedral *= factor;
    s.energy_improper *= factor;
    s.kinetic *= factor;
    s.conserved *= factor;
    for (int k = 0; k < 6; ++k) {
        s.stress[k] *= factor;
        s.stress_kinetic[k] *= factor;
    }
}
__global__ void import_frame(Frame source, Frame target, Scales s,
                             Box source_box, Box* cache) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        *target.scalars = *source.scalars;
        scale_scalars(*target.scalars, s.energy);
        *cache = scaled_box(source_box.current(), s.length);
        for (int k = 0; k < 9; ++k)
            target.box[k] = cache->cell[k];
    }
    if (i >= source.n_atoms_)
        return;
    store_position(target.atoms.x, target.atoms.x_corr, i,
                   load_position(source.atoms.x, source.atoms.x_corr, i)
                       * s.length);
    store_position(target.atoms.y, target.atoms.y_corr, i,
                   load_position(source.atoms.y, source.atoms.y_corr, i)
                       * s.length);
    store_position(target.atoms.z, target.atoms.z_corr, i,
                   load_position(source.atoms.z, source.atoms.z_corr, i)
                       * s.length);
    target.atoms.mass[i] =
        source.atoms.mass ? source.atoms.mass[i] * s.mass : 0.;
    target.atoms.charge[i] =
        source.atoms.charge ? source.atoms.charge[i] * s.charge : 0.;
    target.atoms.vx[i] = source.atoms.vx ? source.atoms.vx[i] * s.velocity : 0.;
    target.atoms.vy[i] = source.atoms.vy ? source.atoms.vy[i] * s.velocity : 0.;
    target.atoms.vz[i] = source.atoms.vz ? source.atoms.vz[i] * s.velocity : 0.;
    const fixed_t* input[3] = {source.atoms.fx, source.atoms.fy,
                               source.atoms.fz};
    fixed_t* output[3] = {target.atoms.fx, target.atoms.fy, target.atoms.fz};
    for (int k = 0; k < 3; ++k) {
        const double value =
            input[k] ? decode_force<double>(input[k][i]) * s.force : 0.;
        if (!encode_force(value, output[k][i]))
            raise_force_fault(target.force_fault, ForceFault::kConversion, i);
    }
}
__global__ void export_frame(Frame source, Frame target, Scales s,
                             bool forces) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        *target.scalars = *source.scalars;
        scale_scalars(*target.scalars, 1. / s.energy);
    }
    if (i >= source.n_atoms_)
        return;
    if (target.atoms.charge)
        target.atoms.charge[i] = source.atoms.charge[i] / s.charge;
    if (!forces)
        return;
    const fixed_t* input[3] = {source.atoms.fx, source.atoms.fy,
                               source.atoms.fz};
    fixed_t* output[3] = {target.atoms.fx, target.atoms.fy, target.atoms.fz};
    for (int k = 0; k < 3; ++k)
        if (output[k]) {
            const double value = decode_force<double>(input[k][i]) / s.force;
            if (!encode_force(value, output[k][i]))
                raise_force_fault(target.force_fault, ForceFault::kConversion,
                                  i);
        }
}
} // namespace
struct ElectronicFrame::Storage {
    Frame native;
    Box host_box;
    Box* device_box = nullptr;
    Scales scales;
    std::vector<void*> allocations;
    ~Storage() {
        for (auto* p : allocations)
            cudaFree(p);
    }
    template <class T>
    T* allocate(std::size_t count) {
        T* p = nullptr;
        ATV_CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
        allocations.push_back(p);
        return p;
    }
};
ElectronicFrame::ElectronicFrame() = default;
ElectronicFrame::~ElectronicFrame() = default;
void ElectronicFrame::configure(Context& context, Frame& parent,
                                bool electronic) {
    storage_.reset();
    if (!electronic)
        return;
    const auto& source = context.units.physical();
    const auto native = ATV::units::PhysicalUnits::from_style("electronic");
    if (source.length().scale == native.length().scale
        && source.energy().scale == native.energy().scale
        && source.mass().scale == native.mass().scale
        && source.charge().scale == native.charge().scale)
        return;
    auto data = std::make_unique<Storage>();
    const auto factor = [](ATV::units::Unit a, ATV::units::Unit b) {
        return ATV::units::Conversion(a, b).scale();
    };
    data->scales = {factor(source.length(), native.length()),
                    factor(source.mass(), native.mass()),
                    factor(source.velocity(), native.velocity()),
                    factor(source.charge(), native.charge()),
                    factor(source.force(), native.force()),
                    factor(source.energy(), native.energy())};
    data->native = parent;
    auto& f = data->native;
    const auto count = std::max(1, parent.capacity());
    f.atoms.x = data->allocate<pos_t>(count);
    f.atoms.y = data->allocate<pos_t>(count);
    f.atoms.z = data->allocate<pos_t>(count);
    if constexpr (kPositionCorrection) {
        f.atoms.x_corr = data->allocate<pos_t>(count);
        f.atoms.y_corr = data->allocate<pos_t>(count);
        f.atoms.z_corr = data->allocate<pos_t>(count);
    }
    f.atoms.vx = data->allocate<mixed_t>(count);
    f.atoms.vy = data->allocate<mixed_t>(count);
    f.atoms.vz = data->allocate<mixed_t>(count);
    f.atoms.mass = data->allocate<mixed_t>(count);
    f.atoms.charge = data->allocate<force_t>(count);
    f.atoms.fx = data->allocate<fixed_t>(3 * count);
    f.atoms.fy = f.atoms.fx + count;
    f.atoms.fz = f.atoms.fy + count;
    f.scalars = data->allocate<SystemScalars>(1);
    // Faults belong to the execution owner. Aliasing its fault record also
    // avoids racing a scalar copy against per-row encoding failures.
    f.force_fault = parent.force_fault;
    if (!f.force_fault) {
        f.force_fault = data->allocate<ForceFault>(1);
        ATV_CUDA_CHECK(
            cudaMemsetAsync(f.force_fault, 0, sizeof(ForceFault),
                            static_cast<cudaStream_t>(context.stream)));
    }
    f.box = data->allocate<geom_t>(9);
    data->device_box = data->allocate<Box>(1);
    storage_ = std::move(data);
}
ElectronicFrame::Scope::Scope(ElectronicFrame& adapter, Context& context,
                              Frame& parent, bool refresh)
    : adapter_(adapter), context_(context), parent_(parent) {
    if (!adapter_.storage_)
        return;
    parent_snapshot_.emplace(parent);
    saved_box_.emplace(context.host_box);
    auto& s = *adapter_.storage_;
    if (parent.capacity() > s.native.capacity())
        throw std::logic_error(
            "electronic unit adapter capacity changed; rebuild the solver");
    s.native.n_atoms_ = parent.n_atoms_;
    s.native.n_local_ = parent.n_local_;
    s.native.n_ghost_ = parent.n_ghost_;
    if (refresh) {
        import_frame<<<std::max(1, (parent.n_atoms() + 127) / 128), 128, 0,
                       static_cast<cudaStream_t>(context.stream)>>>(
            parent, s.native, s.scales, *saved_box_, s.device_box);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
    s.host_box = scaled_box(*saved_box_, s.scales.length);
    s.host_box.resident = s.device_box;
    context.host_box = s.host_box;
    saved_units_.emplace(std::move(context.units));
    inherited_ = context.exec.inherited_units;
    if (!inherited_)
        context.exec.inherited_units = &*saved_units_;
    context.units = ATV::units::UnitSystem::electronic();
    context.units.dielectric = saved_units_->dielectric;
    // Keep the Backend's one Frame address stable. Its resident view is
    // temporarily native, so borrowed child stages and backend dispatch
    // observe the same Frame without creating a second execution owner.
    parent_ = s.native;
}
ElectronicFrame::Scope::~Scope() {
    if (adapter_.storage_) {
        parent_ = *parent_snapshot_;
        context_.units = std::move(*saved_units_);
        context_.exec.inherited_units = inherited_;
        context_.host_box = *saved_box_;
    }
}
Frame& ElectronicFrame::Scope::frame() { return parent_; }
void ElectronicFrame::Scope::publish() {
    if (!adapter_.storage_)
        return;
    auto& s = *adapter_.storage_;
    export_frame<<<std::max(1, (parent_.n_atoms() + 127) / 128), 128, 0,
                   static_cast<cudaStream_t>(context_.stream)>>>(
        s.native, *parent_snapshot_, s.scales, context_.exec.forces_requested);
    ATV_CUDA_CHECK(cudaGetLastError());
}
} // namespace ATV::cuda::unit_boundary

/// @file cuda/potentials/periodic_torsion.cu
/// @brief The table-driven periodic torsion kernel (charmm / fourier / cvff).

#include "atomiverse/core/dihedral_geom.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/energy_term.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/potentials/periodic_torsion.cuh"

#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace ATV {
namespace cuda {

namespace {

constexpr int kTorsionBlock = 128;

struct TableView {
    const int* offset;
    const double* K;
    const int* n;
    const double* cos_delta;
    const double* sin_delta;
    const int* active;
};

struct TupleView {
    const int *i, *j, *k, *l, *type;
};

// One tuple per thread in FP64 (the CPU loop's arithmetic): geometry, the
// type's term sum, the dE/dφ → dV/dcos map and the four forces; energy and
// the six virial components reduce in one CUB collective.
__global__ __launch_bounds__(kTorsionBlock, 4) void periodic_torsion_k(
    const pos_t* __restrict__ x, const pos_t* __restrict__ y,
    const pos_t* __restrict__ z, fixed_t* __restrict__ fx,
    fixed_t* __restrict__ fy, fixed_t* __restrict__ fz, ForceFault* fault,
    EnergyTerm energy, acc_t* __restrict__ stress, TupleView tuples,
    TableView table, cuda::Box box_arg, int n_tuples) {
    const auto box = box_arg.current().as<double>();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    acc_t v = 0;
    acc_t my_v[6] = {};
    DeferredForceAdds<12> adds;
    if (q < n_tuples && table.active[tuples.type[q]]) {
        const int i1 = tuples.i[q], i2 = tuples.j[q], i3 = tuples.k[q],
                  i4 = tuples.l[q], t = tuples.type[q];
        dihedral_geom::Geometry g;
        auto min_image = [&box](double& ax, double& ay, double& az) {
            box.min_image(ax, ay, az);
        };
        if (dihedral_geom::fill(x[i1], y[i1], z[i1], x[i2], y[i2], z[i2], x[i3],
                                y[i3], z[i3], x[i4], y[i4], z[i4], min_image,
                                g)) {
            double e, dE_dphi, d2E_dphi2;
            torsion_sum(table.offset[t], table.offset[t + 1], table.K, table.n,
                        table.cos_delta, table.sin_delta, g.cos_phi, g.sin_phi,
                        e, dE_dphi, d2E_dphi2);
            const double dV_dcos = dihedral_geom::dV_dcos_from_dE_dphi(
                dE_dphi, g.sin_phi, g.cos_phi, d2E_dphi2);
            double f1x, f1y, f1z, f2x, f2y, f2z, f3x, f3y, f3z, f4x, f4y, f4z;
            dihedral_geom::forces_from_dV_dcos(g, dV_dcos, f1x, f1y, f1z, f2x,
                                               f2y, f2z, f3x, f3y, f3z, f4x,
                                               f4y, f4z);
            v = acc_t(e);
            adds.add(0, fx, i1, f1x, fault);
            adds.add(1, fy, i1, f1y, fault);
            adds.add(2, fz, i1, f1z, fault);
            adds.add(3, fx, i2, f2x, fault);
            adds.add(4, fy, i2, f2y, fault);
            adds.add(5, fz, i2, f2z, fault);
            adds.add(6, fx, i3, f3x, fault);
            adds.add(7, fy, i3, f3y, fault);
            adds.add(8, fz, i3, f3z, fault);
            adds.add(9, fx, i4, f4x, fault);
            adds.add(10, fy, i4, f4y, fault);
            adds.add(11, fz, i4, f4z, fault);
            // CPU Voigt6: (−b1, F1) + (b2, F3 + F4) + (b3, F4).
            const double g34x = f3x + f4x, g34y = f3y + f4y, g34z = f3z + f4z;
            my_v[0] = acc_t(-g.b1x * f1x + g.b2x * g34x + g.b3x * f4x);
            my_v[1] = acc_t(-g.b1y * f1y + g.b2y * g34y + g.b3y * f4y);
            my_v[2] = acc_t(-g.b1z * f1z + g.b2z * g34z + g.b3z * f4z);
            my_v[3] = acc_t(-g.b1x * f1y + g.b2x * g34y + g.b3x * f4y);
            my_v[4] = acc_t(-g.b1x * f1z + g.b2x * g34z + g.b3x * f4z);
            my_v[5] = acc_t(-g.b1y * f1z + g.b2y * g34z + g.b3y * f4z);
        }
    }
    // Null energy/stress: the step has no eflag/vflag (LAMMPS ev_set).
    if (energy || stress != nullptr) {
        ::cuda::std::array<acc_t, 7> my_vals = {
            v, my_v[0], my_v[1], my_v[2], my_v[3], my_v[4], my_v[5]};
        using Reduce =
            cub::BlockReduce<::cuda::std::array<acc_t, 7>, kTorsionBlock>;
        __shared__ typename Reduce::TempStorage reduction;
        const auto out =
            Reduce(reduction).Reduce(my_vals, SumComponents<acc_t, 7>{});
        if (threadIdx.x == 0) {
            if (energy)
                energy.add(out[0]);
            if (stress != nullptr) {
#pragma unroll
                for (int c = 0; c < 6; ++c)
                    atomicAdd(&stress[c], out[c + 1]);
            }
        }
    }
    adds.check(fault);
}

template <class T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    const std::size_t bytes = std::max<std::size_t>(1, host.size()) * sizeof(T);
    ATV_CUDA_CHECK(cudaMalloc(&device, bytes));
    if (!host.empty())
        ATV_CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                                  cudaMemcpyHostToDevice));
    return device;
}

} // namespace

struct PeriodicTorsionSet::State {
    int* offset = nullptr;
    double* K = nullptr;
    int* n = nullptr;
    double* cos_delta = nullptr;
    double* sin_delta = nullptr;
    int* active = nullptr;
    void release() {
        cudaFree(offset);
        cudaFree(K);
        cudaFree(n);
        cudaFree(cos_delta);
        cudaFree(sin_delta);
        cudaFree(active);
        offset = n = active = nullptr;
        K = cos_delta = sin_delta = nullptr;
    }
    ~State() { release(); }
};

PeriodicTorsionSet::PeriodicTorsionSet(
    std::vector<std::vector<TorsionTerm>> terms, Topology topology,
    const char* style)
    : table_(terms, style), topology_(topology), style_(style),
      state_(std::make_unique<State>()) {}

PeriodicTorsionSet::~PeriodicTorsionSet() = default;
PeriodicTorsionSet::PeriodicTorsionSet(PeriodicTorsionSet&&) noexcept = default;
PeriodicTorsionSet&
PeriodicTorsionSet::operator=(PeriodicTorsionSet&&) noexcept = default;

void PeriodicTorsionSet::build(const BuildContext& ctx) {
    auto& f = ctx.frame;
    const bool improper = topology_ == Topology::Improper;
    const int count = improper ? f.n_impropers() : f.n_dihedrals();
    // Build-time check (one D2H, allowed): every tuple type has a row.
    if (count > 0) {
        std::vector<int> types(static_cast<std::size_t>(count));
        ATV_CUDA_CHECK(cudaMemcpy(
            types.data(), improper ? f.impropers.type : f.dihedrals.type,
            count * sizeof(int), cudaMemcpyDeviceToHost));
        for (int t : types)
            if (t < 0 || t >= table_.n_types())
                throw std::runtime_error(
                    std::string(style_) + ": topology type " + std::to_string(t)
                    + " has no coefficients ("
                    + std::to_string(table_.n_types()) + " types given)");
    }
    state_->release();
    state_->offset = upload(table_.offset);
    state_->K = upload(table_.K);
    state_->n = upload(table_.n);
    state_->cos_delta = upload(table_.cos_delta);
    state_->sin_delta = upload(table_.sin_delta);
    std::vector<int> active(table_.active.begin(), table_.active.end());
    state_->active = upload(active);
}

void PeriodicTorsionSet::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const bool improper = topology_ == Topology::Improper;
    const int count = improper ? f.n_impropers() : f.n_dihedrals();
    if (count == 0)
        return;
    if (!state_->offset)
        throw std::logic_error(std::string(style_) + ": launch before build");
    const TupleView tuples =
        improper
            ? TupleView{f.impropers.atomi, f.impropers.atomj, f.impropers.atomk,
                        f.impropers.atoml, f.impropers.type}
            : TupleView{f.dihedrals.atomi, f.dihedrals.atomj, f.dihedrals.atomk,
                        f.dihedrals.atoml, f.dihedrals.type};
    const TableView table{state_->offset,    state_->K,         state_->n,
                          state_->cos_delta, state_->sin_delta, state_->active};
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    periodic_torsion_k<<<grid_size(count, kTorsionBlock), kTorsionBlock, 0,
                         stream>>>(
        f.atoms.x, f.atoms.y, f.atoms.z, f.atoms.fx, f.atoms.fy, f.atoms.fz,
        f.force_fault,
        energy_term(f.scalars,
                    improper ? EnergyKind::Improper : EnergyKind::Dihedral,
                    ctx.energy_requested),
        ctx.stress_requested ? &f.scalars->stress[0] : nullptr, tuples, table,
        ctx.runtime.host_box, count);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void PeriodicTorsionSet::destroy() { state_->release(); }

DihedralCharmm::DihedralCharmm(std::vector<double> K, std::vector<int> n,
                               std::vector<double> delta)
    : set_(charmm_torsion_terms(K, n, delta),
           PeriodicTorsionSet::Topology::Dihedral, "DihedralCharmm") {}

DihedralFourier::DihedralFourier(std::vector<std::vector<TorsionTerm>> terms)
    : set_(std::move(terms), PeriodicTorsionSet::Topology::Dihedral,
           "DihedralFourier") {}

ImproperCvff::ImproperCvff(std::vector<double> K, std::vector<int> d,
                           std::vector<int> n)
    : set_(cvff_torsion_terms(K, d, n), PeriodicTorsionSet::Topology::Improper,
           "ImproperCvff") {}

} // namespace cuda
} // namespace ATV

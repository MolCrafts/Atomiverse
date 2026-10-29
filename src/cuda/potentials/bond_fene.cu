/// @file src/cuda/potentials/bond_fene.cu
/// @brief CUDA BondFene — Kremer–Grest, same `fene_term` as the CPU bond.

#include "atomiverse/core/fene.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/potentials/bond_fene.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

namespace ATV {
namespace cuda {
namespace kernels {

__global__ __launch_bounds__(ATV::cuda::kDefaultBlock, 4) void fene_bond_k(
    const pos_t* __restrict__ x, const pos_t* __restrict__ y,
    const pos_t* __restrict__ z, fixed_t* __restrict__ fx,
    fixed_t* __restrict__ fy, fixed_t* __restrict__ fz, ForceFault* fault,
    acc_t* __restrict__ energy, acc_t* __restrict__ stress,
    const int* __restrict__ bi, const int* __restrict__ bj,
    const int* __restrict__ bt, const force_t* __restrict__ K,
    const force_t* __restrict__ R0, int n_types, cuda::Box box_arg,
    int n_bonds) {
    const auto box = box_arg.current().as<force_t>();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    acc_t v = 0;
    acc_t my_v[6] = {};
    DeferredForceAdds<6> adds;

    if (idx < n_bonds) {
        int i = bi[idx], j = bj[idx], t = bt[idx];
        force_t dx = x[j] - x[i];
        force_t dy = y[j] - y[i];
        force_t dz = z[j] - z[i];
        box.min_image(dx, dy, dz);
        const force_t r = sqrt(dx * dx + dy * dy + dz * dz);

        const bool typed = t >= 0 && t < n_types;
        const FeneTerm<force_t> term =
            typed ? fene_term(r, K[t], R0[t])
                  : FeneTerm<force_t>{force_t(0), force_t(0), true};
        if (!typed || term.overstretched) {
            raise_force_fault(fault, ForceFault::kBond, i);
        } else {
            v = acc_t(term.energy);
            // `dx` is r_j - r_i, so fpair * dx is the force on j.
            const force_t f_x = term.fpair * dx;
            const force_t f_y = term.fpair * dy;
            const force_t f_z = term.fpair * dz;
            adds.add(0, fx, j, f_x, fault);
            adds.add(1, fy, j, f_y, fault);
            adds.add(2, fz, j, f_z, fault);
            adds.add(3, fx, i, -f_x, fault);
            adds.add(4, fy, i, -f_y, fault);
            adds.add(5, fz, i, -f_z, fault);
            my_v[0] = acc_t(dx * f_x);
            my_v[1] = acc_t(dy * f_y);
            my_v[2] = acc_t(dz * f_z);
            my_v[3] = acc_t(dx * f_y);
            my_v[4] = acc_t(dx * f_z);
            my_v[5] = acc_t(dy * f_z);
        }
    }

    ::cuda::std::array<acc_t, 7> my_vals = {v,       my_v[0], my_v[1], my_v[2],
                                            my_v[3], my_v[4], my_v[5]};
    // Null energy/stress: the step has no eflag/vflag (LAMMPS ev_set). The
    // pointers are uniform, so skipping the collective is safe.
    if (energy != nullptr || stress != nullptr) {
        using Reduce =
            cub::BlockReduce<::cuda::std::array<acc_t, 7>, kDefaultBlock>;
        __shared__ typename Reduce::TempStorage reduction;
        const auto out =
            Reduce(reduction).Reduce(my_vals, SumComponents<acc_t, 7>{});
        if (threadIdx.x == 0) {
            if (energy != nullptr)
                atomicAdd(energy, out[0]);
            if (stress != nullptr) {
#pragma unroll
                for (int c = 0; c < 6; ++c)
                    atomicAdd(&stress[c], out[c + 1]);
            }
        }
    }
    adds.check(fault);
}

} // namespace kernels

struct BondFene::State {
    force_t* d_K = nullptr;
    force_t* d_R0 = nullptr;
    int n_types = 0;
};

BondFene::BondFene(std::vector<double> K, std::vector<double> R0)
    : K_(std::move(K)), R0_(std::move(R0)), state_(std::make_unique<State>()) {
    if (K_.size() != R0_.size() || K_.empty())
        throw std::invalid_argument(
            "BondFene: K and R0 must match and be non-empty");
    for (std::size_t t = 0; t < K_.size(); ++t) {
        if (!(K_[t] > 0.0) || !(R0_[t] > 0.0))
            throw std::invalid_argument("BondFene: K and R0 must be > 0");
    }
}

BondFene::~BondFene() = default;
BondFene::BondFene(BondFene&&) noexcept = default;
BondFene& BondFene::operator=(BondFene&&) noexcept = default;

void BondFene::build(const BuildContext& /*ctx*/) {
    int nt = static_cast<int>(K_.size());
    state_->n_types = nt;
    const std::vector<force_t> device_K(K_.begin(), K_.end());
    const std::vector<force_t> device_R0(R0_.begin(), R0_.end());
    cudaMalloc(&state_->d_K, nt * sizeof(force_t));
    cudaMalloc(&state_->d_R0, nt * sizeof(force_t));
    cudaMemcpy(state_->d_K, device_K.data(), nt * sizeof(force_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(state_->d_R0, device_R0.data(), nt * sizeof(force_t),
               cudaMemcpyHostToDevice);
}

void BondFene::compute(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    int nb = f.n_bonds();
    if (nb == 0)
        return;
    int block = kDefaultBlock;
    int grid = grid_size(static_cast<int>(nb));
    kernels::fene_bond_k<<<grid, block, 0, stream>>>(
        f.atoms.x, f.atoms.y, f.atoms.z, f.atoms.fx, f.atoms.fy, f.atoms.fz,
        f.force_fault, ctx.energy_requested ? &f.scalars->energy : nullptr,
        ctx.stress_requested ? &f.scalars->stress[0] : nullptr, f.bonds.atomi,
        f.bonds.atomj, f.bonds.type, state_->d_K, state_->d_R0, state_->n_types,
        ctx.runtime.host_box, nb);
}

void BondFene::destroy(const DestroyContext& /*ctx*/) {
    cudaFree(state_->d_K);
    cudaFree(state_->d_R0);
    *state_ = {};
}

} // namespace cuda
} // namespace ATV

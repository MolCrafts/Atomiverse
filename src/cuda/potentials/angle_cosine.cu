/// @file cuda/potentials/angle_cosine.cu
/// @brief CUDA AngleCosine — V = K(1 + cos θ) (kernel + class).
///
/// Port of LAMMPS `src/MOLECULE/angle_cosine.cpp` (`AngleCosine::compute`),
/// checkout SHA e8c4c278a8. LAMMPS: Copyright (2003) Sandia Corporation,
/// GNU General Public License. Launch shape, fixed-point force deposit and
/// the energy/virial block reduction follow `cuda::AngleHarmonic`.

#include "atomiverse/core/angle_cosine.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/energy_term.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/potentials/angle_cosine.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace ATV {
namespace cuda {

struct AngleCosine::State {
    force_t* d_K = nullptr;
    int n_types = 0;
};

AngleCosine::AngleCosine(std::vector<double> K)
    : K_(std::move(K)), state_(std::make_unique<State>()) {
    if (K_.empty())
        throw std::invalid_argument("AngleCosine: K must be non-empty");
    for (double k : K_)
        if (!std::isfinite(k))
            throw std::invalid_argument("AngleCosine: K must be finite");
}

AngleCosine::~AngleCosine() = default;
AngleCosine::AngleCosine(AngleCosine&&) noexcept = default;
AngleCosine& AngleCosine::operator=(AngleCosine&&) noexcept = default;

namespace {

__global__ __launch_bounds__(ATV::cuda::kDefaultBlock, 4) void cosine_angle_k(
    const pos_t* __restrict__ x, const pos_t* __restrict__ y,
    const pos_t* __restrict__ z, fixed_t* __restrict__ fx,
    fixed_t* __restrict__ fy, fixed_t* __restrict__ fz, ForceFault* fault,
    EnergyTerm energy, acc_t* __restrict__ stress, const int* __restrict__ ai,
    const int* __restrict__ aj, const int* __restrict__ ak,
    const int* __restrict__ at, const force_t* __restrict__ K,
    cuda::Box box_arg, int n_angles) {
    const auto box = box_arg.current().as<force_t>();
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Tail lanes contribute zero and still participate in the CUB collective.
    acc_t v = 0;
    acc_t my_v[6] = {};
    DeferredForceAdds<9> adds;

    if (idx < n_angles) {
        const int i = ai[idx], j = aj[idx], k = ak[idx];
        // LAMMPS delx1 = x[i1] - x[i2], delx2 = x[i3] - x[i2] (i2 = centre).
        force_t d1[3] = {x[i] - x[j], y[i] - y[j], z[i] - z[j]};
        force_t d2[3] = {x[k] - x[j], y[k] - y[j], z[k] - z[j]};
        box.min_image(d1[0], d1[1], d1[2]);
        box.min_image(d2[0], d2[1], d2[2]);

        const AngleCosineTerm<force_t> term =
            angle_cosine_term<force_t>(K[at[idx]], d1, d2);
        v = acc_t(term.energy);

        adds.add(0, fx, i, term.f1[0], fault);
        adds.add(1, fy, i, term.f1[1], fault);
        adds.add(2, fz, i, term.f1[2], fault);
        adds.add(3, fx, k, term.f3[0], fault);
        adds.add(4, fy, k, term.f3[1], fault);
        adds.add(5, fz, k, term.f3[2], fault);
        adds.add(6, fx, j, -(term.f1[0] + term.f3[0]), fault);
        adds.add(7, fy, j, -(term.f1[1] + term.f3[1]), fault);
        adds.add(8, fz, j, -(term.f1[2] + term.f3[2]), fault);

        // LAMMPS ev_tally: v_ab = d1_a F1_b + d2_a F3_b, Voigt xx yy zz xy xz yz.
        my_v[0] = acc_t(d1[0] * term.f1[0] + d2[0] * term.f3[0]);
        my_v[1] = acc_t(d1[1] * term.f1[1] + d2[1] * term.f3[1]);
        my_v[2] = acc_t(d1[2] * term.f1[2] + d2[2] * term.f3[2]);
        my_v[3] = acc_t(d1[0] * term.f1[1] + d2[0] * term.f3[1]);
        my_v[4] = acc_t(d1[0] * term.f1[2] + d2[0] * term.f3[2]);
        my_v[5] = acc_t(d1[1] * term.f1[2] + d2[1] * term.f3[2]);
    }

    ::cuda::std::array<acc_t, 7> my_vals = {v,       my_v[0], my_v[1], my_v[2],
                                            my_v[3], my_v[4], my_v[5]};
    // Null energy/stress: the step has no eflag/vflag (LAMMPS ev_set). The
    // pointers are uniform, so skipping the collective is safe.
    if (energy || stress != nullptr) {
        using Reduce =
            cub::BlockReduce<::cuda::std::array<acc_t, 7>, kDefaultBlock>;
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

} // anonymous namespace

void AngleCosine::build(const BuildContext& ctx) {
    const int nt = static_cast<int>(K_.size());
    state_->n_types = nt;

    // Build-time type check (a launch never branches on host data). The
    // default `supports_input_changes` rejects identity edits, so the angle
    // list cannot change under a built component.
    const int na = ctx.frame.n_angles();
    if (na > 0) {
        std::vector<int> types(static_cast<std::size_t>(na));
        ATV_CUDA_CHECK(cudaMemcpy(types.data(), ctx.frame.angles.type,
                                  types.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
        for (int t : types)
            if (t < 0 || t >= nt)
                throw std::invalid_argument(
                    "AngleCosine: angle type " + std::to_string(t)
                    + " outside the K table of " + std::to_string(nt));
    }

    const std::vector<force_t> K_dev(K_.begin(), K_.end());
    ATV_CUDA_CHECK(cudaMalloc(&state_->d_K, nt * sizeof(force_t)));
    ATV_CUDA_CHECK(cudaMemcpy(state_->d_K, K_dev.data(), nt * sizeof(force_t),
                              cudaMemcpyHostToDevice));
}

void AngleCosine::compute(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    const int na = f.n_angles();
    if (na == 0)
        return;
    const int block = kDefaultBlock;
    const int grid = grid_size(na);
    cosine_angle_k<<<grid, block, 0, stream>>>(
        f.atoms.x, f.atoms.y, f.atoms.z, f.atoms.fx, f.atoms.fy, f.atoms.fz,
        f.force_fault,
        energy_term(f.scalars, EnergyKind::Angle, ctx.energy_requested),
        ctx.stress_requested ? &f.scalars->stress[0] : nullptr, f.angles.atomi,
        f.angles.atomj, f.angles.atomk, f.angles.type, state_->d_K,
        ctx.runtime.host_box, na);
}

void AngleCosine::destroy(const DestroyContext& /*ctx*/) {
    if (state_->d_K)
        cudaFree(state_->d_K);
    *state_ = {};
}

} // namespace cuda
} // namespace ATV

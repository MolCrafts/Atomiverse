/// @file cuda/potentials/angle_harmonic.cu
/// @brief CUDA AngleHarmonic — V = K(θ − θ₀)² (kernel + class).

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/potentials/angle_harmonic.cuh"
#include <cmath>
#include <cuda_runtime.h>

namespace ATV {
namespace cuda {

struct AngleHarmonic::State {
    force_t* d_K = nullptr;
    force_t* d_theta0 = nullptr; // in radians
    int n_types = 0;
};

AngleHarmonic::AngleHarmonic(std::vector<double> K, std::vector<double> theta0)
    : K_(std::move(K)), theta0_(std::move(theta0)),
      state_(std::make_unique<State>()) {}

AngleHarmonic::~AngleHarmonic() = default;
AngleHarmonic::AngleHarmonic(AngleHarmonic&&) noexcept = default;
AngleHarmonic& AngleHarmonic::operator=(AngleHarmonic&&) noexcept = default;

namespace {

__global__ __launch_bounds__(ATV::cuda::kDefaultBlock, 4) void harmonic_angle_k(
    const pos_t* __restrict__ x, const pos_t* __restrict__ y,
    const pos_t* __restrict__ z, fixed_t* __restrict__ fx,
    fixed_t* __restrict__ fy, fixed_t* __restrict__ fz, ForceFault* fault,
    acc_t* __restrict__ energy, acc_t* __restrict__ stress,
    const int* __restrict__ ai, const int* __restrict__ aj,
    const int* __restrict__ ak, const int* __restrict__ at,
    const force_t* __restrict__ K, const force_t* __restrict__ theta0,
    cuda::Box box_arg, int n_angles) {
    const auto box = box_arg.current().as<force_t>();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Tail lanes contribute zero and still participate in the CUB collective.
    acc_t v = 0;
    acc_t my_v[6] = {};
    DeferredForceAdds<9> adds;

    if (idx < n_angles) {
        int i = ai[idx], j = aj[idx], k = ak[idx];
        int type = at[idx];

        // Vectors from center atom j
        force_t rij_x = x[i] - x[j], rij_y = y[i] - y[j], rij_z = z[i] - z[j];
        force_t rkj_x = x[k] - x[j], rkj_y = y[k] - y[j], rkj_z = z[k] - z[j];

        // Minimum image via cached inv_l on the by-value Box
        // (cuda-md-review-cleanup-04-box-by-value-pbc).
        box.min_image(rij_x, rij_y, rij_z);
        box.min_image(rkj_x, rkj_y, rkj_z);

        force_t rij2 = rij_x * rij_x + rij_y * rij_y + rij_z * rij_z;
        force_t rkj2 = rkj_x * rkj_x + rkj_y * rkj_y + rkj_z * rkj_z;
        force_t rij_inv = rsqrt(rij2);
        force_t rkj_inv = rsqrt(rkj2);

        force_t cos_theta =
            (rij_x * rkj_x + rij_y * rkj_y + rij_z * rkj_z) * rij_inv * rkj_inv;
        cos_theta = fmin(fmax(cos_theta, force_t(-1)), force_t(1));
        force_t theta = acos(cos_theta);
        force_t dtheta = theta - theta0[type];

        // V = K(θ - θ0)^2
        v = acc_t(K[type] * dtheta * dtheta);

        // dV/dθ = 2K(θ - θ0)
        force_t sin_theta = sqrt(force_t(1) - cos_theta * cos_theta);
        constexpr force_t kSinEpsilon = force_t(1e-6);
        if (sin_theta < kSinEpsilon)
            sin_theta = kSinEpsilon;
        force_t prefactor = -force_t(2) * K[type] * dtheta / sin_theta;

        // LAMMPS angle_harmonic.cpp force decomposition (verbatim port):
        //   a11 = a·cos / r_ij²; a22 = a·cos / r_kj²; a12 = -a / (r_ij·r_kj)
        //   F_i = a11·r_ij + a12·r_kj
        //   F_k = a22·r_kj + a12·r_ij
        //   F_j = -F_i - F_k  (Newton 3rd / translation invariance)
        force_t rij_inv2 = rij_inv * rij_inv;
        force_t rkj_inv2 = rkj_inv * rkj_inv;
        force_t cross_inv = rij_inv * rkj_inv;

        force_t a11 = prefactor * cos_theta * rij_inv2;
        force_t a22 = prefactor * cos_theta * rkj_inv2;
        force_t a12 = -prefactor * cross_inv;

        force_t fi_x = a11 * rij_x + a12 * rkj_x;
        force_t fi_y = a11 * rij_y + a12 * rkj_y;
        force_t fi_z = a11 * rij_z + a12 * rkj_z;

        force_t fk_x = a22 * rkj_x + a12 * rij_x;
        force_t fk_y = a22 * rkj_y + a12 * rij_y;
        force_t fk_z = a22 * rkj_z + a12 * rij_z;

        adds.add(0, fx, i, fi_x, fault);
        adds.add(1, fy, i, fi_y, fault);
        adds.add(2, fz, i, fi_z, fault);
        adds.add(3, fx, k, fk_x, fault);
        adds.add(4, fy, k, fk_y, fault);
        adds.add(5, fz, k, fk_z, fault);
        adds.add(6, fx, j, -(fi_x + fk_x), fault);
        adds.add(7, fy, j, -(fi_y + fk_y), fault);
        adds.add(8, fz, j, -(fi_z + fk_z), fault);

        // Angle virial (LAMMPS ev_tally convention):
        //   v_ab = r_ij_a·F_i_b + r_kj_a·F_k_b  (atom j contributes
        //   nothing extra thanks to translation invariance.)
        my_v[0] = acc_t(rij_x * fi_x + rkj_x * fk_x);
        my_v[1] = acc_t(rij_y * fi_y + rkj_y * fk_y);
        my_v[2] = acc_t(rij_z * fi_z + rkj_z * fk_z);
        my_v[3] = acc_t(rij_x * fi_y + rkj_x * fk_y);
        my_v[4] = acc_t(rij_x * fi_z + rkj_x * fk_z);
        my_v[5] = acc_t(rij_y * fi_z + rkj_y * fk_z);
    }

    // Reduce energy and six virial components in one CUB collective.
    ::cuda::std::array<acc_t, 7> my_vals = {v,       my_v[0], my_v[1], my_v[2],
                                            my_v[3], my_v[4], my_v[5]};

    using Reduce =
        cub::BlockReduce<::cuda::std::array<acc_t, 7>, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const auto out =
        Reduce(reduction).Reduce(my_vals, SumComponents<acc_t, 7>{});
    if (threadIdx.x == 0) {
        atomicAdd(energy, out[0]);
        if (stress != nullptr) {
#pragma unroll
            for (int c = 0; c < 6; ++c)
                atomicAdd(&stress[c], out[c + 1]);
        }
    }
    adds.check(fault);
}

} // anonymous namespace

void AngleHarmonic::build(const BuildContext& /*ctx*/) {
    int nt = static_cast<int>(K_.size());
    state_->n_types = nt;

    // theta0_ is already in RADIANS by API contract (matches the CPU
    // sibling, which uses theta0_[type] directly and is tested via the
    // PE benchmark passing `109.5° * π/180`). The previous extra
    // multiplication by `kDegToRad` here shrank θ₀ by another factor of
    // π/180, so e.g. 109.5°→1.91 rad became 1.91·0.01745≈0.0334 rad.
    // Every angle then evaluated V ≈ K·(1.91)² instead of K·(0)² at
    // equilibrium → ~9× over-count vs CPU/LAMMPS in stage 1 of the MSU
    // PE deck, plus mis-aligned restoring forces.
    std::vector<force_t> theta0_dev(nt);
    const std::vector<force_t> K_dev(K_.begin(), K_.end());
    for (int i = 0; i < nt; ++i)
        theta0_dev[i] = static_cast<force_t>(theta0_[i]);

    cudaMalloc(&state_->d_K, nt * sizeof(force_t));
    cudaMalloc(&state_->d_theta0, nt * sizeof(force_t));
    cudaMemcpy(state_->d_K, K_dev.data(), nt * sizeof(force_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(state_->d_theta0, theta0_dev.data(), nt * sizeof(force_t),
               cudaMemcpyHostToDevice);
}

void AngleHarmonic::compute(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    int na = f.n_angles();
    if (na == 0)
        return;

    int block = kDefaultBlock;
    int grid = grid_size(static_cast<int>(na));
    constexpr int shmem = 0;
    harmonic_angle_k<<<grid, block, shmem, stream>>>(
        f.atoms.x, f.atoms.y, f.atoms.z, f.atoms.fx, f.atoms.fy, f.atoms.fz,
        f.force_fault, &f.scalars->energy, &f.scalars->stress[0],
        f.angles.atomi, f.angles.atomj, f.angles.atomk, f.angles.type,
        state_->d_K, state_->d_theta0, ctx.runtime.host_box, na);
}

void AngleHarmonic::destroy(const DestroyContext& /*ctx*/) {
    cudaFree(state_->d_K);
    cudaFree(state_->d_theta0);
    *state_ = {};
}

} // namespace cuda
} // namespace ATV

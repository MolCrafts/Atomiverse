/// @file cuda/potentials/angle_harmonic.cu
/// @brief CUDA AngleHarmonic — V = K(θ − θ₀)² (kernel + class).

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/potentials/angle_harmonic.cuh"
#include <cmath>
#include <cuda_runtime.h>

namespace ATV {
namespace cuda {

struct AngleHarmonic::State {
    work_t* d_K = nullptr;
    work_t* d_theta0 = nullptr; // in radians
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
    const real_t* __restrict__ x, const real_t* __restrict__ y,
    const real_t* __restrict__ z, real_t* __restrict__ fx,
    real_t* __restrict__ fy, real_t* __restrict__ fz,
    prec_t* __restrict__ energy, prec_t* __restrict__ stress,
    const int* __restrict__ ai, const int* __restrict__ aj,
    const int* __restrict__ ak, const int* __restrict__ at,
    const work_t* __restrict__ K, const work_t* __restrict__ theta0,
    cuda::Box box, int n_angles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // See bond_harmonic.cu for the rationale: out-of-range threads must
    // stay alive through the block-level energy / virial reductions
    // with v=0 and my_v=0 so __syncthreads() inside the reduction is
    // hit by every thread. The previous `if (idx >= n_angles) return;`
    // left those threads' `sdata` entries UNINITIALISED — sum was
    // garbage, atomicAdd-ed into `energy` and `stress[]` once per block.
    work_t v = work_t(0);
    work_t my_v[6] = {work_t(0), work_t(0), work_t(0),
                      work_t(0), work_t(0), work_t(0)};

    if (idx < n_angles) {
        int i = ai[idx], j = aj[idx], k = ak[idx];
        int type = at[idx];

        // Vectors from center atom j
        work_t rij_x = x[i] - x[j], rij_y = y[i] - y[j], rij_z = z[i] - z[j];
        work_t rkj_x = x[k] - x[j], rkj_y = y[k] - y[j], rkj_z = z[k] - z[j];

        // Minimum image via cached inv_l on the by-value Box
        // (cuda-md-review-cleanup-04-box-by-value-pbc).
        box.min_image(rij_x, rij_y, rij_z);
        box.min_image(rkj_x, rkj_y, rkj_z);

        work_t rij2 = rij_x * rij_x + rij_y * rij_y + rij_z * rij_z;
        work_t rkj2 = rkj_x * rkj_x + rkj_y * rkj_y + rkj_z * rkj_z;
        work_t rij_inv = rsqrt(rij2);
        work_t rkj_inv = rsqrt(rkj2);

        work_t cos_theta =
            (rij_x * rkj_x + rij_y * rkj_y + rij_z * rkj_z) * rij_inv * rkj_inv;
        cos_theta = fmin(fmax(cos_theta, work_t(-1)), work_t(1));
        work_t theta = acos(cos_theta);
        work_t dtheta = theta - theta0[type];

        // V = K(θ - θ0)^2
        v = K[type] * dtheta * dtheta;

        // dV/dθ = 2K(θ - θ0)
        work_t sin_theta = sqrt(work_t(1) - cos_theta * cos_theta);
        constexpr work_t kSinEpsilon = work_t(1e-6);
        if (sin_theta < kSinEpsilon)
            sin_theta = kSinEpsilon;
        work_t prefactor = -work_t(2) * K[type] * dtheta / sin_theta;

        // LAMMPS angle_harmonic.cpp force decomposition (verbatim port):
        //   a11 = a·cos / r_ij²; a22 = a·cos / r_kj²; a12 = -a / (r_ij·r_kj)
        //   F_i = a11·r_ij + a12·r_kj
        //   F_k = a22·r_kj + a12·r_ij
        //   F_j = -F_i - F_k  (Newton 3rd / translation invariance)
        work_t rij_inv2 = rij_inv * rij_inv;
        work_t rkj_inv2 = rkj_inv * rkj_inv;
        work_t cross_inv = rij_inv * rkj_inv;

        work_t a11 = prefactor * cos_theta * rij_inv2;
        work_t a22 = prefactor * cos_theta * rkj_inv2;
        work_t a12 = -prefactor * cross_inv;

        work_t fi_x = a11 * rij_x + a12 * rkj_x;
        work_t fi_y = a11 * rij_y + a12 * rkj_y;
        work_t fi_z = a11 * rij_z + a12 * rkj_z;

        work_t fk_x = a22 * rkj_x + a12 * rij_x;
        work_t fk_y = a22 * rkj_y + a12 * rij_y;
        work_t fk_z = a22 * rkj_z + a12 * rij_z;

        atomicAdd(&fx[i], static_cast<real_t>(fi_x));
        atomicAdd(&fy[i], static_cast<real_t>(fi_y));
        atomicAdd(&fz[i], static_cast<real_t>(fi_z));
        atomicAdd(&fx[k], static_cast<real_t>(fk_x));
        atomicAdd(&fy[k], static_cast<real_t>(fk_y));
        atomicAdd(&fz[k], static_cast<real_t>(fk_z));
        atomicAdd(&fx[j], static_cast<real_t>(-(fi_x + fk_x)));
        atomicAdd(&fy[j], static_cast<real_t>(-(fi_y + fk_y)));
        atomicAdd(&fz[j], static_cast<real_t>(-(fi_z + fk_z)));

        // Angle virial (LAMMPS ev_tally convention):
        //   v_ab = r_ij_a·F_i_b + r_kj_a·F_k_b  (atom j contributes
        //   nothing extra thanks to translation invariance.)
        my_v[0] = rij_x * fi_x + rkj_x * fk_x;
        my_v[1] = rij_y * fi_y + rkj_y * fk_y;
        my_v[2] = rij_z * fi_z + rkj_z * fk_z;
        my_v[3] = rij_x * fi_y + rkj_x * fk_y;
        my_v[4] = rij_x * fi_z + rkj_x * fk_z;
        my_v[5] = rij_y * fi_z + rkj_y * fk_z;
    }

    // Packed warp-shuffle reduction: 7 scalars (energy + 6 stress
    // components) via cg_reduce
    // (cuda-md-review-cleanup-05-reductions-and-atomics).
    extern __shared__ work_t warp_scratch[]; // per-warp staging cells
    work_t my_vals[7] = {v,       my_v[0], my_v[1], my_v[2],
                         my_v[3], my_v[4], my_v[5]};
    work_t out[7];
    block_reduce_sum_n<7>(my_vals, warp_scratch, out);
    if (threadIdx.x == 0) {
        atomicAdd(energy, static_cast<prec_t>(out[0]));
        if (stress != nullptr) {
#pragma unroll
            for (int c = 0; c < 6; ++c)
                atomicAdd(&stress[c], static_cast<prec_t>(out[c + 1]));
        }
    }
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
    std::vector<work_t> theta0_dev(nt);
    const std::vector<work_t> K_dev(K_.begin(), K_.end());
    for (int i = 0; i < nt; ++i)
        theta0_dev[i] = static_cast<work_t>(theta0_[i]);

    cudaMalloc(&state_->d_K, nt * sizeof(work_t));
    cudaMalloc(&state_->d_theta0, nt * sizeof(work_t));
    cudaMemcpy(state_->d_K, K_dev.data(), nt * sizeof(work_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(state_->d_theta0, theta0_dev.data(), nt * sizeof(work_t),
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
    int shmem = warps_per_block(block) * sizeof(work_t);
    harmonic_angle_k<<<grid, block, shmem, stream>>>(
        f.atoms.x, f.atoms.y, f.atoms.z, f.atoms.fx, f.atoms.fy, f.atoms.fz,
        &f.scalars->energy, &f.scalars->stress[0], f.angles.atomi,
        f.angles.atomj, f.angles.atomk, f.angles.type, state_->d_K,
        state_->d_theta0, ctx.runtime.host_box, na);
}

void AngleHarmonic::destroy(const DestroyContext& /*ctx*/) {
    cudaFree(state_->d_K);
    cudaFree(state_->d_theta0);
    *state_ = {};
}

} // namespace cuda
} // namespace ATV

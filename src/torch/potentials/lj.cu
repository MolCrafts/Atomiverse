/// @file torch/md/lj.cu
/// @brief Differentiable LJ operator -- forward reuses cuda kernel, backward is analytical.

#ifdef ATV_HAS_TORCH

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/potentials/lj_kernel.cuh"
#include "atomiverse/torch/potentials/lj.cuh"

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

namespace ATV {
namespace torch {

namespace {

/// @brief Backward kernel: compute dE/d(epsilon), dE/d(sigma) from pair distances.
///
/// dE/d(epsilon) = sum_pairs 4 * (sr12 - sr6) - v_shift/epsilon
/// dE/d(sigma) = sum_pairs 4*epsilon * (-12*sigma^11/r^12 + 6*sigma^5/r^6)
__global__ void lj_param_grad(cuda::real_t* __restrict__ grad_eps_out,
                              cuda::real_t* __restrict__ grad_sig_out,
                              const cuda::real_t* __restrict__ pdist,
                              const int* __restrict__ n_pairs_ptr,
                              cuda::work_t epsilon, cuda::work_t sigma,
                              cuda::work_t cutoff, cuda::work_t v_shift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = *n_pairs_ptr;

    cuda::work_t de = cuda::work_t(0);
    cuda::work_t ds = cuda::work_t(0);
    if (idx < n_pairs) {
        cuda::work_t r = static_cast<cuda::work_t>(pdist[idx]);
        if (r < cutoff
            && r >= static_cast<cuda::work_t>(cuda::kernels::kMinPairDist)) {
            cuda::work_t inv_r = cuda::work_t(1) / r;
            cuda::work_t sr = sigma * inv_r;
            cuda::work_t sr2 = sr * sr;
            cuda::work_t sr6 = sr2 * sr2 * sr2;
            cuda::work_t sr12 = sr6 * sr6;

            // dV/d(epsilon) = 4*(sr12 - sr6) - v_shift/epsilon
            de = cuda::work_t(4) * (sr12 - sr6) - v_shift / epsilon;

            // dV/d(sigma): d/dsigma [4*eps*(sigma/r)^12 - 4*eps*(sigma/r)^6]
            //   = 4*eps * (12*sigma^11/r^12 - 6*sigma^5/r^6)
            cuda::work_t inv_sig = cuda::work_t(1) / sigma;
            ds = cuda::work_t(4) * epsilon
                 * (cuda::work_t(12) * sr12 * inv_sig
                    - cuda::work_t(6) * sr6 * inv_sig);
        }
    }

    // Block reduction for grad_eps
    extern __shared__ cuda::work_t sdata[];
    cuda::work_t* s_eps = sdata;
    cuda::work_t* s_sig = sdata + blockDim.x;
    s_eps[threadIdx.x] = de;
    s_sig[threadIdx.x] = ds;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_eps[threadIdx.x] += s_eps[threadIdx.x + s];
            s_sig[threadIdx.x] += s_sig[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (s_eps[0] != cuda::work_t(0))
            atomicAdd(grad_eps_out, static_cast<cuda::real_t>(s_eps[0]));
        if (s_sig[0] != cuda::work_t(0))
            atomicAdd(grad_sig_out, static_cast<cuda::real_t>(s_sig[0]));
    }
}

/// @brief autograd::Function for LJ forward+backward.
struct LJFunction : public ::torch::autograd::Function<LJFunction> {
    static ::torch::autograd::variable_list
    forward(::torch::autograd::AutogradContext* ctx, Tensor x, Tensor y,
            Tensor z, Tensor fx, Tensor fy, Tensor fz, Tensor energy,
            Tensor pair_i, Tensor pair_j, Tensor pair_dx, Tensor pair_dy,
            Tensor pair_dz, Tensor pair_dist, Tensor n_pairs_t, Tensor epsilon,
            Tensor sigma, double cutoff, int64_t n_atoms, int64_t max_pairs) {
        cuda::real_t eps_val = epsilon.item<cuda::real_t>();
        cuda::real_t sig_val = sigma.item<cuda::real_t>();
        cuda::real_t cut_val = static_cast<cuda::real_t>(cutoff);
        cuda::real_t v_shift =
            cuda::kernels::lj_shift(eps_val, sig_val, cut_val);

        auto stream = ::c10::cuda::getCurrentCUDAStream().stream();
        int block = cuda::kDefaultBlock;
        int grid = cuda::grid_size(static_cast<int>(max_pairs));

        cuda::kernels::
            lj_force<<<grid, block, block * sizeof(cuda::real_t), stream>>>(
                fx.data_ptr<cuda::real_t>(), fy.data_ptr<cuda::real_t>(),
                fz.data_ptr<cuda::real_t>(), energy.data_ptr<cuda::prec_t>(),
                pair_i.data_ptr<int>(), pair_j.data_ptr<int>(),
                pair_dx.data_ptr<cuda::real_t>(),
                pair_dy.data_ptr<cuda::real_t>(),
                pair_dz.data_ptr<cuda::real_t>(),
                pair_dist.data_ptr<cuda::real_t>(), eps_val, sig_val, cut_val,
                v_shift, n_pairs_t.data_ptr<int>());

        // Save for backward
        ctx->save_for_backward({pair_i, pair_j, pair_dx, pair_dy, pair_dz,
                                pair_dist, n_pairs_t, epsilon, sigma, fx, fy,
                                fz});
        ctx->saved_data["cutoff"] = cutoff;
        ctx->saved_data["n_atoms"] = n_atoms;
        ctx->saved_data["max_pairs"] = max_pairs;
        ctx->saved_data["v_shift"] = static_cast<double>(v_shift);

        return {energy};
    }

    static ::torch::autograd::variable_list
    backward(::torch::autograd::AutogradContext* ctx,
             ::torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto pair_i = saved[0], pair_j = saved[1];
        auto pair_dx = saved[2], pair_dy = saved[3], pair_dz = saved[4];
        auto pair_dist = saved[5], n_pairs_t = saved[6];
        auto epsilon = saved[7], sigma = saved[8];
        auto fx = saved[9], fy = saved[10], fz = saved[11];

        auto grad_energy = grad_outputs[0]; // scalar
        double ge = grad_energy.item<double>();

        double cutoff = ctx->saved_data["cutoff"].toDouble();
        int64_t n_atoms = ctx->saved_data["n_atoms"].toInt();
        int64_t max_pairs = ctx->saved_data["max_pairs"].toInt();
        cuda::work_t v_shift =
            static_cast<cuda::work_t>(ctx->saved_data["v_shift"].toDouble());

        // dE/dx = -fx (force is negative gradient of energy)
        // These are already computed in forward; scale by grad_energy
        Tensor grad_x = Tensor();
        Tensor grad_y = Tensor();
        Tensor grad_z = Tensor();
        if (ge != 0.0) {
            grad_x = -fx * ge;
            grad_y = -fy * ge;
            grad_z = -fz * ge;
        }

        // dE/d(epsilon) and dE/d(sigma) via custom kernel
        Tensor grad_eps = Tensor();
        Tensor grad_sig = Tensor();

        cuda::work_t eps_val =
            static_cast<cuda::work_t>(epsilon.item<double>());
        cuda::work_t sig_val = static_cast<cuda::work_t>(sigma.item<double>());

        if (epsilon.requires_grad() || sigma.requires_grad()) {
            auto opts = ::torch::TensorOptions()
                            .dtype(kRealDtype)
                            .device(::torch::kCUDA);
            auto d_grad_eps = ::torch::zeros({1}, opts);
            auto d_grad_sig = ::torch::zeros({1}, opts);

            auto stream = ::c10::cuda::getCurrentCUDAStream().stream();
            int block = cuda::kDefaultBlock;
            int grid = cuda::grid_size(static_cast<int>(max_pairs));
            lj_param_grad<<<grid, block, 2 * block * sizeof(cuda::work_t),
                            stream>>>(
                d_grad_eps.data_ptr<cuda::real_t>(),
                d_grad_sig.data_ptr<cuda::real_t>(),
                pair_dist.data_ptr<cuda::real_t>(), n_pairs_t.data_ptr<int>(),
                eps_val, sig_val, static_cast<cuda::work_t>(cutoff), v_shift);

            if (epsilon.requires_grad())
                grad_eps = d_grad_eps * ge;
            if (sigma.requires_grad())
                grad_sig = d_grad_sig * ge;
        }

        // Return grads in same order as forward args:
        // x, y, z, fx, fy, fz, energy, pair_i, pair_j,
        // pair_dx, pair_dy, pair_dz, pair_dist, n_pairs_t,
        // epsilon, sigma, cutoff, n_atoms, max_pairs
        return {
            grad_x,   grad_y,   grad_z,   Tensor(), Tensor(), Tensor(),
            Tensor(),           // fx, fy, fz, energy
            Tensor(), Tensor(), // pair_i, pair_j
            Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), // pair_dx..n_pairs
            grad_eps, grad_sig, Tensor(), Tensor(), Tensor()}; // cutoff, n_atoms, max_pairs
    }
};

} // anonymous namespace

Tensor lj(Frame& frame, const NeighborList& neighbors, const Tensor& epsilon,
          const Tensor& sigma, double cutoff) {
    int na = frame.n_atoms();

    // Pair list is owned by the NeighborList, not by the Frame.
    const NeighborList::PairList& pairs = neighbors.pairs();
    int np = pairs.n_pairs();

    // Estimate max pairs for grid sizing
    int64_t max_pairs =
        np > 0 ? static_cast<int64_t>(np) : static_cast<int64_t>(na) * 50;

    // n_pairs as device tensor (for kernel)
    auto int_opts =
        ::torch::TensorOptions().dtype(kIntDtype).device(::torch::kCUDA);
    Tensor n_pairs_t;
    if (pairs.pair_i.defined() && pairs.pair_i.numel() > 0) {
        // If n_pairs is set, create a device scalar
        n_pairs_t = ::torch::full({1}, static_cast<int64_t>(np), int_opts);
    } else {
        n_pairs_t = ::torch::zeros({1}, int_opts);
    }

    // Zero energy before accumulation
    frame.energy.zero_();

    auto result = LJFunction::apply(
        frame.atoms.x, frame.atoms.y, frame.atoms.z, frame.atoms.fx,
        frame.atoms.fy, frame.atoms.fz, frame.energy, pairs.pair_i,
        pairs.pair_j, pairs.pair_dx, pairs.pair_dy, pairs.pair_dz,
        pairs.pair_dist, n_pairs_t, epsilon, sigma, static_cast<double>(cutoff),
        static_cast<int64_t>(na), max_pairs);

    return result[0];
}

} // namespace torch
} // namespace ATV

#endif // ATV_HAS_TORCH

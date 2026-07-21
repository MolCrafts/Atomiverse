/// @file torch/bonded/harmonic_bond.cu
/// @brief Differentiable harmonic bond -- forward reuses cuda kernel, backward analytical.

#ifdef ATV_HAS_TORCH

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/potentials/harmonic_bond_kernel.cuh"
#include "atomiverse/torch/potentials/harmonic_bond.cuh"

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

namespace ATV {
namespace torch {

namespace {

/// @brief Backward kernel for harmonic bond parameter gradients.
///
/// Computes dE/dK[t] = sum_bonds_of_type_t (r - r0[t])^2
///          dE/dr0[t] = sum_bonds_of_type_t -2*K[t]*(r - r0[t])
/// Also computes dE/dx for each atom.
__global__ void harmonic_bond_backward(
    const cuda::real_t* __restrict__ x, const cuda::real_t* __restrict__ y,
    const cuda::real_t* __restrict__ z, cuda::real_t* __restrict__ grad_x,
    cuda::real_t* __restrict__ grad_y, cuda::real_t* __restrict__ grad_z,
    cuda::real_t* __restrict__ grad_K, cuda::real_t* __restrict__ grad_r0,
    const int* __restrict__ bi, const int* __restrict__ bj,
    const int* __restrict__ bt, const cuda::real_t* __restrict__ K,
    const cuda::real_t* __restrict__ r0, const cuda::real_t* __restrict__ box,
    cuda::work_t grad_energy, int n_bonds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bonds)
        return;

    int i = bi[idx], j = bj[idx], t = bt[idx];

    cuda::work_t dx =
        static_cast<cuda::work_t>(x[j]) - static_cast<cuda::work_t>(x[i]);
    cuda::work_t dy =
        static_cast<cuda::work_t>(y[j]) - static_cast<cuda::work_t>(y[i]);
    cuda::work_t dz =
        static_cast<cuda::work_t>(z[j]) - static_cast<cuda::work_t>(z[i]);

    // Minimum image convention (orthorhombic)
    cuda::work_t bx = static_cast<cuda::work_t>(box[0]);
    cuda::work_t by = static_cast<cuda::work_t>(box[4]);
    cuda::work_t bz = static_cast<cuda::work_t>(box[8]);
    dx -= bx * round(dx / bx);
    dy -= by * round(dy / by);
    dz -= bz * round(dz / bz);

    cuda::work_t r = sqrt(dx * dx + dy * dy + dz * dz);
    cuda::work_t dr = r - static_cast<cuda::work_t>(r0[t]);

    // dE/dx: F = -2K(r-r0) * r_hat, dE/dx = -F
    cuda::work_t f_over_r =
        cuda::work_t(2) * static_cast<cuda::work_t>(K[t]) * dr / r;
    cuda::real_t dEdx_j =
        static_cast<cuda::real_t>(f_over_r * dx * grad_energy);
    cuda::real_t dEdy_j =
        static_cast<cuda::real_t>(f_over_r * dy * grad_energy);
    cuda::real_t dEdz_j =
        static_cast<cuda::real_t>(f_over_r * dz * grad_energy);

    atomicAdd(&grad_x[j], dEdx_j);
    atomicAdd(&grad_y[j], dEdy_j);
    atomicAdd(&grad_z[j], dEdz_j);
    atomicAdd(&grad_x[i], -dEdx_j);
    atomicAdd(&grad_y[i], -dEdy_j);
    atomicAdd(&grad_z[i], -dEdz_j);

    // dE/dK[t] = (r - r0)^2
    atomicAdd(&grad_K[t], static_cast<cuda::real_t>(dr * dr * grad_energy));

    // dE/dr0[t] = -2*K*(r - r0)
    atomicAdd(&grad_r0[t],
              static_cast<cuda::real_t>(cuda::work_t(-2)
                                        * static_cast<cuda::work_t>(K[t]) * dr
                                        * grad_energy));
}

struct HarmonicBondFunction
    : public ::torch::autograd::Function<HarmonicBondFunction> {
    static ::torch::autograd::variable_list
    forward(::torch::autograd::AutogradContext* ctx, Tensor x, Tensor y,
            Tensor z, Tensor fx, Tensor fy, Tensor fz, Tensor energy,
            Tensor box, Tensor bond_i, Tensor bond_j, Tensor bond_type,
            Tensor K, Tensor r0, int64_t n_bonds) {
        auto stream = ::c10::cuda::getCurrentCUDAStream().stream();
        int block = cuda::kDefaultBlock;
        int grid = cuda::grid_size(static_cast<int>(n_bonds));

        cuda::kernels::harmonic_bond_k<<<
            grid, block, block * sizeof(cuda::real_t), stream>>>(
            x.data_ptr<cuda::real_t>(), y.data_ptr<cuda::real_t>(),
            z.data_ptr<cuda::real_t>(), fx.data_ptr<cuda::real_t>(),
            fy.data_ptr<cuda::real_t>(), fz.data_ptr<cuda::real_t>(),
            energy.data_ptr<cuda::prec_t>(), bond_i.data_ptr<int>(),
            bond_j.data_ptr<int>(), bond_type.data_ptr<int>(),
            K.data_ptr<cuda::real_t>(), r0.data_ptr<cuda::real_t>(),
            box.data_ptr<cuda::real_t>(), static_cast<int>(n_bonds));

        ctx->save_for_backward(
            {x, y, z, box, bond_i, bond_j, bond_type, K, r0});
        ctx->saved_data["n_bonds"] = n_bonds;

        return {energy};
    }

    static ::torch::autograd::variable_list
    backward(::torch::autograd::AutogradContext* ctx,
             ::torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto x = saved[0], y = saved[1], z = saved[2], box = saved[3];
        auto bond_i = saved[4], bond_j = saved[5], bond_type = saved[6];
        auto K = saved[7], r0 = saved[8];

        int64_t n_bonds = ctx->saved_data["n_bonds"].toInt();
        cuda::work_t ge =
            static_cast<cuda::work_t>(grad_outputs[0].item<double>());

        int64_t n_atoms = x.numel();
        int64_t n_types = K.numel();

        auto opts =
            ::torch::TensorOptions().dtype(kRealDtype).device(x.device());
        auto grad_x = ::torch::zeros({n_atoms}, opts);
        auto grad_y = ::torch::zeros({n_atoms}, opts);
        auto grad_z = ::torch::zeros({n_atoms}, opts);
        auto grad_K = ::torch::zeros({n_types}, opts);
        auto grad_r0 = ::torch::zeros({n_types}, opts);

        auto stream = ::c10::cuda::getCurrentCUDAStream().stream();
        int block = cuda::kDefaultBlock;
        int grid = cuda::grid_size(static_cast<int>(n_bonds));

        harmonic_bond_backward<<<grid, block, 0, stream>>>(
            x.data_ptr<cuda::real_t>(), y.data_ptr<cuda::real_t>(),
            z.data_ptr<cuda::real_t>(), grad_x.data_ptr<cuda::real_t>(),
            grad_y.data_ptr<cuda::real_t>(), grad_z.data_ptr<cuda::real_t>(),
            grad_K.data_ptr<cuda::real_t>(), grad_r0.data_ptr<cuda::real_t>(),
            bond_i.data_ptr<int>(), bond_j.data_ptr<int>(),
            bond_type.data_ptr<int>(), K.data_ptr<cuda::real_t>(),
            r0.data_ptr<cuda::real_t>(), box.data_ptr<cuda::real_t>(), ge,
            static_cast<int>(n_bonds));

        // Return grads: x, y, z, fx, fy, fz, energy, box,
        //               bond_i, bond_j, bond_type, K, r0, n_bonds
        return {grad_x,   grad_y,   grad_z,   Tensor(),
                Tensor(), Tensor(), Tensor(), Tensor(), // fx,fy,fz,energy,box
                Tensor(), Tensor(), Tensor(),  // bond_i, bond_j, bond_type
                grad_K,   grad_r0,  Tensor()}; // K, r0, n_bonds
    }
};

} // anonymous namespace

Tensor harmonic_bond(Frame& frame, const Tensor& K, const Tensor& r0) {
    int nb = frame.n_bonds();
    if (nb == 0)
        return frame.energy;

    frame.energy.zero_();

    auto result = HarmonicBondFunction::apply(
        frame.atoms.x, frame.atoms.y, frame.atoms.z, frame.atoms.fx,
        frame.atoms.fy, frame.atoms.fz, frame.energy, frame.box,
        frame.bonds.atomi, frame.bonds.atomj, frame.bonds.type, K, r0,
        static_cast<int64_t>(nb));

    return result[0];
}

} // namespace torch
} // namespace ATV

#endif // ATV_HAS_TORCH

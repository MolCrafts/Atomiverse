/// @file cuda/scf/uhf_fock.cu
/// @brief Device UHF Fock: s/p integrals + V^σ = J[Pα+Pβ] − K[P^σ].
///
/// PySCF `scf.uhf.get_veff`. The (ik|jl) index matches `build_G_kernel` in
/// rhf_fock.cu so the closed-shell reduction is the RHF factor
/// ½ Pij Pkl − ¼ Pik Pjl when Pα = Pβ = P/2.

#ifdef ATV_HAS_HF

#include "atomiverse/cuda/scf/internal/uhf_fock.cuh"

#include "atomiverse/core/gto.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/scf/device_integrals.cuh"
#include "cublas_dispatch.cuh"

#include "atomiverse/cuda/frame/types.cuh"

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {
namespace {

__global__ void
uhf_scale_columns_by_rsqrt_kernel(prec_t* __restrict__ U,
                                  const prec_t* __restrict__ lambda, int n,
                                  prec_t thresh) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (i >= n || j >= n)
        return;
    const prec_t lam = lambda[j];
    const prec_t s = (lam > thresh) ? rsqrt(lam) : prec_t(0);
    U[i + j * n] *= s;
}

__global__ void uhf_pack_atom_pos_bohr_kernel(prec_t* __restrict__ atom_pos,
                                              const real_t* __restrict__ x_A,
                                              const real_t* __restrict__ y_A,
                                              const real_t* __restrict__ z_A,
                                              int natm, prec_t a2b) {
    const int a = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (a >= natm)
        return;
    const int o = a * 3;
    atom_pos[o + 0] = static_cast<prec_t>(x_A[a]) * a2b;
    atom_pos[o + 1] = static_cast<prec_t>(y_A[a]) * a2b;
    atom_pos[o + 2] = static_cast<prec_t>(z_A[a]) * a2b;
}

__global__ void uhf_expand_shell_centers_kernel(
    prec_t* __restrict__ shell_centers, const int* __restrict__ atom_of,
    const prec_t* __restrict__ atom_pos, int n_shells) {
    const int s = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (s >= n_shells)
        return;
    const int a = atom_of[s];
    shell_centers[s * 3 + 0] = atom_pos[a * 3 + 0];
    shell_centers[s * 3 + 1] = atom_pos[a * 3 + 1];
    shell_centers[s * 3 + 2] = atom_pos[a * 3 + 2];
}

__global__ void
uhf_nuclear_repulsion_pairs_kernel(double* __restrict__ v_nuc_out,
                                   const prec_t* __restrict__ atom_pos,
                                   const int* __restrict__ atom_Z, int natm) {
    const int npairs = natm * (natm - 1) / 2;
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    double contrib = 0.0;
    if (idx < npairs) {
        int i = idx;
        int a = 0;
        int row = natm - 1;
        while (i >= row && row > 0) {
            i -= row;
            ++a;
            --row;
        }
        const int b = a + 1 + i;
        const prec_t* Ra = atom_pos + a * 3;
        const prec_t* Rb = atom_pos + b * 3;
        const double dx = static_cast<double>(Ra[0] - Rb[0]);
        const double dy = static_cast<double>(Ra[1] - Rb[1]);
        const double dz = static_cast<double>(Ra[2] - Rb[2]);
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > 0.0)
            contrib = static_cast<double>(atom_Z[a] * atom_Z[b]) / sqrt(r2);
    }
    __shared__ double warp_scratch[8];
    const double block_sum =
        cuda::block_reduce_sum<double>(contrib, warp_scratch);
    if (threadIdx.x == 0 && block_sum != 0.0)
        atomicAdd(v_nuc_out, block_sum);
}

/// F^σ_ij = H_ij + Σ_kl (ij|kl)(Pα+Pβ)_kl − Σ_kl (ik|jl) P^σ_kl.
__global__ void uhf_build_fock_kernel(prec_t* __restrict__ Fa,
                                      prec_t* __restrict__ Fb,
                                      const prec_t* __restrict__ H,
                                      const prec_t* __restrict__ Pa,
                                      const prec_t* __restrict__ Pb,
                                      const prec_t* __restrict__ ERI, int n) {
    const int total = n * n;
    const int n2 = n * n;
    const int n3 = n2 * n;
    const int stride = static_cast<int>(blockDim.x * gridDim.x);
    for (int ij = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         ij < total; ij += stride) {
        const int j = ij / n;
        const int i = ij - j * n;
        prec_t jtot = prec_t(0);
        prec_t ka = prec_t(0);
        prec_t kb = prec_t(0);
        for (int l = 0; l < n; ++l) {
            const int base_ij_l = i + j * n + l * n3;
            for (int k = 0; k < n; ++k) {
                const prec_t pa = static_cast<prec_t>(Pa[k + l * n]);
                const prec_t pb = static_cast<prec_t>(Pb[k + l * n]);
                const prec_t eri_j =
                    static_cast<prec_t>(ERI[base_ij_l + k * n2]);
                const prec_t eri_k =
                    static_cast<prec_t>(ERI[i + k * n + j * n2 + l * n3]);
                jtot += (pa + pb) * eri_j;
                ka += pa * eri_k;
                kb += pb * eri_k;
            }
        }
        const prec_t h = static_cast<prec_t>(H[ij]);
        Fa[ij] = static_cast<prec_t>(h + jtot - ka);
        Fb[ij] = static_cast<prec_t>(h + jtot - kb);
    }
}

void alloc_spin_mats(UhfBeta& b, int n, cudaStream_t stream) {
    const int nn = n * n;
    const size_t bytes = static_cast<size_t>(nn) * sizeof(prec_t);
    cudaMalloc(&b.H, bytes);
    cudaMalloc(&b.P, bytes);
    cudaMalloc(&b.P_old, bytes);
    cudaMalloc(&b.C, bytes);
    cudaMalloc(&b.C_prime, bytes);
    cudaMalloc(&b.eig, static_cast<size_t>(n) * sizeof(prec_t));
    cudaMalloc(&b.work, bytes);
    cudaMalloc(&b.F_prime, bytes);
    cudaMemsetAsync(b.P, 0, bytes, stream);
    cudaMemsetAsync(b.P_old, 0, bytes, stream);
    cudaMemsetAsync(b.C_prime, 0, bytes, stream);
}

} // namespace

UHFFockDevice::UHFFockDevice(int n_shells, int n_ao, int n_prim,
                             const double* alpha_mol, const double* coeff_mol,
                             const int* L_mol, const int* atom_of_mol,
                             int n_alpha, int n_beta)
    : n_shells_(n_shells), n_ao_(n_ao), n_prim_(n_prim), n_alpha_(n_alpha),
      n_beta_(n_beta) {
    const std::size_t nflat =
        static_cast<std::size_t>(n_shells) * static_cast<std::size_t>(n_prim);
    alpha_mol_.assign(alpha_mol, alpha_mol + nflat);
    coeff_mol_.assign(coeff_mol, coeff_mol + nflat);
    L_mol_.assign(L_mol, L_mol + n_shells);
    atom_of_mol_.assign(atom_of_mol, atom_of_mol + n_shells);
}

void UHFFockDevice::build(const SolverContext& ctx, SCFWorkspace& ws,
                          UhfBeta& beta) {
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    const int natm = ctx.frame.n_atoms();
    const int n_shells = n_shells_;
    const int n = n_ao_;
    const int nn = n * n;
    const int n4 = nn * nn;

    for (int s = 0; s < n_shells; ++s) {
        if (L_mol_[static_cast<std::size_t>(s)] > 1) {
            throw std::invalid_argument(
                "cuda::scf::UHF: missing kernel launch_eri_sp for angular "
                "momentum l>1 (shell "
                + std::to_string(s) + " has l="
                + std::to_string(L_mol_[static_cast<std::size_t>(s)])
                + "). Use cpu::scf::UHF.");
        }
    }
    if (n_alpha_ > n || n_beta_ > n) {
        throw std::invalid_argument(
            "cuda::scf::UHF: occupation exceeds the AO count (n_ao="
            + std::to_string(n) + ", n_alpha=" + std::to_string(n_alpha_)
            + ", n_beta=" + std::to_string(n_beta_) + ").");
    }

    ws.n = n;
    ws.n_occ = n_alpha_;
    beta.n_occ = n_beta_;
    ws.converged = false;
    ws.n_shells = n_shells;
    ws.n_prim = n_prim_;
    ws.natm = natm;
    ws.pure_s_shells = true;
    for (int s = 0; s < n_shells; ++s) {
        if (L_mol_[static_cast<std::size_t>(s)] != 0) {
            ws.pure_s_shells = false;
            break;
        }
    }

    std::vector<int> h_L(static_cast<std::size_t>(n_shells));
    std::vector<int> h_ao_start(static_cast<std::size_t>(n_shells));
    std::vector<int> h_atom_of(static_cast<std::size_t>(n_shells));
    std::vector<int> h_ao_shell(static_cast<std::size_t>(n));
    std::vector<int> h_ao_cart(static_cast<std::size_t>(n));
    int ao = 0;
    for (int s = 0; s < n_shells; ++s) {
        const int L = L_mol_[static_cast<std::size_t>(s)];
        h_L[static_cast<std::size_t>(s)] = L;
        h_atom_of[static_cast<std::size_t>(s)] =
            atom_of_mol_[static_cast<std::size_t>(s)];
        h_ao_start[static_cast<std::size_t>(s)] = ao;
        const int ncart = (L + 1) * (L + 2) / 2;
        for (int c = 0; c < ncart; ++c) {
            h_ao_shell[static_cast<std::size_t>(ao)] = s;
            h_ao_cart[static_cast<std::size_t>(ao)] = c;
            ++ao;
        }
    }
    if (ao != n) {
        throw std::runtime_error(
            "UHFFockDevice::build: n_ao mismatch vs shell L list");
    }

    const std::size_t nflat =
        static_cast<std::size_t>(n_shells) * static_cast<std::size_t>(n_prim_);
    std::vector<prec_t> h_alpha(nflat), h_coeff(nflat);
    for (int s = 0; s < n_shells; ++s) {
        const int L = h_L[static_cast<std::size_t>(s)];
        for (int p = 0; p < n_prim_; ++p) {
            const std::size_t k = static_cast<std::size_t>(s * n_prim_ + p);
            const double a = alpha_mol_[k];
            const double c = coeff_mol_[k];
            h_alpha[k] = static_cast<prec_t>(a);
            const double norm =
                (L == 0) ? ATV::phys_norm_s(a) : ATV::phys_norm_p(a);
            h_coeff[k] = static_cast<prec_t>(c * norm);
        }
    }

    cudaMalloc(&ws.d_shell_alpha, nflat * sizeof(prec_t));
    cudaMalloc(&ws.d_shell_coeff, nflat * sizeof(prec_t));
    cudaMalloc(&ws.d_shell_center, n_shells * 3 * sizeof(prec_t));
    cudaMalloc(&ws.d_atom_Z, natm * sizeof(int));
    cudaMalloc(&ws.d_atom_pos, natm * 3 * sizeof(prec_t));
    int* d_ao_start = nullptr;
    cudaMalloc(&ws.d_shell_L, n_shells * sizeof(int));
    cudaMalloc(&d_ao_start, n_shells * sizeof(int));
    cudaMalloc(&ws.d_shell_atom, n_shells * sizeof(int));
    cudaMalloc(&ws.d_ao_shell, n * sizeof(int));
    cudaMalloc(&ws.d_ao_cart, n * sizeof(int));

    cudaMemcpyAsync(ws.d_shell_alpha, h_alpha.data(), nflat * sizeof(prec_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ws.d_shell_coeff, h_coeff.data(), nflat * sizeof(prec_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ws.d_shell_L, h_L.data(), n_shells * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_ao_start, h_ao_start.data(), n_shells * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ws.d_shell_atom, h_atom_of.data(), n_shells * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ws.d_ao_shell, h_ao_shell.data(), n * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ws.d_ao_cart, h_ao_cart.data(), n * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ws.d_atom_Z, ctx.frame.atoms.type,
                    static_cast<std::size_t>(natm) * sizeof(int),
                    cudaMemcpyDeviceToDevice, stream);

    const prec_t a2b = static_cast<prec_t>(ATV::units::kAngstromInBohr);
    uhf_pack_atom_pos_bohr_kernel<<<cuda::grid_size(natm, cuda::kDefaultBlock),
                                    cuda::kDefaultBlock, 0, stream>>>(
        ws.d_atom_pos, ctx.frame.atoms.x, ctx.frame.atoms.y, ctx.frame.atoms.z,
        natm, a2b);
    uhf_expand_shell_centers_kernel<<<cuda::grid_size(n_shells,
                                                      cuda::kDefaultBlock),
                                      cuda::kDefaultBlock, 0, stream>>>(
        ws.d_shell_center, ws.d_shell_atom, ws.d_atom_pos, n_shells);

    cudaMalloc(&ws.d_V_nuc, sizeof(double));
    cudaMemsetAsync(ws.d_V_nuc, 0, sizeof(double), stream);
    if (natm >= 2) {
        const int npairs = natm * (natm - 1) / 2;
        uhf_nuclear_repulsion_pairs_kernel<<<
            cuda::grid_size(npairs, cuda::kDefaultBlock), cuda::kDefaultBlock,
            0, stream>>>(ws.d_V_nuc, ws.d_atom_pos, ws.d_atom_Z, natm);
    }
    ws.V_nuc = 0.0;

    const size_t mat_bytes = static_cast<size_t>(nn) * sizeof(prec_t);
    cudaMalloc(&ws.H, mat_bytes);
    cudaMalloc(&ws.P, mat_bytes);
    cudaMalloc(&ws.P_old, mat_bytes);
    cudaMalloc(&ws.C, mat_bytes);
    cudaMalloc(&ws.C_prime, mat_bytes);
    cudaMemsetAsync(ws.C_prime, 0, mat_bytes, stream);
    cudaMalloc(&ws.eig, static_cast<size_t>(n) * sizeof(prec_t));
    cudaMalloc(&ws.work, mat_bytes);
    cudaMalloc(&ws.Hcore, mat_bytes);
    cudaMalloc(&ws.S, mat_bytes);
    cudaMalloc(&ws.X, mat_bytes);
    cudaMalloc(&ws.F_prime, mat_bytes);
    cudaMalloc(&ws.ERI, static_cast<size_t>(n4) * sizeof(prec_t));
    cudaMalloc(&d_temp_, mat_bytes);
    cudaMalloc(&d_T_, mat_bytes);
    cudaMalloc(&d_V_, mat_bytes);
    cudaMalloc(&ws.d_residual, sizeof(prec_t));
    cudaMalloc(&ws.d_scalar_scratch, 2 * sizeof(prec_t));
    {
        int* h_map = nullptr;
        cudaHostAlloc(reinterpret_cast<void**>(&h_map), sizeof(int),
                      cudaHostAllocMapped);
        ws.h_converged_mapped = h_map;
        *ws.h_converged_mapped = 0;
        void* d_ptr = nullptr;
        cudaHostGetDevicePointer(&d_ptr, ws.h_converged_mapped, 0);
        ws.d_converged = static_cast<int*>(d_ptr);
    }
    ws.device_sp_shells = true;
    cudaMalloc(&ws.d_W, mat_bytes);
    ATV_CUDA_CHECK(cudaMalloc(&ws.d_force_Ptotal, mat_bytes));
    ATV_CUDA_CHECK(cudaMalloc(&ws.d_force_Wbeta, mat_bytes));
    if constexpr (kMixedPrecision)
        ATV_CUDA_CHECK(cudaMalloc(&ws.d_force_gradient,
                                  std::size_t(natm) * 3 * sizeof(prec_t)));
    cudaMemsetAsync(ws.P, 0, mat_bytes, stream);
    cudaMemsetAsync(ws.P_old, 0, mat_bytes, stream);
    alloc_spin_mats(beta, n, stream);

    launch_1e_integrals_sp(ws.S, d_T_, d_V_, ws.d_shell_alpha, ws.d_shell_coeff,
                           ws.d_shell_center, ws.d_shell_L, d_ao_start,
                           ws.d_ao_shell, ws.d_ao_cart, ws.d_atom_Z,
                           ws.d_atom_pos, n_shells, n, n_prim_, natm, stream);
    launch_eri_sp(ws.ERI, ws.d_shell_alpha, ws.d_shell_coeff, ws.d_shell_center,
                  ws.d_shell_L, d_ao_start, ws.d_ao_shell, ws.d_ao_cart,
                  n_shells, n, n_prim_, stream);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    cublas_handle_ = handle;

    cudaMemcpyAsync(ws.Hcore, d_T_, mat_bytes, cudaMemcpyDeviceToDevice,
                    stream);
    prec_t one = prec_t(1);
    detail::axpy(handle, nn, &one, d_V_, 1, ws.Hcore, 1);

    cusolverDnHandle_t solver;
    cusolverDnCreate(&solver);
    cusolverDnSetStream(solver, stream);
    cusolver_handle_ = solver;

    cudaMemcpyAsync(ws.X, ws.S, mat_bytes, cudaMemcpyDeviceToDevice, stream);
    int lwork = 0;
    prec_t* d_eigenvalues = nullptr;
    cudaMalloc(&d_eigenvalues, static_cast<size_t>(n) * sizeof(prec_t));
    int* d_info = nullptr;
    cudaMalloc(&d_info, sizeof(int));
    detail::syevd_bufferSize(solver, CUSOLVER_EIG_MODE_VECTOR,
                             CUBLAS_FILL_MODE_LOWER, n, ws.X, n, d_eigenvalues,
                             &lwork);
    prec_t* d_work_solver = nullptr;
    cudaMalloc(&d_work_solver, static_cast<size_t>(lwork) * sizeof(prec_t));
    detail::syevd(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n,
                  ws.X, n, d_eigenvalues, d_work_solver, lwork, d_info);

    cudaMemcpyAsync(d_temp_, ws.X, mat_bytes, cudaMemcpyDeviceToDevice, stream);
    constexpr int kBx = 32;
    constexpr int kBy = 8;
    dim3 block(kBx, kBy);
    dim3 grid((n + kBx - 1) / kBx, (n + kBy - 1) / kBy);
    uhf_scale_columns_by_rsqrt_kernel<<<grid, block, 0, stream>>>(
        d_temp_, d_eigenvalues, n, static_cast<prec_t>(1.0e-8));
    prec_t zero = prec_t(0);
    detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &one, d_temp_, n,
                 ws.X, n, &zero, d_V_, n);
    cudaMemcpyAsync(ws.X, d_V_, mat_bytes, cudaMemcpyDeviceToDevice, stream);

    cudaStreamSynchronize(stream);
    cudaFree(d_eigenvalues);
    cudaFree(d_work_solver);
    cudaFree(d_info);
    cudaFree(d_ao_start);
}

void UHFFockDevice::step(const SolverContext& ctx, SCFWorkspace& ws,
                         UhfBeta& beta) {
    if (ws.converged)
        return;
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto handle = static_cast<cublasHandle_t>(cublas_handle_);
    cublasSetStream(handle, stream);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    const int n = ws.n;
    const int nn = n * n;
    const int threads = cuda::kDefaultBlock;
    const int blocks = cuda::grid_size(nn, threads);
    uhf_build_fock_kernel<<<blocks, threads, 0, stream>>>(
        ws.work, beta.work, ws.Hcore, ws.P, beta.P, ws.ERI, n);

    const prec_t one = prec_t(1);
    const prec_t zero = prec_t(0);
    auto to_prime = [&](prec_t* F, prec_t* Fp, prec_t* Hslot) {
        detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, F, n,
                     ws.X, n, &zero, d_temp_, n);
        detail::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, ws.X, n,
                     d_temp_, n, &zero, Fp, n);
        cudaMemcpyAsync(Hslot, Fp, static_cast<size_t>(nn) * sizeof(prec_t),
                        cudaMemcpyDeviceToDevice, stream);
    };
    to_prime(ws.work, ws.F_prime, ws.H);
    to_prime(beta.work, beta.F_prime, beta.H);
    (void)ctx;
}

void UHFFockDevice::destroy(const SolverContext&, SCFWorkspace& ws) {
    ATV_CUDA_CHECK(cudaFree(ws.d_force_gradient));
    ATV_CUDA_CHECK(cudaFree(ws.d_force_Ptotal));
    ATV_CUDA_CHECK(cudaFree(ws.d_force_Wbeta));
    ws.d_force_gradient = ws.d_force_Ptotal = ws.d_force_Wbeta = nullptr;
    if (cublas_handle_) {
        cublasDestroy(static_cast<cublasHandle_t>(cublas_handle_));
        cublas_handle_ = nullptr;
    }
    if (cusolver_handle_) {
        cusolverDnDestroy(static_cast<cusolverDnHandle_t>(cusolver_handle_));
        cusolver_handle_ = nullptr;
    }
    cudaFree(d_temp_);
    d_temp_ = nullptr;
    cudaFree(d_T_);
    d_T_ = nullptr;
    cudaFree(d_V_);
    d_V_ = nullptr;
}

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV

#endif

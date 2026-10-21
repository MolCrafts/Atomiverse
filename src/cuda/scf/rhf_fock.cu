/// @file cuda/scf/rhf_fock.cu
/// @brief internal::RHFFock (host-matrix upload) and RHFFockDevice (GPU
///        integrals). Device Löwdin X uses scale_columns_by_rsqrt_kernel
///        (CUDA skill patterns: coalesced col-major, 32×8 launch, thresh).

#ifdef ATV_HAS_HF

#include "atomiverse/core/gto.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/internal/cart2sph.h"
#include "atomiverse/cpu/scf/internal/cartesian_shell_pack.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/scf/device_integrals.cuh"
#include "atomiverse/cuda/scf/internal/rhf_fock.cuh"
#include "cublas_dispatch.cuh"
#include <type_traits>

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/reducers/xnorm.hpp>
#include <xtensor/views/xview.hpp>

namespace ATV {
namespace cuda {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace scf {
namespace internal {

// Level-shift diagonal kernel lives in diis_mixer.cu (CDIIS owns shift).

// ── Kernel: Löwdin column scale U[:,j] *= rsqrt(λ_j) (or 0 if λ_j ≤ thr) ─
// Skills: fuse-elementwise (scale+rsqrt), optimize-global-memory (col-major
// coalesced on row = threadIdx.x), choose-launch-configuration (32×8 block),
// handle-boundary-conditions (i<n && j<n guards), write-numerically-stable
// (drop near-null overlap directions).
// Layout: U is column-major n×n; lambda length n.
// After this, X = U_scaled * U^T implements U diag(λ^{-1/2}) U^T without a
// host-side diagonal matrix.
__global__ void scale_columns_by_rsqrt_kernel(prec_t* __restrict__ U,
                                              const prec_t* __restrict__ lambda,
                                              int n, prec_t thresh) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (i >= n || j >= n)
        return;
    // Broadcast λ_j within a column (j fixed along y); row i is stride-1 in
    // col-major so consecutive threadIdx.x load/store contiguous U(i,j).
    const prec_t lam = lambda[j];
    const prec_t s = (lam > thresh) ? rsqrt(lam) : prec_t(0);
    U[i + j * n] *= s;
}

// ── Wave 2: geometry pack without host D2H ─────────────────────────────
// Frame stores SoA positions in Å. Integral kernels want [natm×3] Bohr and
// pure-s shell centers (= nuclear positions). One thread per atom.
// Skills: choose-launch-configuration (1D 256), handle-boundary (a < natm),
// optimize-global-memory (coalesced SoA loads; stores write 3 consecutive
// prec_t per atom — small write stride).
// Pack nuclear positions only; shell centers filled by expand_s_shells_kernel.
__global__ void pack_atom_pos_bohr_kernel(
    prec_t* __restrict__ atom_pos, // [natm * 3] Bohr
    const real_t* __restrict__ x_A, const real_t* __restrict__ y_A,
    const real_t* __restrict__ z_A, int natm, prec_t a2b) {
    const int a = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (a >= natm)
        return;
    const prec_t xb = static_cast<prec_t>(x_A[a]) * a2b;
    const prec_t yb = static_cast<prec_t>(y_A[a]) * a2b;
    const prec_t zb = static_cast<prec_t>(z_A[a]) * a2b;
    const int o = a * 3;
    atom_pos[o + 0] = xb;
    atom_pos[o + 1] = yb;
    atom_pos[o + 2] = zb;
}

// Wave 3: expand per-atom s-shell table to the full molecule shell list.
// One thread per molecular shell s ∈ [0, natm * nsh_atom).
//   atom = s / nsh_atom, ish = s % nsh_atom
//   α/c from alpha_atom[ish * n_prim + p] (table block for one atom)
//   center = atom_pos[atom] (Bohr, already packed)
// Skills: choose-launch-configuration, handle-boundary, optimize-global-memory
// (coalesced along p when n_prim is small — sequential p per thread is fine).
__global__ void expand_s_shells_kernel(prec_t* __restrict__ alpha_out,
                                       prec_t* __restrict__ coeff_out,
                                       prec_t* __restrict__ shell_centers,
                                       const prec_t* __restrict__ alpha_atom,
                                       const prec_t* __restrict__ coeff_atom,
                                       const prec_t* __restrict__ atom_pos,
                                       int natm, int nsh_atom, int n_prim) {
    const int s = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int n_shells = natm * nsh_atom;
    if (s >= n_shells)
        return;
    const int atom = s / nsh_atom;
    const int ish = s % nsh_atom;
    const int base_out = s * n_prim;
    const int base_in = ish * n_prim;
    for (int p = 0; p < n_prim; ++p) {
        alpha_out[base_out + p] = alpha_atom[base_in + p];
        coeff_out[base_out + p] = coeff_atom[base_in + p];
    }
    const int oa = atom * 3;
    const int os = s * 3;
    shell_centers[os + 0] = atom_pos[oa + 0];
    shell_centers[os + 1] = atom_pos[oa + 1];
    shell_centers[os + 2] = atom_pos[oa + 2];
}

// Nuclear repulsion V_nuc = Σ_{a<b} Za Zb / |Ra−Rb| [Hartree].
// One thread per unordered pair; multi-block partials reduced with atomics
// (order-insensitive for the double sum; pair set is exact).
// Skills: choose-launch-configuration, handle-boundary, write-numerically-stable
// (skip r=0).
__global__ void
nuclear_repulsion_pairs_kernel(double* __restrict__ v_nuc_out,
                               const prec_t* __restrict__ atom_pos,
                               const int* __restrict__ atom_Z, int natm) {
    const int npairs = natm * (natm - 1) / 2;
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    double contrib = 0.0;
    if (idx < npairs) {
        // Decode linear pair index → (a,b) with a < b.
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
        if (r2 > 0.0) {
            const double r = sqrt(r2);
            contrib = static_cast<double>(atom_Z[a] * atom_Z[b]) / r;
        }
    }
    // Block reduce then one atomicAdd per block (fewer atomics than per-thread).
    __shared__ double warp_scratch[/* max warps in 256 */ 8];
    const double block_sum =
        cuda::block_reduce_sum<double>(contrib, warp_scratch);
    if (threadIdx.x == 0 && block_sum != 0.0)
        atomicAdd(v_nuc_out, block_sum);
}

// ── Kernel: build G(P) from ERI ──────────────────────────────────────
// G_ij = sum_kl P_kl * [(ij|kl) - 0.5*(ik|jl)]
// P already includes factor 2 from RHF closed-shell occupancy.
// Coalesced P[k+l*n] along k for fixed l. Do NOT early-out on a sticky
// converged flag: callers zero G then add Hcore+G; skipping the write
// yields F=Hcore and corrupts the public energy if the host loop still runs.
__global__ void build_G_kernel(prec_t* __restrict__ G,
                               const prec_t* __restrict__ P,
                               const prec_t* __restrict__ ERI, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    const int n2 = n * n;
    const int n3 = n2 * n;
    for (int ij = idx; ij < total; ij += blockDim.x * gridDim.x) {
        int j = ij / n; // column-major: ij = i + j*n
        int i = ij % n;
        prec_t val = prec_t(0);
        for (int l = 0; l < n; ++l) {
            // ERI[i,j,k,l] col-major: i + j*n + k*n² + l*n³
            const int base_ij_l = i + j * n + l * n3;
            for (int k = 0; k < n; ++k) {
                const int ij_kl = base_ij_l + k * n2;
                const int ik_jl = i + k * n + j * n2 + l * n3;
                val += static_cast<prec_t>(P[k + l * n])
                       * (static_cast<prec_t>(ERI[ij_kl])
                          - prec_t(0.5) * static_cast<prec_t>(ERI[ik_jl]));
            }
        }
        G[ij] = val;
    }
}

void RHFFock::build(const SolverContext& ctx, SCFWorkspace& ws) {
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto& scf = ws;
    int n = n_basis_;
    int nn = n * n;
    int n4 = nn * nn;

    // Set workspace metadata
    scf.n = n;
    scf.n_occ = n_occ_;
    scf.V_nuc = V_nuc_;
    scf.converged = false;

    // Allocate device matrices
    cudaMalloc(&scf.H, nn * sizeof(prec_t));
    cudaMalloc(&scf.P, nn * sizeof(prec_t));
    cudaMalloc(&scf.P_old, nn * sizeof(prec_t));
    cudaMalloc(&scf.C, nn * sizeof(prec_t));
    cudaMalloc(&scf.C_prime, nn * sizeof(prec_t));
    cudaMemsetAsync(scf.C_prime, 0, nn * sizeof(prec_t), stream);
    cudaMalloc(&scf.eig, n * sizeof(prec_t));
    cudaMalloc(&scf.work, nn * sizeof(prec_t));
    cudaMalloc(&scf.Hcore, nn * sizeof(prec_t));
    cudaMalloc(&scf.S, nn * sizeof(prec_t));
    cudaMalloc(&scf.X, nn * sizeof(prec_t));
    cudaMalloc(&scf.F_prime, nn * sizeof(prec_t));
    cudaMalloc(&scf.ERI, n4 * sizeof(prec_t));
    cudaMalloc(&d_temp_, nn * sizeof(prec_t));

    // Zero P
    cudaMemsetAsync(scf.P, 0, nn * sizeof(prec_t), stream);
    cudaMemsetAsync(scf.P_old, 0, nn * sizeof(prec_t), stream);

    // Upload Hcore (row-major xtensor -> column-major device)
    // For symmetric matrices, row-major == column-major, so direct copy works.
    std::vector<prec_t> buf(nn);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            buf[i + j * n] = Hcore_(i, j); // col-major
    ATV_CUDA_CHECK(cudaMemcpy(scf.Hcore, buf.data(), nn * sizeof(prec_t),
                              cudaMemcpyHostToDevice));

    // Upload S for device CDIIS error e = Xᵀ (FPS − SPF) X.
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            buf[i + j * n] = S_(i, j);
    ATV_CUDA_CHECK(cudaMemcpy(scf.S, buf.data(), nn * sizeof(prec_t),
                              cudaMemcpyHostToDevice));

    // Upload ERI (row-major 4D xtensor -> column-major 4D device)
    std::vector<prec_t> eri_buf(n4);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    eri_buf[i + j * n + k * nn + l * nn * n] = ERI_(i, j, k, l);
    ATV_CUDA_CHECK(cudaMemcpy(scf.ERI, eri_buf.data(), n4 * sizeof(prec_t),
                              cudaMemcpyHostToDevice));

    // Compute X = S^{-1/2} on host using xtensor-blas
    auto [s_eig, U] = xt::linalg::eigh(S_);
    xt::xtensor<work_t, 2> s_inv_sqrt = xt::zeros<work_t>({n, n});
    for (int i = 0; i < n; ++i)
        s_inv_sqrt(i, i) = work_t(1) / std::sqrt(s_eig(i));
    auto X_host =
        xt::linalg::dot(xt::linalg::dot(U, s_inv_sqrt), xt::transpose(U));

    // Upload X (col-major)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            buf[i + j * n] = X_host(i, j);
    ATV_CUDA_CHECK(cudaMemcpy(scf.X, buf.data(), nn * sizeof(prec_t),
                              cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    cublas_handle_ = handle;

    cudaStreamSynchronize(stream);
}

void RHFFock::step(const SolverContext& ctx, SCFWorkspace& ws) {
    auto& scf = ws;
    if (scf.converged)
        return;

    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto handle = static_cast<cublasHandle_t>(cublas_handle_);
    int n = scf.n;
    int nn = n * n;

    // Step 1: G(P) from ERI
    // Use work buffer for G
    cudaMemsetAsync(scf.work, 0, nn * sizeof(prec_t), stream);
    int threads = cuda::kDefaultBlock;
    int blocks = cuda::grid_size(nn, threads);
    build_G_kernel<<<blocks, threads, 0, stream>>>(scf.work, scf.P, scf.ERI, n);

    // Step 2: F = Hcore + G (H serves as F)
    cudaMemcpyAsync(scf.H, scf.Hcore, nn * sizeof(prec_t),
                    cudaMemcpyDeviceToDevice, stream);
    prec_t one = prec_t(1);
    detail::axpy(handle, nn, &one, scf.work, 1, scf.H, 1);

    // Step 3: F' = X^T * F * X
    // temp = F * X
    prec_t zero = prec_t(0);
    detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, scf.H, n,
                 scf.X, n, &zero, d_temp_, n);
    // F' = X^T * temp
    detail::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, scf.X, n,
                 d_temp_, n, &zero, scf.F_prime, n);

    // EigenSolver will diagonalize F_prime (stored where H usually goes for toy-scf).
    // Copy F' -> H so EigenSolver picks it up via scf.H
    // Actually, EigenSolver reads from scf.H and writes to scf.C/eig.
    // We need it to read F_prime. Swap: copy F_prime into the slot EigenSolver reads.
    // EigenSolver copies scf.H -> scf.C then diag in-place. So set scf.H = F_prime.
    // But we need scf.H (Fock matrix F) later for energy. Save F in work temporarily.
    // Simpler: F_prime is separate. Copy F_prime -> H for EigenSolver.
    // After EigenSolver runs, H contains F_prime (overwritten by copy to C).
    // The real Fock F is in d_temp_ ... no, d_temp_ was overwritten.
    // Let me just swap: save F somewhere, put F_prime in H.
    // We can use P_old as temp (it's only used later in build_density).
    // Actually, simpler: EigenSolver reads scf.H. So we just overwrite scf.H with F_prime.
    // We need F for energy later → store it in work.
    cudaMemcpyAsync(scf.work, scf.H, nn * sizeof(prec_t),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(scf.H, scf.F_prime, nn * sizeof(prec_t),
                    cudaMemcpyDeviceToDevice, stream);
    // Now: scf.H = F' (seed for EigenSolver), scf.work = unshifted AO F.
    // Device CDIIS (DIISMixer) extrapolates work, rebuilds F', and applies
    // the Saunders–Hillier level shift before diagonalization — matching
    // CPU MixStage order (DIIS on unshifted F, then level shift).
    (void)level_shift_;
    (void)ctx;
}

void RHFFock::destroy(const SolverContext& /*ctx*/, SCFWorkspace& /*ws*/) {
    if (cublas_handle_) {
        cublasDestroy(static_cast<cublasHandle_t>(cublas_handle_));
        cublas_handle_ = nullptr;
    }
    cudaFree(d_temp_);
    d_temp_ = nullptr;
}

} // namespace internal
} // namespace scf

// ══════════════════════════════════════════════════════════════════════
// RHFFockDevice — CUDA-only path: GPU integral kernels, zero CPU integrals
// Wave 4: molecular s+p shell list + launch_*_sp (AO-indexed)
// ══════════════════════════════════════════════════════════════════════

// Expand shell centers from atom_of[s] → atom_pos (Bohr). One thread / shell.
__global__ void expand_shell_centers_kernel(prec_t* __restrict__ shell_centers,
                                            const int* __restrict__ atom_of,
                                            const prec_t* __restrict__ atom_pos,
                                            int n_shells) {
    const int s = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (s >= n_shells)
        return;
    const int a = atom_of[s];
    shell_centers[s * 3 + 0] = atom_pos[a * 3 + 0];
    shell_centers[s * 3 + 1] = atom_pos[a * 3 + 1];
    shell_centers[s * 3 + 2] = atom_pos[a * 3 + 2];
}

namespace scf {
namespace internal {

RHFFockDevice::RHFFockDevice(int n_shells, int n_ao, int n_prim,
                             const double* alpha_mol, const double* coeff_mol,
                             const int* L_mol, const int* atom_of_mol,
                             int n_occ, double level_shift)
    : n_shells_(n_shells), n_ao_(n_ao), n_prim_(n_prim), n_occ_(n_occ),
      level_shift_(level_shift), pure_s_broadcast_(false) {
    const std::size_t nflat =
        static_cast<std::size_t>(n_shells) * static_cast<std::size_t>(n_prim);
    alpha_mol_.assign(alpha_mol, alpha_mol + nflat);
    coeff_mol_.assign(coeff_mol, coeff_mol + nflat);
    L_mol_.assign(L_mol, L_mol + n_shells);
    atom_of_mol_.assign(atom_of_mol, atom_of_mol + n_shells);
}

RHFFockDevice::RHFFockDevice(int n_shells, int n_ao, const double* alpha,
                             const double* coeff, int n_prim_total,
                             const int* nprim, const int* prim_off,
                             const int* L_mol, const int* atom_of_mol,
                             int n_occ, double level_shift, bool spherical)
    : n_shells_(n_shells), n_ao_(n_ao), n_prim_(0), n_occ_(n_occ),
      level_shift_(level_shift), pure_s_broadcast_(false), ragged_(true),
      spherical_(spherical) {
    if (n_prim_total < 1 || nprim == nullptr || prim_off == nullptr)
        throw std::invalid_argument(
            "RHFFockDevice: ragged Cartesian shells need a primitive list");
    alpha_mol_.assign(alpha, alpha + n_prim_total);
    coeff_mol_.assign(coeff, coeff + n_prim_total);
    L_mol_.assign(L_mol, L_mol + n_shells);
    atom_of_mol_.assign(atom_of_mol, atom_of_mol + n_shells);
    nprim_.assign(nprim, nprim + n_shells);
    prim_off_.assign(prim_off, prim_off + n_shells);
    for (int L : L_mol_) {
        if (L < 0 || L > cpu::scf::internal::kCudaCartesianLgt1Max) {
            throw std::invalid_argument(
                std::string("cuda::scf::RHF: missing kernel ")
                + cpu::scf::internal::kMissingKernelEriG
                + " for angular momentum l=" + std::to_string(L));
        }
    }
}

RHFFockDevice::RHFFockDevice(int n_shells_per_atom, int n_prim,
                             const double* alpha_atom, const double* coeff_atom,
                             int n_occ, int natm, double level_shift)
    : n_prim_(n_prim), n_occ_(n_occ), level_shift_(level_shift),
      pure_s_broadcast_(false), n_shells_per_atom_(n_shells_per_atom) {
    // Expand pure-s atom block to molecular shells immediately.
    n_shells_ = natm * n_shells_per_atom;
    n_ao_ = n_shells_; // pure-s: 1 AO / shell
    const std::size_t nflat_atom = static_cast<std::size_t>(n_shells_per_atom)
                                   * static_cast<std::size_t>(n_prim);
    alpha_mol_.resize(static_cast<std::size_t>(n_shells_)
                      * static_cast<std::size_t>(n_prim));
    coeff_mol_.resize(alpha_mol_.size());
    L_mol_.assign(static_cast<std::size_t>(n_shells_), 0);
    atom_of_mol_.resize(static_cast<std::size_t>(n_shells_));
    for (int a = 0; a < natm; ++a) {
        for (int ish = 0; ish < n_shells_per_atom; ++ish) {
            const int s = a * n_shells_per_atom + ish;
            atom_of_mol_[static_cast<std::size_t>(s)] = a;
            for (int p = 0; p < n_prim; ++p) {
                const std::size_t dst =
                    static_cast<std::size_t>(s * n_prim + p);
                const std::size_t src =
                    static_cast<std::size_t>(ish * n_prim + p);
                alpha_mol_[dst] = alpha_atom[src];
                coeff_mol_[dst] = coeff_atom[src];
            }
        }
    }
    (void)nflat_atom;
}

RHFFockDevice::RHFFockDevice(int n_shells_per_atom, int n_prim,
                             const double* alpha_atom, const double* coeff_atom,
                             int n_occ, double level_shift)
    : n_prim_(n_prim), n_occ_(n_occ), level_shift_(level_shift),
      pure_s_broadcast_(true), n_shells_per_atom_(n_shells_per_atom) {
    // Defer molecular expand until build() knows natm.
    const std::size_t nflat = static_cast<std::size_t>(n_shells_per_atom)
                              * static_cast<std::size_t>(n_prim);
    alpha_mol_.assign(alpha_atom, alpha_atom + nflat);
    coeff_mol_.assign(coeff_atom, coeff_atom + nflat);
}

// Cartesian → spherical sandwich. Same algebra as
// IntegralWorkspace::c2s in integrals.cpp (CartToSph::block, row
// rescaled by 1/sqrt(S_ii), column normalized in the Cartesian overlap).
// Four-index order matches the column-major ERI
// (i + j n + k n^2 + l n^3) contracted one index at a time.
void apply_c2s_lgt1(cublasHandle_t handle, cudaStream_t stream,
                    const std::vector<int>& shell_L,
                    const std::vector<int>& ao_start, prec_t* S_cart,
                    prec_t* T_cart, prec_t* V_cart, prec_t* ERI_cart, prec_t* S,
                    prec_t* T, prec_t* V, prec_t* ERI, int n_cart, int n_sph) {
    static_assert(std::is_same<prec_t, double>::value,
                  "spherical sandwich is double precision");
    const std::size_t n2 =
        static_cast<std::size_t>(n_cart) * static_cast<std::size_t>(n_cart);
    std::vector<double> hS(n2);
    cudaMemcpyAsync(hS.data(), S_cart, n2 * sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    auto Sij = [&](int i, int j) {
        return hS[static_cast<std::size_t>(i + j * n_cart)];
    };
    cpu::scf::internal::CartToSph tables;
    std::vector<double> hC(static_cast<std::size_t>(n_cart)
                               * static_cast<std::size_t>(n_sph),
                           0.0);
    int col = 0;
    for (int s = 0; s < static_cast<int>(shell_L.size()); ++s) {
        const int L = shell_L[static_cast<std::size_t>(s)];
        const int ao0 = ao_start[static_cast<std::size_t>(s)];
        const int nc = (L + 1) * (L + 2) / 2;
        const int ns = 2 * L + 1;
        const auto& m = tables.block(L);
        for (int j = 0; j < ns; ++j) {
            std::vector<double> v(static_cast<std::size_t>(nc));
            for (int i = 0; i < nc; ++i) {
                const double diag = Sij(ao0 + i, ao0 + i);
                if (!(diag > 0.0)) {
                    throw std::runtime_error(
                        "cuda::scf::RHF: Cartesian overlap diagonal is not "
                        "positive; cannot build the spherical transform");
                }
                v[static_cast<std::size_t>(i)] = m(i, j) / std::sqrt(diag);
            }
            double nrm2 = 0.0;
            for (int i = 0; i < nc; ++i)
                for (int k = 0; k < nc; ++k)
                    nrm2 += v[static_cast<std::size_t>(i)]
                            * Sij(ao0 + i, ao0 + k)
                            * v[static_cast<std::size_t>(k)];
            if (!(nrm2 > 0.0)) {
                throw std::runtime_error(
                    "cuda::scf::RHF: spherical column norm is not positive");
            }
            const double inv = 1.0 / std::sqrt(nrm2);
            for (int i = 0; i < nc; ++i)
                hC[static_cast<std::size_t>((ao0 + i) + (col + j) * n_cart)] =
                    v[static_cast<std::size_t>(i)] * inv;
        }
        col += ns;
    }
    if (col != n_sph) {
        throw std::runtime_error(
            "cuda::scf::RHF: spherical AO count does not match shell L");
    }
    prec_t* dC = nullptr;
    cudaMalloc(&dC, hC.size() * sizeof(prec_t));
    cudaMemcpyAsync(dC, hC.data(), hC.size() * sizeof(prec_t),
                    cudaMemcpyHostToDevice, stream);
    prec_t* temp = nullptr;
    cudaMalloc(&temp, static_cast<std::size_t>(n_cart)
                          * static_cast<std::size_t>(n_sph) * sizeof(prec_t));
    const prec_t one = 1;
    const prec_t zero = 0;
    auto sand2 = [&](prec_t* src, prec_t* dst) {
        detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_cart, n_sph, n_cart,
                     &one, src, n_cart, dC, n_cart, &zero, temp, n_cart);
        detail::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n_sph, n_sph, n_cart,
                     &one, dC, n_cart, temp, n_cart, &zero, dst, n_sph);
    };
    sand2(S_cart, S);
    sand2(T_cart, T);
    sand2(V_cart, V);

    const int n_c2 = n_cart * n_cart;
    const int n_c3 = n_c2 * n_cart;
    const std::size_t n4c =
        static_cast<std::size_t>(n_c2) * static_cast<std::size_t>(n_c2);
    prec_t* scratch = nullptr;
    cudaMalloc(&scratch, n4c * sizeof(prec_t));
    // Contract i (fastest): (n_sph × n_cart^3) = C^T * ERI.
    detail::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n_sph, n_c3, n_cart, &one,
                 dC, n_cart, ERI_cart, n_cart, &zero, scratch, n_sph);
    // Contract j, one (k,l) block at a time.
    cublasDgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n_sph, n_sph, n_cart, &one, scratch,
        n_sph, static_cast<long long>(n_sph) * n_cart, dC, n_cart, 0, &zero,
        ERI_cart, n_sph, static_cast<long long>(n_sph) * n_sph, n_c2);
    // Contract k, one l block at a time.
    const int ns2 = n_sph * n_sph;
    cublasDgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, ns2, n_sph, n_cart, &one, ERI_cart,
        ns2, static_cast<long long>(ns2) * n_cart, dC, n_cart, 0, &zero,
        scratch, ns2, static_cast<long long>(ns2) * n_sph, n_cart);
    // Contract l into the spherical tensor.
    const int ns3 = ns2 * n_sph;
    detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ns3, n_sph, n_cart, &one,
                 scratch, ns3, dC, n_cart, &zero, ERI, ns3);
    cudaStreamSynchronize(stream);
    cudaFree(temp);
    cudaFree(scratch);
    cudaFree(dC);
}

void RHFFockDevice::build(const SolverContext& ctx, SCFWorkspace& ws) {
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto& scf = ws;
    int natm = ctx.frame.n_atoms();
    int* d_nprim = nullptr;
    int* d_prim_off = nullptr;

    // Pure-s broadcast ctor: expand atom block → molecular shells now.
    if (pure_s_broadcast_) {
        const int nsh_atom = n_shells_per_atom_;
        const std::vector<double> atom_alpha = alpha_mol_;
        const std::vector<double> atom_coeff = coeff_mol_;
        n_shells_ = natm * nsh_atom;
        n_ao_ = n_shells_;
        alpha_mol_.resize(static_cast<std::size_t>(n_shells_)
                          * static_cast<std::size_t>(n_prim_));
        coeff_mol_.resize(alpha_mol_.size());
        L_mol_.assign(static_cast<std::size_t>(n_shells_), 0);
        atom_of_mol_.resize(static_cast<std::size_t>(n_shells_));
        for (int a = 0; a < natm; ++a) {
            for (int ish = 0; ish < nsh_atom; ++ish) {
                const int s = a * nsh_atom + ish;
                atom_of_mol_[static_cast<std::size_t>(s)] = a;
                for (int p = 0; p < n_prim_; ++p) {
                    const std::size_t dst =
                        static_cast<std::size_t>(s * n_prim_ + p);
                    const std::size_t src =
                        static_cast<std::size_t>(ish * n_prim_ + p);
                    alpha_mol_[dst] = atom_alpha[src];
                    coeff_mol_[dst] = atom_coeff[src];
                }
            }
        }
        pure_s_broadcast_ = false; // expanded once per build cycle
    }

    const int n_shells = n_shells_;
    int n_cart = 0;
    int n_sph = 0;
    int max_l = 0;
    for (int s = 0; s < n_shells; ++s) {
        const int L = L_mol_[static_cast<std::size_t>(s)];
        n_cart += (L + 1) * (L + 2) / 2;
        n_sph += 2 * L + 1;
        if (L > max_l)
            max_l = L;
    }
    if (n_cart != n_ao_) {
        throw std::runtime_error(
            "RHFFockDevice::build: n_ao mismatch vs shell L list");
    }
    const int n = spherical_ ? n_sph : n_cart;
    int nn = n * n;
    int n4 = nn * nn;

    scf.n = n;
    scf.n_occ = n_occ_;
    scf.converged = false;
    scf.n_shells = n_shells;
    scf.n_prim = n_prim_;
    scf.natm = natm;
    scf.pure_s_shells = true;
    for (int s = 0; s < n_shells; ++s) {
        if (L_mol_[static_cast<std::size_t>(s)] != 0) {
            scf.pure_s_shells = false;
            break;
        }
    }

    // Host AO maps + phys-normed contraction
    std::vector<int> h_L(static_cast<std::size_t>(n_shells));
    std::vector<int> h_ao_start(static_cast<std::size_t>(n_shells));
    std::vector<int> h_atom_of(static_cast<std::size_t>(n_shells));
    std::vector<int> h_ao_shell(static_cast<std::size_t>(n_cart));
    std::vector<int> h_ao_cart(static_cast<std::size_t>(n_cart));
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
    if (ao != n_cart) {
        throw std::runtime_error(
            "RHFFockDevice::build: Cartesian AO count mismatch vs shell L");
    }

    const std::size_t nflat = ragged_ ? alpha_mol_.size()
                                      : static_cast<std::size_t>(n_shells)
                                            * static_cast<std::size_t>(n_prim_);
    std::vector<prec_t> h_alpha(nflat), h_coeff(nflat);
    for (int s = 0; s < n_shells; ++s) {
        const int L = h_L[static_cast<std::size_t>(s)];
        const int off =
            ragged_ ? prim_off_[static_cast<std::size_t>(s)] : s * n_prim_;
        const int np = ragged_ ? nprim_[static_cast<std::size_t>(s)] : n_prim_;
        for (int p = 0; p < np; ++p) {
            const std::size_t k = static_cast<std::size_t>(off + p);
            const double a = alpha_mol_[k];
            const double c = coeff_mol_[k];
            h_alpha[k] = static_cast<prec_t>(a);
            h_coeff[k] = static_cast<prec_t>(
                c * cpu::scf::internal::contraction_norm(L, a));
        }
    }

    // ── 1. Device geometry + shell SoA ────────────────────────────────
    cudaMalloc(&scf.d_shell_alpha, nflat * sizeof(prec_t));
    cudaMalloc(&scf.d_shell_coeff, nflat * sizeof(prec_t));
    cudaMalloc(&scf.d_shell_center, n_shells * 3 * sizeof(prec_t));
    cudaMalloc(&scf.d_atom_Z, natm * sizeof(int));
    cudaMalloc(&scf.d_atom_pos, natm * 3 * sizeof(prec_t));

    int* d_ao_start = nullptr;
    cudaMalloc(&scf.d_shell_L, n_shells * sizeof(int));
    cudaMalloc(&d_ao_start, n_shells * sizeof(int));
    cudaMalloc(&scf.d_shell_atom, n_shells * sizeof(int));
    cudaMalloc(&scf.d_ao_shell, n_cart * sizeof(int));
    cudaMalloc(&scf.d_ao_cart, n_cart * sizeof(int));

    cudaMemcpyAsync(scf.d_shell_alpha, h_alpha.data(), nflat * sizeof(prec_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(scf.d_shell_coeff, h_coeff.data(), nflat * sizeof(prec_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(scf.d_shell_L, h_L.data(), n_shells * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_ao_start, h_ao_start.data(), n_shells * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(scf.d_shell_atom, h_atom_of.data(), n_shells * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(scf.d_ao_shell, h_ao_shell.data(), n_cart * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(scf.d_ao_cart, h_ao_cart.data(), n_cart * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(scf.d_atom_Z, ctx.frame.atoms.type,
                    static_cast<std::size_t>(natm) * sizeof(int),
                    cudaMemcpyDeviceToDevice, stream);

    const prec_t a2b = static_cast<prec_t>(ATV::units::kAngstromInBohr);
    {
        const int block = cuda::kDefaultBlock;
        const int grid = cuda::grid_size(natm, block);
        pack_atom_pos_bohr_kernel<<<grid, block, 0, stream>>>(
            scf.d_atom_pos, ctx.frame.atoms.x, ctx.frame.atoms.y,
            ctx.frame.atoms.z, natm, a2b);
    }
    {
        const int block = cuda::kDefaultBlock;
        const int grid = cuda::grid_size(n_shells, block);
        expand_shell_centers_kernel<<<grid, block, 0, stream>>>(
            scf.d_shell_center, scf.d_shell_atom, scf.d_atom_pos, n_shells);
    }

    // V_nuc: device-only reduction (Wave 7 — no mid-build D2H/H2D rewrite).
    // Host `scf.V_nuc` is filled post-SCF in RhfScfCycle::finish from d_V_nuc.
    // Energy during SCF reads d_V_nuc on device (RHFDensity).
    cudaMalloc(&scf.d_V_nuc, sizeof(double));
    cudaMemsetAsync(scf.d_V_nuc, 0, sizeof(double), stream);
    if (natm >= 2) {
        const int npairs = natm * (natm - 1) / 2;
        const int block = cuda::kDefaultBlock;
        const int grid = cuda::grid_size(npairs, block);
        nuclear_repulsion_pairs_kernel<<<grid, block, 0, stream>>>(
            scf.d_V_nuc, scf.d_atom_pos, scf.d_atom_Z, natm);
    }
    scf.V_nuc = 0.0;

    // ── 2. Allocate SCF workspace matrices ────────────────────────────
    cudaMalloc(&scf.H, nn * sizeof(prec_t));
    cudaMalloc(&scf.P, nn * sizeof(prec_t));
    cudaMalloc(&scf.P_old, nn * sizeof(prec_t));
    cudaMalloc(&scf.C, nn * sizeof(prec_t));
    cudaMalloc(&scf.C_prime, nn * sizeof(prec_t));
    cudaMemsetAsync(scf.C_prime, 0, nn * sizeof(prec_t), stream);
    cudaMalloc(&scf.eig, n * sizeof(prec_t));
    cudaMalloc(&scf.work, nn * sizeof(prec_t));
    cudaMalloc(&scf.Hcore, nn * sizeof(prec_t));
    cudaMalloc(&scf.S, nn * sizeof(prec_t));
    cudaMalloc(&scf.X, nn * sizeof(prec_t));
    cudaMalloc(&scf.F_prime, nn * sizeof(prec_t));
    cudaMalloc(&scf.ERI, n4 * sizeof(prec_t));
    cudaMalloc(&d_temp_, nn * sizeof(prec_t));
    cudaMalloc(&d_T_, nn * sizeof(prec_t));
    cudaMalloc(&d_V_, nn * sizeof(prec_t));
    // Device control scalars + force W (no mid-SCF scientific D2H).
    cudaMalloc(&scf.d_residual, sizeof(prec_t));
    cudaMalloc(&scf.d_scalar_scratch, sizeof(prec_t));
    // Mapped convergence flag: device writes, host polls without cudaMemcpy.
    {
        int* h_map = nullptr;
        cudaHostAlloc(reinterpret_cast<void**>(&h_map), sizeof(int),
                      cudaHostAllocMapped);
        scf.h_converged_mapped = h_map;
        *scf.h_converged_mapped = 0;
        void* d_ptr = nullptr;
        cudaHostGetDevicePointer(&d_ptr, scf.h_converged_mapped, 0);
        scf.d_converged = static_cast<int*>(d_ptr);
    }
    // Device forces (pure-s and mixed s/p) need W after SCF.
    // Ragged l≤2 energy does not own launch_rhf_forces_lgt1.
    scf.device_sp_shells = !ragged_;
    if (!ragged_) {
        ATV_CUDA_CHECK(cudaMalloc(&scf.d_W, nn * sizeof(prec_t)));
        if constexpr (kMixedPrecision)
            ATV_CUDA_CHECK(cudaMalloc(&scf.d_force_gradient,
                                      std::size_t(natm) * 3 * sizeof(prec_t)));
    }

    cudaMemsetAsync(scf.P, 0, nn * sizeof(prec_t), stream);
    cudaMemsetAsync(scf.P_old, 0, nn * sizeof(prec_t), stream);

    // ── 3. Device integral kernels (s+p MD/OS; pure-s is the l=0 subset) ─
    // Use the unified sp path for all device RHF so pure-s and mixed share
    // one contraction (avoids dual-kernel norm/layout drift).
    if (!ragged_) {
        ATV::cuda::scf::launch_1e_integrals_sp(
            scf.S, d_T_, d_V_, scf.d_shell_alpha, scf.d_shell_coeff,
            scf.d_shell_center, scf.d_shell_L, d_ao_start, scf.d_ao_shell,
            scf.d_ao_cart, scf.d_atom_Z, scf.d_atom_pos, n_shells, n, n_prim_,
            natm, stream);
        ATV::cuda::scf::launch_eri_sp(
            scf.ERI, scf.d_shell_alpha, scf.d_shell_coeff, scf.d_shell_center,
            scf.d_shell_L, d_ao_start, scf.d_ao_shell, scf.d_ao_cart, n_shells,
            n, n_prim_, stream);
    } else {
        cudaMalloc(&d_nprim, n_shells * sizeof(int));
        cudaMalloc(&d_prim_off, n_shells * sizeof(int));
        cudaMemcpyAsync(d_nprim, nprim_.data(), n_shells * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_prim_off, prim_off_.data(), n_shells * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        prec_t* S_tgt = scf.S;
        prec_t* T_tgt = d_T_;
        prec_t* V_tgt = d_V_;
        prec_t* ERI_tgt = scf.ERI;
        prec_t* S_cart_d = nullptr;
        prec_t* T_cart_d = nullptr;
        prec_t* V_cart_d = nullptr;
        prec_t* ERI_cart_d = nullptr;
        if (spherical_) {
            const std::size_t n2c = static_cast<std::size_t>(n_cart)
                                    * static_cast<std::size_t>(n_cart);
            const std::size_t n4c = n2c * n2c;
            cudaMalloc(&S_cart_d, n2c * sizeof(prec_t));
            cudaMalloc(&T_cart_d, n2c * sizeof(prec_t));
            cudaMalloc(&V_cart_d, n2c * sizeof(prec_t));
            cudaMalloc(&ERI_cart_d, n4c * sizeof(prec_t));
            S_tgt = S_cart_d;
            T_tgt = T_cart_d;
            V_tgt = V_cart_d;
            ERI_tgt = ERI_cart_d;
        }
        ATV::cuda::scf::launch_1e_integrals_lgt1(
            S_tgt, T_tgt, V_tgt, scf.d_shell_alpha, scf.d_shell_coeff, d_nprim,
            d_prim_off, scf.d_shell_center, scf.d_shell_L, scf.d_ao_shell,
            scf.d_ao_cart, scf.d_atom_Z, scf.d_atom_pos, n_cart, natm, max_l,
            stream);
        ATV::cuda::scf::launch_eri_lgt1(
            ERI_tgt, scf.d_shell_alpha, scf.d_shell_coeff, d_nprim, d_prim_off,
            scf.d_shell_center, scf.d_shell_L, d_ao_start, n_shells, n_cart,
            max_l, stream);
        if (spherical_) {
            cublasHandle_t handle_sph;
            cublasCreate(&handle_sph);
            cublasSetStream(handle_sph, stream);
            cublas_handle_ = handle_sph;
            apply_c2s_lgt1(handle_sph, stream, h_L, h_ao_start, S_cart_d,
                           T_cart_d, V_cart_d, ERI_cart_d, scf.S, d_T_, d_V_,
                           scf.ERI, n_cart, n);
            cudaFree(S_cart_d);
            cudaFree(T_cart_d);
            cudaFree(V_cart_d);
            cudaFree(ERI_cart_d);
        }
    }

    // ── 4. Hcore = T + V ──────────────────────────────────────────────
    cublasHandle_t handle;
    if (spherical_) {
        handle = static_cast<cublasHandle_t>(cublas_handle_);
    } else {
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
        cublas_handle_ = handle;
    }

    cudaMemcpyAsync(scf.Hcore, d_T_, nn * sizeof(prec_t),
                    cudaMemcpyDeviceToDevice, stream);
    prec_t one = prec_t(1);
    detail::axpy(handle, nn, &one, d_V_, 1, scf.Hcore, 1);

    // ── 5. Löwdin X = S^{-1/2} on device ──────────────────────────────
    cusolverDnHandle_t solver;
    cusolverDnCreate(&solver);
    cusolverDnSetStream(solver, stream);
    cusolver_handle_ = solver;

    // Keep scf.S intact for CDIIS; syevd overwrites a copy in scf.X.
    cudaMemcpyAsync(scf.X, scf.S, nn * sizeof(prec_t), cudaMemcpyDeviceToDevice,
                    stream);

    int lwork = 0;
    prec_t* d_eigenvalues;
    cudaMalloc(&d_eigenvalues, n * sizeof(prec_t));
    int* d_info;
    cudaMalloc(&d_info, sizeof(int));

    detail::syevd_bufferSize(solver, CUSOLVER_EIG_MODE_VECTOR,
                             CUBLAS_FILL_MODE_LOWER, n, scf.X, n, d_eigenvalues,
                             &lwork);
    prec_t* d_work_solver;
    cudaMalloc(&d_work_solver, lwork * sizeof(prec_t));

    detail::syevd(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n,
                  scf.X, n, d_eigenvalues, d_work_solver, lwork, d_info);

    cudaMemcpyAsync(d_temp_, scf.X, nn * sizeof(prec_t),
                    cudaMemcpyDeviceToDevice, stream);
    constexpr int kBx = 32;
    constexpr int kBy = 8;
    dim3 block(kBx, kBy);
    dim3 grid((n + kBx - 1) / kBx, (n + kBy - 1) / kBy);
    const prec_t lindep_thresh = static_cast<prec_t>(1.0e-8);
    scale_columns_by_rsqrt_kernel<<<grid, block, 0, stream>>>(
        d_temp_, d_eigenvalues, n, lindep_thresh);

    prec_t zero = prec_t(0);
    detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &one, d_temp_, n,
                 scf.X, n, &zero, d_V_, n);
    cudaMemcpyAsync(scf.X, d_V_, nn * sizeof(prec_t), cudaMemcpyDeviceToDevice,
                    stream);

    // Free Löwdin / AO-start scratch after stream work (no scientific D2H).
    // cudaFree is device-wide; one stream fence first keeps free after last use.
    // AO maps (L, ao_shell, ao_cart) stay live for device s/p forces after SCF.
    cudaStreamSynchronize(stream);
    cudaFree(d_eigenvalues);
    cudaFree(d_work_solver);
    cudaFree(d_info);
    cudaFree(d_ao_start);
    cudaFree(d_nprim);
    cudaFree(d_prim_off);
    // Wave 7: d_V_nuc stays as the sole nuclear-repulsion source for SCF energy;
    // host scf.V_nuc mirrored only in RhfScfCycle::finish post-pass.
}

void RHFFockDevice::step(const SolverContext& ctx, SCFWorkspace& ws) {
    auto& scf = ws;
    if (scf.converged)
        return;

    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto handle = static_cast<cublasHandle_t>(cublas_handle_);
    int n = scf.n;
    int nn = n * n;

    cudaMemsetAsync(scf.work, 0, nn * sizeof(prec_t), stream);
    int threads = cuda::kDefaultBlock;
    int blocks = cuda::grid_size(nn, threads);
    build_G_kernel<<<blocks, threads, 0, stream>>>(scf.work, scf.P, scf.ERI, n);

    cudaMemcpyAsync(scf.H, scf.Hcore, nn * sizeof(prec_t),
                    cudaMemcpyDeviceToDevice, stream);
    prec_t one = prec_t(1);
    detail::axpy(handle, nn, &one, scf.work, 1, scf.H, 1);

    prec_t zero = prec_t(0);
    detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, scf.H, n,
                 scf.X, n, &zero, d_temp_, n);
    detail::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, scf.X, n,
                 d_temp_, n, &zero, scf.F_prime, n);

    cudaMemcpyAsync(scf.work, scf.H, nn * sizeof(prec_t),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(scf.H, scf.F_prime, nn * sizeof(prec_t),
                    cudaMemcpyDeviceToDevice, stream);
    // Level shift + CDIIS live in DIISMixer (after this step).
    (void)level_shift_;
    (void)ctx;
}

void RHFFockDevice::destroy(const SolverContext& /*ctx*/, SCFWorkspace& ws) {
    ATV_CUDA_CHECK(cudaFree(ws.d_force_gradient));
    ws.d_force_gradient = nullptr;
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

#endif // ATV_HAS_HF
